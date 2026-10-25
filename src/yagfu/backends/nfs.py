"""GNFS backend: msieve polyselect -> cuda-sieve sieving -> msieve postproc.

Ported from bench_factor.py's run_nfs (validated on this hardware), with
the per-size parameter tables generalized by interpolation (params.py).
Pipeline shape:

* polyselect: msieve -np with a wall-clock budget (SIGINT cooperative cap)
  for >= 85 digits; classic base-m polynomial below that (this msieve
  build emits no GPU stage-1 candidates under ~85 digits).
* sieving: cuda-sieve bands with an adaptive width, special-q from alim/2,
  merged into msieve.dat; msieve -nc1 verdict drives the loop.
* postprocessing: -nc2 matbuild, GPU Lanczos -nc2, sqrt -nc3.
"""

from __future__ import annotations

import math
import re
import signal
import subprocess
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path

from ..config import RunContext
from ..fmt import fmt_count, fmt_dur
from ..params import (
    NFS_MSIEVE_POLY_MIN_DIGITS,
    nfs_params,
    poly_budget,
)
from . import find_bin
from .runner import (
    count_lines,
    read_log,
    run_cmd,
    run_cmd_progress,
    run_procs_parallel,
)

NP_TIMEOUT_S = 2 * 3600
SIEVE_TIMEOUT_S = 6 * 3600
NC_TIMEOUT_S = 2 * 3600
LA_TIMEOUT_S = 12 * 3600
SQRT_TIMEOUT_S = 2 * 3600


# ---------------------------------------------------------------------------
# polynomials
# ---------------------------------------------------------------------------

def _degree(coeffs: list[int]) -> int:
    for i in range(len(coeffs) - 1, -1, -1):
        if coeffs[i]:
            return i
    return 0


def _iroot(n: int, k: int) -> int:
    x = 1 << ((n.bit_length() + k - 1) // k + 1)
    while True:
        y = ((k - 1) * x + n // x ** (k - 1)) // k
        if y >= x:
            break
        x = y
    while x ** k > n:
        x -= 1
    return x


def base_m_poly(n: int, degree: int) -> tuple[list[int], int]:
    """Classic base-m polynomial: write n in base m ~ n^(1/degree).

    Tries a handful of m near the exact root and keeps the one whose
    coefficient profile is flattest under the natural skew.
    """
    best = None
    m0 = _iroot(n, degree)
    for m in range(m0 - 2, m0 + 3):
        if m < 2:
            continue
        coeffs, t = [], n
        while t:
            coeffs.append(t % m)
            t //= m
        if len(coeffs) > degree + 1:
            continue
        coeffs += [0] * (degree + 1 - len(coeffs))
        if coeffs[degree] == 0:
            continue
        skew = (abs(coeffs[0]) / abs(coeffs[degree])) ** (1.0 / degree)
        spread = max(math.log2(max(1, abs(c) * skew ** (degree - i)))
                     for i, c in enumerate(coeffs))
        if best is None or spread < best[0]:
            best = (spread, coeffs, m)
    _, coeffs, m = best
    return coeffs, m


def _parse_msieve_fb(text: str) -> dict[str, str]:
    fb = {}
    for line in text.splitlines():
        parts = line.split()
        if len(parts) == 2:
            fb[parts[0]] = parts[1]
    return fb


def _fb_to_job(fb: dict[str, str], rlim: int, alim: int, lpb: int) -> str:
    deg = max(int(k[1:]) for k in fb if k[0] == "A" and k[1:].isdigit())
    lines = [f"n: {fb['N']}", f"skew: {fb['SKEW']}"]
    for i in range(deg, -1, -1):
        lines.append(f"c{i}: {fb.get('A' + str(i), '0')}")
    lines += [f"Y0: {fb['R0']}", f"Y1: {fb['R1']}"]
    lines += [
        f"rlim: {rlim}", f"alim: {alim}",
        f"lpbr: {lpb}", f"lpba: {lpb}",
        f"mfbr: {2 * lpb}", f"mfba: {2 * lpb}",
        "rlambda: 2.5", "alambda: 2.5",
    ]
    return "\n".join(lines) + "\n"


def _job_to_msieve_fb(job_text: str) -> tuple[str, str]:
    """GGNFS .job -> (msieve.fb text, worktodo.ini text)."""
    kv: dict[str, str] = {}
    deg = 0
    for line in job_text.splitlines():
        parts = line.split()
        if len(parts) == 2:
            kv[parts[0].rstrip(":")] = parts[1]
    for k in kv:
        m = re.fullmatch(r"c(\d+)", k)
        if m:
            deg = max(deg, int(m.group(1)))
    fb = [f"N {kv['n']}", f"SKEW {kv['skew']}", f"R0 {kv['Y0']}", f"R1 {kv['Y1']}"]
    for i in range(deg + 1):
        fb.append(f"A{i} {kv.get('c' + str(i), '0')}")
    fb += [f"FRMAX {kv['rlim']}", f"FAMAX {kv['alim']}",
           f"SRLPMAX {1 << int(kv['lpbr'])}", f"SALPMAX {1 << int(kv['lpba'])}"]
    return "\n".join(fb) + "\n", kv["n"] + "\n"


# ---------------------------------------------------------------------------
# sieving helpers
# ---------------------------------------------------------------------------

def _run_band_parallel(cmd_base, cwd: Path, n_procs: int, round_idx: int,
                       timeout: float, beat=None, beat_s: float = 120.0):
    """Split one q band into n_procs chunks running concurrently on the GPU.

    cmd_base ends with a (q_lo, q_hi) placeholder tuple for --qrange.
    `beat(elapsed_s)` fires every beat_s while the band runs (long bands
    would otherwise be silent for minutes).  Returns (rc, wall,
    [relation file names]).
    """
    q_lo, q_hi = cmd_base[-1]
    base = cmd_base[:-1]
    width = q_hi - q_lo
    n_procs = max(1, min(n_procs, width // 1500 or 1))
    chunk = width // n_procs
    jobs, names = [], []
    for i in range(n_procs):
        a = q_lo + i * chunk
        b = q_hi if i == n_procs - 1 else a + chunk
        rels = f"band{round_idx}.{i}.rels"
        names.append(rels)
        cmd = [str(c) for c in base] + ["--qrange", f"{a}:{b}",
                                        "--relations", rels,
                                        "--log", f"band{round_idx}.{i}.log"]
        jobs.append((cmd, cwd / f"band{round_idx}.{i}.stdout", None))
    results = run_procs_parallel(jobs, cwd, timeout, beat=beat,
                                 beat_s=beat_s)
    rc = max(rc for rc, _ in results)
    wall = max(w for _, w in results)
    return rc, wall, names


# ---------------------------------------------------------------------------
# the pipeline
# ---------------------------------------------------------------------------

@dataclass
class NfsResult:
    factors: list[int] = field(default_factory=list)
    ok: bool = False
    stages: dict[str, float] = field(default_factory=dict)
    poly_e: float | None = None
    poly_method: str | None = None
    params: tuple = ()               # (rlim, alim, lpb, logI, initial_raw)
    raw_relations: int = 0
    bands: int = 0
    error: str | None = None


class PolyselectJob:
    """msieve -np running in the background while GPU ECM continues.

    The engine starts one as soon as a number is known to be NFS-bound;
    run_nfs later consumes the finished job instead of running its own
    polyselect.  A watcher thread enforces the wall-clock budget
    (cooperative SIGINT, then terminate, then kill); Ctrl+C, timeouts and
    factor-found cancels never leave the msieve process alive.
    """

    def __init__(self, n: int, ctx: RunContext, wd: Path,
                 background: bool = True):
        msieve = find_bin("msieve", ctx.bin_dir)
        wd.mkdir(parents=True, exist_ok=True)
        # never inherit a stale poly from an earlier cancelled attempt
        for name in ("msieve.fb", "np.log"):
            try:
                (wd / name).unlink()
            except OSError:
                pass
        self.n, self.ctx, self.wd = n, ctx, wd
        self.budget = poly_budget(len(str(n)), ctx.poly_budget_s)
        self.fb: dict[str, str] | None = None
        self.poly_e: float | None = None
        self.poly_method: str | None = None
        self.error: str | None = None
        self.wall_s = 0.0
        self._retried = False
        self._t0 = time.monotonic()
        self._spawn(msieve)
        if background:
            ctx.log(f"nfs: polyselect started in background "
                    f"(budget {fmt_dur(self.budget)})")
        else:
            ctx.log(f"nfs: polyselect (msieve -np, "
                    f"budget {fmt_dur(self.budget)})")
        if self.budget > 0:
            threading.Thread(target=self._budget_watch, daemon=True).start()

    def _spawn(self, msieve) -> None:
        cmd = [msieve, "-np", "-t", str(self.ctx.threads), "-v", str(self.n)]
        f = open(self.wd / "np.log", "wb")
        try:
            self._p = subprocess.Popen([str(c) for c in cmd],
                                       cwd=str(self.wd),
                                       stdout=f, stderr=subprocess.STDOUT)
        finally:
            f.close()

    def _budget_watch(self) -> None:
        p = self._p
        try:
            p.wait(timeout=self.budget)
            return
        except subprocess.TimeoutExpired:
            pass
        if p.poll() is None:
            p.send_signal(signal.SIGINT)
            try:
                p.wait(timeout=90)
            except subprocess.TimeoutExpired:
                self._term_kill()

    def _term_kill(self) -> None:
        p = self._p
        if p.poll() is None:
            p.terminate()
            try:
                p.wait(timeout=10)
            except subprocess.TimeoutExpired:
                p.kill()
                p.wait()

    def running(self) -> bool:
        return self._p.poll() is None

    def wait(self, timeout: float | None = None) -> "PolyselectJob":
        """Block until polyselect is done (including the uncapped retry
        when the budget expired before any candidate was saved)."""
        try:
            self._p.wait(timeout=timeout or NP_TIMEOUT_S)
        except subprocess.TimeoutExpired:
            self._term_kill()
            self._finish()
            return self
        if (self.budget > 0 and not self._retried
                and not (self.wd / "msieve.fb").exists()):
            # the SIGINT arrived before any candidate was saved
            self._retried = True
            self._spawn(find_bin("msieve", self.ctx.bin_dir))
            try:
                self._p.wait(timeout=NP_TIMEOUT_S)
            except subprocess.TimeoutExpired:
                self._term_kill()
        self._finish()
        return self

    def _finish(self) -> None:
        self.wall_s = time.monotonic() - self._t0
        fb_path = self.wd / "msieve.fb"
        if not fb_path.exists():
            self.error = "polyselect failed (np.log)"
            return
        self.fb = _parse_msieve_fb(read_log(fb_path))
        self.poly_method = "msieve -np"
        m = re.search(r"combined = ([\d.e+-]+)", read_log(self.wd / "np.log"))
        if m:
            self.poly_e = float(m.group(1))

    def cancel(self) -> None:
        """Stop the polyselect (unless it already produced a polynomial)
        and drop partial artifacts; idempotent."""
        p = getattr(self, "_p", None)
        if p is not None and p.poll() is None:
            self._term_kill()
        if p is not None and p.poll() is not None and self.fb is None:
            for name in ("msieve.fb", "np.log"):
                try:
                    (self.wd / name).unlink()
                except OSError:
                    pass


def run_nfs(n: int, ctx: RunContext, wd: Path,
            poly: PolyselectJob | None = None) -> NfsResult:
    """Factor n with GNFS inside wd.

    `poly` is a PolyselectJob the engine started while ECM was still
    running; its (already computed) polynomial is used instead of running
    polyselect again."""
    res = NfsResult(params=nfs_params(len(str(n))))
    msieve = find_bin("msieve", ctx.bin_dir)
    cuda_sieve = find_bin("cuda-sieve", ctx.bin_dir)
    wd.mkdir(parents=True, exist_ok=True)
    digits = len(str(n))
    nproc = str(ctx.threads)
    rlim, alim, lpb, logI, initial_raw = res.params
    ctx.log(f"nfs: c{digits}, rlim={rlim:,}, alim={alim:,}, lpb={lpb}, "
            f"logI={logI}, target_raw~{initial_raw:,}")

    # ---- stage 1: polynomial selection ---------------------------------
    t = time.monotonic()
    if digits >= NFS_MSIEVE_POLY_MIN_DIGITS:
        if poly is None:
            poly = PolyselectJob(n, ctx, wd, background=False)
        poly.wait()
        res.stages["polyselect_s"] = poly.wall_s
        if poly.fb is None:
            res.error = poly.error
            ctx.log(f"nfs: {res.error}")
            return res
        fb = poly.fb
        res.poly_e, res.poly_method = poly.poly_e, poly.poly_method
        if poly.poly_e is not None:
            ctx.log(f"nfs: poly E={poly.poly_e:g} "
                    f"(msieve -np, {fmt_dur(poly.wall_s)})")
        else:
            ctx.log(f"nfs: msieve -np poly ({fmt_dur(poly.wall_s)})")
    else:
        degree = 3 if digits < 80 else 4
        coeffs, mm = base_m_poly(n, degree)
        fb = {"N": str(n),
              "SKEW": f"{(abs(coeffs[0]) / abs(coeffs[_degree(coeffs)])) ** 0.25:.2f}",
              "R0": str(-mm), "R1": "1"}
        for i, c in enumerate(coeffs):
            fb[f"A{i}"] = str(c)
        res.poly_method = f"base-m deg-{degree}"
        ctx.log(f"nfs: base-m degree-{degree} poly (m={mm})")
        res.stages["polyselect_s"] = time.monotonic() - t


    # ---- stage 2: job file + msieve.fb/worktodo.ini ---------------------
    job_text = _fb_to_job(fb, rlim, alim, lpb)
    job_path = wd / f"c{digits}.job"
    job_path.write_text(job_text)
    fb_text, ini_text = _job_to_msieve_fb(job_text)
    (wd / "msieve.fb").write_text(fb_text)
    (wd / "worktodo.ini").write_text(ini_text)

    # ---- stage 3: sieving + filtering feedback loop ----------------------
    bands: list[tuple[int, int, list[str], int, float]] = []
    q_next = alim // 2
    q_first = q_next
    q_cap = 6 * q_first
    target_raw = initial_raw
    have_raw = 0
    round_idx = 0
    while True:
        ctx.check_deadline()
        round_idx += 1
        if bands:
            qlo_, qhi_, _, cnt, _ = bands[-1]
            per_int = max(cnt / (qhi_ - qlo_), 1e-9)
            need = target_raw - have_raw
            width = int(need / per_int * 1.05) + 2000
        else:
            width = 30_000  # probe band
        qhi = min(q_next + width, q_cap)
        if qhi <= q_next:
            q_cap += 2 * q_first  # relax the duplicate-control window
            qhi = min(q_next + width, q_cap)
            ctx.log(f"nfs: q window exhausted, cap relaxed to {q_cap}")
        ctx.log(f"nfs: round {round_idx}{' (probe)' if not bands else ''}: "
                f"q {q_next}:{qhi}"
                + (f" (have {have_raw:,}, target {target_raw:,})"
                   if bands else ""))
        base = [cuda_sieve, "--pipeline", "--cofactor",
                "--poly", job_path.name, "--logI", logI, "--region",
                min(logI, 14), "--device", ctx.device, (q_next, qhi)]
        try:
            rc, wall, relnames = _run_band_parallel(
                base, wd, ctx.sieve_procs, round_idx, SIEVE_TIMEOUT_S,
                beat=lambda el: ctx.log(f"nfs: sieving round {round_idx}, "
                                        f"{fmt_dur(el)} elapsed"))
        except subprocess.TimeoutExpired:
            res.error = "sieve band timed out"
            ctx.log(f"nfs: {res.error}")
            return res
        res.stages["sieve_s"] = res.stages.get("sieve_s", 0.0) + wall
        if rc != 0:
            res.error = f"sieve band {round_idx} failed (rc={rc})"
            ctx.log(f"nfs: {res.error}")
            return res
        cnt = sum(count_lines(wd / f) for f in relnames)
        bands.append((q_next, qhi, relnames, cnt, 0.0))
        have_raw += cnt
        q_next = qhi
        rate = cnt / wall if wall > 0 else 0.0
        if rate > 0 and target_raw > have_raw:
            ctx.log(f"nfs: +{cnt:,} rels in {fmt_dur(wall)} "
                    f"({fmt_count(rate)}/s); have {fmt_count(have_raw)} / "
                    f"target {fmt_count(target_raw)}, "
                    f"ETA {fmt_dur((target_raw - have_raw) / rate)}")
        elif rate > 0:
            ctx.log(f"nfs: +{cnt:,} rels in {fmt_dur(wall)} "
                    f"({fmt_count(rate)}/s); have {fmt_count(have_raw)}, "
                    f"filtering")

        with open(wd / "msieve.dat", "wb") as out:
            for _, _, files, _, _ in bands:
                for f in files:
                    out.write((wd / f).read_bytes())
        t = time.monotonic()
        rc, _ = run_cmd([msieve, "-s", "msieve.dat", "-l",
                         f"nc1-r{round_idx}.log", "-nc1", "-t", nproc, "-v"],
                        wd, wd / f"nc1-r{round_idx}.stdout",
                        timeout=NC_TIMEOUT_S)
        res.stages["filter_s"] = res.stages.get("filter_s", 0.0) \
            + time.monotonic() - t
        out_text = read_log(wd / f"nc1-r{round_idx}.stdout")
        m = re.search(r"found (\d+) cycles, need (\d+)", out_text)
        if m and int(m.group(1)) >= int(m.group(2)) \
                and "too few cycles" not in out_text:
            break
        m = re.search(r"filtering wants (\d+) more relations", out_text)
        if m and round_idx < 10:
            deficit = int(m.group(1))
            # msieve's hint is a floor; cycles grow superlinearly — grow the
            # target geometrically (YAFU's filt-bump analog)
            target_raw = have_raw + max(int(deficit * 1.2),
                                        int(have_raw * ctx.topup_frac))
            continue
        res.error = "filtering did not converge"
        ctx.log(f"nfs: {res.error}")
        return res

    # ---- stage 4: matrix build + Lanczos + sqrt ---------------------------
    t = time.monotonic()
    run_cmd([msieve, "-s", "msieve.dat", "-l", "nc2-mat.log",
             "-nc2", "only_matbuild=1", "-t", nproc, "-v"],
            wd, wd / "nc2-mat.stdout", timeout=NC_TIMEOUT_S)
    res.stages["matbuild_s"] = time.monotonic() - t
    t = time.monotonic()
    # msieve prints its own "linear algebra completed ... (X%, ETA ...)"
    # updates (carriage-return paced) while it runs; forward them live —
    # throttled to one per minute — instead of sitting silent for hours
    last_la = [0.0]

    def _la_lines(lines):
        for l in lines:
            if "linear algebra" not in l:
                continue
            if l.startswith("commencing") or time.monotonic() - last_la[0] >= 60:
                ctx.log(f"nfs: {l.strip()}")
                last_la[0] = time.monotonic()

    rc, _ = run_cmd_progress(
        [msieve, "-s", "msieve.dat", "-l", "nc2-la.log",
         "-nc2", "-t", nproc, "-v"], wd, wd / "nc2-la.stdout",
        timeout=LA_TIMEOUT_S, every_s=30,
        env_extra={"CUDA_VISIBLE_DEVICES": ctx.device},
        on_lines=_la_lines)
    res.stages["linalg_s"] = time.monotonic() - t
    if rc == -9:
        res.error = "linear algebra timed out"
        ctx.log(f"nfs: {res.error}")
        return res
    if rc != 0:
        res.error = f"linear algebra failed (rc={rc})"
        ctx.log(f"nfs: {res.error}")
        return res
    t = time.monotonic()
    try:
        rc, _ = run_cmd([msieve, "-s", "msieve.dat", "-l", "nc3.log",
                         "-nc3", "-t", nproc, "-v"], wd, wd / "nc3.stdout",
                        timeout=SQRT_TIMEOUT_S)
    except subprocess.TimeoutExpired:
        res.error = "sqrt timed out"
        ctx.log(f"nfs: {res.error}")
        return res
    res.stages["sqrt_s"] = time.monotonic() - t
    log3 = read_log(wd / "nc3.stdout")
    factors = [int(x) for x in re.findall(r"p\d+ factor: (\d+)", log3)]
    if not factors:
        factors = [int(x) for x in re.findall(r"Found [a-z]+ factor of \d+ digits: (\d+)", log3)]
    res.factors = [f for f in factors if n % f == 0]
    res.ok = bool(res.factors)
    res.raw_relations = have_raw
    res.bands = len(bands)
    ctx.log(f"nfs: complete, {len(res.factors)} factor(s), "
            f"{len(bands)} sieve band(s), {have_raw:,} relations")
    return res
