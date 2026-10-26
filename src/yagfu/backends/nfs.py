"""GNFS backend: msieve polyselect -> cuda-sieve sieving -> msieve postproc.

Ported from bench_factor.py's run_nfs (validated on this hardware), with
the per-size parameter tables generalized by interpolation (params.py).
Pipeline shape:

* polyselect: msieve -np with a wall-clock budget (SIGINT cooperative cap)
  for >= 85 digits, sharded one process per GPU over disjoint
  leading-coefficient ranges; classic base-m polynomial below that (this
  msieve build emits no GPU stage-1 candidates under ~85 digits).
* sieving: cuda-sieve bands with an adaptive width, special-q from alim/2,
  merged into msieve.dat; msieve -nc1 verdict drives the loop.
* postprocessing: -nc2 matbuild, GPU Lanczos -nc2, sqrt -nc3.
"""

from __future__ import annotations

import json
import math
import os
import re
import shutil
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
#: total wall cap for a polyselect: msieve's internal deadline only ends
#: stage 1 (its stage-2 optimization keeps going, and the CPU-seconds
#: estimate does not account for GPU stage-1 speed), so the search is
#: trusted to finish on its own but never past this
POLY_MAX_WAIT_S = 2 * 3600
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
                       timeout: float, devices: tuple[str, ...] = ("0",),
                       beat=None, beat_s: float = 120.0):
    """Split one q band into chunks running concurrently, one GPU each.

    n_procs is per device: chunk i round-robins onto devices[i % len]
    (adjacent q ranges land on different cards), each pinned to its card
    via CUDA_VISIBLE_DEVICES and addressing it as ordinal 0.  devices
    entries are physical ids.

    cmd_base ends with a (q_lo, q_hi) placeholder tuple for --qrange.
    `beat(elapsed_s)` fires every beat_s while the band runs (long bands
    would otherwise be silent for minutes).  Returns (rc, wall,
    [relation file names]).
    """
    q_lo, q_hi = cmd_base[-1]
    base = cmd_base[:-1]
    width = q_hi - q_lo
    k = max(1, len(devices))
    n_procs = max(1, min(n_procs * k, width // 1500 or 1))
    chunk = width // n_procs
    jobs, envs, names = [], [], []
    for i in range(n_procs):
        a = q_lo + i * chunk
        b = q_hi if i == n_procs - 1 else a + chunk
        rels = f"band{round_idx}.{i}.rels"
        names.append(rels)
        cmd = [str(c) for c in base] + ["--qrange", f"{a}:{b}",
                                        "--relations", rels,
                                        "--log", f"band{round_idx}.{i}.log"]
        jobs.append((cmd, cwd / f"band{round_idx}.{i}.stdout", None))
        envs.append({"CUDA_VISIBLE_DEVICES": devices[i % k]})
    results = run_procs_parallel(jobs, cwd, timeout, beat=beat,
                                 beat_s=beat_s, envs=envs)
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


def _poly_coeff_max(n: int) -> int:
    """msieve's default leading-coefficient search bound (poly_skew.c
    find_poly_core): degree picked from the size, then
    coeff_max = N^(1/(degree*(degree-1))) / 3 — same float arithmetic as
    the C, so the values match what a plain -np prints."""
    digits = math.log10(n)
    degree = 4 if digits < 108 else (5 if digits < 220 else 6)
    return int(float(n) ** (1.0 / (degree * (degree - 1))) / 3.0)


def _poly_ranges(n: int, k: int) -> list[tuple[int, int] | None]:
    """Split the default leading-coefficient range into k contiguous
    integer chunks; [None] means "one process, default range".  msieve
    only ever searches multiples of an ad-sieve multiplier and rounds
    each begin up to one, so arbitrary integer boundaries still partition
    the searched set with no gaps and no overlap."""
    if k <= 1:
        return [None]
    top = _poly_coeff_max(n)
    width = top // k
    if width == 0:
        return [None]
    out = []
    for i in range(k):
        lo = 1 if i == 0 else i * width + 1
        hi = (i + 1) * width if i < k - 1 else top
        if lo <= hi:
            out.append((lo, hi))
    return out


class PolyselectJob:
    """msieve -np polynomial search, sharded across the GPUs when there
    are several: each card runs one process over a disjoint slice of the
    leading-coefficient range, and the best-scoring shard wins.

    The winner's msieve.fb/np.log are promoted to the workdir root, so
    run_nfs's downstream stages and the resume paths see exactly the
    single-process layout.  A watcher thread per process enforces the
    total wall cap — cooperative SIGINT first, so msieve writes out its
    best candidate — and the same wall cap buys k-fold coverage, not an
    earlier exit.  Interrupts and timeouts never leave a process alive;
    an interrupted search still leaves the salvaged best-so-far fb for a
    rerun to reuse.
    """

    def __init__(self, n: int, ctx: RunContext, wd: Path,
                 engine: str = "gpu_cubsort"):
        msieve = find_bin("msieve", ctx.bin_dir)
        wd.mkdir(parents=True, exist_ok=True)
        # never inherit a stale poly from an earlier cancelled attempt
        for name in ("msieve.fb", "np.log"):
            try:
                (wd / name).unlink()
            except OSError:
                pass
        self.n, self.ctx, self.wd = n, ctx, wd
        # trunk msieve has a stage-1 engine registry; the GPU cubsort
        # engine is the fast default, the CPU Gerbicz reference is the
        # always-compiled-in fallback a wholly poly-less search retries on
        self.engine = engine
        self._engine_retried = False
        self.budget = poly_budget(len(str(n)), ctx.poly_budget_s)
        # an explicit --poly-budget is the total wall cap; otherwise the
        # estimate above is informational and POLY_MAX_WAIT_S applies
        cap = ctx.poly_budget_s if (
            ctx.poly_budget_s is not None and ctx.poly_budget_s > 0
        ) else POLY_MAX_WAIT_S
        self.fb: dict[str, str] | None = None
        self.poly_e: float | None = None
        self.poly_method: str | None = None
        self.error: str | None = None
        self.wall_s = 0.0
        self._t0 = time.monotonic()
        self._deadline_at = self._t0 + cap
        self._cap_logged = False
        # [(subdir, process)]: one private subdir per sharded card, or
        # the bare workdir for a single-process search
        self._procs: list[tuple[Path, subprocess.Popen]] = []
        self._shard(msieve)
        if self.sharded:
            ctx.log(f"nfs: polyselect sharded across {len(self._procs)} "
                    f"GPUs (leading coeffs 1..{self._coeff_top:,} split "
                    f"{len(self._procs)} ways, -t {self._threads} each, "
                    f"~{fmt_dur(self.budget)} expected, cap {fmt_dur(cap)})")
        else:
            ctx.log(f"nfs: polyselect (msieve -np, ~{fmt_dur(self.budget)} "
                    f"expected, cap {fmt_dur(cap)})")
        self._watch()

    def _shard(self, msieve) -> None:
        """Spawn one msieve -np per card: the default coefficient range
        split k ways when sharding, whole and unpinned-from-args on a lone
        card (exact single-process behavior).  A sharded process gets a
        fresh private subdir — msieve reads msieve.fb *before* searching,
        and a leftover file for the same N would skip the search."""
        k = len(self.ctx.devices) if self.engine == "gpu_cubsort" else 1
        rng = _poly_ranges(self.n, k)
        self._procs = []
        self._cards = list(zip(self.ctx.devices[:len(rng)], rng))
        self.sharded = len(rng) > 1
        self._coeff_top = _poly_coeff_max(self.n)
        # GPU engines cap their producer pool at 4 anyway; -t also sizes
        # the stage-2/3 CPU pool, so sharded processes divide the cores
        self._threads = (str(self.ctx.threads) if len(rng) == 1 else
                         str(max(2, self.ctx.threads // len(rng))))
        for i, (dev, r) in enumerate(self._cards):
            d = self.wd if r is None else self.wd / f"poly-{i}"
            if r is not None:
                shutil.rmtree(d, ignore_errors=True)
                d.mkdir(parents=True)
            # one nfs_args token, space-separated: a comma anywhere would
            # trip msieve's legacy "X,Y" coefficient-range parse
            args = f"stage1_engine={self.engine}"
            if r is not None:
                args += f" min_coeff={r[0]} max_coeff={r[1]}"
            cmd = [msieve, "-np", args, "-t", self._threads, "-v",
                   str(self.n)]
            f = open(d / "np.log", "wb")
            try:
                # msieve -np has no device flag; each process sees exactly
                # its card as ordinal 0 via CUDA_VISIBLE_DEVICES
                p = subprocess.Popen([str(c) for c in cmd],
                                     cwd=str(d),
                                     stdout=f, stderr=subprocess.STDOUT,
                                     env={**os.environ,
                                          **self.ctx.pin_env(dev)})
            finally:
                f.close()
            self._procs.append((d, p))

    def engine_retry(self) -> bool:
        """A GPU search where NO card produced a polynomial gets one retry
        on the CPU Gerbicz engine — always compiled in, and msieve's own
        default.  An unlucky empty range slice is not retried: with other
        shards holding a polynomial, CPU hours on the empty slice buy
        nothing.  The wall cap restarts; True when a process runs."""
        if self._engine_retried or self.engine == "cpu_gerbicz":
            return False
        if any(p.poll() is None for _, p in self._procs):
            return False                          # still running
        if self.sharded:
            empty = [d.name for d, _ in self._procs
                     if not (d / "msieve.fb").exists()]
            if len(empty) < len(self._procs):
                if empty:
                    self.ctx.log(f"nfs: polyselect shard(s) {', '.join(empty)}"
                                 " found no polynomial; keeping the others'")
                return False
        elif (self.wd / "msieve.fb").exists():
            return False
        self._engine_retried = True
        self.engine = "cpu_gerbicz"
        self._t0 = time.monotonic()
        self._deadline_at = self._t0 + POLY_MAX_WAIT_S
        self._cap_logged = False
        self.ctx.log("nfs: GPU polyselect found nothing; "
                     "retrying on the CPU Gerbicz engine")
        self._shard(find_bin("msieve", self.ctx.bin_dir))
        self._watch()
        return True

    def _watch(self) -> None:
        """Enforce the total wall cap on every -np process."""
        for _, p in self._procs:
            threading.Thread(target=self._deadline_watch, args=(p,),
                             daemon=True).start()

    def _deadline_watch(self, p) -> None:
        """At the cap, SIGINT msieve so it saves its best-so-far polynomial
        (msieve.fb) on the way out; escalate only if it hangs."""
        try:
            p.wait(timeout=max(self._deadline_at - time.monotonic(), 0.0))
            return
        except subprocess.TimeoutExpired:
            pass
        if p.poll() is None:
            if not self._cap_logged:
                self._cap_logged = True
                self.ctx.log(f"nfs: polyselect hit its wall cap "
                             f"({fmt_dur(time.monotonic() - self._t0)}); "
                             f"interrupting for the best-so-far polynomial")
            p.send_signal(signal.SIGINT)
            # the cooperative unwind can be slow: a single stage-2 root
            # optimization at c13x sizes runs for minutes, and killing
            # mid-unwind discards the whole search (measured: a 2h search
            # lost to a 90s grace)
            try:
                p.wait(timeout=600)
            except subprocess.TimeoutExpired:
                self._term_kill(p)

    def _term_kill(self, p) -> None:
        if p.poll() is None:
            p.terminate()
            try:
                p.wait(timeout=10)
            except subprocess.TimeoutExpired:
                p.kill()
                p.wait()

    def wait(self, timeout: float | None = None) -> "PolyselectJob":
        """Block until every process ends (the wall-cap watchers SIGINT
        them at the deadline so each saves its best-so-far polynomial),
        merge the shards, and only a search where no card produced a
        polynomial retries once on the CPU engine.  Any exception (Ctrl+C
        included) tears the processes down and salvages the best-so-far
        polynomial before propagating."""
        try:
            self._join(timeout)
            if self.sharded:
                self._merge()
            if self.engine_retry():
                self._join(timeout)
            self._finish()
        except BaseException:
            self.cancel(drop_poly=False)
            raise
        return self

    def _join(self, timeout: float | None) -> None:
        for _, p in self._procs:
            try:
                p.wait(timeout=timeout or NP_TIMEOUT_S)
            except subprocess.TimeoutExpired:
                self._term_kill(p)

    def _merge(self) -> None:
        """Promote the best-scoring shard result to the workdir root: the
        polynomial (msieve.fb) plus its log (np.log carries the
        combined-E line the finish/resume paths parse).  Also called from
        cancel(): msieve writes its best on SIGTERM, so an interrupted
        search still salvages the best-so-far polynomial."""
        scored = []
        for d, _ in self._procs:
            if d == self.wd or not (d / "msieve.fb").exists():
                continue
            scored.append((_poly_e_from_log(d) or -1.0, d))
        if not scored:
            return
        _, best = max(scored)
        es = ", ".join(f"{e:g}" if e >= 0 else "?" for e, _ in scored)
        self.ctx.log(f"nfs: polyselect shards E = [{es}]; best from "
                     f"{best.name}")
        shutil.copyfile(best / "msieve.fb", self.wd / "msieve.fb")
        shutil.copyfile(best / "np.log", self.wd / "np.log")

    def _finish(self) -> None:
        self.wall_s = time.monotonic() - self._t0
        fb_path = self.wd / "msieve.fb"
        if not fb_path.exists():
            self.error = "polyselect failed (np.log)"
            return
        self.fb = _parse_msieve_fb(read_log(fb_path))
        self.poly_method = "msieve -np"
        m = re.search(r"combined = ([\d.e+-]+)",
                      read_log(self.wd / "np.log"))
        if m:
            self.poly_e = float(m.group(1))

    def cancel(self, drop_poly: bool = True) -> None:
        """Stop every process and drop partial artifacts; idempotent.

        drop_poly=False keeps the merged best-so-far msieve.fb on disk:
        an interrupted run then resumes with the polynomial instead of
        re-searching (msieve.fb holds msieve's best confirmed candidate —
        a mid-run interrupt leaves a usable, if not yet optimal,
        polynomial behind)."""
        for _, p in self._procs:
            if p.poll() is None:
                self._term_kill(p)
        if self.sharded:
            self._merge()
        if all(p.poll() is not None for _, p in self._procs) \
                and drop_poly and self.fb is None:
            for name in ("msieve.fb", "np.log"):
                try:
                    (self.wd / name).unlink()
                except OSError:
                    pass
            for d in self.wd.glob("poly-*"):
                shutil.rmtree(d, ignore_errors=True)


def _append_band_rels(wd: Path, relnames: list[str]) -> None:
    """Append one round's band files into msieve.dat, then delete them:
    relations live in exactly one copy on disk."""
    with open(wd / "msieve.dat", "ab") as out:
        for f in relnames:
            out.write((wd / f).read_bytes())
    for f in relnames:
        try:
            (wd / f).unlink()
        except OSError:
            pass


_NFS_STATE = "nfs-state.json"


def _save_nfs_state(wd: Path, n: int, **kv) -> None:
    kv["n"] = str(n)
    (wd / _NFS_STATE).write_text(json.dumps(kv))


def _load_nfs_state(wd: Path, n: int) -> dict | None:
    """Checkpoint of the sieve loop for this exact n, or None."""
    p = wd / _NFS_STATE
    if not p.exists():
        return None
    try:
        s = json.loads(p.read_text())
    except (OSError, ValueError):
        return None
    if s.get("n") != str(n) or "q_next" not in s:
        return None
    return s


def _poly_e_from_log(wd: Path) -> float | None:
    """Best-polynomial score from a polyselect log, if present."""
    p = wd / "np.log"
    if not p.exists():
        return None
    m = re.search(r"combined = ([\d.e+-]+)", read_log(p))
    return float(m.group(1)) if m else None


def run_nfs(n: int, ctx: RunContext, wd: Path) -> NfsResult:
    """Factor n with GNFS inside wd (polyselect runs here when nothing
    reusable is on disk)."""
    res = NfsResult(params=nfs_params(len(str(n))))
    msieve = find_bin("msieve", ctx.bin_dir)
    cuda_sieve = find_bin("cuda-sieve", ctx.bin_dir)
    wd.mkdir(parents=True, exist_ok=True)
    digits = len(str(n))
    nproc = str(ctx.threads)
    rlim, alim, lpb, logI, initial_raw = res.params
    ctx.log(f"nfs: c{digits}, rlim={rlim:,}, alim={alim:,}, lpb={lpb}, "
            f"logI={logI}, target_raw~{initial_raw:,}")

    # ---- stage 1: polynomial selection (or reuse from a previous run) ----
    t = time.monotonic()
    job_path = wd / f"c{digits}.job"
    reused = False
    if job_path.exists():
        job_text = job_path.read_text()
        if f"n: {n}" in job_text:
            reused = True
            res.poly_method = "msieve -np (resumed)"
            e = _poly_e_from_log(wd)
            if e is not None:
                res.poly_e = e
            ctx.log("nfs: reusing polynomial from workdir"
                    + (f" (E={res.poly_e:g})" if res.poly_e else ""))
    if not reused and digits >= NFS_MSIEVE_POLY_MIN_DIGITS:
        fb = None
        if (wd / "msieve.fb").exists():
            # polyselect finished (or was budget-capped) but the job file
            # never got written — the fb is still usable
            cand = _parse_msieve_fb(read_log(wd / "msieve.fb"))
            if cand.get("N") == str(n):
                fb = cand
                res.poly_method = "msieve -np (resumed)"
                e = _poly_e_from_log(wd)
                if e is not None:
                    res.poly_e = e
                ctx.log("nfs: reusing msieve.fb from workdir"
                        + (f" (E={res.poly_e:g})" if res.poly_e else ""))
        if fb is None:
            poly = PolyselectJob(n, ctx, wd)
            poly.wait()
            if poly.fb is None:
                res.error = poly.error
                ctx.log(f"nfs: {res.error}")
                return res
            fb = poly.fb
            res.poly_e, res.poly_method = poly.poly_e, poly.poly_method
            if res.poly_e is not None:
                ctx.log(f"nfs: poly E={res.poly_e:g} "
                        f"(msieve -np, {fmt_dur(poly.wall_s)})")
            else:
                ctx.log("nfs: msieve -np poly")
    if not reused and digits < NFS_MSIEVE_POLY_MIN_DIGITS:
        degree = 3 if digits < 80 else 4
        coeffs, mm = base_m_poly(n, degree)
        fb = {"N": str(n),
              "SKEW": f"{(abs(coeffs[0]) / abs(coeffs[_degree(coeffs)])) ** 0.25:.2f}",
              "R0": str(-mm), "R1": "1"}
        for i, c in enumerate(coeffs):
            fb[f"A{i}"] = str(c)
        res.poly_method = f"base-m deg-{degree}"
        ctx.log(f"nfs: base-m degree-{degree} poly (m={mm})")
    res.stages["polyselect_s"] = 0.0 if reused else time.monotonic() - t

    # ---- stage 2: job file + msieve.fb/worktodo.ini ---------------------
    if not reused:
        job_text = _fb_to_job(fb, rlim, alim, lpb)
        job_path.write_text(job_text)
    fb_text, ini_text = _job_to_msieve_fb(job_text)
    (wd / "msieve.fb").write_text(fb_text)
    (wd / "worktodo.ini").write_text(ini_text)

    # ---- stage 3: sieving + filtering feedback loop ----------------------
    # resumable: the polynomial (job file) and every sieved relation
    # (msieve.dat) survive a kill; nfs-state.json pins q_next/target/have
    state = _load_nfs_state(wd, n)
    if state is not None and not (wd / "msieve.dat").exists():
        state = None                # checkpoint without its relations: redo
    q_first = alim // 2
    if state is None:
        q_next = q_first
        q_cap = 6 * q_first
        target_raw = initial_raw
        have_raw = 0
        round_idx = 0
        bands_count = 0
        last_cnt = last_width = 0
        try:
            (wd / "msieve.dat").unlink()   # stateless leftovers: restart
        except OSError:
            pass
    else:
        q_next = int(state["q_next"])
        q_cap = int(state.get("q_cap", 6 * q_first))
        target_raw = int(state["target_raw"])
        have_raw = int(state["have_raw"])
        round_idx = int(state.get("round_idx", 0))
        bands_count = int(state.get("bands", 0))
        last_cnt = int(state.get("last_cnt", 0))
        last_width = int(state.get("last_width", 0))
        ctx.log(f"nfs: resuming: {have_raw:,} relations in msieve.dat "
                f"(round {round_idx}, q from {q_next:,}, "
                f"target {target_raw:,})")
    while True:
        ctx.check_deadline()
        round_idx += 1
        if bands_count:
            per_int = max(last_cnt / last_width, 1e-9)
            need = target_raw - have_raw
            width = int(need / per_int * 1.05) + 2000
        else:
            width = 30_000  # probe band
        qhi = min(q_next + width, q_cap)
        if qhi <= q_next:
            q_cap += 2 * q_first  # relax the duplicate-control window
            qhi = min(q_next + width, q_cap)
            ctx.log(f"nfs: q window exhausted, cap relaxed to {q_cap}")
        ctx.log(f"nfs: round {round_idx}{' (probe)' if not bands_count else ''}: "
                f"q {q_next}:{qhi}"
                + (f" (have {have_raw:,}, target {target_raw:,})"
                   if bands_count else "")
                + (f" [{ctx.sieve_procs} proc x {len(ctx.devices)} gpu]"
                   if len(ctx.devices) > 1 else ""))
        base = [cuda_sieve, "--pipeline", "--cofactor",
                "--poly", job_path.name, "--logI", logI, "--region",
                min(logI, 14), "--device", 0, (q_next, qhi)]
        for stale in wd.glob(f"band{round_idx}.*"):
            # a killed run may have left this round's partial outputs; the
            # same round index repeats on resume, so never append to them
            stale.unlink(missing_ok=True)
        try:
            rc, wall, relnames = _run_band_parallel(
                base, wd, ctx.sieve_procs, round_idx, SIEVE_TIMEOUT_S,
                ctx.devices)
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
        _append_band_rels(wd, relnames)     # one copy on disk, then delete
        last_cnt, last_width = cnt, qhi - q_next
        have_raw += cnt
        q_next = qhi
        bands_count += 1
        _save_nfs_state(wd, n, q_next=q_next, q_cap=q_cap,
                        target_raw=target_raw, have_raw=have_raw,
                        round_idx=round_idx, bands=bands_count,
                        last_cnt=last_cnt, last_width=last_width)
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
            _save_nfs_state(wd, n, q_next=q_next, q_cap=q_cap,
                            target_raw=target_raw, have_raw=have_raw,
                            round_idx=round_idx, bands=bands_count,
                            last_cnt=last_cnt, last_width=last_width)
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
        env_extra=ctx.pin_env(),
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
    res.bands = bands_count
    ctx.log(f"nfs: complete, {len(res.factors)} factor(s), "
            f"{bands_count} sieve band(s), {have_raw:,} relations")
    return res
