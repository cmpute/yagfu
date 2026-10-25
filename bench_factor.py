#!/usr/bin/env python3
"""bench_factor.py — GPU integer-factorization method benchmark for yagfu.

Measures end-to-end wall time of the two GPU factorization pipelines staged
in bin/, on semiprimes of a chosen digit count:

  mpqs  bin/cuda-mpqs                    SIQS/MPQS, single process, all stages
  nfs   bin/msieve (polyselect) +        GNFS: polyselect -> cuda-sieve
        bin/cuda-sieve (sieving) +          relation collection -> msieve
        bin/msieve (post-processing)       filtering / linear algebra / sqrt

Usage:
  python3 bench_factor.py gen  --digits 70,80,90,100 [--seed 20261025]
  python3 bench_factor.py run  --method mpqs --digits 70,80,90,100
  python3 bench_factor.py run  --method nfs  --digits 70,80,90,100
  python3 bench_factor.py run  --method all  --digits 70,80,90,100
  python3 bench_factor.py report
  python3 bench_factor.py show  --digits 90 --method nfs

Notes
-----
* Semiprimes are p*q with p,q of half the digit count (e.g. c100 = p50*q50),
  generated with a fixed seed so runs are reproducible (benchmarks/numbers.json).
* NFS polynomial selection uses msieve -np (GPU stage 1) where its parameter
  table covers the size (>= ~85 digits).  Below that this msieve build finds
  no stage-1 candidates (its deg-4 table starts at 80 "digits" = log10(n) >= 80
  and the GPU lattice search emits nothing), so for smaller inputs the tool
  falls back to a classic base-m degree-4 polynomial with a tiny local search.
  This is recorded per-run in the results as poly_method.
* NFS relation targets start from a per-size table and are then grown in a
  feedback loop: msieve -nc1 literally prints "filtering wants N more
  relations", and the tool sieves that many more (x1.2) in the next q band.
* Everything runs sequentially on one GPU (--device), so the two methods get
  the machine to themselves, comparably.

Only dependency beyond the stdlib: gmpy2 (used by the yagfu driver anyway).
"""

from __future__ import annotations

import argparse
import json
import math
import re
import shutil
import signal
import subprocess
import sys
import time
from pathlib import Path

try:
    import gmpy2
    from gmpy2 import mpz
except ImportError:  # pragma: no cover
    print("error: gmpy2 is required (pip install gmpy2)", file=sys.stderr)
    sys.exit(1)

# ---------------------------------------------------------------------------
# layout
# ---------------------------------------------------------------------------

REPO = Path(__file__).resolve().parent
BENCH_DIR = REPO / "benchmarks"
NUMBERS_JSON = BENCH_DIR / "numbers.json"
RESULTS_JSON = BENCH_DIR / "results.json"

BIN = REPO / "bin"
MSIEVE = BIN / "msieve"
CUDA_SIEVE = BIN / "cuda-sieve"
CUDA_MPQS = BIN / "cuda-mpqs"
JOB_TO_MSIEVE = REPO / "cuda-sieve" / "job_to_msieve.sh"  # informational; conversion is done in Python

NPROC = max(1, (shutil.which("nproc") and int(subprocess.run(["nproc"], capture_output=True, text=True).stdout.strip() or 1)) or 1)

# ---------------------------------------------------------------------------
# per-size configuration
# ---------------------------------------------------------------------------

# SIQS: cuda-mpqs auto-picks factor-base bound F from bit length; the user
# guide says "always enable large primes for inputs > 80 digits, starting
# bound 10*F..100*F".  We pin lp1 = 100 * F_auto (F_auto from the USER_GUIDE
# table) and leave everything else at defaults.
MPQS_LP1 = {70: None, 80: 150_000_000, 90: 300_000_000, 100: 300_000_000}

# NFS sieve parameters per digit count.  2LP both sides (mfb = 2*lpb),
# equal lims, special-q on the algebraic side (bench default), q band starts
# at alim/2.  initial_raw is the first sieving target in raw relations; the
# msieve feedback loop tops it up as needed.
NFS_PARAMS = {
    # digits: (rlim, alim, lpb, logI, initial_raw)
    70:  (350_000, 350_000, 23, 12,   800_000),
    80:  (800_000, 800_000, 24, 13, 1_600_000),
    90:  (1_500_000, 1_500_000, 26, 13, 8_000_000),
    100: (3_000_000, 3_000_000, 27, 14, 15_000_000),
}
# msieve -np's own deg-4 table starts at log10(n) = 80; the GPU stage-1 search
# has been verified to emit candidates at 90/100 digits but none at 70/80.
NFS_MSIEVE_POLY_MIN_DIGITS = 85

# ---- polyselect budget heuristic (YAFU-style time budget, GPU-recalibrated) --
#
# msieve's built-in polyselect deadlines are calibrated for CPU sieving and are
# wildly oversized once cuda-sieve does the sieving: at c100 msieve's table
# allows 12 minutes while the whole GPU sieve takes ~1 minute.  YAFU's rule is
# 8% of the *predicted total GNFS time*; with everything but sieving measured
# in seconds here, that degenerates to ~8% of the predicted SIEVE time.
#
#   T_sieve(d) ~= 59s * 3^((d - 100)/10)
#
# anchored on our measured runs (c90: 31 s, c100: 59 s) and consistent with the
# cuda-sieve RUNBOOK's complete-band figures once their heavier lpb is folded
# in (c123 ~18 min, c151 ~6.5 h).  Marginal analysis agrees: with E ~ t^0.09
# (measured at c100: -2.4% E for -23% search time; the classic "a few % per
# doubling"), minimizing  T_poly + T_sieve(T_poly)  puts the optimum near 9-10%
# of the realized sieve time.
#
# The floor exists because polyselect cannot be arbitrarily fast: stage 1
# exhausts the leading-coefficient space in ~8 s at c100 and the CPU root
# optimization of the found candidates drains after the deadline (measured
# ~60-130 s at c100).  YAFU's quality-bar early exit ("avg" = 1.036 * E0) is
# NOT ported: this msieve fork prints an e0 baseline that our best measured
# polys only reach 0.81-0.90 of, so that gate would never trigger; the E/e0
# ratio is recorded per-run as an advisory quality metric instead.
POLY_BETA = 0.08          # fraction of predicted sieve time
POLY_FLOOR_S = 60.0       # seconds; stage-1 space + rootopt drain minimum


def sieve_time_estimate(digits):
    """Predicted cuda-sieve wall time (seconds) for a digits-digit GNFS job."""
    return 59.0 * 3.0 ** ((digits - 100) / 10.0)


def poly_budget(digits, measured_sieve_s=None, override=None):
    if override is not None:            # explicit --poly-budget; 0 = msieve default
        return override
    t = POLY_BETA * (measured_sieve_s if measured_sieve_s is not None
                     else sieve_time_estimate(digits))
    return max(POLY_FLOOR_S, t)


def _last_measured_sieve_s(digits):
    """Adaptive arm: reuse our own measured sieve time at this size, if any."""
    if not RESULTS_JSON.exists():
        return None
    for r in json.loads(RESULTS_JSON.read_text()):
        if (r.get("method") == "nfs" and r.get("digits") == digits
                and r.get("stages", {}).get("sieve_s")):
            return r["stages"]["sieve_s"]
    return None


def fmt_secs(x: float) -> str:
    if x is None:
        return "-"
    if x < 90:
        return f"{x:.1f}s"
    m, s = divmod(int(round(x)), 60)
    if m < 90:
        return f"{m}m{s:02d}s"
    h, m = divmod(m, 60)
    return f"{h}h{m:02d}m{s:02d}s"


def run_cmd(cmd, cwd, log_path, env_extra=None, timeout=None):
    """Run a subprocess, capturing merged output to log_path; return (rc, wall)."""
    import os
    env = dict(os.environ)
    if env_extra:
        env.update(env_extra)
    t0 = time.monotonic()
    with open(log_path, "wb") as f:
        p = subprocess.run(
            [str(c) for c in cmd], cwd=str(cwd), stdout=f,
            stderr=subprocess.STDOUT, env=env, timeout=timeout,
        )
    return p.returncode, time.monotonic() - t0


def run_cmd_wall_cap(cmd, cwd, log_path, cap_s, grace_s=90, timeout=3600):
    """Run a subprocess under an outer WALL-CLOCK cap, enforced cooperatively.

    msieve's poly_deadline counts CPU-seconds inside the stage-1 lattice
    sieve, which barely advances while the GPU does the sieving — under GPU
    stage 1 it effectively never binds, so the wall time is controlled here
    instead.  On cap expiry we send SIGINT: msieve's documented cooperative
    stop ("press Ctrl-C to pause") makes -np exit gracefully, keeping the
    best polynomial found so far (verified: 60 s cap at c100 -> exit 0 at
    62 s with a valid msieve.fb).  SIGTERM/SIGKILL escalate only if the
    graceful window is missed.
    """
    import os
    t0 = time.monotonic()
    with open(log_path, "wb") as f:
        p = subprocess.Popen([str(c) for c in cmd], cwd=str(cwd),
                             stdout=f, stderr=subprocess.STDOUT)
        try:
            p.wait(timeout=cap_s)
        except subprocess.TimeoutExpired:
            p.send_signal(signal.SIGINT)
            try:
                p.wait(timeout=grace_s)
            except subprocess.TimeoutExpired:
                p.terminate()
                try:
                    p.wait(timeout=15)
                except subprocess.TimeoutExpired:
                    p.kill()
                    p.wait()
    if time.monotonic() - t0 > timeout:
        return -9, time.monotonic() - t0
    return p.returncode, time.monotonic() - t0


def count_lines(path: Path) -> int:
    """Fast line count via wc -l (relation files reach gigabytes)."""
    out = subprocess.run(["wc", "-l", str(path)], capture_output=True, text=True)
    return int(out.stdout.split()[0])


def run_band_parallel(cmd_base, cwd, n_procs, round_idx, timeout):
    """Split one sieve band into n_procs contiguous q chunks and run them
    concurrently on the same GPU (the per-card multi-process shape the
    production cuda-sieve recipe uses).  cmd_base's --qrange value is a
    (q_lo, q_hi) placeholder tuple; each process gets its own chunk,
    --relations and log files.  Returns (rc, wall, [relation file names])."""
    q_lo, q_hi = cmd_base[-1]          # placeholder carried as a tuple
    base = cmd_base[:-1]
    width = q_hi - q_lo
    n_procs = max(1, min(n_procs, width // 1500 or 1))
    chunk = width // n_procs
    ranges = []
    for i in range(n_procs):
        a = q_lo + i * chunk
        b = q_hi if i == n_procs - 1 else a + chunk
        ranges.append((a, b))
    import os
    t0 = time.monotonic()
    procs, names = [], []
    for i, (a, b) in enumerate(ranges):
        rels = f"band{round_idx}.{i}.rels"
        names.append(rels)
        log = open(cwd / f"band{round_idx}.{i}.stdout", "wb")
        p = subprocess.Popen(
            [str(c) for c in base] + ["--qrange", f"{a}:{b}",
                                      "--relations", rels,
                                      "--log", f"band{round_idx}.{i}.log"],
            cwd=str(cwd), stdout=log, stderr=subprocess.STDOUT)
        procs.append(p)
    deadline = t0 + timeout
    for p in procs:
        p.wait(timeout=max(1.0, deadline - time.monotonic()))
    rc = max(p.returncode for p in procs)
    return rc, time.monotonic() - t0, names


# ---------------------------------------------------------------------------
# semiprime generation
# ---------------------------------------------------------------------------

def cmd_gen(args):
    import random
    digits_list = [int(d) for d in args.digits.split(",")]
    rng = random.Random(args.seed)
    numbers = {}
    for digits in digits_list:
        half = digits // 2
        while True:
            p = _rand_prime(rng, half)
            q = _rand_prime(rng, half)
            if p == q:
                continue
            n = p * q
            if len(str(n)) == digits:
                break
        numbers[str(digits)] = {"n": str(n), "p": str(p), "q": str(q)}
        print(f"c{digits}: N = {n}")
    BENCH_DIR.mkdir(exist_ok=True)
    NUMBERS_JSON.write_text(json.dumps({"seed": args.seed, "numbers": numbers}, indent=1))
    print(f"wrote {NUMBERS_JSON}")


def _rand_prime(rng, digits):
    from gmpy2 import next_prime
    while True:
        lo, hi = 10 ** (digits - 1), 10 ** digits - 1
        start = rng.randrange(lo | 1, hi, 2)
        p = next_prime(mpz(start))
        if len(str(p)) == digits:
            return p


def load_number(digits):
    if not NUMBERS_JSON.exists():
        sys.exit(f"error: {NUMBERS_JSON} missing; run `bench_factor.py gen` first")
    data = json.loads(NUMBERS_JSON.read_text())
    if str(digits) not in data["numbers"]:
        sys.exit(f"error: c{digits} not generated; run `gen --digits {digits}`")
    return data["numbers"][str(digits)]


# ---------------------------------------------------------------------------
# method 1: cuda-mpqs (SIQS)
# ---------------------------------------------------------------------------

def run_mpqs(digits: int, cfg) -> dict:
    entry = load_number(digits)
    n = entry["n"]
    wd = BENCH_DIR / f"mpqs-c{digits}"
    if wd.exists():
        shutil.rmtree(wd)
    (wd / "work").mkdir(parents=True)

    cmd = [CUDA_MPQS, "--N", n, "--device", cfg.device, "--dir", "work", "--verbose"]
    lp1 = MPQS_LP1.get(digits)
    if lp1:
        cmd += ["--lp1_bound", str(lp1)]
    rc, wall = run_cmd(cmd, wd, wd / "run.log", timeout=cfg.mpqs_timeout)
    log = (wd / "run.log").read_text(errors="replace")

    factors = sorted(int(m) for m in re.findall(r"Factor:\s+(\d+)", log))
    ok = (rc == 0 and "SUCCESS: Factorization Verified" in log
          and len(factors) == 2 and mpz(factors[0]) * mpz(factors[1]) == mpz(n)
          and gmpy2.is_prime(mpz(factors[0])) and gmpy2.is_prime(mpz(factors[1])))

    internal = {}
    m = re.search(r"Duration:\s+([\d.]+)ms", log)
    if m:
        internal["sieve_s"] = float(m.group(1)) / 1000
    m = re.search(r"Construction Completed in ([\d.]+)ms", log)
    if m:
        internal["matrix_s"] = float(m.group(1)) / 1000
    m = re.search(r"Total \(All\)\s*:\s*([\d.]+)ms", log)
    if m:
        internal["linalg_s"] = float(m.group(1)) / 1000
    m = re.search(r"Total:\s+([\d.]+)\s*s", log)
    if m:
        internal["pipeline_total_s"] = float(m.group(1))
    m = re.search(r"Selected F=(\d+), M=(\d+)", log)
    if m:
        internal["F"], internal["M"] = int(m.group(1)), int(m.group(2))

    return {
        "method": "mpqs", "digits": digits, "ok": ok, "rc": rc,
        "wall_s": wall, "lp1_bound": lp1, "internal": internal,
        "factors": [str(f) for f in factors],
        "workdir": str(wd.relative_to(REPO)),
        "ts": time.strftime("%Y-%m-%d %H:%M:%S"),
    }


# ---------------------------------------------------------------------------
# method 2: NFS — msieve polyselect + cuda-sieve + msieve postprocessing
# ---------------------------------------------------------------------------

def base_m_poly(n: mpz, degree: int = 4):
    """Classic base-m polynomial: write n in base m, m ~ n^(1/degree).

    Returns (coeffs c_degree..c0, m).  Tries a handful of m near the exact
    integer root and keeps the one whose coefficient size profile is flattest
    under the natural skew |c0/c_deg|^(1/degree).
    """
    best = None
    m0 = int(gmpy2.iroot(n, degree)[0])   # exact floor(n^(1/degree))
    for m in range(m0 - 2, m0 + 3):
        if m < 2:
            continue
        coeffs, t = [], n
        while t:
            coeffs.append(int(t % m))
            t //= m
        if len(coeffs) > degree + 1:
            continue                      # m too small -> higher-degree poly
        coeffs += [0] * (degree + 1 - len(coeffs))
        if coeffs[degree] == 0:
            continue                      # m too large -> lower-degree fit
        skew = (abs(coeffs[0]) / abs(coeffs[degree])) ** (1.0 / degree)
        spread = max(math.log2(max(1, abs(c) * skew ** (degree - i)))
                     for i, c in enumerate(coeffs))
        if best is None or spread < best[0]:
            best = (spread, coeffs, m)
    _, coeffs, m = best
    return coeffs, m


def fb_to_job(fb: dict, params) -> str:
    """msieve.fb dict (N/SKEW/R0/R1/A0..An) -> GGNFS .job text with sieve params."""
    deg = max(int(k[1:]) for k in fb if k[0] == "A" and k[1:].isdigit())
    lines = [f"n: {fb['N']}", f"skew: {fb['SKEW']}"]
    for i in range(deg, -1, -1):
        lines.append(f"c{i}: {fb.get('A' + str(i), '0')}")
    lines += [f"Y0: {fb['R0']}", f"Y1: {fb['R1']}"]
    rlim, alim, lpb = params
    lines += [
        f"rlim: {rlim}", f"alim: {alim}",
        f"lpbr: {lpb}", f"lpba: {lpb}",
        f"mfbr: {2 * lpb}", f"mfba: {2 * lpb}",
        "rlambda: 2.5", "alambda: 2.5",
    ]
    return "\n".join(lines) + "\n"


def job_to_msieve_fb(job_text: str):
    """Python port of cuda-sieve/job_to_msieve.sh: .job text -> (msieve.fb, worktodo.ini)."""
    kv = {}
    deg = 0
    for line in job_text.splitlines():
        parts = line.split()
        if len(parts) == 2:
            kv[parts[0].rstrip(":")] = parts[1]
    for k, v in kv.items():
        m = re.fullmatch(r"c(\d+)", k)
        if m:
            deg = max(deg, int(m.group(1)))
    sr_lpmax = str(1 << int(kv["lpbr"]))
    sa_lpmax = str(1 << int(kv["lpba"]))
    fb = [f"N {kv['n']}", f"SKEW {kv['skew']}", f"R0 {kv['Y0']}", f"R1 {kv['Y1']}"]
    for i in range(deg + 1):
        fb.append(f"A{i} {kv.get('c' + str(i), '0')}")
    fb += [f"FRMAX {kv['rlim']}", f"FAMAX {kv['alim']}",
           f"SRLPMAX {sr_lpmax}", f"SALPMAX {sa_lpmax}"]
    return "\n".join(fb) + "\n", kv["n"] + "\n"


def run_nfs(digits: int, cfg) -> dict:
    entry = load_number(digits)
    n = entry["n"]
    wd = BENCH_DIR / f"nfs-c{digits}"
    if wd.exists():
        shutil.rmtree(wd)
    wd.mkdir(parents=True)
    (wd / "n.txt").write_text(n + "\n")

    rlim, alim, lpb, logI, initial_raw = NFS_PARAMS[digits]
    params3 = (rlim, alim, lpb)
    res = {
        "method": "nfs", "digits": digits, "ok": False, "rc": None,
        "wall_s": 0.0, "stages": {}, "workdir": str(wd.relative_to(REPO)),
        "lpb": lpb, "rlim": rlim, "alim": alim, "logI": logI,
        "sieve_procs": getattr(cfg, "sieve_procs", 1) or 1,
        "ts": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    t_total0 = time.monotonic()

    # ---- stage 1: polynomial selection ------------------------------------
    if digits >= NFS_MSIEVE_POLY_MIN_DIGITS:
        budget = poly_budget(digits, _last_measured_sieve_s(digits),
                             override=getattr(cfg, "poly_budget", None))
        res["poly_budget_s"] = budget
        cmd = [MSIEVE, "-np", "-t", NPROC, "-v", n]
        if budget > 0:
            # wall cap enforced by the outer layer (see run_cmd_wall_cap);
            # msieve's own poly_deadline is kept as a large backstop only
            rc, wall = run_cmd_wall_cap(cmd, wd, wd / "np.log",
                                        cap_s=budget, timeout=cfg.np_timeout)
            # SIGINT before any candidate is saved leaves no msieve.fb;
            # retry uncapped rather than fail the whole run
            if not (wd / "msieve.fb").exists():
                rc, wall2 = run_cmd(cmd, wd, wd / "np.log", timeout=cfg.np_timeout)
                wall += wall2
        else:
            rc, wall = run_cmd(cmd, wd, wd / "np.log", timeout=cfg.np_timeout)
        res["stages"]["polyselect_s"] = wall
        res["poly_method"] = (f"msieve -np (wall budget {int(budget)}s, SIGINT-capped)"
                              if budget > 0 else "msieve -np (msieve default deadline)")
        if rc != 0 or not (wd / "msieve.fb").exists():
            res["rc"] = rc
            res["error"] = "polyselect failed; see np.log"
            return _finish_nfs(res, t_total0, wd)
        fb = {}
        for line in (wd / "msieve.fb").read_text().splitlines():
            parts = line.split()
            if len(parts) == 2:
                fb[parts[0]] = parts[1]
        np_log = (wd / "np.log").read_text(errors="replace")
        m = re.search(r"combined = ([\d.e+-]+)", np_log)
        if m:
            res["poly_e"] = float(m.group(1))
        m = re.search(r"expecting poly E from ([\d.e+-]+) to", np_log)
        if m and res.get("poly_e"):
            res["poly_e0"] = float(m.group(1))
            res["poly_quality"] = round(res["poly_e"] / res["poly_e0"], 3)
    else:
        t0 = time.monotonic()
        degree = 3 if digits < 80 else 4   # classic deg3->deg4 GNFS crossover
        coeffs, mm = base_m_poly(mpz(n), degree)
        fb = {
            "N": n,
            "SKEW": f"{(abs(coeffs[0]) / abs(coeffs[degree_(coeffs)])) ** 0.25:.2f}",
            "R0": str(-mm), "R1": "1",
        }
        for i, c in enumerate(coeffs):
            fb[f"A{i}"] = str(c)
        res["poly_method"] = f"base-m (m={mm})"
        res["stages"]["polyselect_s"] = time.monotonic() - t0

    # ---- stage 2: job file + msieve.fb/worktodo.ini ------------------------
    job_text = fb_to_job(fb, params3)
    (wd / f"c{digits}.job").write_text(job_text)
    fb_text, ini_text = job_to_msieve_fb(job_text)
    (wd / "msieve.fb").write_text(fb_text)
    (wd / "worktodo.ini").write_text(ini_text)

    # ---- stage 3: sieving + filtering feedback loop ------------------------
    # YAFU-shaped closed loop: probe band measures the yield (rels per q-range
    # integer), each later band is sized to the current deficit from the
    # measured rate, and msieve's filter verdict (found >= needed cycles) is
    # the only stop criterion.  On filter failure the target grows
    # geometrically (msieve's "wants N more" hint is a fixed floor while
    # cycles grow superlinearly near the singleton-cascade knee; c90 measured
    # "wants 1M more" at 4.57M raw but converged only at 7.37M).
    bands = []          # list of (qlo, qhi, [rels files], count, wall)
    q_next = alim // 2              # CADO-convention special-q start
    q_first = q_next
    q_cap = 6 * q_first             # duplicate-control window (RUNBOOK rule)
    target_raw = initial_raw
    have_raw = 0
    round_idx = 0
    while True:
        round_idx += 1
        # width: aim for the deficit using the measured rels/integer of the
        # last band; the first band is a small probe that also serves to
        # measure the rate.
        if bands:
            qlo_, qhi_, _, cnt, _ = bands[-1]
            per_int = max(cnt / (qhi_ - qlo_), 1e-9)
            need = target_raw - have_raw
            width = int(need / per_int * 1.05) + 2000
        else:
            width = 30_000
        qhi = min(q_next + width, q_cap)
        if qhi <= q_next:
            # ran out of duplicate-control window; relax it rather than loop
            q_cap += 2 * q_first
            qhi = min(q_next + width, q_cap)
            print(f"    [nfs c{digits}] q window exhausted, relaxed cap to {q_cap}")
        if bands:
            # YAFU-style ETA: projected q where the current target is met
            est_q_end = q_next + (target_raw - have_raw) / max(per_int, 1e-9)
            print(f"    [nfs c{digits}] round {round_idx}: q {q_next}:{qhi} "
                  f"(have {have_raw:,}, target {target_raw:,}, "
                  f"est. q_end ~{est_q_end:,.0f})", flush=True)
        base = [CUDA_SIEVE, "--pipeline", "--cofactor",
                "--poly", f"c{digits}.job", "--logI", logI, "--region", min(logI, 14),
                "--device", cfg.device, (q_next, qhi)]   # qrange placeholder
        rc, wall, relnames = run_band_parallel(
            base, wd, getattr(cfg, "sieve_procs", 1) or 1, round_idx,
            cfg.sieve_timeout)
        res["stages"].setdefault("sieve_s", 0.0)
        res["stages"]["sieve_s"] += wall
        if rc != 0:
            res["rc"] = rc
            res["error"] = f"sieve band {round_idx} failed"
            return _finish_nfs(res, t_total0, wd)
        cnt = sum(count_lines(wd / f) for f in relnames)
        bands.append((q_next, qhi, relnames, cnt, wall))
        have_raw += cnt
        q_next = qhi

        # merge bands -> msieve.dat, attempt filtering
        with open(wd / "msieve.dat", "wb") as out:
            for _, _, files, _, _ in bands:
                for f in files:
                    out.write((wd / f).read_bytes())
        rc, wall = run_cmd([MSIEVE, "-s", "msieve.dat", "-l", f"nc1-r{round_idx}.log",
                            "-nc1", "-t", NPROC, "-v"], wd,
                           wd / f"nc1-r{round_idx}.stdout", timeout=cfg.nc_timeout)
        res["stages"].setdefault("filter_s", 0.0)
        res["stages"]["filter_s"] += wall
        out_text = (wd / f"nc1-r{round_idx}.stdout").read_text(errors="replace")
        m = re.search(r"found (\d+) cycles, need (\d+)", out_text)
        if m and int(m.group(1)) >= int(m.group(2)) and "too few cycles" not in out_text:
            res["cycles"] = [int(m.group(1)), int(m.group(2))]
            break
        m = re.search(r"filtering wants (\d+) more relations", out_text)
        if m and round_idx < 10:
            deficit = int(m.group(1))
            # msieve's "wants N more" hint is a fixed floor and the cycle
            # count grows superlinearly in relations, so grow geometrically
            # (YAFU's -filt_bump analog, defaulted larger: filter passes are
            # the expensive part once sieving is on the GPU).
            topup_frac = getattr(cfg, "topup_frac", 0.5)
            target_raw = have_raw + max(int(deficit * 1.2),
                                        int(have_raw * topup_frac))
            continue
        res["rc"] = rc
        res["error"] = "filtering did not converge"
        return _finish_nfs(res, t_total0, wd)

    res["raw_relations"] = have_raw
    res["bands"] = [{"q": f"{a}:{b}", "rels": c, "s": round(w, 1)} for a, b, _, c, w in bands]

    # ---- stage 4: matrix build + Lanczos + sqrt ----------------------------
    rc, wall = run_cmd([MSIEVE, "-s", "msieve.dat", "-l", "nc2-mat.log",
                        "-nc2", "only_matbuild=1", "-t", NPROC, "-v"],
                       wd, wd / "nc2-mat.stdout", timeout=cfg.nc_timeout)
    res["stages"]["matbuild_s"] = wall
    rc, wall = run_cmd([MSIEVE, "-s", "msieve.dat", "-l", "nc2-la.log",
                        "-nc2", "-t", NPROC, "-v"], wd, wd / "nc2-la.stdout",
                       timeout=cfg.la_timeout, env_extra={"CUDA_VISIBLE_DEVICES": cfg.device})
    res["stages"]["linalg_s"] = wall
    if rc != 0:
        res["rc"] = rc
        res["error"] = "linear algebra failed"
        return _finish_nfs(res, t_total0, wd)
    m = re.search(r"matrix is (\d+) x (\d+).*?weight (\d+)", (wd / "nc2-la.stdout").read_text(errors="replace"))
    if m:
        res["matrix"] = [int(m.group(1)), int(m.group(2)), int(m.group(3))]

    rc, wall = run_cmd([MSIEVE, "-s", "msieve.dat", "-l", "nc3.log",
                        "-nc3", "-t", NPROC, "-v"], wd, wd / "nc3.stdout",
                       timeout=cfg.sqrt_timeout)
    res["stages"]["sqrt_s"] = wall

    log3 = (wd / "nc3.stdout").read_text(errors="replace")
    factors = sorted(int(x) for x in re.findall(r"p\d+ factor: (\d+)", log3))
    res["factors"] = [str(f) for f in factors]
    res["rc"] = rc
    res["ok"] = (rc == 0 and len(factors) == 2
                 and mpz(factors[0]) * mpz(factors[1]) == mpz(n)
                 and gmpy2.is_prime(mpz(factors[0])))
    return _finish_nfs(res, t_total0, wd)


def degree_(coeffs):
    for i in range(len(coeffs) - 1, -1, -1):
        if coeffs[i]:
            return i
    return 0


def _finish_nfs(res, t0, wd):
    res["wall_s"] = time.monotonic() - t0
    res.setdefault("factors", [])
    return res


# ---------------------------------------------------------------------------
# results store / report
# ---------------------------------------------------------------------------

def save_result(res):
    RESULTS_JSON.parent.mkdir(exist_ok=True)
    data = json.loads(RESULTS_JSON.read_text()) if RESULTS_JSON.exists() else []
    data = [r for r in data if not (r["digits"] == res["digits"] and r["method"] == res["method"])]
    data.append(res)
    RESULTS_JSON.write_text(json.dumps(data, indent=1))


def cmd_report(_args):
    if not RESULTS_JSON.exists():
        sys.exit("no results yet")
    data = json.loads(RESULTS_JSON.read_text())
    data.sort(key=lambda r: (r["digits"], r["method"]))
    hdr = (f"{'N':>5} {'method':>6} {'ok':>3} {'total':>10} "
           f"{'poly':>8} {'sieve':>8} {'filter':>8} {'mat/LA/sqrt':>16}  detail")
    print(hdr)
    print("-" * len(hdr))
    for r in data:
        st = r.get("stages", {})
        if r["method"] == "mpqs":
            poly = fmt_secs(None)
            sieve = fmt_secs(r.get("internal", {}).get("sieve_s"))
            filt = fmt_secs(None)
            mls = fmt_secs(r.get("internal", {}).get("linalg_s"))
            detail = f"F={r.get('internal', {}).get('F')} lp1={r.get('lp1_bound') or 'off'}"
        else:
            poly = fmt_secs(st.get("polyselect_s"))
            sieve = fmt_secs(st.get("sieve_s"))
            filt = fmt_secs(st.get("filter_s"))
            mls = "/".join(fmt_secs(st.get(k)) for k in ("matbuild_s", "linalg_s", "sqrt_s"))
            detail = (f"raw={r.get('raw_relations', 0):,} bands={len(r.get('bands', []))} "
                      f"E={r.get('poly_e', '-')}")
            if r.get("poly_quality"):
                detail += f" ({r['poly_quality']:.2f}x e0, budget {int(r.get('poly_budget_s', 0))}s)"
        print(f"{r['digits']:>5} {r['method']:>6} {'✓' if r['ok'] else '✗':>3} "
              f"{fmt_secs(r['wall_s']):>10} {poly:>8} {sieve:>8} {filt:>8} {mls:>16}  {detail}")
        if not r["ok"]:
            err = r.get("error", f"rc={r.get('rc')}")
            print(f"{'':>12} ERROR: {err}   (see {r.get('workdir')})")


def cmd_show(args):
    data = json.loads(RESULTS_JSON.read_text()) if RESULTS_JSON.exists() else []
    for r in data:
        if r["digits"] == args.digits and r["method"] == args.method:
            print(json.dumps(r, indent=1))
            return
    sys.exit("no such result")


# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("gen", help="generate semiprimes")
    p.add_argument("--digits", default="70,80,90,100")
    p.add_argument("--seed", type=int, default=20261025)
    p.set_defaults(fn=cmd_gen)

    p = sub.add_parser("run", help="run benchmarks")
    p.add_argument("--method", choices=["mpqs", "nfs", "all"], required=True)
    p.add_argument("--digits", default="70,80,90,100")
    p.add_argument("--device", default="0")
    p.add_argument("--mpqs-timeout", type=float, default=6 * 3600)
    p.add_argument("--np-timeout", type=float, default=2 * 3600)
    p.add_argument("--sieve-timeout", type=float, default=6 * 3600)
    p.add_argument("--nc-timeout", type=float, default=2 * 3600)
    p.add_argument("--la-timeout", type=float, default=12 * 3600)
    p.add_argument("--sqrt-timeout", type=float, default=2 * 3600)
    p.add_argument("--poly-budget", type=float, default=None,
                   help="polyselect deadline in seconds (overrides the "
                        "heuristic; 0 = msieve's built-in table)")
    p.add_argument("--sieve-procs", type=int, default=1,
                   help="cuda-sieve processes per GPU; each sieve band is "
                        "split into this many contiguous q chunks running "
                        "concurrently on the same card (production recipe "
                        "uses 8/card)")
    p.add_argument("--topup-frac", type=float, default=0.5,
                   help="filter-failure relation top-up growth: new target = "
                        "have + max(1.2x msieve hint, topup_frac x have). "
                        "YAFU's 5%% bump analog; larger converges in fewer "
                        "filter rounds, smaller oversieves less")
    p.set_defaults(fn=cmd_run)

    p = sub.add_parser("report", help="print results table")
    p.set_defaults(fn=cmd_report)

    p = sub.add_parser("show", help="dump one result as JSON")
    p.add_argument("--digits", type=int, required=True)
    p.add_argument("--method", required=True)
    p.set_defaults(fn=cmd_show)

    args = ap.parse_args()
    args.fn(args)


def cmd_run(args):
    for digits in [int(d) for d in args.digits.split(",")]:
        methods = ["mpqs", "nfs"] if args.method == "all" else [args.method]
        for method in methods:
            print(f"=== {method} c{digits} ===", flush=True)
            fn = run_mpqs if method == "mpqs" else run_nfs
            res = fn(digits, args)
            save_result(res)
            status = "OK" if res["ok"] else f"FAILED ({res.get('error', 'rc=' + str(res.get('rc')))}"
            print(f"    {status}, wall {fmt_secs(res['wall_s'])}", flush=True)


if __name__ == "__main__":
    main()
