#!/usr/bin/env python3
"""bench_factor — SIQS vs NFS benchmark and crossover estimator.

Drives the yagfu package backends directly (no engine/scheduler): for each
digit count, runs cuda-mpqs (SIQS) and/or the msieve + cuda-sieve GNFS
pipeline on a fixed-seed semiprime, records wall times, and fits the
classic exponential models T(d) = A·e^(k·d) (YAFU tune's approach) to
estimate the SIQS -> NFS crossover digits — the recommended --xover.

Usage:
  python3 scripts/bench_factor.py gen  --digits 70,80,90,100 [--seed 20261025]
  python3 scripts/bench_factor.py run  --method all --digits 70,80,90,100
  python3 scripts/bench_factor.py report
  python3 scripts/bench_factor.py crossover

Everything reuses the package: backends.siqs / backends.nfs do the running
(with their stage timings), params.py picks per-size parameters, and
primality checks are yagfu's own BPSW — no third-party dependencies.
"""

from __future__ import annotations

import argparse
import json
import random
import re
import shutil
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from yagfu.backends.nfs import run_nfs            # noqa: E402
from yagfu.backends.siqs import run_siqs          # noqa: E402
from yagfu.config import RunContext               # noqa: E402
from yagfu.params import mpqs_lp1_bound, nfs_params, poly_budget  # noqa: E402
from yagfu.primes import is_probable_prime        # noqa: E402

BENCH_DIR = ROOT / "benchmarks"
NUMBERS_JSON = BENCH_DIR / "numbers.json"
RESULTS_JSON = BENCH_DIR / "results.json"

RUN_ROOT = BENCH_DIR / "runs"         # per-run workdirs (gitignored)


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def fmt_secs(x: float | None) -> str:
    if x is None:
        return "-"
    if x < 90:
        return f"{x:.1f}s"
    m, s = divmod(int(round(x)), 60)
    if m < 90:
        return f"{m}m{s:02d}s"
    h, m = divmod(m, 60)
    return f"{h}h{m:02d}m{s:02d}s"


def load_results() -> list[dict]:
    if not RESULTS_JSON.exists():
        sys.exit("no results yet; run `bench_factor.py run` first")
    return json.loads(RESULTS_JSON.read_text())


def save_result(res: dict) -> None:
    RESULTS_JSON.parent.mkdir(exist_ok=True)
    data = load_results() if RESULTS_JSON.exists() else []
    data = [r for r in data if not (r["digits"] == res["digits"]
                                    and r["method"] == res["method"])]
    data.append(res)
    RESULTS_JSON.write_text(json.dumps(data, indent=1))


def make_ctx(args) -> RunContext:
    from yagfu.config import resolve_devices
    devices = resolve_devices(
        tuple(d.strip() for d in args.device.split(",") if d.strip()))
    return RunContext(
        device=devices[0],
        devices=devices,
        sieve_procs=args.sieve_procs,
        poly_budget_s=args.poly_budget,
        topup_frac=args.topup_frac,
        verbose=1 if args.verbose else 0,
    )


# ---------------------------------------------------------------------------
# semiprime generation
# ---------------------------------------------------------------------------

def _rand_prime(rng: random.Random, digits: int) -> int:
    from yagfu.arith import next_prime

    while True:
        start = rng.randrange(10 ** (digits - 1) | 1, 10 ** digits, 2)
        p = next_prime(start)
        if len(str(p)) == digits:
            return p


def cmd_gen(args) -> None:
    digits_list = [int(d) for d in args.digits.split(",")]
    rng = random.Random(args.seed)
    numbers = {}
    for digits in digits_list:
        # odd digit counts split floor/ceil (42+43 for c85): two floor-half
        # primes multiply to at most digits-1 digits and never match
        lo, hi = digits // 2, digits - digits // 2
        while True:
            p, q = _rand_prime(rng, lo), _rand_prime(rng, hi)
            if p != q and len(str(p * q)) == digits:
                break
        numbers[str(digits)] = {"n": str(p * q), "p": str(p), "q": str(q)}
        print(f"c{digits}: N = {p * q}")
    BENCH_DIR.mkdir(exist_ok=True)
    NUMBERS_JSON.write_text(json.dumps({"seed": args.seed,
                                        "numbers": numbers}, indent=1))
    print(f"wrote {NUMBERS_JSON}")


def load_number(digits: int) -> dict:
    if not NUMBERS_JSON.exists():
        sys.exit(f"error: {NUMBERS_JSON} missing; run `gen` first")
    data = json.loads(NUMBERS_JSON.read_text())
    if str(digits) not in data["numbers"]:
        sys.exit(f"error: c{digits} not generated; "
                 f"run `gen --digits {digits}`")
    return data["numbers"][str(digits)]


# ---------------------------------------------------------------------------
# the two methods (thin wrappers around the yagfu backends)
# ---------------------------------------------------------------------------

def bench_siqs(digits: int, ctx: RunContext, args) -> dict:
    entry = load_number(digits)
    n = int(entry["n"])
    wd = RUN_ROOT / f"mpqs-c{digits}"
    if wd.exists():
        shutil.rmtree(wd)
    wd.mkdir(parents=True)

    t0 = time.monotonic()
    r = run_siqs(n, ctx, wd)
    ok = (r.ok and len(r.factors) == 2
          and r.factors[0] * r.factors[1] == n
          and all(is_probable_prime(f) for f in r.factors))
    return {
        "method": "siqs", "digits": digits, "ok": ok, "wall_s": r.wall_s,
        "error": r.error, "internal": r.internal,
        "factors": [str(f) for f in r.factors],
        "ts": time.strftime("%Y-%m-%d %H:%M:%S"),
        "total_wall_s": time.monotonic() - t0,
    }


def bench_nfs(digits: int, ctx: RunContext, args) -> dict:
    entry = load_number(digits)
    n = int(entry["n"])
    wd = RUN_ROOT / f"nfs-c{digits}"
    if wd.exists():
        shutil.rmtree(wd)
    wd.mkdir(parents=True)

    t0 = time.monotonic()
    r = run_nfs(n, ctx, wd)
    ok = bool(r.ok and r.factors
              and all(n % f == 0 for f in r.factors)
              and all(is_probable_prime(f) for f in r.factors))
    return {
        "method": "nfs", "digits": digits, "ok": ok, "error": r.error,
        "stages": {k: round(v, 1) for k, v in r.stages.items()},
        "poly_e": r.poly_e, "poly_method": r.poly_method,
        "params": list(r.params), "raw_relations": r.raw_relations,
        "bands": r.bands, "factors": [str(f) for f in r.factors],
        "ts": time.strftime("%Y-%m-%d %H:%M:%S"),
        "wall_s": time.monotonic() - t0,
    }


# ---------------------------------------------------------------------------
# tune: YAFU-style crossover estimation WITHOUT full factorizations
# ---------------------------------------------------------------------------

#: Rough RTX 4090 calibrations from completed runs (c70-c100, idle GPU).
#: NFS postprocessing (filtering+matbuild+LA+sqrt) as a fraction of sieve
#: time; NFS relation top-up factor (filtering asks for more than the
#: initial target).
NFS_POST_FRAC = 1.10
NFS_TOPUP = 1.25

#: Full-job raw-relation targets, for the probe's ETA only: the runtime
#: keeps no such table (its sieve loop derives the target from the
#: measured excess curve); the probe needs a model to turn a measured
#: band rate into a predicted job time.
_BENCH_TARGET_RELS = ((90, 8_000_000), (100, 15_000_000), (110, 40_000_000),
                      (120, 100_000_000), (130, 300_000_000),
                      (140, 800_000_000), (150, 2_000_000_000))


def _target_rels(digits: int) -> int:
    if digits <= _BENCH_TARGET_RELS[0][0]:
        return _BENCH_TARGET_RELS[0][1]
    for (d0, r0), (d1, r1) in zip(_BENCH_TARGET_RELS,
                                  _BENCH_TARGET_RELS[1:]):
        if d0 <= digits <= d1:
            t = (digits - d0) / (d1 - d0)
            return int(round(r0 + t * (r1 - r0)))
    return _BENCH_TARGET_RELS[-1][1]

_RE_TARGET_RELS = re.compile(r"Target Relations: (\d+)")
_RE_EST = {
    "sieve_s": re.compile(r"Sieve time:\s+([\d.]+)\s*s"),
    "matrix_s": re.compile(r"Matrix time:\s+([\d.]+)\s*s"),
    "linalg_s": re.compile(r"LinAlg time:\s+([\d.]+)\s*s"),
    "total_s": re.compile(r"Total estimate:\s+([\d.]+)\s*s"),
}


def probe_siqs(n: int, ctx: RunContext, wd: Path) -> dict:
    """cuda-mpqs --estimate_only: the tool's own truncated-sieve projection.

    One ~2 s probe prints its structural estimate (sieve / matrix / LA
    / total) — no factorization runs at all.
    """
    from yagfu.backends import find_bin
    from yagfu.backends.runner import read_log, run_cmd

    wd.mkdir(parents=True, exist_ok=True)
    binary = find_bin("cuda-mpqs", ctx.bin_dir)
    cmd = [binary, "--N", n, "--device", 0, "--dir", "work",
           "--estimate_only"]
    lp1 = mpqs_lp1_bound(len(str(n)))
    if lp1:
        cmd += ["--lp1_bound", lp1]
    rc, wall = run_cmd(cmd, wd, wd / "probe.log",
                       env_extra=ctx.pin_env())
    log = read_log(wd / "probe.log")
    out: dict = {}
    for key, rex in _RE_EST.items():
        m = rex.search(log)
        if m:
            out[key] = round(float(m.group(1)), 2)
    m = _RE_TARGET_RELS.search(log)
    if m:
        out["target_rels"] = int(m.group(1))
    if "total_s" not in out:
        return {"error": f"estimate parse failed rc={rc} "
                         f"(see {wd}/probe.log)"}
    out["est_total_s"] = out["total_s"]
    out["probe_wall_s"] = round(wall, 2)
    return out


def probe_nfs(n: int, ctx: RunContext, wd: Path, band_width: int,
              poly_cap_s: float) -> dict:
    """Short polyselect + one cuda-sieve band -> rate -> extrapolated total.

    YAFU's tune analog for NFS: extrapolate the sieve time from a measured
    band, keep the polyselect budget separate, add the postprocessing
    fraction.  Filtering/LA/sqrt never run.
    """
    from yagfu.backends import find_bin
    from yagfu.backends.nfs import (
        NFS_MSIEVE_POLY_MIN_DIGITS,
        _fb_to_job,
        _parse_msieve_fb,
    )
    from yagfu.backends.runner import count_lines, read_log, run_cmd, run_cmd_wall_cap

    wd.mkdir(parents=True, exist_ok=True)
    digits = len(str(n))
    msieve = find_bin("msieve", ctx.bin_dir)
    sieve_bin = find_bin("cuda-sieve", ctx.bin_dir)
    rlim, alim, lpb, logI = nfs_params(digits)

    # short polyselect; the sieve rate depends on polynomial quality, so a
    # probe needs the real -np search.  Below ~90 digits -np emits nothing
    # and NFS probing is not offered (base-m's rate is ~60x lower and
    # would poison the fit).
    t0 = time.monotonic()
    if digits < NFS_MSIEVE_POLY_MIN_DIGITS:
        return {"error": "msieve -np emits nothing below ~90 digits; "
                         "NFS probing unsupported at this size"}
    budget = min(poly_cap_s, max(10.0, poly_budget(digits)))
    run_cmd_wall_cap([msieve, "-np", "-t", ctx.threads, "-v", str(n)],
                     wd, wd / "np.log", cap_s=budget,
                     env_extra=ctx.pin_env())
    if not (wd / "msieve.fb").exists():
        run_cmd([msieve, "-np", "-t", ctx.threads, "-v", str(n)],
                wd, wd / "np.log", env_extra=ctx.pin_env())
    if not (wd / "msieve.fb").exists():
        return {"error": f"polyselect failed (see {wd}/np.log)"}
    fb = _parse_msieve_fb(read_log(wd / "msieve.fb"))
    poly_method = "msieve -np (short)"
    poly_s = time.monotonic() - t0

    job = _fb_to_job(fb, rlim, alim, lpb)
    (wd / "probe.job").write_text(job)
    q0 = alim // 2
    rels_name = "probe.rels"
    cmd = [sieve_bin, "--pipeline", "--cofactor", "--poly", "probe.job",
           "--logI", logI, "--region", min(logI, 14), "--device", 0,
           "--qrange", f"{q0}:{q0 + band_width}",
           "--relations", rels_name, "--log", "probe.sieve.log"]
    try:
        rc, band_s = run_cmd(cmd, wd, wd / "probe.stdout",
                             env_extra=ctx.pin_env())
    except Exception as e:  # noqa: BLE001
        return {"error": f"band failed: {e!r}"}
    got = count_lines(wd / rels_name)
    if rc != 0 or got == 0:
        return {"error": f"band rc={rc}, {got} relations "
                         f"(see {wd}/probe.stdout)"}
    rate = got / band_s
    target_rels = _target_rels(digits)
    est_sieve = target_rels * NFS_TOPUP / rate
    return {
        "poly_s": round(poly_s, 1), "poly_method": poly_method,
        "band_rels": got,
        "band_s": round(band_s, 1), "rate_rels_s": round(rate, 1),
        "target_rels": int(target_rels * NFS_TOPUP),
        "est_sieve_s": round(est_sieve, 1),
        "est_total_s": round(poly_s + est_sieve * (1 + NFS_POST_FRAC), 1),
    }


def cmd_tune(args) -> None:
    ctx = make_ctx(args)
    digits_list = [int(d) for d in args.digits.split(",")]
    siqs_pts: list[tuple[float, float]] = []
    nfs_pts: list[tuple[float, float]] = []
    records = []
    for digits in digits_list:
        n = int(load_number(digits)["n"])   # fixed numbers.json, no gen here
        print(f"=== tune c{digits} ===", flush=True)
        wd = RUN_ROOT / f"tune-siqs-c{digits}"
        if wd.exists():
            shutil.rmtree(wd)
        r = probe_siqs(n, ctx, wd)
        print(f"    siqs: {r}", flush=True)
        if "est_total_s" in r:
            siqs_pts.append((digits, r["est_total_s"]))
            records.append({"method": "siqs", "digits": digits, **r})
        wd = RUN_ROOT / f"tune-nfs-c{digits}"
        if wd.exists():
            shutil.rmtree(wd)
        r = probe_nfs(n, ctx, wd, args.band_width, args.poly_cap)
        print(f"    nfs:  {r}", flush=True)
        if "est_total_s" in r:
            nfs_pts.append((digits, r["est_total_s"]))
            records.append({"method": "nfs", "digits": digits, **r})
        if not args.keep:
            shutil.rmtree(RUN_ROOT / f"tune-siqs-c{digits}",
                          ignore_errors=True)
            shutil.rmtree(RUN_ROOT / f"tune-nfs-c{digits}",
                          ignore_errors=True)

    if len(siqs_pts) < 2 or len(nfs_pts) < 2:
        sys.exit("tune: need >= 2 successful probes per method")

    (ln_a1, k1), (ln_a2, k2) = _fit_exp(siqs_pts), _fit_exp(nfs_pts)
    print()
    print(f"fit:  T_siqs = {_fmt_model(ln_a1, k1)}   "
          f"T_nfs = {_fmt_model(ln_a2, k2)}")
    if k1 <= k2:
        sys.exit("tune: SIQS not growing faster than NFS — no crossing")
    d_star = (ln_a2 - ln_a1) / (k1 - k2)
    print(f"crossover: ~{d_star:.1f} digits "
          f"(per-digit growth: x{k1 * 10:.1f} vs x{k2 * 10:.1f} per 10 digits)")
    print(f"suggested: yagfu --xover {round(d_star)}")
    BENCH_DIR.mkdir(exist_ok=True)
    (BENCH_DIR / "tune.json").write_text(json.dumps(
        {"records": records,
         "fit": {"siqs": [ln_a1, k1], "nfs": [ln_a2, k2]},
         "crossover": d_star}, indent=1))
    print(f"wrote {BENCH_DIR / 'tune.json'}")


# ---------------------------------------------------------------------------
# crossover from completed runs (YAFU tune's model: T(d) = A * exp(k*d))
# ---------------------------------------------------------------------------

def _fit_exp(points: list[tuple[float, float]]) -> tuple[float, float]:
    """Least-squares fit of ln T = lnA + k*d; returns (lnA, k)."""
    import math

    n = len(points)
    xd = [d for d, _ in points]
    yd = [math.log(t) for _, t in points]
    mx = sum(xd) / n
    my = sum(yd) / n
    var = sum((x - mx) ** 2 for x in xd)
    cov = sum((x - mx) * (y - my) for x, y in zip(xd, yd))
    k = cov / var
    return my - k * mx, k


def cmd_crossover(_args=None) -> float | None:
    data = load_results()
    pts = {m: sorted((r["digits"], r["wall_s"]) for r in data
                     if r["method"] == m and r.get("ok"))
           for m in ("siqs", "nfs")}
    for m, p in pts.items():
        if len(p) < 2:
            print(f"crossover: need >= 2 successful {m} runs "
                  f"(have {len(p)})")
            return None

    (ln_a1, k1), (ln_a2, k2) = _fit_exp(pts["siqs"]), _fit_exp(pts["nfs"])
    if k1 <= k2:
        print("crossover: SIQS time is not growing faster than NFS — "
              "no crossing ahead")
        return None
    d_star = (ln_a2 - ln_a1) / (k1 - k2)
    print(f"fit:  T_siqs = {_fmt_model(ln_a1, k1)}   "
          f"T_nfs = {_fmt_model(ln_a2, k2)}")
    print(f"crossover: SIQS and NFS tie at ~{d_star:.1f} digits "
          f"(per-digit growth: x{k1 * 10:.1f} vs x{k2 * 10:.1f} per 10 digits)")

    # honest bracket from the raw measurements
    s, f = dict(pts["siqs"]), dict(pts["nfs"])
    prev = None
    for d in sorted(set(s) & set(f)):
        winner = "siqs" if s[d] < f[d] else "nfs"
        if prev and prev[1] != winner:
            print(f"measured flip between c{prev[0]} and c{d} "
                  f"({prev[1]} -> {winner})")
        prev = (d, winner)
    print(f"suggested: yagfu --xover {round(d_star)}")
    return d_star


def _fmt_model(ln_a: float, k: float) -> str:
    import math

    return f"{math.exp(ln_a):.3g}*exp({k:.4f}*d)"


# ---------------------------------------------------------------------------
# report
# ---------------------------------------------------------------------------

def cmd_report(_args) -> None:
    data = sorted(load_results(), key=lambda r: (r["digits"], r["method"]))
    hdr = (f"{'N':>5} {'method':>6} {'ok':>3} {'total':>9} "
           f"{'poly':>8} {'sieve':>8} {'filter':>8} {'LA/sqrt':>12}  detail")
    print(hdr)
    print("-" * len(hdr))
    for r in data:
        st = r.get("stages") or {}
        if r["method"] == "siqs":
            poly = fmt_secs(None)
            sieve = fmt_secs(r.get("internal", {}).get("sieve_s"))
            filt = fmt_secs(None)
            la = fmt_secs(r.get("internal", {}).get("linalg_s"))
            detail = f"F={r.get('internal', {}).get('F')}"
        else:
            poly = fmt_secs(st.get("polyselect_s"))
            sieve = fmt_secs(st.get("sieve_s"))
            filt = fmt_secs(st.get("filter_s"))
            la = "/".join(fmt_secs(st.get(k)) for k in ("linalg_s", "sqrt_s"))
            detail = (f"raw={r.get('raw_relations', 0):,} "
                      f"E={r.get('poly_e') or '-'}")
        print(f"{r['digits']:>5} {r['method']:>6} {'✓' if r['ok'] else '✗':>3} "
              f"{fmt_secs(r['wall_s']):>9} {poly:>8} {sieve:>8} {filt:>8} "
              f"{la:>12}  {detail}")
        if not r["ok"]:
            print(f"{'':>12} ERROR: {r.get('error')}")


# ---------------------------------------------------------------------------

def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.
                                 RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("gen", help="generate fixed-seed semiprimes")
    p.add_argument("--digits", default="85,90,95,100,105,110")
    p.add_argument("--seed", type=int, default=20261025)
    p.set_defaults(fn=cmd_gen)

    p = sub.add_parser("run", help="run benchmarks")
    p.add_argument("--method", choices=["siqs", "nfs", "all"], required=True)
    p.add_argument("--digits", default="85,90,95,100,105,110")
    p.add_argument("--device", default="0")
    p.add_argument("--sieve-procs", type=int, default=1)
    p.add_argument("--poly-budget", type=float, default=None)
    p.add_argument("--topup-frac", type=float, default=1.0)
    p.add_argument("--keep", action="store_true",
                   help="keep per-run workdirs")
    p.add_argument("-v", "--verbose", action="store_true")
    p.set_defaults(fn=cmd_run)

    p = sub.add_parser("report", help="print results table")
    p.set_defaults(fn=cmd_report)

    p = sub.add_parser("tune", help="YAFU-style fast crossover estimate: "
                                     "truncated rate probes + structural "
                                     "extrapolation on the fixed numbers.json "
                                     "semiprimes (run `gen` first); no full "
                                     "factorization is completed")
    p.add_argument("--digits", default="85,90,95,100,105,110")
    p.add_argument("--band-width", type=int, default=5000,
                   help="NFS probe q-band width")
    p.add_argument("--poly-cap", type=float, default=40.0,
                   help="NFS probe polyselect budget cap in seconds")
    p.add_argument("--device", default="0")
    p.add_argument("--sieve-procs", type=int, default=1)
    p.add_argument("--poly-budget", type=float, default=None)
    p.add_argument("--topup-frac", type=float, default=1.0)
    p.add_argument("--keep", action="store_true")
    p.add_argument("-v", "--verbose", action="store_true")
    p.set_defaults(fn=cmd_tune)

    p = sub.add_parser("crossover", help="fit T(d) from COMPLETED runs "
                                         "and estimate the SIQS -> NFS "
                                         "crossover")
    p.set_defaults(fn=cmd_crossover)

    args = ap.parse_args()
    args.fn(args)


def cmd_run(args) -> None:
    ctx = make_ctx(args)
    digits_list = [int(d) for d in args.digits.split(",")]
    methods = ["siqs", "nfs"] if args.method == "all" else [args.method]
    for digits in digits_list:
        for method in methods:
            print(f"=== {method} c{digits} ===", flush=True)
            res = (bench_siqs if method == "siqs" else bench_nfs)(
                digits, ctx, args)
            save_result(res)
            status = ("OK" if res["ok"]
                      else f"FAILED ({res.get('error')})")
            print(f"    {status}, wall {fmt_secs(res['wall_s'])}", flush=True)
            if not args.keep:
                wd = RUN_ROOT / f"{'mpqs' if method == 'siqs' else 'nfs'}-c{digits}"
                shutil.rmtree(wd, ignore_errors=True)
    if args.method == "all":
        cmd_crossover()


if __name__ == "__main__":
    main()
