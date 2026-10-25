"""yagfu command-line interface.

    yagfu 123456789...            factor one number
    yagfu -f numbers.txt          factor one number per line
    yagfu -db 12345...            consult and report to factordb.com
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
import tempfile
import time
from pathlib import Path

from . import __version__
from .config import RunContext, cleanup_workdir
from .engine import Engine, FactorResult, FoundFactor
from .fdb import (
    ECM_REPORT_MAX_DIGITS,
    ECM_REPORT_MIN_DIGITS,
    FactorDbClient,
    read_token,
)
from .primes import is_probable_prime


# ---------------------------------------------------------------------------
# CLI definition
# ---------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        prog="yagfu",
        description="yagfu — GPU integer factorization "
                    "(TD -> rho -> GPU ECM -> SIQS/NFS)",
        epilog="examples:\n"
               "  yagfu 170141183460469231731687303715884105727\n"
               "  yagfu -f composites.txt -db\n",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("number", nargs="?", help="decimal number to factor")
    ap.add_argument("-f", "--file", help="text file with one number per line")
    ap.add_argument("-db", "--factor-db", dest="factor_db", action="store_true",
                    help="fetch known factorizations from factordb.com, factor "
                         "what remains, and report new factors back")
    ap.add_argument("--plan", choices=("light", "normal", "deep"),
                    default="normal", help="ECM pretest depth (default normal)")
    ap.add_argument("--xover", type=int, default=105,
                    help="SIQS -> NFS crossover in digits (default 105)")
    ap.add_argument("--device", default="0", help="CUDA device (default 0)")
    ap.add_argument("--threads", type=int, default=None,
                    help="CPU threads for msieve stages (default nproc-2)")
    ap.add_argument("--sieve-procs", type=int, default=1,
                    help="concurrent cuda-sieve processes per band (default 1)")
    ap.add_argument("--bin-dir", help="directory with the backend binaries")
    ap.add_argument("--workdir", help="base scratch directory "
                                      "(default: a temp dir). Per-number "
                                      "workdirs are auto-removed when a "
                                      "number finishes; kept on Ctrl+C or "
                                      "with --keep")
    ap.add_argument("--keep", action="store_true",
                    help="keep per-number workdirs even when a number "
                         "finishes normally")
    ap.add_argument("--timeout", type=float, default=None,
                    help="per-number wall-clock budget in seconds")
    ap.add_argument("--ecm-only", action="store_true",
                    help="never run SIQS/NFS: keep raising the ECM level "
                         "until the number factors or Ctrl+C (the t65 level "
                         "is sampled forever once reached)")
    ap.add_argument("--force", choices=("siqs", "nfs"), default=None,
                    help=argparse.SUPPRESS)  # debug: skip the selector
    ap.add_argument("--ecm-stage2-maxmem", type=int, default=None, metavar="MB",
                    help="gpuecm2 stage-2 GPU memory cap in MB (default: 90%% "
                         "of the GPU's total VRAM; 0 = no cap, gpuecm2's own "
                         "free-memory budgeting)")
    ap.add_argument("--poly-budget", type=float, default=None,
                    help="NFS polyselect deadline in seconds (0 = msieve default)")
    ap.add_argument("--topup-frac", type=float, default=1.0,
                    help="NFS relation target growth per filter round: "
                         "new = have + max(1.2x deficit, frac x have). GPU "
                         "sieving is cheap here while -nc1 passes are not, "
                         "so growing fast saves filter rounds (default 1.0)")
    ap.add_argument("--db-token", default=None,
                    help="factordb token file (default ~/.factordb_token)")
    ap.add_argument("--db-no-report", action="store_true",
                    help="with -db: only query, never report")
    ap.add_argument("--db-dry-run", action="store_true",
                    help="with -db: print requests instead of sending")
    ap.add_argument("--json", action="store_true",
                    help="one JSON object per input line on stdout")
    ap.add_argument("-v", "--verbose", action="count", default=0,
                    help="progress logging to stderr (repeat for more)")
    ap.add_argument("-q", "--quiet", action="store_true")
    ap.add_argument("--version", action="version",
                    version=f"yagfu {__version__}")
    return ap


# ---------------------------------------------------------------------------
# input handling
# ---------------------------------------------------------------------------

def load_numbers(args) -> list[int]:
    if args.file:
        if args.number:
            sys.exit("error: pass either a number or -f FILE, not both")
        numbers = []
        for lineno, line in enumerate(Path(args.file).read_text().splitlines(), 1):
            s = line.strip()
            if not s or s.startswith("#"):
                continue
            try:
                numbers.append(int(s))
            except ValueError:
                sys.exit(f"error: {args.file}:{lineno}: not a decimal integer: {s!r}")
        if not numbers:
            sys.exit(f"error: {args.file} contains no numbers")
        return numbers
    if not args.number:
        sys.exit(build_parser().format_usage())
    try:
        return [int(args.number.strip())]
    except ValueError:
        sys.exit(f"error: not a decimal integer: {args.number!r}")


# ---------------------------------------------------------------------------
# factordb integration
# ---------------------------------------------------------------------------

def _db_factor_list(client: FactorDbClient, n: int, fid: str | None):
    """Prime factors factordb already knows for n (decimals)."""
    if fid is None:
        return []
    decs = client.get_factor_decimals(fid)
    out = []
    for d in decs:
        try:
            v = int(d)
        except ValueError:
            continue
        if 1 < v < n and n % v == 0:
            out.append(v)
    return out


def factor_with_db(n: int, client: FactorDbClient, engine: Engine, wd: Path,
                   report: bool = True) -> tuple[FactorResult, list[FoundFactor], dict]:
    """factordb-aware driver.

    Returns (engine result, known-db factors merged in, fdb summary dict).
    The engine result's factor list contains every prime factor of n
    (db-known ones included, source 'fdb')."""
    fdb_info: dict = {}
    rec = client.get_number(n)
    known: list[int] = []
    to_factor = n
    status = rec["status"] if rec else None
    fid = rec["fid"] if rec else None
    fdb_info["status"] = status

    if status in ("P", "PR"):
        known = [n]
        to_factor = 1
    elif status in ("FF", "CF"):
        # divide out the primes factordb already knows; whatever remains
        # composite is the part we must factor ourselves
        known = sorted(set(_db_factor_list(client, n, fid)))
        rem = n
        for f in known:
            while rem % f == 0:
                rem //= f
        if rem > 1 and is_probable_prime(rem):
            known.append(rem)
            rem = 1
        to_factor = rem

    if to_factor > 1:
        res = engine.factor(to_factor, wd)
    else:
        res = FactorResult(n=to_factor)
        res.complete = True

    # merge: db-known primes + locally found primes
    merged: list[FoundFactor] = [FoundFactor(f, "fdb", f < 4_759_123_141)
                                 for f in known]
    local_new = [f for f in res.factors]
    merged.extend(local_new)
    prod = 1
    for f in merged:
        prod *= f.value
    merged_complete = prod == n and all(is_probable_prime(f.value) for f in merged)

    if report and res.factors and not res.interrupted \
            and status not in ("FF", "P", "PR"):
        new_vals = [f.value for f in local_new]
        client.report_factors(fid, n, new_vals)
        fdb_info["reported"] = len(new_vals)
        # ECM parameter reports for locally-found ECM factors (25..100 digits)
        fmap = client.factor_fid_map(fid) if fid else {}
        reported_ecm = 0
        for f in local_new:
            d = len(str(f.value))
            if f.ecm is None or not (ECM_REPORT_MIN_DIGITS <= d <= ECM_REPORT_MAX_DIGITS):
                continue
            ffid = fmap.get(str(f.value))
            if not ffid:
                # unknown to the map yet — resolve via a fresh query
                sub = client.get_number(f.value)
                ffid = sub["fid"] if sub else None
            if ffid:
                client.report_ecm(ffid, f.ecm.b1, f.ecm.sigma,
                                  f.ecm.b2 if f.ecm.stage >= 2 else None)
                reported_ecm += 1
        if reported_ecm:
            fdb_info["ecm_reported"] = reported_ecm

    merged.sort(key=lambda f: f.value)
    return res, merged, fdb_info


# ---------------------------------------------------------------------------
# output
# ---------------------------------------------------------------------------

def _fmt_factors(factors: list[FoundFactor]) -> str:
    return " * ".join(f"{f.value} ({f.tag}, {f.source}" +
                      (f", sigma={f.ecm.sigma}, B1={f.ecm.b1}" if f.ecm else "")
                      + ")" for f in factors)


def _merged_complete(n: int, factors: list[FoundFactor]) -> bool:
    prod = 1
    for f in factors:
        if not is_probable_prime(f.value):
            return False
        prod *= f.value
    return prod == n


def emit(n: int, res: FactorResult, merged: list[FoundFactor] | None,
         fdb_info: dict | None, elapsed: float, as_json: bool, wd_kept: str | None,
         quiet: bool) -> bool:
    factors = merged if merged is not None else res.factors
    complete = _merged_complete(n, factors) if merged is not None else res.complete
    if as_json:
        obj = {
            "n": str(n),
            "complete": bool(complete),
            "factors": [{"value": str(f.value), "tag": f.tag,
                         "proven": f.proven, "source": f.source,
                         **({"ecm": {"b1": f.ecm.b1, "b2": f.ecm.b2,
                                     "sigma": f.ecm.sigma,
                                     "stage": f.ecm.stage}}
                            if f.ecm else {})} for f in factors],
            "elapsed_s": round(elapsed, 3),
            "stages": {k: round(v, 3) for k, v in res.stages.items()},
            "counts": res.counts,
            "fdb": fdb_info,
            "error": res.error,
        }
        if wd_kept:
            obj["workdir"] = wd_kept
        print(json.dumps(obj), flush=True)
    else:
        if complete:
            print(f"{n} = " + " * ".join(str(f.value) for f in factors), flush=True)
            if not quiet:
                tags = " ".join(f.tag for f in factors)
                print(f"  [{tags}]  {elapsed:.1f}s"
                      + (f"  fdb={fdb_info}" if fdb_info else ""), flush=True)
        else:
            partial = " * ".join(str(f.value) for f in factors) or "-"
            print(f"{n}: FAILED ({res.error}) in {elapsed:.1f}s", flush=True)
            print(f"  partial: {partial}", flush=True)
            if wd_kept:
                print(f"  logs kept in {wd_kept}", flush=True)
    return bool(complete)


# ---------------------------------------------------------------------------

def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    if args.quiet and args.json:
        pass  # json lines still go to stdout; -q only mutes the detail line
    numbers = load_numbers(args)
    kwargs = dict(
        bin_dir=Path(args.bin_dir) if args.bin_dir else None,
        device=args.device,
        sieve_procs=args.sieve_procs,
        plan=args.plan,
        xover=args.xover,
        verbose=args.verbose,
        quiet=args.quiet,
        keep_workdir=args.keep,
        timeout_s=args.timeout,
        force_method=args.force,
        ecm_only=args.ecm_only,
        topup_frac=args.topup_frac,
    )
    if args.threads:
        kwargs["threads"] = args.threads
    if args.ecm_stage2_maxmem is not None:
        kwargs["ecm_stage2_maxmem_mb"] = args.ecm_stage2_maxmem
    if args.poly_budget is not None:
        kwargs["poly_budget_s"] = args.poly_budget
    ctx = RunContext(**kwargs)
    engine = Engine(ctx)

    fdb_client = None
    if args.factor_db:
        token = read_token(args.db_token)
        fdb_client = FactorDbClient(
            token=token, dry_run=args.db_dry_run,
            log=(lambda msg: ctx.log(msg)) if not args.quiet else None)

    if args.workdir:
        base = Path(args.workdir).expanduser()
        base.mkdir(parents=True, exist_ok=True)
        owns_base = False
    else:
        base = Path(tempfile.mkdtemp(prefix="yagfu-"))
        owns_base = True

    failures = 0
    interrupted = False
    try:
        for idx, n in enumerate(numbers):
            wd = base / f"{idx:03d}-{str(n)[:12]}"
            if wd.exists():
                shutil.rmtree(wd)
            wd.mkdir(parents=True)
            ctx.log(f"=== [{idx + 1}/{len(numbers)}] c{len(str(n))} "
                    f"workdir {wd} ===")
            t0 = time.monotonic()
            stopped = False   # this number ended via Ctrl+C: keep its wd
            try:
                if fdb_client is not None:
                    res, merged, fdb_info = factor_with_db(
                        n, fdb_client, engine, wd,
                        report=not args.db_no_report)
                    ok = emit(n, res, merged, fdb_info, time.monotonic() - t0,
                              args.json, str(wd) if res.interrupted else None,
                              args.quiet)
                else:
                    res = engine.factor(n, wd)
                    ok = emit(n, res, None, None, time.monotonic() - t0,
                              args.json, str(wd) if res.interrupted else None,
                              args.quiet)
                if not ok:
                    failures += 1
                if res.interrupted:
                    interrupted = True
                    stopped = True
                    break
            except KeyboardInterrupt:
                interrupted = True
                stopped = True
                break
            except Exception as e:  # noqa: BLE001
                failures += 1
                print(f"{n}: CRASHED: {e!r}", flush=True)
            finally:
                # scratch is scratch: a finished number (success, failure,
                # crash) cleans up; only an interrupted one is kept
                cleanup_workdir(ctx, wd, stopped)
    except KeyboardInterrupt:
        interrupted = True  # e.g. between numbers, or during fdb calls
    finally:
        if interrupted:
            print("yagfu: interrupted by user; partial results above",
                  file=sys.stderr, flush=True)
        if owns_base:
            if interrupted:
                ctx.log(f"yagfu: interrupted-run workdirs kept under {base}")
            elif not ctx.keep_workdir:
                shutil.rmtree(base, ignore_errors=True)

    if interrupted:
        return 130  # 128 + SIGINT
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
