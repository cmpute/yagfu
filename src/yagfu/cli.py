"""yagfu command-line interface.

    yagfu 123456789...            factor one number
    yagfu -f numbers.txt          factor one number or id:<fid> per line
    yagfu -db 12345...            consult and report to factordb.com
    yagfu -db -snfs id:110000...  factor a factordb record by id, using
                                  its SNFS suggestion when it has one
"""

from __future__ import annotations

import argparse
import json
import re
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path

from . import __version__
from .backends.nfs import (
    _score_snfs_candidates,
    bench_gnfs_polyselect,
    parse_snfs_suggestion,
    snfs_job_from_file,
)
from .config import RunContext, cleanup_workdir, resolve_devices
from .params import snfs_gnfs_equiv
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
               "  yagfu -f composites.txt -db\n"
               "  yagfu --bench-nfs id:1100000008650831857\n",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("number", nargs="?",
                    help="decimal number to factor, or id:<fid> for a "
                         "factordb record lookup (with -db)")
    ap.add_argument("-f", "--file", help="text file with one number or "
                                         "id:<fid> per line")
    ap.add_argument("-db", "--factor-db", dest="factor_db", action="store_true",
                    help="fetch known factorizations from factordb.com, factor "
                         "what remains, and report new factors back")
    ap.add_argument("-snfs", "--snfs", dest="snfs", action="store_true",
                    help="with -db: when factordb suggests an SNFS "
                         "polynomial for the number, factor it with SNFS "
                         "after the ECM pretest (ECM target and sieve "
                         "parameters follow the SNFS difficulty); numbers "
                         "without a suggestion keep the usual SIQS/NFS "
                         "routing")
    ap.add_argument("--plan", choices=("light", "normal", "deep"),
                    default="normal", help="ECM pretest depth (default normal)")
    ap.add_argument("--xover", type=int, default=100,
                    help="SIQS -> NFS crossover in digits (default 100)")
    ap.add_argument("--device", default="0",
                    help='CUDA device, a comma-separated list, or "all" '
                         "for every card visible in this shell: ECM stage "
                         "1, NFS sieving and polyselect fan out across "
                         "them; ECM stage 2 and Lanczos use the first "
                         "device (SIQS goes cluster across all cards "
                         "from c80)")
    ap.add_argument("--threads", type=int, default=None,
                    help="CPU threads for msieve stages (default nproc-2)")
    ap.add_argument("--sieve-procs", type=int, default=1,
                    help="concurrent cuda-sieve processes per band, per "
                         "GPU (default 1)")
    ap.add_argument("--bin-dir", help="directory with the backend binaries")
    ap.add_argument("--workdir", help="base scratch directory "
                                      "(default: a temp dir). A number's "
                                      "workdir is removed only when it "
                                      "factors completely; kept on failure, "
                                      "Ctrl+C, or --keep")
    ap.add_argument("--keep", action="store_true",
                    help="keep per-number workdirs even on success")
    ap.add_argument("--timeout", type=float, default=None,
                    help="per-number wall-clock budget in seconds")
    ecm_mode = ap.add_mutually_exclusive_group()
    ecm_mode.add_argument("--ecm-only", action="store_true",
                          help="never run SIQS/NFS: keep raising the ECM "
                               "level until the number factors or Ctrl+C "
                               "(the t65 level is sampled forever once "
                               "reached)")
    ecm_mode.add_argument("--no-ecm", action="store_true",
                          help="skip the ECM stage entirely: TD -> rho -> "
                               "SIQS/NFS (handy for reruns when the ECM "
                               "pretest is known to be useless)")
    ap.add_argument("--ecm-t-level", metavar="LO-HI", default=None,
                    help="run the ECM pretest over exactly this t-level "
                         "range, e.g. 38.0-44.0: the ledger is credited to "
                         "t38 without curves (below it nothing runs) and "
                         "the ladder climbs to t44, landing exactly on it "
                         "(each stage runs only its overlap's share, "
                         "earlier stages' cross-level credit counted); "
                         "overrides --plan")
    ap.add_argument("--force", choices=("siqs", "nfs"), default=None,
                    help=argparse.SUPPRESS)  # debug: skip the selector
    ap.add_argument("--ecm-stage2-maxmem", type=int, default=None, metavar="MB",
                    help="gpuecm2 stage-2 GPU memory cap in MB (default: 90%% "
                         "of the GPU's total VRAM; 0 = no cap, gpuecm2's own "
                         "free-memory budgeting)")
    ap.add_argument("--poly-budget", type=float, default=None,
                    help="NFS polyselect hard wall cap in seconds: one "
                         "full-range search, SIGINT at the cap, keep the "
                         "best-so-far polynomial.  Without it the search "
                         "stops at the quality target (see --poly-frac), "
                         "or when the best E stalls near it, or when the "
                         "whole coefficient range is searched")
    ap.add_argument("--bench-nfs", metavar="N|id:FID", default=None,
                    help="score the best SNFS polynomial against a timed "
                         "GNFS polyselect on this number and exit — no "
                         "factoring runs.  SNFS polys come from the "
                         "factordb suggestion (re-ranked by normalized "
                         "Murphy E, exactly like -snfs); GNFS runs "
                         "msieve -np for --bench-nfs-time seconds.  "
                         "Prints both E values and which method to use")
    ap.add_argument("--bench-nfs-time", type=float, default=60.0,
                    metavar="S", dest="bench_nfs_time",
                    help="GNFS polyselect wall limit for --bench-nfs "
                         "(default 60)")
    ap.add_argument("--poly-frac", type=float, default=0.65,
                    help="polyselect quality gate as a fraction of the E "
                         "msieve says to expect (default 0.65: msieve's bar "
                         "assumes a search several times longer than a "
                         "bounded GPU one; 1.0 insists on the full bar)")
    ap.add_argument("--topup-frac", type=float, default=1.0,
                    help="NFS ladder step: the next -nc1 waits for at most "
                         "one topup step past the last filter's data "
                         "(raw x (1 + frac)); every rung is capped at this "
                         "even when the estimator aims lower (default 1.0 = "
                         "double each filter round)")
    ap.add_argument("--target-safety", type=float, default=1.05,
                    help="NFS margin over the requirement extrapolated from "
                         "the measured excess curve; the extrapolation "
                         "already lands on the high side, this covers the "
                         "residual (default 1.05)")
    ap.add_argument("--wave-max", type=int, default=16_000_000,
                    help="NFS max relations per sieve wave: the verdict "
                         "reaction granularity, ~2 min of 4-GPU sieving "
                         "(default 16000000)")
    ap.add_argument("--sieve-logI", type=int, default=None, metavar="N",
                    dest="sieve_logI",
                    help="NFS: override the params table's siever width "
                         "(log2 I).  A job limited by the special-q window "
                         "needs a wider rectangle even at a worse "
                         "per-relation cost.  Default: the params table")
    ap.add_argument("--sieve-J", type=int, default=None, metavar="N",
                    dest="sieve_J",
                    help="NFS: cuda-sieve J (sieve height).  Default "
                         "2^(logI-1).  Pin it when raising --sieve-logI to "
                         "widen the a-range without widening the b-range")
    ap.add_argument("--db-token", default=None,
                    help="factordb token file (default ~/.factordb_token)")
    ap.add_argument("--db-no-report", action="store_true",
                    help="with -db: only query, never report")
    ap.add_argument("--db-dry-run", action="store_true",
                    help="with -db: print requests instead of sending")
    ap.add_argument("--snfs-job", help=argparse.SUPPRESS)  # debug: feed an
                    # GGNFS-format SNFS job file directly, no factordb
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

def load_numbers(args) -> list[int | tuple[str, str]]:
    """Decimal ints, or ("id", fid) items naming factordb records.

    The id:<fid> input form is a -db feature; without -db it is rejected."""
    def parse(s: str, where: str) -> int | tuple[str, str]:
        if s.startswith("id:"):
            if not args.factor_db:
                sys.exit(f"error: {where}: {s!r} needs -db (factordb lookup)")
            fid = s[3:].strip()
            if not fid.isdigit():
                sys.exit(f"error: {where}: bad factordb id: {s!r}")
            return ("id", fid)
        try:
            return int(s)
        except ValueError:
            sys.exit(f"error: {where}: not a decimal integer or id:N: {s!r}")

    if args.file:
        if args.number:
            sys.exit("error: pass either a number or -f FILE, not both")
        numbers = []
        for lineno, line in enumerate(Path(args.file).read_text().splitlines(), 1):
            s = line.strip()
            if not s or s.startswith("#"):
                continue
            numbers.append(parse(s, f"{args.file}:{lineno}"))
        if not numbers:
            sys.exit(f"error: {args.file} contains no numbers")
        return numbers
    if not args.number:
        sys.exit(build_parser().format_usage())
    return [parse(args.number.strip(), "argument")]


# ---------------------------------------------------------------------------
# factordb integration
# ---------------------------------------------------------------------------

def _db_factor_list(client: FactorDbClient, n: int, fid: str | None):
    """Prime factors factordb already knows for n (decimals).

    A CF record may store composite factors (e.g. n = p * c131); those must
    stay in the remainder for the engine to factor, never count as known
    primes — otherwise the remainder hits 1, the engine never runs, and the
    composite in the merged list reads as a permanent "incomplete"."""
    if fid is None:
        return []
    decs = client.get_factor_decimals(fid)
    out = []
    for d in decs:
        try:
            v = int(d)
        except ValueError:
            continue
        if 1 < v < n and n % v == 0 and is_probable_prime(v):
            out.append(v)
    return out


def factor_with_db(n: int, client: FactorDbClient, engine: Engine, wd: Path,
                   report: bool = True, rec: dict | None = None,
                   use_snfs: bool = False
                   ) -> tuple[FactorResult, list[FoundFactor], dict]:
    """factordb-aware driver.

    Returns (engine result, known-db factors merged in, fdb summary dict).
    The engine result's factor list contains every prime factor of n
    (db-known ones included, source 'fdb').  `rec` may carry a record
    already fetched (get_number_by_id path), skipping the first query.

    With use_snfs, a SNFS suggestion in the record is handed to the engine
    when it applies to the number we factor ourselves (the suggestion maps
    the whole record; a remainder that lost db-known factors would need
    its own polynomial and keeps the usual routing)."""
    fdb_info: dict = {}
    if rec is None:
        rec = client.get_number(n, snfs=use_snfs)
    known: list[int] = []
    to_factor = n
    status = rec["status"] if rec else None
    fid = rec["fid"] if rec else None
    fdb_info["status"] = status
    if fid is not None:
        fdb_info["id"] = str(fid)

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

    snfs = None
    if use_snfs and to_factor > 1 and to_factor == n:
        snfs = parse_snfs_suggestion(rec.get("snfs") if rec else None, n,
                                     fid=fid)
        if snfs is None:
            engine.ctx.logv("snfs: no usable factordb suggestion in the "
                            "record")
        else:
            fdb_info["snfs"] = round(snfs.scaled, 2)
            e = f", E {snfs.e:g}" if snfs.e else ""
            engine.ctx.log(f"snfs: factordb suggests "
                           f"{snfs.form or 'a polynomial'} "
                           f"(difficulty {snfs.difficulty:g}, scaled "
                           f"{snfs.scaled:g} ~ gnfs "
                           f"c{snfs_gnfs_equiv(snfs.scaled):g}{e})")

    if to_factor > 1:
        res = engine.factor(to_factor, wd, snfs=snfs)
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


def _verify_devices(devices: tuple[str, ...]) -> None:
    """Fail fast on a typo'd --device when nvidia-smi can check it.

    Best-effort: without a usable nvidia-smi (GPU-less machines, CI) the
    check is skipped — the backends will surface their own errors then.
    """
    try:
        out = subprocess.run(["nvidia-smi", "-L"], capture_output=True,
                             text=True, timeout=15)
    except (OSError, subprocess.SubprocessError):
        return
    if out.returncode != 0:
        return
    lines = [l for l in out.stdout.splitlines() if l.strip()]
    if not lines:
        return
    known: set[str] = set()
    for i, line in enumerate(lines):
        known.add(str(i))
        m = re.search(r"\(UUID:\s*GPU-([\w-]+)\)", line)
        if m:
            known.add(m.group(1))
    for d in devices:
        uuid = d[4:] if d.startswith("GPU-") else d
        if d not in known and uuid not in known:
            sys.exit(f"error: device {d!r} not found by nvidia-smi")


# ---------------------------------------------------------------------------
# --bench-nfs: SNFS vs GNFS by Murphy E, no factoring
# ---------------------------------------------------------------------------

def bench_nfs(args, ctx: RunContext, client: FactorDbClient) -> int:
    """Score the best SNFS polynomial against a timed GNFS polyselect and
    recommend a method, then exit.  Nothing is factored: the SNFS side is
    the -snfs poly selection (factordb suggestion re-ranked by normalized
    Murphy E via score-poly), the GNFS side is msieve -np under a plain
    wall limit (--bench-nfs-time).

    Both Es are reported in the phi.kyleaskine.com normalization (Murphy E
    * 1.6^(6 - degree)), which is what makes polys of different degrees —
    here typically an SNFS degree 4-6 against msieve's GNFS degree pick —
    comparable; the recommendation goes by it."""
    spec = args.bench_nfs.strip()
    fid = None
    n: int | None = None
    if spec.startswith("id:"):
        fid = spec[3:].strip()
        if not fid.isdigit():
            print(f"error: bad factordb id: {spec!r}", file=sys.stderr)
            return 2
    else:
        try:
            n = int(spec)
        except ValueError:
            print(f"error: not a decimal integer or id:N: {spec!r}",
                  file=sys.stderr)
            return 2
    if args.bench_nfs_time <= 0:
        print("error: --bench-nfs-time must be positive", file=sys.stderr)
        return 2

    rec: dict | None = None
    if fid is not None:
        rec = client.get_number_by_id(fid, snfs=True)
        if rec is None or not rec.get("decimal"):
            print(f"id:{fid}: factordb record unavailable", flush=True)
            return 1
        n = int(rec["decimal"])
    else:
        # best effort: a plain number benches as GNFS unless factordb
        # carries an SNFS suggestion for it
        rec = client.get_number(n, snfs=True)

    digits = len(str(n))
    status = (rec or {}).get("status")
    print(f"bench-nfs: c{digits}" + (f" (id:{fid})" if fid else "")
          + (f", factordb status {status}" if status else ""), flush=True)
    if status in ("P", "PR"):
        print("  the record is already prime; nothing to bench", flush=True)
        return 1

    if args.workdir:
        wd = Path(args.workdir).expanduser() / "bench-nfs"
    else:
        wd = Path(tempfile.mkdtemp(prefix="yagfu-bench-nfs-"))
    wd.mkdir(parents=True, exist_ok=True)
    try:
        return _bench_nfs_report(args, ctx, client, n, fid, rec, wd)
    finally:
        if args.keep:
            ctx.log(f"bench: kept {wd}")
        else:
            shutil.rmtree(wd, ignore_errors=True)


def _bench_nfs_report(args, ctx: RunContext, client: FactorDbClient,
                      n: int, fid: str | None, rec: dict | None,
                      wd: Path) -> int:
    limit = args.bench_nfs_time

    # ---- SNFS side: the -snfs selection path, scored ----
    snfs = parse_snfs_suggestion((rec or {}).get("snfs"), n, fid=fid) \
        if rec else None
    snfs_e = snfs_deg = None
    snfs_method = None
    if snfs is None:
        print("  snfs: no factordb suggestion for this number", flush=True)
    else:
        e0 = f", factordb E {snfs.e:g}" if snfs.e else ""
        print(f"  snfs: suggestion {snfs.form or '(polynomial)'}: "
              f"difficulty {snfs.difficulty:g}, scaled {snfs.scaled:g} "
              f"(~ gnfs c{snfs_gnfs_equiv(snfs.scaled):g}{e0})", flush=True)
        scored = _score_snfs_candidates([snfs, *snfs.alternates], ctx, wd)
        for cand, ne, _sk in scored:
            deg = len(cand.coeffs) - 1
            print(f"    degree-{deg} candidate: normalized Murphy E "
                  f"{ne:.3e}", flush=True)
        if scored:
            best, snfs_e, _sk = scored[0]
            snfs_deg = len(best.coeffs) - 1
            snfs_method = "score-poly"
        elif snfs.e:
            # no helper: factordb's own score is in the same units as
            # score-poly's raw Murphy E — normalize it the same way
            snfs_deg = len(snfs.coeffs) - 1
            snfs_e = float(snfs.e) * 1.6 ** (6 - snfs_deg)
            snfs_method = "factordb e"
            print("    (no score-poly helper; factordb's E, same "
                  "normalization)", flush=True)
        else:
            print("    no score-poly helper and no factordb E",
                  flush=True)

    # ---- GNFS side: a plain timed polyselect ----
    b = bench_gnfs_polyselect(n, ctx, wd / "gnfs", limit)
    gnfs_e = gnfs_deg = None
    gnfs_raw = b["e"] if b else None
    gnfs_bar = b["bar"] if b else None
    if b and gnfs_raw is not None and b["degree"]:
        gnfs_deg = b["degree"]
        gnfs_e = gnfs_raw * 1.6 ** (6 - gnfs_deg)
    print(f"  gnfs: msieve -np polyselect, {limit:g}s limit"
          + (f", degree {gnfs_deg}" if gnfs_deg else "")
          + (f": E {gnfs_raw:.3e}" if gnfs_raw is not None
             else ": no polynomial in the limit"), flush=True)
    if gnfs_bar is not None:
        if gnfs_raw is not None:
            print(f"    msieve expects E from {gnfs_bar:.3e} for a full "
                  f"search; this run reached {gnfs_raw / gnfs_bar:.0%} "
                  f"of it", flush=True)
        else:
            print(f"    msieve expects E from {gnfs_bar:.3e} for a full "
                  f"search; nothing reached", flush=True)

    # ---- verdict ----
    print(f"  best SNFS Murphy E: "
          + (f"{snfs_e:.3e} (normalized, degree {snfs_deg}, "
             f"{snfs_method})" if snfs_e is not None else "unavailable"),
          flush=True)
    print(f"  best GNFS Murphy E: "
          + (f"{gnfs_e:.3e} (normalized, degree {gnfs_deg}, "
             f"msieve -np)" if gnfs_e is not None else "unavailable"),
          flush=True)
    if snfs_e is None and gnfs_e is None:
        print("  no usable E on either side; cannot recommend", flush=True)
        return 1
    if gnfs_e is None:
        pick, why = "SNFS", "GNFS polyselect produced no polynomial"
    elif snfs_e is None:
        pick, why = "GNFS", "no SNFS polynomial available"
    else:
        hi, lo = max(snfs_e, gnfs_e), min(snfs_e, gnfs_e)
        pick = "SNFS" if snfs_e >= gnfs_e else "GNFS"
        why = f"normalized Murphy E ratio {hi / lo:.2f}x"
        if hi / lo < 1.15:
            why += " — too close to call, either sieves fine"
    print(f"  recommendation: {pick} ({why})", flush=True)
    if gnfs_raw is not None and gnfs_bar is not None and gnfs_raw < gnfs_bar:
        print(f"  note: a {limit:g}s GNFS polyselect undershoots a full "
              f"search; the GNFS side improves with --bench-nfs-time",
              flush=True)
    return 0


# ---------------------------------------------------------------------------

def _make_ctx(args) -> RunContext:
    raw_devices = tuple(d.strip() for d in args.device.split(",") if d.strip())
    if not raw_devices:
        sys.exit("error: --device needs at least one ordinal")
    try:
        devices = resolve_devices(raw_devices)
    except ValueError as e:
        sys.exit(f"error: {e}")
    _verify_devices(devices)
    ecm_t_level = None
    if args.ecm_t_level is not None:
        if args.ecm_only or args.no_ecm:
            ap.error("--ecm-t-level does not combine with --ecm-only/--no-ecm")
        mt = re.fullmatch(r"(\d+(?:\.\d+)?)\s*-\s*(\d+(?:\.\d+)?)",
                          args.ecm_t_level.strip())
        if mt is None:
            ap.error("--ecm-t-level wants LO-HI in digits, e.g. 38.0-44.0")
        lo, hi = float(mt.group(1)), float(mt.group(2))
        if hi > 65.0:
            ap.error("--ecm-t-level: the ECM ladder tops out at t65")
        if lo > hi:
            ap.error("--ecm-t-level: LO must not exceed HI")
        ecm_t_level = (lo, hi)
    kwargs = dict(
        bin_dir=Path(args.bin_dir) if args.bin_dir else None,
        device=devices[0],
        devices=devices,
        sieve_procs=args.sieve_procs,
        plan=args.plan,
        xover=args.xover,
        verbose=args.verbose,
        quiet=args.quiet,
        keep_workdir=args.keep,
        timeout_s=args.timeout,
        force_method=args.force,
        ecm_only=args.ecm_only,
        no_ecm=args.no_ecm,
        ecm_t_level=ecm_t_level,
        topup_frac=args.topup_frac,
        target_safety=args.target_safety,
        wave_max=args.wave_max,
        sieve_logI=args.sieve_logI,
        sieve_J=args.sieve_J,
    )
    if args.threads:
        kwargs["threads"] = args.threads
    if args.ecm_stage2_maxmem is not None:
        kwargs["ecm_stage2_maxmem_mb"] = args.ecm_stage2_maxmem
    if args.poly_budget is not None:
        kwargs["poly_budget_s"] = args.poly_budget
    kwargs["poly_frac"] = args.poly_frac
    return RunContext(**kwargs)


def _bench_nfs_entry(args) -> int:
    """--bench-nfs dispatch: scores polys and exits, no factoring."""
    ctx = _make_ctx(args)
    client = FactorDbClient(
        token=read_token(args.db_token), dry_run=args.db_dry_run,
        log=(lambda msg: ctx.log(msg)) if not args.quiet else None)
    try:
        return bench_nfs(args, ctx, client)
    except KeyboardInterrupt:
        print("yagfu: interrupted by user", file=sys.stderr, flush=True)
        return 130


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    if args.quiet and args.json:
        pass  # json lines still go to stdout; -q only mutes the detail line
    if args.snfs_job and args.factor_db:
        sys.exit("error: pass either --snfs-job or -db, not both")
    if args.snfs and not args.factor_db and not args.snfs_job:
        sys.exit("error: -snfs needs -db (the SNFS suggestion comes from "
                 "factordb) or --snfs-job")
    if args.bench_nfs:
        return _bench_nfs_entry(args)
    numbers = load_numbers(args)

    snfs_direct = None
    if args.snfs_job:
        try:
            snfs_direct = snfs_job_from_file(
                Path(args.snfs_job).read_text())
        except OSError as e:
            sys.exit(f"error: cannot read {args.snfs_job}: {e}")
        if snfs_direct is None:
            sys.exit(f"error: {args.snfs_job}: no SNFS polynomial found "
                     "(need an n:/polynomial job file)")
        if len(numbers) != 1 or numbers[0] != snfs_direct.n:
            sys.exit(f"error: {args.snfs_job} is a job for "
                     f"{snfs_direct.n}; pass exactly that number")

    ctx = _make_ctx(args)
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
    kept_any = False            # any workdir survives (failure/interrupt)
    try:
        for idx, item in enumerate(numbers):
            # an id:<fid> input resolves to its record (and number) first;
            # factor_with_db then reuses the record instead of re-querying
            rec = None
            if isinstance(item, tuple):
                fid = item[1]
                rec = fdb_client.get_number_by_id(fid, snfs=args.snfs)
                if rec is None or not rec.get("decimal"):
                    failures += 1
                    print(f"id:{fid}: FAILED (factordb record unavailable)",
                          flush=True)
                    continue
                n = int(rec["decimal"])
            else:
                n = item
            wd = base / f"{idx:03d}-{str(n)[:12]}"
            # an existing workdir is a resume candidate, never wiped
            wd.mkdir(parents=True, exist_ok=True)
            ctx.log(f"=== [{idx + 1}/{len(numbers)}] c{len(str(n))} "
                    f"workdir {wd} ===")
            t0 = time.monotonic()
            keep_wd = True      # failure/interrupt default; success clears it
            try:
                if fdb_client is not None:
                    res, merged, fdb_info = factor_with_db(
                        n, fdb_client, engine, wd,
                        report=not args.db_no_report, rec=rec,
                        use_snfs=args.snfs)
                    ok = emit(n, res, merged, fdb_info, time.monotonic() - t0,
                              args.json,
                              str(wd) if not _merged_complete(n, merged) else None,
                              args.quiet)
                else:
                    res = engine.factor(n, wd, snfs=snfs_direct)
                    ok = emit(n, res, None, None, time.monotonic() - t0,
                              args.json,
                              str(wd) if not res.complete else None,
                              args.quiet)
                if res.interrupted:
                    interrupted = True
                    break
                if ok:
                    keep_wd = False       # success: scratch is scratch
                else:
                    failures += 1         # failed: keep for post-mortem
            except KeyboardInterrupt:
                interrupted = True
                break
            except Exception as e:  # noqa: BLE001
                failures += 1
                print(f"{n}: CRASHED: {e!r}", flush=True)
                print(f"  logs kept in {wd}", flush=True)
            finally:
                kept_any |= keep_wd
                # only a complete factorization cleans up after itself
                cleanup_workdir(ctx, wd, keep_wd)
    except KeyboardInterrupt:
        interrupted = True  # e.g. between numbers, or during fdb calls
    finally:
        if interrupted:
            print("yagfu: interrupted by user; partial results above",
                  file=sys.stderr, flush=True)
        if owns_base:
            if kept_any or ctx.keep_workdir:
                ctx.log(f"yagfu: kept workdirs under {base}")
            else:
                shutil.rmtree(base, ignore_errors=True)

    if interrupted:
        return 130  # 128 + SIGINT
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
