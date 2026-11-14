"""GNFS backend: msieve polyselect -> cuda-sieve sieving -> msieve postproc.

Ported from bench_factor.py's run_nfs (validated on this hardware), with
the per-size parameter tables generalized by interpolation (params.py).
With a supplied SNFS polynomial (SnfsJob: factordb's get_number `snfs`
suggestion or a --snfs-job file) polyselect is skipped entirely and the
parameters come from the SNFS difficulty table instead.
Pipeline shape:

* polyselect: one msieve -np per GPU over disjoint leading-coefficient
  slices for >= 85 digits; our msieve fork streams every new overall-
  best E to its log, and yagfu SIGINTs the search the moment the best E
  meets the bar msieve itself prints — else the whole range is searched
  (an explicit --poly-budget reverts to a hard wall cap).  Classic
  base-m polynomial below that (this msieve build emits no GPU stage-1
  candidates under ~85 digits).
* sieving: cuda-sieve waves with an adaptive width, special-q from alim/2,
  streamed to per-wave files and merged into msieve.dat; the msieve -nc1
  verdict drives the loop and runs in the background (it only ever reads
  msieve.dat, and msieve.dat only grows while no -nc1 is reading it — so
  the stop check stays exact while the filter cost hides under GPU time).
  One wave is always in flight, so a verdict that lands mid-wave acts on
  the NEXT wave immediately, and "enough" additionally kills the wave in
  flight and drops the queue: oversieving past the enough-point costs at
  most one wave.  How much data the next -nc1 waits for is the estimator's
  rung: one topup step past the last filter (the ladder), capped by the
  requirement extrapolated from the measured excess-vs-unique curve —
  -nc1 re-reads everything each round, so the filter count, not the wave
  count, is the serial cost.
* postprocessing: -nc2 matbuild, GPU Lanczos -nc2, sqrt -nc3.
"""

from __future__ import annotations

import json
import math
import os
import re
import shutil
import signal
import struct
import subprocess
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from types import SimpleNamespace

from ..config import RunContext
from ..fmt import fmt_count, fmt_dur
from ..params import (
    NFS_MIN_DIGITS,
    NFS_MSIEVE_POLY_MIN_DIGITS,
    SNFS_2LP_UNMEASURED_ABOVE,
    nfs_params,
    poly_budget,
    snfs_lp_structure,
    snfs_sieve_params,
)
from . import BinNotFound, find_bin
from .runner import (
    count_lines,
    read_log,
    run_cmd,
    run_cmd_progress,
    run_procs_parallel,
)

NP_TIMEOUT_S = 2 * 3600
#: wall cap for the explicit --poly-budget mode, the CPU-engine retry,
#: and the fallback when the logs carry no E bar (an older msieve
#: binary).  The default quality-gated search has no cap: it ends with
#: the bar or with msieve's own range
POLY_MAX_WAIT_S = 2 * 3600
#: how often the watcher re-reads the shard logs for the running best E
POLY_POLL_S = 20.0
#: stop when the best E has not improved for this long and already sits
#: this close to the bar: the best E grows only logarithmically with
#: search effort, so a stalled record means hours more search for
#: percent-level gains (measured: GPU searches stall at 60-70% of the
#: printed bar well before the coefficient range is exhausted)
POLY_STALL_S = 30 * 60
POLY_STALL_FLOOR = 0.5
SIEVE_TIMEOUT_S = 6 * 3600
#: wall-clock target for one sieve wave.  A wave is sized by its relation
#: target *and* capped so the last completed wave's measured wall/width
#: ratio predicts no more than this; the cap is 3/4 of SIEVE_TIMEOUT_S, so a
#: slow q band or an operator geometry change cannot reach the hard timeout
#: (a timed-out wave resumes from its checkpoint, but it spends the engine's
#: single same-selection retry).
WAVE_TARGET_S = int(4.5 * 3600)
#: relations/q grows sublinearly with the sieve area: doubling the area
#: measured 1.50x relations here (1.3-1.5x in bench/RUNBOOK), i.e.
#: area^0.59.  Used only to rescale the first wave after an operator changes
#: logI/J, when the state's relations/q still describes the old geometry.
GEOM_YIELD_EXP = 0.6
NC_TIMEOUT_S = 2 * 3600
#: The Lanczos wall time is a property of the matrix (dimensions x weight)
#: and the hardware, not a constant: the 15,253- c214 run in ws-15c253
#: (43.06M dims, 78.5 nnz/col) needed ~28h, and the old 12h cap killed a
#: perfectly healthy run at 42%.  There is deliberately no absolute cap:
#: the stage is bounded by *liveness* — the Lanczos prints a progress line
#: every few thousand dimensions and rewrites msieve.dat.chk every ~1.5M
#: dimensions (hourly at that size), so LA_STALL_S of silence means a
#: stalled device, not a long job.  A killed attempt resumes from the
#: checkpoint with -ncr instead of failing the job.
LA_STALL_S = 2 * 3600
LA_MAX_ATTEMPTS = 3
#: Same liveness reasoning for the square root: one dependency (multiply
#: every relation, then the q-adic Newton iteration) took ~28 min on the
#: c214 and msieve may try up to 64, so a fixed 2h cap could not even
#: finish the c221 run (2h20m for 5 dependencies).  The phases pace
#: progress lines; SQRT_STALL_S still catches a real hang.
SQRT_STALL_S = 2 * 3600


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
# SNFS: polynomials supplied from outside (factordb's get_number `snfs`
# suggestion, or a hand-written --snfs-job file).  yagfu never searches
# for SNFS polynomials itself; a factordb suggestion carries several
# candidate polys — the ones with a root mod n are kept and re-ranked by
# normalized Murphy E with the bundled score-poly helper (factordb orders
# by scaled difficulty, which is not the same ranking), and the winner's
# per-side sieve profile (rlim/alim/lpb/mfb, 3LP on the special-q side)
# comes from snfs_sieve_params.  The special-q side (factordb's `side`)
# reaches the cuda-sieve command.
# ---------------------------------------------------------------------------

@dataclass
class SnfsJob:
    n: int
    coeffs: list[int]          # c0..cd, ascending
    y0: int
    y1: int
    skew: float
    difficulty: float          # raw SNFS difficulty
    scaled: float              # scaled difficulty: the param-selection size
    side: str = "a"            # special-q side factordb wants: "a" | "r"
    e: float | None = None     # polyselect score, when factordb reports one
    form: str | None = None    # special form, e.g. "859465723^19-1"
    fid: object = None         # factordb record the suggestion came from
    alternates: list = field(default_factory=list)
    """other usable polys from the same factordb suggestion (same order
    factordb ranks them); filled by parse_snfs_suggestion, empty for
    --snfs-job files.  run_nfs re-ranks them by Murphy E when the
    score-poly helper is available."""


def poly_has_root_mod_n(job: SnfsJob) -> bool:
    """True when f(-Y0/Y1) = 0 (mod n) exactly: the polynomial divides
    n's algebraic form.  factordb's suggestion lists polys of the source
    *expression*, which need not all divide the stored cofactor (its own
    note: "N candidate(s) of the numerator do not divide the quotient"),
    so this is checked before anything is scored or sieved."""
    deg = len(job.coeffs) - 1
    n = job.n
    if math.gcd(abs(job.y1), n) != 1:
        return False
    num, den = (-job.y0) % n, job.y1 % n
    acc = 0
    for i, c in enumerate(job.coeffs):
        acc = (acc + c * pow(num, i, n) * pow(den, deg - i, n)) % n
    return acc == 0


def parse_snfs_suggestion(raw: object, n: int,
                          fid: object = None) -> SnfsJob | None:
    """factordb get_number's `snfs` object -> best SnfsJob for n, or None.

    The object is {"n": decimal, "polys": [...]} with the polys ordered
    best-first (factordb puts the lowest scaled difficulty first).  Every
    parseable poly that has a root mod n is kept: the first becomes the
    returned job and the rest ride along in `alternates` for the sieve
    stage to re-rank by Murphy E (factordb orders by scaled difficulty,
    which is not the same thing)."""
    if not isinstance(raw, dict) or str(raw.get("n", "")) != str(n):
        return None
    jobs: list[SnfsJob] = []
    for p in raw.get("polys") or []:
        if not isinstance(p, dict):
            continue
        try:
            coeffs = [int(c) for c in p["c"]]
            if not coeffs or coeffs[-1] == 0:
                continue
            difficulty = float(p["difficulty"])
            scaled = float(p.get("scaled", difficulty))
            job = SnfsJob(n=n, coeffs=coeffs,
                          y0=int(p["y0"]), y1=int(p["y1"]),
                          skew=float(p["skew"]),
                          difficulty=difficulty, scaled=scaled,
                          side=p.get("side", "a"),
                          e=p.get("e"), form=p.get("form"), fid=fid)
        except (KeyError, TypeError, ValueError):
            continue
        if job.side not in ("a", "r"):
            job.side = "a"
        if not poly_has_root_mod_n(job):
            continue
        jobs.append(job)
    if not jobs:
        return None
    best, jobs[0].alternates = jobs[0], jobs[1:]
    return best


def _snfs_poly_text(cand: SnfsJob) -> str:
    deg = len(cand.coeffs) - 1
    lines = [f"n: {cand.n}"]
    for i in range(deg, -1, -1):
        lines.append(f"c{i}: {cand.coeffs[i]}")
    lines += [f"Y1: {cand.y1}", f"Y0: {cand.y0}"]
    return "\n".join(lines) + "\n"


def _score_snfs_candidates(cands: list[SnfsJob], ctx,
                           wd: Path) -> list[tuple[SnfsJob, float, float]]:
    """score-poly each candidate: [(poly, normalized Murphy E, skew)],
    best first.

    The normalized E (Murphy E scaled by 1.6^(6 - degree), the
    phi.kyleaskine.com / msieve polyselect convention) is what makes
    polys of different degrees comparable; the scorer's golden-section
    E-optimal skew replaces the candidate's own.  Candidates above
    msieve's MAX_POLY_DEGREE (8) and failed scorings are dropped — the
    callers see exactly what re-ranking would act on.  Empty when the
    helper is missing or every scoring failed: the sieve stage then
    keeps factordb's order and the bench reports the factordb score."""
    try:
        scorer = find_bin("score-poly", ctx.bin_dir)
    except BinNotFound:
        ctx.log("nfs: snfs scoring skipped (no score-poly helper)")
        return []
    scored: list[tuple[SnfsJob, float, float | None]] = []
    for cand in cands:
        deg = len(cand.coeffs) - 1
        if deg > 8:      # msieve's MAX_POLY_DEGREE
            continue
        path = wd / f"snfs-cand-d{deg}-{id(cand) % 10000}.poly"
        path.write_text(_snfs_poly_text(cand))
        try:
            out = subprocess.run([scorer, str(path)], capture_output=True,
                                 text=True, timeout=600, check=True).stdout
            m_e = re.search(r"murphyE ([\d.eE+-]+)", out)
            m_s = re.search(r"skew ([\d.]+)", out)
            if not m_e:
                raise ValueError("no murphyE in score-poly output")
            e = float(m_e.group(1))
            skew = min(max(float(m_s.group(1)), 0.01), 1e4) if m_s \
                else cand.skew
            scored.append((cand, e * 1.6 ** (6 - deg), skew))
        except (subprocess.SubprocessError, ValueError, OSError) as exc:
            ctx.log(f"nfs: snfs candidate degree {deg} scoring failed "
                    f"({exc}); excluded from re-ranking")
    for old in wd.glob("snfs-cand-*.poly"):
        old.unlink(missing_ok=True)
    scored.sort(key=lambda t: -t[1])
    return scored


def _rank_snfs_candidates(job: SnfsJob, ctx, wd: Path) -> SnfsJob:
    """Re-rank the suggestion's candidates by normalized Murphy E.

    factordb orders its polys by scaled difficulty, which treats every
    degree alike; the normalized Murphy E (see _score_snfs_candidates)
    is the better predictor of relations per special-q.  Any failure
    falls back to the factordb order, so sieving never blocks on the
    helper."""
    if not job.alternates:
        return job
    scored = _score_snfs_candidates([job, *job.alternates], ctx, wd)
    if not scored:
        return job
    winner, _norm_e, skew = scored[0]
    if winner is not job:
        # keep the parse-time identity (fid, form, ...) of the winner but
        # preserve the original job's alternates list shape
        winner.skew = skew
        winner.alternates = [c for c, _, _ in scored if c is not winner]
        ctx.log("nfs: snfs re-ranking picks a poly other than factordb's "
                "scaled-difficulty favorite")
    else:
        winner.skew = skew
        winner.alternates = job.alternates
    ctx.log("nfs: snfs candidates by normalized Murphy E: " + ", ".join(
        f"d{len(c.coeffs) - 1} E={e:.3e}" for c, e, _ in scored))
    return winner


def snfs_job_from_file(text: str, fid: object = None) -> SnfsJob | None:
    """Parse a GGNFS-format SNFS job file (the shape factordb's snfs.php
    serves: n/type/size/skew/cN/Y0/Y1/params, comments carrying the
    difficulty and the special-q side).  None without an n:/polynomial."""
    kv: dict[str, str] = {}
    deg = -1
    difficulty = scaled = None
    side = "a"
    for line in text.splitlines():
        s = line.strip()
        if s.startswith("#"):
            m = re.search(r"difficulty:\s*([\d.]+)", s)
            if m and difficulty is None:
                difficulty = float(m.group(1))
            m = re.search(r"scaled difficulty:\s*([\d.]+)", s)
            if m:
                scaled = float(m.group(1))
            if "rational side" in s:
                side = "r"
            continue
        parts = s.split()
        if len(parts) != 2:
            continue
        k, v = parts[0].rstrip(":"), parts[1]
        kv[k] = v
        m = re.fullmatch(r"c(\d+)", k)
        if m:
            deg = max(deg, int(m.group(1)))
    if "n" not in kv or deg < 1 or "Y0" not in kv or "Y1" not in kv:
        return None
    try:
        job = SnfsJob(n=int(kv["n"]),
                      coeffs=[int(kv.get(f"c{i}", 0)) for i in range(deg + 1)],
                      y0=int(kv["Y0"]), y1=int(kv["Y1"]),
                      skew=float(kv.get("skew", "1.0")),
                      difficulty=(difficulty if difficulty is not None
                                  else float(kv["size"]) if "size" in kv
                                  else 0.0),
                      scaled=scaled if scaled is not None else 0.0,
                      side=side, fid=fid)
    except ValueError:
        return None
    if job.scaled <= 0:
        job.scaled = job.difficulty
    return job


def _snfs_job_text(job: SnfsJob, sieve_params: tuple) -> str:
    """GGNFS job file for a supplied SNFS polynomial.

    sieve_params is snfs_sieve_params's per-side profile (rlim, alim,
    lpbr, lpba, mfbr, mfba, rlambda, alambda, logI): the special-q side
    carries a 3LP mfb when the difficulty calls for it, and cuda-sieve
    reads every one of these from the job file."""
    (rlim, alim, lpbr, lpba, mfbr, mfba, rlambda, alambda, _logI) = \
        sieve_params
    deg = len(job.coeffs) - 1
    lines = [f"n: {job.n}"]
    if job.form:
        lines.append(f"# {job.form}, difficulty: {job.difficulty:g}, "
                     f"scaled difficulty: {job.scaled:g}")
    if job.side == "r":
        lines.append("# special-q on the rational side")
    lines += ["type: snfs", f"size: {round(job.difficulty)}",
              f"skew: {job.skew:.4f}"]
    for i in range(deg, -1, -1):
        lines.append(f"c{i}: {job.coeffs[i]}")
    lines += [f"Y0: {job.y0}", f"Y1: {job.y1}",
              f"rlim: {rlim}", f"alim: {alim}",
              f"lpbr: {lpbr}", f"lpba: {lpba}",
              f"mfbr: {mfbr}", f"mfba: {mfba}",
              f"rlambda: {rlambda}", f"alambda: {alambda}"]
    return "\n".join(lines) + "\n"


# ---------------------------------------------------------------------------
# sieving helpers
# ---------------------------------------------------------------------------

def _attempt_stdout(wd: Path, name: str) -> Path:
    """A fresh stdout capture per attempt: a relaunched wave or filter
    reuses its tag/seq, so the plain name would overwrite the previous
    attempt's capture — the only place a failed run's own error lines
    land (the child appends to its own --log, but the captured stdout is
    truncated on open).  The first attempt keeps the plain name; the k-th
    relaunch writes <name>.attempt<k>."""
    if not (wd / name).exists():
        return wd / name
    k = 2
    while (wd / f"{name}.attempt{k}").exists():
        k += 1
    return wd / f"{name}.attempt{k}"


def has_sieved_relations(wd: Path) -> bool:
    """True when a previous NFS run left relations in the workdir: a
    non-empty msieve.dat, or a checkpoint recording appended relations
    or filter points.  Such a workdir belongs to the selection that
    produced them — the relations, the dup cache and the sieve's
    checkpoint fingerprints all bake in the polynomial, the lim/lpb and
    the special-q side — so a fallback that reselects sieve params must
    not run on top of it."""
    try:
        if (wd / "msieve.dat").stat().st_size > 0:
            return True
    except OSError:
        pass
    try:
        s = json.loads((wd / _NFS_STATE).read_text())
    except (OSError, ValueError):
        return False
    return bool(s.get("appended") or s.get("have_raw") or s.get("filters"))


def _la_resumable(wd: Path) -> bool:
    """True when msieve.dat.chk is a Lanczos checkpoint for msieve.dat.mat.

    Content, not mtimes: the checkpoint header carries the vector length
    (max_n, the matrix's column count) and the packed-matrix header carries
    max_ncols, so a matching pair is a checkpoint written for this matrix.
    msieve itself refuses a mismatch ("unexpected vector size"), so this only
    decides whether to *try* -ncr; it never has to be exact about identity.
    """
    mat = wd / "msieve.dat.mat"
    chk = wd / "msieve.dat.chk"
    try:
        with open(chk, "rb") as f:
            hdr = f.read(16)
        with open(mat, "rb") as f:
            mhdr = f.read(12)
    except OSError:
        return False
    if len(hdr) < 16 or len(mhdr) < 12:
        return False
    max_n = struct.unpack("<IIII", hdr)[0]
    max_ncols = struct.unpack("<III", mhdr)[2]
    return max_n > 0 and max_n == max_ncols


def _post_processing_resume(wd: Path) -> bool:
    """True when the workdir's last -nc1 already said "enough".

    A converged filter writes msieve.dat.cyc; a shortfall writes no cycles
    (the run keeps sieving, which touches msieve.dat).  So cycles newer than
    the relation file mean the cycles are current and there is nothing left
    to sieve or filter.  Re-filtering instead would rewrite them with the
    batched merge, whose matrix is not guaranteed to be the one an existing
    Lanczos checkpoint was built against (the checkpoint carries no matrix
    fingerprint), discarding hours of linear algebra.
    """
    dat = wd / "msieve.dat"
    cyc = wd / "msieve.dat.cyc"
    try:
        if not cyc.exists() or not dat.exists():
            return False
        if dat.stat().st_mtime >= cyc.stat().st_mtime:
            return False
    except OSError:
        return False
    # a wave file still on disk was sieved but never appended: let the
    # normal resume path queue and append it first
    return not any(wd.glob("w*.rels")) and not any(wd.glob("band*.rels"))


def _run_band_parallel(cmd_base, cwd: Path, n_procs: int, tag: str,
                       timeout: float, devices: tuple[str, ...] = ("0",),
                       beat=None, beat_s: float = 120.0, stop=None,
                       poll_s: float = 2.0):
    """Split one q band into chunks running concurrently, one GPU each.

    n_procs is per device: chunk i round-robins onto devices[i % len]
    (adjacent q ranges land on different cards), each pinned to its card
    via CUDA_VISIBLE_DEVICES and addressing it as ordinal 0.  devices
    entries are physical ids.

    cmd_base ends with a (q_lo, q_hi) placeholder tuple for --qrange.
    Output files are tagged `tag` (the wave id), not a round number.
    `beat(elapsed_s)` fires every beat_s while the band runs (long bands
    would otherwise be silent for minutes).  `stop` () -> bool and
    `poll_s` pass through to run_procs_parallel: the wave re-checks stop
    every poll_s and is killed wholesale when it fires.  Returns (rc,
    wall, [relation file names]).
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
        rels = f"{tag}.{i}.rels"
        names.append(rels)
        cmd = [str(c) for c in base] + ["--qrange", f"{a}:{b}",
                                        "--relations", rels,
                                        "--log", f"{tag}.{i}.log"]
        jobs.append((cmd, _attempt_stdout(cwd, f"{tag}.{i}.stdout"), None))
        envs.append({"CUDA_VISIBLE_DEVICES": devices[i % k]})
    results = run_procs_parallel(jobs, cwd, timeout, beat=beat,
                                 beat_s=beat_s, stop=stop, poll_s=poll_s,
                                 envs=envs)
    rc = max(rc for rc, _ in results)
    wall = max(w for _, w in results)
    return rc, wall, names


#: give up on convergence after this many shortfall verdicts; the ladder
#: from the probe to a c150+ requirement is ~11-12 rungs, so leave room
FILTER_MAX = 30


def _read_verdict(text: str) -> dict:
    """Parse one -nc1 -v log: the stop/shortfall verdict plus the raw
    measurements the rung estimator consumes.  Everything but the verdict
    lines is optional (older msieve builds, sparse logs); the field names
    follow msieve-cuda's filter output:
      found <D> duplicates and <U> unique relations     duplicate.c
      keeping <J> ideals with weight <= 200,
        target excess is <T>                            common/filter/singleton.c
      pass <k>: found <S> singletons; <R> relations
        and <I> ideals remain                           gnfs/filter/singleton.c
      found <X> cycles, need <Y> / filtering wants <N> more relations
    """
    v: dict = {}
    m = re.search(r"found (\d+) cycles, need (\d+)", text)
    if m:
        v["cycles"], v["need"] = int(m.group(1)), int(m.group(2))
    m = re.search(r"filtering wants (\d+) more relations", text)
    if m:
        v["hint"] = int(m.group(1))
    m = re.search(r"found (\d+) duplicates and (\d+) unique relations", text)
    if m:
        v["unique"] = int(m.group(2))
    ms = re.findall(r"keeping (\d+) ideals with weight <= \d+, "
                    r"target excess is (\d+)", text)
    if ms:
        v["ideals_kept"], v["target_excess"] = (int(x) for x in ms[-1])
    ms = re.findall(r"pass \d+: found \d+ singletons; (\d+) relations "
                    r"and (\d+) ideals remain", text)
    if ms:
        v["pass_relations"], v["pass_ideals"] = (int(x) for x in ms[-1])
    return v


def _measured_excess(v: dict) -> int | None:
    """Excess at the filter's fixed point: exact after singleton removal
    when the pass lines are there, else unique minus weight-kept ideals
    (pre-singleton; a monotone stand-in with the same crossover)."""
    if "pass_relations" in v:
        return v["pass_relations"] - v["pass_ideals"]
    if "unique" in v and "ideals_kept" in v:
        return v["unique"] - v["ideals_kept"]
    return None


def _filter_rung(filters: list[tuple], topup_frac: float,
                 safety: float) -> tuple[int | None, str]:
    """When to run the next -nc1, as an appended-relation threshold.

    The base rung is the ladder: one topup step past the last filter's
    snapshot — each -nc1 re-reads everything, so filter count is the
    serial cost and steps stay geometric.  Once at least two filters
    measured the excess-vs-unique curve and it has turned over
    (positive slope), the rung is capped by the extrapolated requirement:
    the ideal count saturates, so the curve is convex and the secant
    underestimates the slope ahead — linear extrapolation to excess ==
    target_excess lands on the high side of the true crossover, which is
    the safe side (surplus sieving is cheap and hidden; falling short
    costs another full -nc1 pass).  `safety` covers the residual between
    purge excess and merge-stage cycles.  Returns (rung, "ladder"|"estimate").
    """
    if not filters:
        return None, "ladder"           # probe: filter as soon as anything lands
    _, raw, unique, excess, texc = filters[-1]
    # the 1.02 floor keeps a degenerate topup from refiltering the same
    # data forever
    ladder = max(int(raw * 1.02), int(raw * (1.0 + topup_frac)))
    if len(filters) < 2 or unique is None or excess is None or texc is None:
        return ladder, "ladder"
    _, _, u0, e0, _ = filters[-2]
    if u0 is None or e0 is None or unique <= u0:
        return ladder, "ladder"
    slope = (excess - e0) / (unique - u0)
    if slope <= 1e-3:
        return ladder, "ladder"
    u_need = unique + (texc - excess) / slope
    ratio = min(1.0, max(0.5, unique / raw))     # measured unique/raw, floored
    est = u_need / ratio * safety
    return max(int(raw * 1.02), min(ladder, int(est))), "estimate"


def _wave_width(rels: int, last_cnt: int, last_width: int,
                last_wall: float, geom_scale: float = 1.0) -> int:
    """Width, in q units, of the next sieve wave for a target relation count.

    The target is priced with the last completed wave's measured
    relations/q.  After an operator changes logI/J that measurement still
    describes the old geometry, so `geom_scale` (the old/new area ratio
    raised to GEOM_YIELD_EXP) rescales the width for the new one.  The width
    is then capped by the same wave's measured wall/width ratio so the
    prediction lands under WAVE_TARGET_S: q-driven cost drift and geometry
    changes both move the per-q wall, and the measurement follows one wave
    behind.  A state without a per-q measurement (last_width == 0) falls
    back to the probe width - slow to adapt, but never oversized.
    """
    if last_cnt > 0 and last_width > 0:
        per_int = max(last_cnt / last_width, 1e-9)
        width = int(rels / per_int * 1.05) + 2000
    else:
        width = 30_000
    width = max(int(width * geom_scale), 2000)
    if last_wall > 0 and last_width > 0:
        wall_cap = int(last_width * (WAVE_TARGET_S / last_wall))
        width = min(width, max(wall_cap, 2000))
    return max(width, 2000)


def _geom_area_scale(old_logI: int, old_J: int, logI: int, J: int) -> float:
    """Width rescale for a resume whose state was sieved at another area.

    relations/q follows area^GEOM_YIELD_EXP, so the same relation target
    needs width * (old_area/new_area)^exp at the new geometry.  Clamped to
    [0.5, 2.0] so a bad state entry cannot size a wave absurdly; the
    measured wall cap is the second line of defence.
    """
    old_area = float(1 << old_logI) * float(old_J)
    new_area = float(1 << logI) * float(J)
    return max(0.5, min(2.0, (old_area / new_area) ** GEOM_YIELD_EXP))


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
    params: tuple = ()               # (rlim, alim, lpb, logI)
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


def _term_kill(p: subprocess.Popen) -> None:
    """Terminate a polyselect process; escalate to kill after 10 s."""
    if p.poll() is None:
        p.terminate()
        try:
            p.wait(timeout=10)
        except subprocess.TimeoutExpired:
            p.kill()
            p.wait()


def _polyselect_shard(n: int, ctx: RunContext, base: Path,
                      engine: str = "gpu_cubsort"
                      ) -> tuple[list[tuple[Path, subprocess.Popen]], dict]:
    """Start the msieve -np search for n under `base`: one process per card
    over disjoint slices of msieve's default leading-coefficient range,
    whole and unpinned on a lone card.  Returns the (dir, process) list and
    an info dict (sharded / threads / coeff_top for the caller's logging).

    msieve only ever searches multiples of an ad-sieve multiplier and
    rounds each begin up to one, so the slices partition the searched set
    exactly, with zero cross-card traffic.  A sharded process gets a fresh
    private subdir — msieve reads msieve.fb *before* searching, and a
    leftover file for the same N would skip the search.  Every dir holds
    that process's np.log (stdout) and msieve.log, and — once it exits —
    its best-so-far msieve.fb."""
    msieve = find_bin("msieve", ctx.bin_dir)
    k = len(ctx.devices) if engine == "gpu_cubsort" else 1
    rng = _poly_ranges(n, k)
    info = {"sharded": len(rng) > 1,
            "coeff_top": _poly_coeff_max(n),
            # GPU engines cap their producer pool at 4 anyway; -t also sizes
            # the stage-2/3 CPU pool, so sharded processes divide the cores
            "threads": (str(ctx.threads) if len(rng) == 1 else
                        str(max(2, ctx.threads // len(rng))))}
    procs: list[tuple[Path, subprocess.Popen]] = []
    for i, (dev, r) in enumerate(zip(ctx.devices[:len(rng)], rng)):
        d = base if r is None else base / f"poly-{i}"
        if r is not None:
            shutil.rmtree(d, ignore_errors=True)
            d.mkdir(parents=True)
        # one nfs_args token, space-separated: a comma anywhere would
        # trip msieve's legacy "X,Y" coefficient-range parse
        args = f"stage1_engine={engine}"
        if r is not None:
            args += f" min_coeff={r[0]} max_coeff={r[1]}"
        cmd = [msieve, "-np", args, "-t", info["threads"], "-v", str(n)]
        f = open(d / "np.log", "wb")
        try:
            # msieve -np has no device flag; each process sees exactly
            # its card as ordinal 0 via CUDA_VISIBLE_DEVICES
            p = subprocess.Popen([str(c) for c in cmd],
                                 cwd=str(d),
                                 stdout=f, stderr=subprocess.STDOUT,
                                 env={**os.environ,
                                      **ctx.pin_env(dev)})
        finally:
            f.close()
        procs.append((d, p))
    return procs, info


class PolyselectJob:
    """msieve -np polynomial search that stops when the polynomial is
    good, not when a timer fires.

    The search spans msieve's whole default leading-coefficient range,
    sharded one process per card over disjoint slices (msieve only ever
    searches multiples of an ad-sieve multiplier and rounds each begin
    up to one, so the slices partition the searched set exactly, with
    zero cross-card traffic).  Our msieve fork streams every new
    overall-best E to its log ("polyselect best E") as the root
    optimizer keeps pace with the sweep, so a watcher polls the logs
    and SIGINTs the whole search the moment the best E meets the bar
    msieve itself prints ("expecting poly E from X to > Y") — the
    processes save their best-so-far polynomial on the way out.  A
    search that never meets the bar ends with msieve's own range;
    --poly-budget (or a missing bar line, i.e. an older msieve binary)
    falls back to the plain wall cap.

    The winner's msieve.fb/np.log are promoted to the workdir root, so
    run_nfs's downstream stages and resume paths see exactly the
    single-process layout.  A search where NO card produced a polynomial
    retries once on the CPU Gerbicz engine — always compiled in, and
    msieve's own default — under the wall cap.  Interrupts and timeouts
    never leave a process alive; an interrupted search still leaves the
    salvaged best-so-far fb for a rerun to reuse.
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
        # an explicit --poly-budget is a hard wall cap; the default mode
        # is quality-gated (the cap is only the no-bar fallback)
        self._hard_cap = (ctx.poly_budget_s is not None
                          and ctx.poly_budget_s > 0)
        self._cap = self.budget if self._hard_cap else POLY_MAX_WAIT_S
        self.fb: dict[str, str] | None = None
        self.poly_e: float | None = None
        self.poly_method: str | None = None
        self.error: str | None = None
        self.wall_s = 0.0
        self._t0 = time.monotonic()
        self._deadline_at = self._t0 + self._cap
        self._cap_logged = False
        self.poll_s = POLY_POLL_S
        self.bar: float | None = None    # "expecting poly E from X"
        # [(subdir, process)]: one private subdir per sharded card, or
        # the bare workdir for a single-process search
        self._procs: list[tuple[Path, subprocess.Popen]] = []
        self._shard(msieve)
        if self.sharded:
            tail = (f"; ~{fmt_dur(self.budget)} expected, "
                    f"cap {fmt_dur(self._cap)}" if self._hard_cap else "")
            ctx.log(f"nfs: polyselect sharded across {len(self._procs)} "
                    f"GPUs (leading coeffs 1..{self._coeff_top:,} split "
                    f"{len(self._procs)} ways, -t {self._threads} each"
                    f"{tail})")
        elif self._hard_cap:
            ctx.log(f"nfs: polyselect (msieve -np, ~{fmt_dur(self.budget)} "
                    f"expected, cap {fmt_dur(self._cap)})")
        else:
            ctx.log("nfs: polyselect (msieve -np)")
        self._watch()

    def _shard(self, msieve) -> None:
        self._procs, info = _polyselect_shard(self.n, self.ctx, self.wd,
                                              self.engine)
        self.sharded = info["sharded"]
        self._coeff_top = info["coeff_top"]
        self._threads = info["threads"]

    def engine_retry(self) -> bool:
        """A GPU search where NO card produced a polynomial gets one retry
        on the CPU Gerbicz engine — always compiled in, and msieve's own
        default.  An unlucky empty range slice is not retried: with other
        shards holding a polynomial, CPU hours on the empty slice buy
        nothing.  The wall cap restarts; True when a process runs."""
        if self._engine_retried or self.engine == "cpu_gerbicz":
            return False
        if (self.wd / "msieve.fb").exists():
            return False                  # the search produced one
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
        threading.Thread(target=self._quality_watch, daemon=True).start()

    def _quality_watch(self) -> None:
        """Poll the shard logs and stop the search: SIGINT as soon as the
        streamed best E clears the quality target — a fraction of msieve's
        printed expectation, which assumes a search several times longer
        than any bounded one — or when the record has stalled above the
        floor.  The processes save their best-so-far polynomial on the
        way out (the cooperative unwind can take minutes, _join's
        escalation handles a hang); the wall cap still applies for the
        modes that have one."""
        bar_warned = False
        best_val: float | None = None
        best_at = 0.0
        while True:
            # list(), not a generator: all() would short-circuit at the
            # first live process and never poll (and reap) exited shards,
            # leaving zombies behind for the whole search
            if all([p.poll() is not None for _, p in self._procs]):
                return
            if self.bar is None:
                for d, _ in self._procs:
                    self.bar = _poly_e_bar_from_log(d)
                    if self.bar is not None:
                        break
            if self.bar is not None:
                best = max((e for e in map(_poly_running_e_from_log,
                                           (d for d, _ in self._procs))
                            if e is not None), default=None)
                if best is not None:
                    if best != best_val:
                        best_val, best_at = best, time.monotonic()
                    target = self.bar * self.ctx.poly_frac
                    if best >= target:
                        self.ctx.log(
                            f"nfs: polyselect reached the quality target "
                            f"(E={best:.3e} >= {self.ctx.poly_frac:.0%} of "
                            f"msieve's {self.bar:.3e}); stopping the search")
                        self._interrupt_all()
                        return
                    if best >= self.bar * POLY_STALL_FLOOR \
                            and time.monotonic() - best_at >= POLY_STALL_S:
                        self.ctx.log(
                            f"nfs: polyselect best E stalled at "
                            f"{best / self.bar:.0%} of msieve's bar for "
                            f"{fmt_dur(time.monotonic() - best_at)}; "
                            f"stopping the search")
                        self._interrupt_all()
                        return
            else:
                if not bar_warned and time.monotonic() - self._t0 > 60:
                    bar_warned = True
                    self.ctx.log("nfs: polyselect: no E bar in the logs; "
                                 "falling back to the wall cap")
                if bar_warned and time.monotonic() >= self._deadline_at:
                    self._cap_hit()
                    return
            if (self._hard_cap or self.engine == "cpu_gerbicz") \
                    and time.monotonic() >= self._deadline_at:
                self._cap_hit()
                return
            try:
                self.ctx.check_deadline()
            except TimeoutError:
                self.ctx.log("nfs: polyselect: run timeout reached; "
                             "keeping the best-so-far polynomial")
                self._interrupt_all()
                return
            time.sleep(self.poll_s)

    def _cap_hit(self) -> None:
        if not self._cap_logged:
            self._cap_logged = True
            self.ctx.log(f"nfs: polyselect hit its wall cap "
                         f"({fmt_dur(time.monotonic() - self._t0)}); "
                         f"interrupting for the best-so-far polynomial")
        self._interrupt_all()

    def _interrupt_all(self) -> None:
        for _, p in self._procs:
            if p.poll() is None:
                p.send_signal(signal.SIGINT)

    def wait(self, timeout: float | None = None) -> "PolyselectJob":
        """Block until the search ends (bar met, range exhausted, wall
        cap — the watcher SIGINTs and msieve saves best-so-far), merge
        the shards, and only a search where no card produced a
        polynomial retries once on the CPU engine.  Any exception
        (Ctrl+C included) tears the processes down and salvages the
        best-so-far polynomial before propagating."""
        try:
            self._join(timeout)
            self._merge()
            if self.engine_retry():
                self._join(timeout)
                self._merge()
            self._finish()
        except BaseException:
            self.cancel(drop_poly=False)
            raise
        return self

    def _join(self, timeout: float | None = NP_TIMEOUT_S) -> None:
        for _, p in self._procs:
            try:
                p.wait(timeout=timeout)
            except subprocess.TimeoutExpired:
                _term_kill(p)

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
                _term_kill(p)
        if self.sharded:
            self._merge()
        if all([p.poll() is not None for _, p in self._procs]) \
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


def _poly_e_bar_from_log(wd: Path) -> float | None:
    """The lower bound of msieve's own E expectation for a completed
    search ("expecting poly E from X to > Y"), or None when absent."""
    for name in ("msieve.log", "np.log"):
        p = wd / name
        if not p.exists():
            continue
        m = re.search(r"expecting poly E from ([\d.e+-]+)", read_log(p))
        if m:
            return float(m.group(1))
    return None


def _poly_running_e_from_log(wd: Path) -> float | None:
    """The running best E our msieve fork streams to its log after every
    new overall best ("polyselect best E"), or None when absent."""
    best = None
    for name in ("msieve.log", "np.log"):
        p = wd / name
        if not p.exists():
            continue
        ms = re.findall(r"polyselect best E = ([\d.e+-]+)", read_log(p))
        if ms:
            e = max(float(m) for m in ms)
            best = e if best is None else max(best, e)
    return best


# ---------------------------------------------------------------------------
# --bench-nfs: score the best SNFS polynomial against a timed GNFS
# polyselect and recommend a method, without sieving anything.  The SNFS
# side is _score_snfs_candidates (the production re-ranking scorer); the
# GNFS side is a plain wall-capped msieve -np — no quality gate, no
# engine retry — so its E is exactly what the limit bought.
# ---------------------------------------------------------------------------

def bench_gnfs_polyselect(n: int, ctx: RunContext, wd: Path,
                          limit_s: float) -> dict | None:
    """A timed GNFS polyselect for --bench-nfs: msieve -np over the default
    leading-coefficient range (sharded per card), SIGINT at limit_s, then
    read the best-so-far score out of the shard logs.  Unlike
    PolyselectJob there is no quality gate and no CPU-engine retry — the
    limit is the only stop, so the returned E is precisely what the time
    bought.

    Returns {"e", "bar", "degree", "fb", "wall_s"}: the best combined E
    across the shards (None when nothing ran), msieve's own expectation
    for a completed search, the winning poly's degree, the merged
    msieve.fb text, and the wall time — or None when polyselect could not
    even start."""
    try:
        find_bin("msieve", ctx.bin_dir)
    except BinNotFound:
        return None
    wd.mkdir(parents=True, exist_ok=True)
    t0 = time.monotonic()
    procs, info = _polyselect_shard(n, ctx, wd)
    if info["sharded"]:
        ctx.log(f"bench: gnfs polyselect sharded across {len(procs)} GPUs, "
                f"{fmt_dur(limit_s)} limit")
    else:
        ctx.log(f"bench: gnfs polyselect ({fmt_dur(limit_s)} limit)")
    deadline = t0 + limit_s
    try:
        while time.monotonic() < deadline:
            if all(p.poll() is not None for _, p in procs):
                break               # whole range searched before the limit
            try:
                ctx.check_deadline()
            except TimeoutError:
                break               # the run-level cap stops us too
            time.sleep(min(POLY_POLL_S,
                           max(0.2, deadline - time.monotonic())))
        if any(p.poll() is None for _, p in procs):
            for _, p in procs:
                if p.poll() is None:
                    p.send_signal(signal.SIGINT)
        for _, p in procs:
            try:
                p.wait(timeout=NP_TIMEOUT_S)
            except subprocess.TimeoutExpired:
                _term_kill(p)
    except BaseException:
        for _, p in procs:
            _term_kill(p)
        raise
    wall_s = time.monotonic() - t0
    # best E per shard: the fork streams every new overall best, and the
    # SIGINT'd processes finish with their final "combined = X" while
    # saving the best-so-far polynomial
    best_e: float | None = None
    best_dir: Path | None = None
    for d, _ in procs:
        cands = [x for x in (_poly_running_e_from_log(d),
                             _poly_e_from_log(d)) if x is not None]
        if cands and (best_e is None or max(cands) > best_e):
            best_e, best_dir = max(cands), d
    bar = None
    for d, _ in procs:
        bar = _poly_e_bar_from_log(d)
        if bar is not None:
            break
    fb = None
    degree = None
    if best_dir is not None and (best_dir / "msieve.fb").exists():
        fb = read_log(best_dir / "msieve.fb")
        if best_dir != wd:
            shutil.copyfile(best_dir / "msieve.fb", wd / "msieve.fb")
        kv = _parse_msieve_fb(fb)
        degs = [int(k[1:]) for k in kv
                if k.startswith("A") and k[1:].isdigit()]
        degree = max(degs) if degs else None
    if best_e is not None:
        ctx.log(f"bench: gnfs polyselect best E {best_e:.3e}"
                + (f" (msieve expects {bar:.3e} from a full search)"
                   if bar is not None else "")
                + f" after {fmt_dur(wall_s)}")
    else:
        ctx.log(f"bench: gnfs polyselect produced no polynomial in "
                f"{fmt_dur(wall_s)}")
    return {"e": best_e, "bar": bar, "degree": degree, "fb": fb,
            "wall_s": wall_s}


# ---------------------------------------------------------------------------
# normscan: cuda-sieve's bench/normscan surveys a q-band's exact-norm width
# and prints per-side quantiles.  The special-q side's MEDIAN is what the 3LP
# promotion criterion consumes when the tool is available; the static-box
# proxy in params.snfs_lp_structure stands in when it is not.
# ---------------------------------------------------------------------------

#: wall cap on one normscan survey (measured 0.2-5 s at the default 100k
#: samples; the cap only has to catch a wedged host)
NORMSCAN_TIMEOUT_S = 300


def _parse_normscan_quantiles(text: str) -> dict[str, float] | None:
    """normscan's per-side quantile line -> the two medians, or None.
    Format from bench/normscan.c: 'per side quantiles: side1 med X p99 Y
    p999 Z ; side0 med ...'"""
    m = re.search(r"per side quantiles: side1 med ([\d.]+) p99 [\d.]+ "
                  r"p999 [\d.]+ ; side0 med ([\d.]+)", text)
    if not m:
        return None
    try:
        return {"side1_med": float(m.group(1)),
                "side0_med": float(m.group(2))}
    except ValueError:              # regex digits always parse; belt+braces
        return None


def _measure_sq_norm_bits(job: SnfsJob, logI: int, alim: int,
                          ctx: RunContext, wd: Path) -> float | None:
    """normscan's measured median exact-norm bits for the special-q side,
    over the job's INITIAL special-q band, or None when the tool is
    unavailable or fails (the caller falls back to the proxy).  The survey
    runs at the geometry the sieve will actually use (logI, J default) and
    on a poly-only job file -- poly_load deliberately ignores unknown
    fields, but the LP lines are not decided yet when this runs."""
    try:
        normscan = find_bin("normscan", ctx.bin_dir)
    except BinNotFound:
        return None
    q_first = alim // 2             # run_nfs's own first special-q
    wd.mkdir(parents=True, exist_ok=True)   # the disk pre-flight runs early
    poly_path = wd / "normscan.job"
    deg = len(job.coeffs) - 1
    lines = [f"n: {job.n}", f"skew: {job.skew:.4f}"]
    lines += [f"c{k}: {job.coeffs[k]}" for k in range(deg, -1, -1)]
    lines += [f"Y0: {job.y0}", f"Y1: {job.y1}"]
    poly_path.write_text("\n".join(lines) + "\n")
    cmd = [str(normscan), "--poly", poly_path.name,
           "--sq-side", "0" if job.side == "r" else "1",
           "--logI", str(int(logI)), "--qmin", str(q_first),
           "--qmax", str(6 * q_first)]
    try:
        proc = subprocess.run(cmd, cwd=wd, capture_output=True, text=True,
                              timeout=NORMSCAN_TIMEOUT_S)
    except (OSError, subprocess.TimeoutExpired):
        return None
    # exit 2/3 (overflow / thin margin) still print quantiles -- a band whose
    # norms overflow the build width is precisely a promote -- so only a
    # missing binary, an unrunnable survey (1), or bad usage (64) falls back
    if proc.returncode not in (0, 2, 3):
        return None
    q = _parse_normscan_quantiles(proc.stdout)
    if q is None:
        return None
    return q["side1_med" if job.side == "a" else "side0_med"]


def snfs_measured_profile(job: SnfsJob, ctx: RunContext,
                          wd: Path) -> tuple[tuple, SnfsLp]:
    """The sieve profile and LP structure for a supplied SNFS polynomial,
    with the 3LP promotion decided on normscan's measured median when the
    tool is available and on the static-box proxy when it is not.  Shared
    by run_nfs (the decision) and the engine's disk estimate, so the
    estimate and the run's verdict cannot disagree."""
    sieve = snfs_sieve_params(job.scaled, job.side, job.coeffs,
                             job.y0, job.y1, ctx.sieve_logI)
    sq_bits = _measure_sq_norm_bits(job, sieve[8], sieve[1], ctx, wd)
    if sq_bits is not None:
        ctx.log(f"nfs: normscan measured sq-side median norm "
                f"{sq_bits:.1f} bits over the initial q band")
        sieve = snfs_sieve_params(job.scaled, job.side, job.coeffs,
                                  job.y0, job.y1, ctx.sieve_logI,
                                  sq_norm_bits=sq_bits)
    lp = snfs_lp_structure(job.scaled, job.side, job.coeffs, job.y0,
                           job.y1, ctx.sieve_logI, sq_norm_bits=sq_bits)
    return sieve, lp


def run_nfs(n: int, ctx: RunContext, wd: Path,
            snfs: SnfsJob | None = None) -> NfsResult:
    """Factor n with GNFS inside wd (polyselect runs here when nothing
    reusable is on disk).  With `snfs`, that supplied polynomial is used
    as-is: no polyselect, parameters from the SNFS difficulty table."""
    digits = len(str(n))
    if snfs is None and digits < NFS_MIN_DIGITS:
        # the table has nothing below c70; the clamped c70 row sieves a
        # small n with norms far under its own bounds, so every candidate
        # survives and the first special-q overflows the cofactor queue.
        res = NfsResult()
        res.error = (f"GNFS is tabulated from c{NFS_MIN_DIGITS}; c{digits} "
                     f"belongs to SIQS or the prework stages")
        ctx.log(f"nfs: {res.error}")
        return res
    msieve = find_bin("msieve", ctx.bin_dir)
    cuda_sieve = find_bin("cuda-sieve", ctx.bin_dir)
    wd.mkdir(parents=True, exist_ok=True)
    sieve = None
    if snfs is not None:
        # pick the best poly first: the profile below keys on the
        # winner's scaled difficulty
        snfs = _rank_snfs_candidates(snfs, ctx, wd)
        sieve, lp = snfs_measured_profile(snfs, ctx, wd)
        (rlim, alim, lpbr, lpba, mfbr, mfba, rlbd, albd,
         logI) = sieve
        lpb = lpbr               # representative lpb for NfsResult.params
        if lp.three_lp_sq:
            ctx.log(f"nfs: sq-side norm ({lp.norm_source}) {lp.norm_bits:.1f}"
                    f" bits, typical cofactor ~{lp.cofactor_bits:.0f} bits > "
                    f"2*{lp.sq_lpb}+{lp.promote_slack:g}: special-q side "
                    f"promoted to 3LP")
        elif snfs.scaled >= SNFS_2LP_UNMEASURED_ABOVE:
            # legal but unmeasured: every deciding point at d>=245 ran 3LP.
            # The escape hatch is real — run_nfs reuses the workdir's job
            # file verbatim on resume, so editing its sq-side mfb to 3LP
            # and restarting bumps the run.
            ctx.log(f"nfs: WARNING: 2LP at d{snfs.scaled:g} is unmeasured "
                    f"(every job that decided at d>=245 ran 3LP, 320-434M "
                    f"raw); if the excess ladder starves, set the workdir "
                    f"job's sq-side mfb to 3LP and restart")
    else:
        rlim, alim, lpb, logI = nfs_params(digits)
    if ctx.sieve_logI is not None:
        # operator override: the tables cap logI at cuda-sieve's per-relation
        # sweet spot, but a job whose special-q window is the binding limit
        # needs relations/q more than per-relation efficiency (RUNBOOK:
        # area and lpb are one joint choice, and the q span is a real cap)
        logI = ctx.sieve_logI
    # the geometry actually handed to cuda-sieve; persisted in the state so a
    # resume can tell whether its measured relations/q still applies
    eff_J = ctx.sieve_J if ctx.sieve_J is not None else (1 << (logI - 1))
    geom = [logI, eff_J]
    res = NfsResult(params=(rlim, alim, lpb, logI))
    nproc = str(ctx.threads)
    ctx.log(f"nfs: c{digits}"
            + (f" snfs (difficulty {snfs.scaled:g}"
               + (f", {snfs.form}" if snfs.form else "")
               + (", special-q on the rational side" if snfs.side == "r"
                  else "")
               + ")"
               if snfs is not None else "")
            + f", rlim={rlim:,}, alim={alim:,}, logI={logI}"
            + (f", lpbr={lpbr}, lpba={lpba}, mfbr={mfbr}, mfba={mfba}"
               if sieve is not None else f", lpb={lpb}"))

    # ---- stage 1: polynomial selection (or reuse from a previous run) ----
    t = time.monotonic()
    job_path = wd / f"c{digits}.job"
    reused = False
    job_text = None
    if snfs is not None:
        if job_path.exists():
            job_text = job_path.read_text()
            if f"n: {n}" in job_text:
                reused = True
                res.poly_method = "snfs (resumed)"
                ctx.log("nfs: reusing snfs job from workdir")
        if not reused:
            job_text = _snfs_job_text(snfs, sieve)
            job_path.write_text(job_text)
            res.poly_method = "snfs (factordb)"
            ctx.log(f"nfs: snfs degree-{len(snfs.coeffs) - 1} poly"
                    + (f" ({snfs.form})" if snfs.form else "")
                    + f", difficulty {snfs.difficulty:g}"
                    + (f", E={snfs.e:g}" if snfs.e else ""))
        res.stages["polyselect_s"] = 0.0
    if not reused and snfs is None and job_path.exists():
        job_text = job_path.read_text()
        if f"n: {n}" in job_text:
            reused = True
            res.poly_method = "msieve -np (resumed)"
            e = _poly_e_from_log(wd)
            if e is not None:
                res.poly_e = e
            ctx.log("nfs: reusing polynomial from workdir"
                    + (f" (E={res.poly_e:g})" if res.poly_e else ""))
    if not reused and snfs is None and digits >= NFS_MSIEVE_POLY_MIN_DIGITS:
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
    if not reused and snfs is None and digits < NFS_MSIEVE_POLY_MIN_DIGITS:
        degree = 3 if digits < 80 else 4
        coeffs, mm = base_m_poly(n, degree)
        fb = {"N": str(n),
              "SKEW": f"{(abs(coeffs[0]) / abs(coeffs[_degree(coeffs)])) ** 0.25:.2f}",
              "R0": str(-mm), "R1": "1"}
        for i, c in enumerate(coeffs):
            fb[f"A{i}"] = str(c)
        res.poly_method = f"base-m deg-{degree}"
        ctx.log(f"nfs: base-m degree-{degree} poly (m={mm})")
    if snfs is None:
        res.stages["polyselect_s"] = 0.0 if reused else time.monotonic() - t

    # ---- stage 2: job file + msieve.fb/worktodo.ini ---------------------
    if not reused and snfs is None:
        job_text = _fb_to_job(fb, rlim, alim, lpb)
        job_path.write_text(job_text)
    fb_text, ini_text = _job_to_msieve_fb(job_text)
    (wd / "msieve.fb").write_text(fb_text)
    (wd / "worktodo.ini").write_text(ini_text)

    # ---- stage 3: sieving + filtering feedback loop ----------------------
    # Resumable: the polynomial (job file) and every sieved relation
    # (msieve.dat) survive a kill; nfs-state.json pins q_next, the
    # appended count and the filter-measurement log; sieved-but-not-yet-
    # appended waves are the w*.rels files still on disk.
    #
    # The loop is a small event pump keeping one wave and at most one
    # -nc1 in flight:
    #   * a wave streams relations into its own files; msieve.dat only
    #     grows while no -nc1 is reading it, so the queue absorbs the
    #     lag and the stop check stays exact;
    #   * a verdict takes effect the moment it lands: the next wave is
    #     aimed at the refreshed rung, and "enough" kills the wave in
    #     flight (its output would be dropped anyway) and empties the
    #     queue — oversieving past the enough-point costs at most one
    #     wave, not one band;
    #   * the next -nc1 waits for the estimator's rung: one topup step
    #     past the last filter (the ladder), capped by the requirement
    #     extrapolated from the measured excess-vs-unique curve.  -nc1
    #     re-reads everything each round, so the filter count — not the
    #     wave count — is the serial cost.
    state = _load_nfs_state(wd, n)
    if state is not None and not (wd / "msieve.dat").exists():
        state = None                # checkpoint without its relations: redo
    q_first = alim // 2
    S = SimpleNamespace(q_next=q_first, q_cap=6 * q_first, appended=0,
                        filters=[], filter_seq=0, wave_seq=0,
                        last_cnt=0, last_width=0, last_wall=0.0,
                        timed_wall=0.0)
    #: one-shot rescale of the first wave after an operator changes the
    #: sieve geometry; 1.0 otherwise.  See _wave_width.
    geom_scale = 1.0
    queued: list[tuple[list[str], int]] = []   # sieved, not appended yet
    if state is None:
        # a stale workdir may hold killed-run leftovers; a fresh start
        # never appends to them.  The .rels.* and .stdout.* arms sweep
        # uncommitted .part/.lock sidecars and per-attempt stdout
        # captures, which the plain patterns miss
        for pat in ("w*.rels", "band*.rels", "w*.rels.*", "band*.rels.*",
                    "w*.log", "band*.log",
                    "w*.stdout", "band*.stdout",
                    "w*.stdout.*", "band*.stdout.*"):
            for stale in wd.glob(pat):
                stale.unlink(missing_ok=True)
        try:
            (wd / "msieve.dat").unlink()   # stateless leftovers: restart
        except OSError:
            pass
    else:
        S.q_next = int(state["q_next"])
        S.q_cap = int(state.get("q_cap", 6 * q_first))
        S.appended = int(state.get("appended", state.get("have_raw", 0)))
        S.filters = [tuple(f) for f in state.get("filters", [])]
        S.filter_seq = int(state.get("filter_seq", state.get("round_idx", 0)))
        S.wave_seq = int(state.get("wave_seq", 0))
        S.last_cnt = int(state.get("last_cnt", 0))
        S.last_width = int(state.get("last_width", 0))
        S.last_wall = float(state.get("last_wall", 0.0) or 0.0)
        S.timed_wall = float(state.get("timed_wall", 0.0) or 0.0)
        for f in sorted(wd.glob("w*.rels")) + sorted(wd.glob("band*.rels")):
            # appended waves are deleted right after the append, so any
            # relation file still on disk was sieved but never appended
            queued.append(([f.name], count_lines(f)))
        ctx.log(f"nfs: resuming: {S.appended:,} relations in msieve.dat, "
                f"{len(queued)} queued wave file(s), "
                f"{len(S.filters)} filter point(s), q from {S.q_next:,}")
        prev_geom = state.get("geometry")
        if prev_geom is None:
            # state written before geometry was recorded: its relations/q
            # may belong to any earlier logI/J, so shrink one wave and let
            # the fresh measurement take over from there
            geom_scale = 0.6
            ctx.log("nfs: state carries no sieve geometry; the first wave "
                    "is sized conservatively")
        else:
            try:
                old_logI, old_J = int(prev_geom[0]), int(prev_geom[1])
            except (TypeError, ValueError, IndexError):
                old_logI, old_J = logI, eff_J
            if (old_logI, old_J) != (logI, eff_J):
                geom_scale = _geom_area_scale(old_logI, old_J, logI, eff_J)
                ctx.log(f"nfs: sieve geometry changed (logI {old_logI}->"
                        f"{logI}, J {old_J}->{eff_J}); the first wave is "
                        f"sized {geom_scale:.2f}x")

    def _snapshot() -> None:
        _save_nfs_state(wd, n, q_next=S.q_next, q_cap=S.q_cap,
                        appended=S.appended,
                        filters=[list(f) for f in S.filters],
                        filter_seq=S.filter_seq, wave_seq=S.wave_seq,
                        last_cnt=S.last_cnt, last_width=S.last_width,
                        last_wall=S.last_wall, timed_wall=S.timed_wall,
                        geometry=geom)

    next_filter_at, _rung_how = _filter_rung(S.filters, ctx.topup_frac,
                                             ctx.target_safety)
    running_filter = None   # (proc, raw snapshot, log path, t0) or None
    wave = None             # {seq, thread, box, q}; at most one in flight
    stop_wave = [False]     # watched by the wave's run_procs_parallel
    waves_appended = 0

    def _launch_wave() -> dict:
        """Size and start the next sieve wave in a worker thread."""
        nonlocal geom_scale
        have_total = S.appended + sum(c for _, c in queued)
        if not S.filters and S.appended == 0 and not queued:
            width = 30_000                                    # probe wave
        else:
            if running_filter is not None:
                # the verdict in flight will set the rung; until it lands
                # assume the ladder step past its snapshot (the estimate
                # can only pull the rung down, and it lands mid-wave)
                goal = running_filter[1] * (1.0 + ctx.topup_frac)
            else:
                goal = next_filter_at if next_filter_at is not None else 1
            remaining = goal - have_total
            # at/over the goal with a filter in flight: keep the GPU fed
            # speculatively — the relations are either consumed by the
            # next rung or dropped whole on "enough", and idling would
            # cost the same wall time with nothing to show for it
            rels = remaining if 0 < remaining < ctx.wave_max else ctx.wave_max
            width = _wave_width(rels, S.last_cnt, S.last_width,
                                S.last_wall, geom_scale)
            geom_scale = 1.0     # one-shot, see the resume geometry check
        qhi = min(S.q_next + width, S.q_cap)
        if qhi <= S.q_next:
            S.q_cap += 2 * q_first   # relax the duplicate-control window
            qhi = min(S.q_next + width, S.q_cap)
            ctx.log(f"nfs: q window exhausted, cap relaxed to {S.q_cap:,}")
        seq = S.wave_seq
        S.wave_seq += 1
        tag = f"w{seq:04d}"
        base = [cuda_sieve, "--pipeline", "--cofactor",
                "--poly", job_path.name, "--logI", logI, "--region",
                min(logI, 14), "--device", 0]
        if ctx.sieve_J is not None:
            # pin J instead of letting it default to 2^(logI-1): raising
            # logI alone widens the a-range, whose extra candidates keep
            # the rational (binding) norms unchanged; widening b instead
            # grows a-bm by a full bit per doubling
            base += ["--J", str(ctx.sieve_J)]
        if snfs is not None and snfs.side == "r":
            base += ["--sq-side", 0]    # SNFS: factordb wants the
                                        # special-q on the rational side
        base += [(S.q_next, qhi)]
        ctx.log(f"nfs: wave {tag}: q {S.q_next}:{qhi} "
                f"(have {fmt_count(have_total)})"
                + (f" [{ctx.sieve_procs} proc x {len(ctx.devices)} gpu]"
                   if len(ctx.devices) > 1 else ""))
        box: dict = {}

        def _run_wave():
            try:
                box["rc"], box["wall"], box["names"] = _run_band_parallel(
                    base, wd, ctx.sieve_procs, tag, SIEVE_TIMEOUT_S,
                    ctx.devices, stop=lambda: stop_wave[0], poll_s=0.25)
            except BaseException as e:   # re-raised on the loop thread
                box["exc"] = e

        th = threading.Thread(target=_run_wave, daemon=True)
        th.start()
        return {"seq": seq, "thread": th, "box": box, "q": (S.q_next, qhi)}

    # A workdir a previous run left at post-processing resumes there: its
    # last -nc1 said "enough" and wrote msieve.dat.cyc, and nothing has been
    # appended since, so there is nothing left to sieve or filter.  Skipping
    # the loop also keeps the cycle set (and therefore the packed matrix)
    # untouched, which is what lets an existing Lanczos checkpoint continue.
    post_only = _post_processing_resume(wd)
    if post_only:
        ctx.log("nfs: filter output is current (msieve.dat.cyc newer than "
                "msieve.dat); resuming at post-processing")

    try:
        while not post_only:
            ctx.check_deadline()

            # reap a finished wave into the queue
            if wave is not None and not wave["thread"].is_alive():
                wave["thread"].join()
                box = wave["box"]
                if "exc" in box:
                    if isinstance(box["exc"], subprocess.TimeoutExpired):
                        res.error = "sieve wave timed out"
                        ctx.log(f"nfs: {res.error}")
                        # the attempt ran for exactly the cap: carry that
                        # wall into the retry's completion accounting, or
                        # the resumed wave looks an order of magnitude
                        # faster than it was and the wall cap loses its
                        # reference
                        S.timed_wall += SIEVE_TIMEOUT_S
                        _snapshot()
                        return res
                    raise box["exc"]
                if stop_wave[0]:
                    wave = None             # the verdict path owns cleanup
                elif box.get("rc", 1) != 0:
                    for f in box.get("names") or []:
                        (wd / f).unlink(missing_ok=True)
                    res.error = (f"sieve wave w{wave['seq']:04d} failed "
                                 f"(rc={box.get('rc')})")
                    ctx.log(f"nfs: {res.error}")
                    return res
                else:
                    names = box["names"]
                    cnt = sum(count_lines(wd / f) for f in names)
                    queued.append((names, cnt))
                    # a resumed wave's own attempt wall only covers the
                    # retry; the timed-out attempt(s) are carried in
                    # timed_wall so rate and the width cap see the truth
                    wall = box.get("wall", 0.0) + S.timed_wall
                    S.timed_wall = 0.0
                    res.stages["sieve_s"] = res.stages.get("sieve_s", 0.0) \
                        + wall
                    S.last_cnt = cnt
                    S.last_width = wave["q"][1] - wave["q"][0]
                    S.last_wall = wall
                    S.q_next = wave["q"][1]
                    have_total = S.appended + sum(c for _, c in queued)
                    rate = cnt / wall if wall > 0 else 0.0
                    if rate > 0:
                        ctx.log(f"nfs: wave w{wave['seq']:04d}: +{cnt:,} "
                                f"rels in {fmt_dur(wall)} "
                                f"({fmt_count(rate)}/s); "
                                f"have {fmt_count(have_total)}")
                    _snapshot()
                    wave = None

            # poll the in-flight filter; its verdict lands here no later
            # than one loop tick after it exits
            if running_filter is not None:
                proc, snapshot, logp, t0 = running_filter
                rc = proc.poll()
                timed_out = False
                if rc is None and time.monotonic() - t0 > NC_TIMEOUT_S:
                    proc.kill()
                    proc.wait()
                    rc = proc.returncode
                    timed_out = True
                if rc is not None:
                    running_filter = None
                    res.stages["filter_s"] = res.stages.get("filter_s", 0.0) \
                        + time.monotonic() - t0
                    if rc != 0:
                        res.error = ("filtering timed out" if timed_out
                                     else f"filtering failed (rc={rc})")
                        ctx.log(f"nfs: {res.error}")
                        return res
                    text = read_log(logp)
                    v = _read_verdict(text)
                    if "cycles" in v and v["cycles"] >= v["need"] \
                            and "too few cycles" not in text:
                        # enough — and everything sieved since this filter
                        # launched is surplus: kill the wave in flight
                        # (its output would be dropped anyway) and empty
                        # the queue
                        stop_wave[0] = True
                        surplus = sum(c for _, c in queued)
                        if wave is not None:
                            wave["thread"].join(timeout=30)
                            for f in wave["box"].get("names") or []:
                                (wd / f).unlink(missing_ok=True)
                            wave = None
                        for names, _ in queued:
                            for f in names:
                                (wd / f).unlink(missing_ok=True)
                        queued.clear()
                        ctx.log(f"nfs: enough at {S.appended:,} relations; "
                                f"wave in flight stopped, queue dropped "
                                f"({surplus:,} surplus relation(s))")
                        break
                    if "cycles" not in v and "hint" not in v:
                        res.error = "filtering did not converge"
                        ctx.log(f"nfs: {res.error}")
                        return res
                    excess = _measured_excess(v)
                    S.filters.append((S.filter_seq, snapshot,
                                      v.get("unique"), excess,
                                      v.get("target_excess")))
                    next_filter_at, rung_how = _filter_rung(
                        S.filters, ctx.topup_frac, ctx.target_safety)
                    ctx.log(f"nfs: filter r{S.filter_seq}: {snapshot:,} raw"
                            + (f" -> {v['unique']:,} unique, excess "
                               f"{excess:,} / {v['target_excess']:,}"
                               if excess is not None
                               and "target_excess" in v else "")
                            + f"; next filter at {next_filter_at:,} "
                              f"({rung_how})")
                    _snapshot()
                    if len(S.filters) >= FILTER_MAX:
                        res.error = "filtering did not converge"
                        ctx.log(f"nfs: {res.error}")
                        return res

            # msieve.dat only grows while no -nc1 is reading it
            if running_filter is None and queued:
                for names, cnt in queued:
                    _append_band_rels(wd, names)
                    S.appended += cnt
                waves_appended += len(queued)
                queued.clear()
                _snapshot()

            # launch a filter once the appended data reaches its rung
            if running_filter is None and S.appended > 0 and (
                    not S.filters
                    or (next_filter_at is not None
                        and S.appended >= next_filter_at)):
                S.filter_seq += 1
                # dup_cache: reuse dedup state across filters (msieve-cuda
                #   merge-batch; writes msieve.dat.dup, ~16 bytes/relation);
                # merge_batch: batched parallel full merge (matrix differs
                #   from the serial merge's, quality statistics identical)
                cmd = [msieve, "-s", "msieve.dat", "-l",
                       f"nc1-r{S.filter_seq}.log", "-nc1",
                       "dup_cache=1,merge_batch=2048", "-t", nproc, "-v"]
                filter_out = _attempt_stdout(
                        wd, f"nc1-r{S.filter_seq}.stdout")
                fh = open(filter_out, "wb")
                try:
                    proc = subprocess.Popen([str(c) for c in cmd],
                                            cwd=str(wd), stdout=fh,
                                            stderr=subprocess.STDOUT)
                finally:
                    fh.close()
                running_filter = (proc, S.appended, filter_out,
                                  time.monotonic())
                ctx.log(f"nfs: filter r{S.filter_seq}: filtering "
                        f"{S.appended:,} relations in the background")

            # keep exactly one wave in flight
            if wave is None:
                wave = _launch_wave()

            time.sleep(0.05)
    finally:
        # a return inside the loop (wave failure, non-convergence) or an
        # exception must not leave wave chunks or a background -nc1
        # running over msieve.dat
        stop_wave[0] = True
        if wave is not None:
            wave["thread"].join(timeout=30)
        if running_filter is not None and running_filter[0].poll() is None:
            try:
                running_filter[0].kill()
                running_filter[0].wait()
            except OSError:
                pass

    # ---- stage 4: matrix build + Lanczos + sqrt ---------------------------
    #
    # The Lanczos wall time is set by the matrix and the hardware, so the
    # stage is bounded by liveness, not by a fixed duration (LA_STALL_S):
    # a killed attempt resumes from msieve.dat.chk with -ncr rather than
    # failing the job.  -ncr reads the packed matrix, so a resume also
    # skips the matbuild pass; a fresh start still builds it, which leaves
    # msieve.dat.mat behind for any later resume.
    chk = wd / "msieve.dat.chk"
    mat = wd / "msieve.dat.mat"
    cyc = wd / "msieve.dat.cyc"

    def _mat_current() -> bool:
        """The packed matrix is at least as new as the cycles it comes from,
        so it is the one the current cycle set produces.  A matbuild after a
        fresh -nc1 (or after an operator replaced the relations) is needed;
        otherwise it would only burn its 10-20 minutes rebuilding the same
        bytes."""
        try:
            return (mat.stat().st_size > 0
                    and mat.stat().st_mtime >= cyc.stat().st_mtime)
        except OSError:
            return False

    t = time.monotonic()
    if _mat_current():
        res.stages["matbuild_s"] = 0.0
    else:
        run_cmd([msieve, "-s", "msieve.dat", "-l", "nc2-mat.log",
                 "-nc2", "only_matbuild=1", "-t", nproc, "-v"],
                wd, wd / "nc2-mat.stdout", timeout=NC_TIMEOUT_S)
        res.stages["matbuild_s"] = time.monotonic() - t
    t = time.monotonic()
    # msieve prints its own "linear algebra completed ... (X%, ETA ...)"
    # updates (carriage-return paced) while it runs; forward the first one
    # (it anchors the run: dimension count + initial ETA) and then one per
    # completed 10% of the dimension count, instead of one per minute
    last_la_decile = [-1]

    def _la_lines(lines):
        for l in lines:
            if "linear algebra" not in l:
                continue
            if l.startswith("commencing"):
                ctx.log(f"nfs: {l.strip()}")
                continue
            m = re.search(r"completed (\d+) of (\d+) dimensions", l)
            if not m:
                continue
            decile = int(m.group(1)) * 10 // int(m.group(2))
            if decile > last_la_decile[0]:
                ctx.log(f"nfs: {l.strip()}")
                last_la_decile[0] = decile

    attempt = 0
    while True:
        attempt += 1
        resume = _la_resumable(wd)
        tag = "" if attempt == 1 else f"-r{attempt}"
        ctx.log("nfs: linear algebra "
                + ("resuming from the checkpoint" if resume else "starting")
                + ("" if attempt == 1
                   else f" (attempt {attempt} of {LA_MAX_ATTEMPTS})"))
        rc, _ = run_cmd_progress(
            [msieve, "-s", "msieve.dat", "-l", f"nc2-la{tag}.log",
             "-ncr" if resume else "-nc2", "-t", nproc, "-v"],
            wd, wd / f"nc2-la{tag}.stdout",
            timeout=None, every_s=30,
            idle_timeout=LA_STALL_S, activity_paths=(chk,),
            env_extra=ctx.pin_env(),
            on_lines=_la_lines)
        if rc == 0:
            break
        why = ("stalled (no progress line or checkpoint for "
               f"{LA_STALL_S // 3600}h)" if rc == -9 else f"rc={rc}")
        if attempt < LA_MAX_ATTEMPTS and _la_resumable(wd):
            ctx.log(f"nfs: linear algebra died ({why}); resuming from the "
                    f"last checkpoint")
            continue
        res.stages["linalg_s"] = time.monotonic() - t
        res.error = f"linear algebra failed ({why})"
        ctx.log(f"nfs: {res.error}")
        return res
    res.stages["linalg_s"] = time.monotonic() - t
    t = time.monotonic()
    rc, _ = run_cmd_progress(
        [msieve, "-s", "msieve.dat", "-l", "nc3.log",
         "-nc3", "-t", nproc, "-v"], wd, wd / "nc3.stdout",
        timeout=None, every_s=30, idle_timeout=SQRT_STALL_S)
    if rc == -9:
        res.error = "sqrt stalled"
        ctx.log(f"nfs: {res.error}")
        return res
    if rc != 0:
        res.error = f"sqrt failed (rc={rc})"
        ctx.log(f"nfs: {res.error}")
        return res
    res.stages["sqrt_s"] = time.monotonic() - t
    log3 = read_log(wd / "nc3.stdout")
    factors = [int(x) for x in re.findall(r"p\d+ factor: (\d+)", log3)]
    if not factors:
        factors = [int(x) for x in re.findall(r"Found [a-z]+ factor of \d+ digits: (\d+)", log3)]
    res.factors = [f for f in factors if n % f == 0]
    res.ok = bool(res.factors)
    res.raw_relations = S.appended
    res.bands = waves_appended
    ctx.log(f"nfs: complete, {len(res.factors)} factor(s), "
            f"{waves_appended} sieve wave(s), {S.appended:,} relations")
    return res
