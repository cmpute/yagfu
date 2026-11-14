"""Per-size parameter tables (bench_factor.py heritage) with interpolation.

The anchor rows were measured/validated on an RTX 4090; between anchors the
values are piecewise-linear.  How many relations a job needs is deliberately
not tabulated — the NFS sieve loop's estimator derives it from the measured
excess curve.
"""

from __future__ import annotations

import math
from typing import NamedTuple

# --------------------------------------------------------------------------
# cuda-mpqs (SIQS): single-large-prime variant bound.
#
# The upstream README warns "single large prime below ~85 digits causes
# 100% sqrt failure", but on this build (1.0.8, git b1a9c45) LP1=150M
# measures fine — and faster — down to c76 (c80: 5.8s vs 7.4s without).
# We enable it from 80 digits, which matches the bench_factor.py
# measurement points (80: 150M, 90/100: 300M) and keeps a margin below
# the smallest verified size.
# --------------------------------------------------------------------------


#: cuda-mpqs's tuning stage refuses smaller inputs ("Minimum supported:
#: ~24 digits (~80 bits)"); below this only TD/rho/ECM can split a number.
MPQS_MIN_DIGITS = 24


def mpqs_lp1_bound(digits: int) -> int | None:
    if digits < 80:
        return None
    if digits < 90:
        return 150_000_000
    return 300_000_000


# --------------------------------------------------------------------------
# GNFS sieve parameters: (rlim, alim, lpb, logI).
# Anchors 70..100 are measured (bench_factor.py); 110..150 track the usual
# GGNFS/CADO tables — logI stays <= 14 (cuda-sieve's sweet spot).
# --------------------------------------------------------------------------

NFS_PARAMS_ANCHORS: tuple[tuple[int, tuple[int, int, int, int]], ...] = (
    (70,  (350_000,     350_000,     23, 12)),
    (80,  (800_000,     800_000,     24, 13)),
    (90,  (1_500_000,   1_500_000,   26, 13)),
    (100, (3_000_000,   3_000_000,   27, 14)),
    (110, (6_000_000,   6_000_000,   28, 14)),
    (120, (12_000_000,  12_000_000,  29, 14)),
    (130, (25_000_000,  25_000_000,  30, 14)),
    (140, (50_000_000,  50_000_000,  30, 14)),
    (150, (100_000_000, 100_000_000, 31, 14)),
)

#: msieve -np's GPU stage-1 search emits nothing below ~90 digits (verified:
#: candidates at 90/100, none at 70/80/85 even uncapped), so the base-m
#: Python poly is used below this.
NFS_MSIEVE_POLY_MIN_DIGITS = 90

#: smallest size the GNFS table covers.  Clamping a smaller composite to
#: the c70 row is not a valid job: with lpb 23 and rlim 350k on a tiny n
#: nearly every (a, b) survives the norm bound, and the sieve dies in its
#: first special-q ("candidates exceeds the cofactor queue").
NFS_MIN_DIGITS = NFS_PARAMS_ANCHORS[0][0]


def _interp(anchors, x: float) -> tuple[int, int, int, int]:
    if x <= anchors[0][0]:
        return anchors[0][1]
    if x >= anchors[-1][0]:
        return anchors[-1][1]
    for (d0, p0), (d1, p1) in zip(anchors, anchors[1:]):
        if d0 <= x <= d1:
            t = (x - d0) / (d1 - d0)
            vals = []
            for a, b in zip(p0, p1):
                v = a + t * (b - a)
                vals.append(int(round(v)) if isinstance(a, int) else v)
            return tuple(vals)  # type: ignore[return-value]
    raise AssertionError("unreachable")


def nfs_params(digits: int) -> tuple[int, int, int, int]:
    """(rlim, alim, lpb, logI) for a digits-digit GNFS job.

    Raises ValueError below the smallest tabulated size instead of clamping:
    a c6 sieved with the c70 row floods the cofactor queue and fails, and a
    number this small is TD/rho/ECM territory, not a GNFS job.
    """
    if digits < NFS_MIN_DIGITS:
        raise ValueError(
            f"GNFS parameters are tabulated from c{NFS_MIN_DIGITS}; "
            f"c{digits} is not a sieve job")
    return _interp(NFS_PARAMS_ANCHORS, digits)


# --------------------------------------------------------------------------
# SNFS sieve parameters: (rlim, alim, lpb, logI), keyed on the SCALED
# difficulty factordb reports (yafu's param-selection size, not the raw
# digit count of the input).  Initial anchors, not measured calibrations:
# rlim/alim/lpb track the usual yafu/NFS@Home SNFS tables — two
# factordb-generated snfs jobs pin the ends (d~170: 7M / lpb 27 / lasieve
# I13, d~270: 85.6M / lpb 31 / I15); logI is capped at 14 (cuda-sieve's
# sweet spot) instead of following lasieve's wider picks.
# --------------------------------------------------------------------------

SNFS_PARAMS_ANCHORS: tuple[tuple[int, tuple[int, int, int, int]], ...] = (
    (100, (500_000,     500_000,     25, 12)),
    (120, (1_200_000,   1_200_000,   26, 13)),
    (140, (3_000_000,   3_000_000,   26, 13)),
    (160, (5_000_000,   5_000_000,   27, 13)),
    (180, (9_000_000,   9_000_000,   27, 13)),
    (200, (20_000_000,  20_000_000,  28, 14)),
    (220, (40_000_000,  40_000_000,  29, 14)),
    (240, (60_000_000,  60_000_000,  30, 14)),
    (260, (80_000_000,  80_000_000,  31, 14)),
    (280, (110_000_000, 110_000_000, 31, 14)),
    (300, (200_000_000, 200_000_000, 32, 14)),
)


def snfs_params(difficulty: float) -> tuple[int, int, int, int]:
    """(rlim, alim, lpb, logI) for an SNFS job of this scaled difficulty.

    Superseded for sieving by snfs_sieve_params (per-side LP profile);
    kept as the classic-table reference for tests and size estimates."""
    return _interp(SNFS_PARAMS_ANCHORS, difficulty)


# --------------------------------------------------------------------------
# SNFS sieve profile for cuda-sieve: per-side limits and large-prime bounds.
#
# The classic symmetric 2LP tables above price sieving like a CPU lasieve:
# per-special-q cost is the scarce resource, so the FB stays small and
# mfb = k*lpb - 1 (the CADO/Kyle convention: 95 = 3*32 - 1,
# 63 = 2*32 - 1).  cuda-sieve's per-q cost is near-flat, so the wall time is
# (relations needed) / (yield per q) and yield per q is what the large
# primes buy.  Kyle Askine's review of the 15^253-1 suggestion (Nov 2026:
# "2LP is bad for this poly, and the factordb lims are really low ...
# especially on your GPU I'd be fine with like 600M/400M at least (and
# bigger still might just be strictly better), and would definitely go
# 95/63 for mfb") calibrated the d275 point; ws-15c253 decided there at
# 361M raw relations with 600M/400M, lpb 32, mfb 95/63.
#
# The LP structure below is graded, because the relation NEED is a property
# of the profile, not of the difficulty alone: at the same scaled d205 and
# the same degree-6 cyclotomic family, the 2LP point (ws-15c261, lpb 28/28,
# FB 25.5M) decided at 45.1M raw while the 3LP point (ws-15c522, lpb 32/32,
# mfb 95/63, FB 101M/69M) decided at 329.5M — 7x the relations and ~8x the
# sieve wall for the same number size.  So:
#
#   * a 2LP baseline ladder grades lpb with the scaled difficulty;
#   * the special-q side is PROMOTED to 3LP (mfb = 3*lpb - 1) only when
#     its norms demand it (snfs_lp_structure's norm criterion) — for the
#     halved-cyclotomic family that is the rational-side jobs, whose
#     |Y0|*B norms grow with the base, while the +-1-coefficient algebraic
#     side stays at the q-lattice scale (~2^166 measured median at d205)
#     regardless of difficulty;
#   * the criterion decides at EVERY difficulty, including the d245-275
#     band where all four deciding points (320-434M raw) ran 3LP under the
#     old blanket rule: a 2LP verdict there is legal but unmeasured, and
#     run_nfs warns about it (SNFS_2LP_UNMEASURED_ABOVE) so a starving
#     excess ladder is read as "bump to 3LP and re-sieve", not as a bug.
#
# rlim/alim still come from SNFS_SIEVE_ANCHORS (the per-side FB philosophy
# above); only the LP bounds are graded here.
# --------------------------------------------------------------------------

SNFS_SIEVE_ANCHORS: tuple[tuple[int, tuple[int, int, int]], ...] = (
    (90,  (4_000_000,     3_000_000,     12)),
    (110, (10_000_000,    7_000_000,     12)),
    (125, (16_000_000,    11_000_000,    13)),
    (145, (25_000_000,    17_000_000,    13)),
    (165, (40_000_000,    27_000_000,    13)),
    (185, (65_000_000,    43_000_000,    13)),
    (205, (100_000_000,   68_000_000,    14)),
    (225, (160_000_000,   108_000_000,   14)),
    (245, (250_000_000,   168_000_000,   14)),
    (265, (380_000_000,   255_000_000,   14)),
    (275, (600_000_000,   400_000_000,   14)),
    (295, (900_000_000,   600_000_000,   14)),
    (320, (1_500_000_000, 1_000_000_000, 14)),
    (350, (2_400_000_000, 1_600_000_000, 14)),
)

#: 2LP baseline: (scaled, sq_lpb, other_lpb) step rows; mfb = 2*lpb - 1.
#: Step bands, not interpolation — LP bounds are bit sizes and are stepped
#: in practice.  The single measured 2LP point is ws-15c261 (d205.48,
#: degree-6 cyclotomic, lpb 28/28 symmetric, deciding 45.1M raw): the rows
#: sit at or one step above it, never below.
SNFS_LPB_LADDER: tuple[tuple[int, int, int], ...] = (
    (0,   29, 29),
    (205, 30, 29),
    (225, 31, 30),
    (245, 32, 32),
)

#: scaled difficulty at or above which a 2LP verdict is legal but has no
#: deciding measurement: every job that decided at d>=245 so far (four of
#: them, 320-434M raw) ran 3LP under the old blanket rule.  The criterion
#: still applies there; run_nfs warns when a job lands on 2LP so a starving
#: excess ladder is read as "bump to 3LP and re-sieve", not as a parameter
#: bug.
SNFS_2LP_UNMEASURED_ABOVE = 245.0

#: promote the special-q side to 3LP (mfb = 3*lpb - 1) when the typical
#: special-q cofactor exceeds the 2LP room (2*lpb) by this many bits.  Two
#: norm sources feed the criterion, each with its own slack:
#:
#: PROXY (static box bound, no normscan): the box misses the q-lattice's
#: sqrt(q) growth, so its gap to reality is large and side-dependent
#: (side a d205: proxy 79.6 vs measured 165.7 bits; side r: 219 vs 236) —
#: the 20 bits only work because both calibration families sit far from
#: its threshold.  Fallback when normscan is unavailable.
#:
#: MEASURED (cuda-sieve bench/normscan, per-side median over the job's
#: initial special-q band): the exact corner bound the siever's TD itself
#: sees.  Calibrated on the two known families -- ws-15c261's (side a,
#: d205: measured cofactor 139.2 bits must stay 2LP; c261 decided 45.1M
#: raw at 2LP lpb 28) and ws-15c265's (side r, d249: measured cofactor
#: 209.5 bits must promote; the family ran and decided 3LP) -- which put
#: the 2*lpb + slack separator at 170 for d205's lpb 30, with 31/39 bits
#: of margin.  NOTE: this slack is the FB-absorption fudge at norm scales
#: the proxy never sees; it is NOT interchangeable with the proxy's 20.
SNFS_3LP_PROMOTE_SLACK = 20.0
SNFS_3LP_PROMOTE_SLACK_MEASURED = 110.0


#: lambdas follow CADO's automatic rule 0.3 + mfb/lpb
def _snfs_lambda(mfb: int, lpb: int) -> float:
    return round(0.3 + mfb / lpb, 2)


def _snfs_sq_norm_bits(coeffs: list[int], y0: int, y1: int, side: str,
                       logI: int) -> float:
    """First-order typical special-q-side norm bits over cuda-sieve's
    rectangle (|a|, b <= 2^(logI-1); cuda-sieve sieves an unskewed box —
    J defaults to 2^(logI-1) regardless of the job's skew).

    Degree-1 side: |Y1*a + Y0*b| is dominated by |Y0|*B and roughly
    symmetric over the box, so half the bound stands in for the median.
    Degree-d side: the full box bound sum|c_i|*(2^(logI-1))^d — a
    deliberate overestimate, because the criterion built on this may only
    PROMOTE, never demote.  bench/normscan measures the exact per-side
    distribution; this proxy only has to separate the calibration
    families, which sit ~80 bits apart."""
    b_half = float(1 << (logI - 1))
    if side == "r":
        bound = (abs(y1) + abs(y0)) * b_half
    else:
        deg = len(coeffs) - 1
        bound = sum(abs(c) for c in coeffs) * b_half ** deg
    return math.log2(max(bound / 2.0, 1.0))


class SnfsLp(NamedTuple):
    """The LP structure snfs_sieve_params hands to the job file, plus the
    norm measurement behind its 3LP decision (for the run log) and the
    three_lp_sq flag the relation estimate keys on."""
    sq_lpb: int
    sq_mfb: int
    other_lpb: int
    other_mfb: int
    three_lp_sq: bool
    norm_bits: float | None
    cofactor_bits: float | None
    norm_source: str | None = None    # "normscan" | "proxy" | None
    promote_slack: float | None = None


def snfs_lp_structure(scaled: float, side: str, coeffs: list[int] | None = None,
                      y0: int | None = None, y1: int | None = None,
                      logI: int | None = None,
                      sq_norm_bits: float | None = None) -> SnfsLp:
    """LP bounds for an SNFS job of this scaled difficulty with factordb's
    special-q `side` ("a" | "r"): the 2LP ladder's LPB step for the
    difficulty and — when the polynomial is supplied — the norm criterion
    that promotes the special-q side to 3LP, at every difficulty.
    `sq_norm_bits` is a normscan-measured median for the special-q side
    (see backends.nfs._measure_sq_norm_bits); without it the static-box
    proxy stands in, with its own — different — slack.  At or above
    SNFS_2LP_UNMEASURED_ABOVE a 2LP verdict is legal but unmeasured;
    run_nfs warns about it."""
    _, alim, _ = _interp(SNFS_SIEVE_ANCHORS, scaled)
    row = SNFS_LPB_LADDER[0]
    for r in SNFS_LPB_LADDER:
        if scaled >= r[0]:
            row = r
    sq_lpb, other_lpb = row[1], row[2]
    norm_bits = cofactor_bits = None
    norm_source = None
    promote_slack = None
    three_lp = False
    if coeffs is not None and y0 is not None:
        if sq_norm_bits is not None:
            norm_bits = float(sq_norm_bits)
            norm_source = "normscan"
            promote_slack = SNFS_3LP_PROMOTE_SLACK_MEASURED
        else:
            norm_bits = _snfs_sq_norm_bits(coeffs, y0, y1, side,
                                           logI if logI is not None
                                           else _interp(SNFS_SIEVE_ANCHORS,
                                                        scaled)[2])
            norm_source = "proxy"
            promote_slack = SNFS_3LP_PROMOTE_SLACK
        # the special-q band starts at alim/2 and its cap relaxes upward;
        # alim is the geometric middle to within a bit (the measured median
        # spans [alim/2, 3*alim], whose middle is 1.22*alim — 0.3 bits)
        cofactor_bits = norm_bits - math.log2(max(float(alim), 2.0))
        three_lp = cofactor_bits > 2 * sq_lpb + promote_slack
    sq_mfb = 3 * sq_lpb - 1 if three_lp else 2 * sq_lpb - 1
    return SnfsLp(sq_lpb, sq_mfb, other_lpb, 2 * other_lpb - 1,
                  three_lp, norm_bits, cofactor_bits,
                  norm_source, promote_slack)


def snfs_sieve_params(scaled: float, side: str,
                      coeffs: list[int] | None = None,
                      y0: int | None = None, y1: int | None = None,
                      logI: int | None = None,
                      sq_norm_bits: float | None = None) -> tuple:
    """Per-side sieve profile (rlim, alim, lpbr, lpba, mfbr, mfba,
    rlambda, alambda, logI) for an SNFS job of this scaled difficulty with
    factordb's special-q `side` ("a" or "r").  Pass the polynomial's
    coeffs/Y0 to arm the 3LP norm promotion and `sq_norm_bits` to drive it
    with a normscan measurement; `logI` overrides the table's rectangle for
    the norm proxy (cuda-sieve --sieve-logI)."""
    rlim, alim, table_logI = _interp(SNFS_SIEVE_ANCHORS, scaled)
    lp = snfs_lp_structure(scaled, side, coeffs, y0, y1, logI, sq_norm_bits)
    if side == "r":
        lpbr, mfbr = lp.sq_lpb, lp.sq_mfb
        lpba, mfba = lp.other_lpb, lp.other_mfb
    else:
        lpba, mfba = lp.sq_lpb, lp.sq_mfb
        lpbr, mfbr = lp.other_lpb, lp.other_mfb
    return (rlim, alim, lpbr, lpba, mfbr, mfba,
            _snfs_lambda(mfbr, lpbr), _snfs_lambda(mfba, lpba),
            logI if logI is not None else table_logI)


# --------------------------------------------------------------------------
# SNFS difficulty -> GNFS-equivalent digit count.
#
# The two scales are different: SNFS exploits small-coefficient
# polynomials, so the same difficulty is easier than GNFS.  yafu's
# empirical conversion (audited from the yafu source):
#
#     gnfs_digits ~= 0.56 * snfs_difficulty + 30
#
# factordb's record pages state exactly this: the 859465723^19-1
# suggestion has difficulty 169.75 yet "costs about as much as GNFS on a
# 125-digit number" (0.56 * 169.75 + 30 = 125.1).  This conversion is for
# *cost comparisons* only — the ECM pretest sizes itself against what the
# sieving job actually costs, and a future SNFS-vs-GNFS choice would use
# it too.  The sieving parameters themselves stay keyed on the SNFS
# difficulty (snfs_params above), like every SNFS parameter table.
# --------------------------------------------------------------------------

def snfs_gnfs_equiv(difficulty: float) -> float:
    """GNFS-equivalent digit count of an SNFS job (yafu: 0.56 D + 30)."""
    return 0.56 * difficulty + 30.0


# --------------------------------------------------------------------------
# NFS disk pre-flight: how big a working set a job of this size should be
# planned against, before anything is sieved.  Relations: c107 and c137 are
# yagfu-measured deciding filters (27.3M / 72.5M raw); above 137 the growth
# is one lpb step (+1 bit ≈ x2 pair space) plus digit growth.  Bytes per
# relation line: c137 measured 126.  Interpolation is log-linear on the
# relation axis (the curve is exponential, linear interpolation would
# undershoot every midpoint).
# --------------------------------------------------------------------------

NFS_RAW_REL_ANCHORS: tuple[tuple[int, float], ...] = (
    (100, 6e6), (107, 27e6), (115, 40e6), (125, 55e6), (137, 72e6),
    (150, 200e6), (160, 400e6), (170, 800e6), (180, 1.6e9),
    (190, 3.2e9), (200, 6.0e9),
)

NFS_BYTES_PER_REL_ANCHORS: tuple[tuple[int, int], ...] = (
    (100, 70), (137, 126), (160, 150), (190, 175), (200, 185),
)


def _interp_log(anchors: tuple[tuple[int, float], ...],
                x: float) -> float:
    """piecewise-linear in log10(value), i.e. exponential between anchors"""
    if x <= anchors[0][0]:
        return anchors[0][1]
    if x >= anchors[-1][0]:
        return anchors[-1][1]
    for (d0, v0), (d1, v1) in zip(anchors, anchors[1:]):
        if d0 <= x <= d1:
            t = (x - d0) / (d1 - d0)
            return 10 ** ((1 - t) * math.log10(v0) + t * math.log10(v1))
    raise AssertionError("unreachable")


def nfs_relations_estimate(digits: int) -> int:
    """raw relations a deciding GNFS of this size collects (pre-ladder)"""
    return int(_interp_log(NFS_RAW_REL_ANCHORS, digits))


# --------------------------------------------------------------------------
# SNFS raw relations: what a deciding run actually collects, keyed on the
# scaled difficulty AND the LP profile the sieve runs (snfs_lp_structure's
# three_lp_sq), because the profile is what sets the need: at the same
# scaled d205 the 3LP point (ws-15c522, lpb 32/32 mfb 95/63) decided at
# 329.5M while the 2LP point (ws-15c261, lpb 28/28) decided at 45.1M —
# the GNFS-equivalent lookup cannot see a 7x spread inside one difficulty.
# The 3LP anchors are measured deciding counts (upper bounds of the true
# need; the disk estimate must cover what actually lands on disk), nearly
# flat across d205-275 because the excess crossover follows the LP/factor-
# base structure rather than the difficulty.  The 2LP table has a single
# measured point; its other rows are flagged extrapolations.  Both tables
# feed only the disk pre-flight — the sieve loop's need is measured by the
# excess ladder, whose extrapolated rungs landed within 0.3% of the
# deciding count on every job that reached one.  Re-anchor per profile
# when a new (scaled, LP structure) deciding count lands.
# --------------------------------------------------------------------------

SNFS_RAW_REL_3LP_ANCHORS: tuple[tuple[int, float], ...] = (
    (205, 330e6),    # ws-15c522 (true need 164-330M)
    (219, 330e6),    # ws-15c558 (decided 320.0M)
    (255, 440e6),    # ws-15c642 (decided 433.9M, true 314-434M)
    (275, 400e6),    # ws-15c253 (decided 361.2M, true 180-361M; padded)
    (300, 500e6),    # extrapolated
)

SNFS_RAW_REL_2LP_ANCHORS: tuple[tuple[int, float], ...] = (
    (125, 10e6),     # unmeasured
    (185, 30e6),     # unmeasured
    (205, 60e6),     # floor: c261 decided 45.1M at lpb 28 / FB 25.5M; the
                     # ladder's 30/29 over the bigger per-side FBs is heavier
    (245, 150e6),    # unmeasured: run_nfs warns when a d>=245 job lands here
    (300, 300e6),    # unmeasured
)


def snfs_relations_estimate(scaled: float, three_lp_sq: bool = False) -> int:
    """raw relations a deciding SNFS run of this scaled difficulty collects,
    for the LP profile snfs_lp_structure picks"""
    if three_lp_sq:
        return int(_interp_log(SNFS_RAW_REL_3LP_ANCHORS, scaled))
    return int(_interp_log(SNFS_RAW_REL_2LP_ANCHORS, scaled))


def nfs_bytes_per_rel(digits: int) -> int:
    return int(_interp_log([(d, float(b)) for d, b in
                            NFS_BYTES_PER_REL_ANCHORS], digits))


def nfs_disk_estimate(digits: int, raw_rels: int | None = None) -> dict:
    """working-set estimate (GB) for an NFS job of this size.

    Components as fractions of the raw relation file: the dup cache writes
    ~16 B/relation (coordinates) + ~2.5 B (bitmaps); the LP intermediate
    peaked at 0.27x the dat on c137 (0.35 planned); matrix/cycles/dep
    ~0.15x; plus a couple of GB of sieve-wave transient.  `raw_rels`
    overrides the GNFS relation lookup — SNFS passes the profile-aware
    snfs_relations_estimate here and keeps the bytes-per-relation model."""
    rels = nfs_relations_estimate(digits) if raw_rels is None else raw_rels
    bpr = nfs_bytes_per_rel(digits)
    dat = rels * bpr / 1e9
    dup = rels * 18 / 1e9
    lp = 0.35 * dat
    mat = 0.15 * dat
    total = dat + dup + lp + mat + 2.0
    return {
        "digits": digits,
        "raw_rels": rels,
        "bytes_per_rel": bpr,
        "dat_gb": dat,
        "dup_gb": dup,
        "lp_gb": lp,
        "mat_gb": mat,
        "total_gb": total,
    }


# --------------------------------------------------------------------------
# NFS polyselect budget: YAFU allots 8% of predicted GNFS time; with GPU
# sieving that degenerates to ~8% of the predicted SIEVE time, floored at
# 60 s (stage-1 space exhaustion + rootopt drain).  T_sieve anchored on the
# measured c90 = 31 s / c100 = 59 s and cuda-sieve RUNBOOK band figures.
# --------------------------------------------------------------------------

POLY_BETA = 0.08
POLY_FLOOR_S = 60.0


def sieve_time_estimate(digits: int) -> float:
    return 59.0 * 3.0 ** ((digits - 100) / 10.0)


def poly_budget(digits: int, override: float | None = None,
                measured_sieve_s: float | None = None) -> float:
    if override is not None:
        return override
    t = POLY_BETA * (measured_sieve_s if measured_sieve_s is not None
                     else sieve_time_estimate(digits))
    return max(POLY_FLOOR_S, t)
