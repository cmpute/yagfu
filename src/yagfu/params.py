"""Per-size parameter tables (bench_factor.py heritage) with interpolation.

The anchor rows were measured/validated on an RTX 4090; between anchors the
values are piecewise-linear.  How many relations a job needs is deliberately
not tabulated — the NFS sieve loop's estimator derives it from the measured
excess curve.
"""

from __future__ import annotations

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
    """(rlim, alim, lpb, logI) for a digits-digit GNFS job."""
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
    """(rlim, alim, lpb, logI) for an SNFS job of this scaled difficulty."""
    return _interp(SNFS_PARAMS_ANCHORS, difficulty)


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
    import math
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


def nfs_bytes_per_rel(digits: int) -> int:
    return int(_interp_log([(d, float(b)) for d, b in
                            NFS_BYTES_PER_REL_ANCHORS], digits))


def nfs_disk_estimate(digits: int) -> dict:
    """working-set estimate (GB) for an NFS job of this size.

    Components as fractions of the raw relation file: the dup cache writes
    ~16 B/relation (coordinates) + ~2.5 B (bitmaps); the LP intermediate
    peaked at 0.27x the dat on c137 (0.35 planned); matrix/cycles/dep
    ~0.15x; plus a couple of GB of sieve-wave transient."""
    rels = nfs_relations_estimate(digits)
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
