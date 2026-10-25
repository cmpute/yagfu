"""Per-size parameter tables (bench_factor.py heritage) with interpolation.

The anchor rows were measured/validated on an RTX 4090; between anchors the
values are piecewise-linear, and the NFS sieve loop's feedback adapts the
relation target anyway.
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
# GNFS sieve parameters: (rlim, alim, lpb, logI, initial_raw_relations).
# Anchors 70..100 are measured (bench_factor.py); 110..150 track the usual
# GGNFS/CADO tables — logI stays <= 14 (cuda-sieve's sweet spot).
# --------------------------------------------------------------------------

NFS_PARAMS_ANCHORS: tuple[tuple[int, tuple[int, int, int, int, int]], ...] = (
    (70,  (350_000,     350_000,     23, 12,   800_000)),
    (80,  (800_000,     800_000,     24, 13, 1_600_000)),
    (90,  (1_500_000,   1_500_000,   26, 13, 8_000_000)),
    (100, (3_000_000,   3_000_000,   27, 14, 15_000_000)),
    (110, (6_000_000,   6_000_000,   28, 14, 40_000_000)),
    (120, (12_000_000,  12_000_000,  29, 14, 100_000_000)),
    (130, (25_000_000,  25_000_000,  30, 14, 300_000_000)),
    (140, (50_000_000,  50_000_000,  30, 14, 800_000_000)),
    (150, (100_000_000, 100_000_000, 31, 14, 2_000_000_000)),
)

#: msieve -np's deg-4 GPU stage-1 table starts at log10(n) = 80; below ~85
#: digits nothing comes out, so the base-m Python poly is used there.
NFS_MSIEVE_POLY_MIN_DIGITS = 85


def nfs_params(digits: int) -> tuple[int, int, int, int, int]:
    """(rlim, alim, lpb, logI, initial_raw) for a digits-digit GNFS job."""
    anchors = NFS_PARAMS_ANCHORS
    if digits <= anchors[0][0]:
        return anchors[0][1]
    if digits >= anchors[-1][0]:
        return anchors[-1][1]
    for (d0, p0), (d1, p1) in zip(anchors, anchors[1:]):
        if d0 <= digits <= d1:
            t = (digits - d0) / (d1 - d0)
            vals = []
            for a, b in zip(p0, p1):
                v = a + t * (b - a)
                vals.append(int(round(v)) if isinstance(a, int) else v)
            return tuple(vals)  # type: ignore[return-value]
    raise AssertionError("unreachable")


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
