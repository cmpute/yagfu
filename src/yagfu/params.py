"""Per-size parameter tables (bench_factor.py heritage) with interpolation.

The anchor rows were measured/validated on an RTX 4090; between anchors the
values are piecewise-linear, and the NFS sieve loop's feedback adapts the
relation target anyway.
"""

from __future__ import annotations

# --------------------------------------------------------------------------
# cuda-mpqs (SIQS): single-large-prime variant bound.  The user guide says
# "always enable large primes for inputs > 80 digits, starting bound
# 10*F..100*F"; below ~75 digits the plain 2LP-free variant wins.
# --------------------------------------------------------------------------

_MPQS_LP1_ANCHORS = ((75, 0), (85, 150_000_000), (95, 300_000_000), (1 << 30, 300_000_000))


def mpqs_lp1_bound(digits: int) -> int | None:
    prev_d, prev_v = _MPQS_LP1_ANCHORS[0]
    for d, v in _MPQS_LP1_ANCHORS[1:]:
        if digits <= d:
            if prev_v == 0 and v == 0:
                return None
            if prev_v == 0:
                # linear ramp from the point where LP1 first turns on
                return None if digits <= prev_d else int(v)
            t = (digits - prev_d) / (d - prev_d)
            return int(prev_v + t * (v - prev_v))
        prev_d, prev_v = d, v
    return _MPQS_LP1_ANCHORS[-1][1]


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
