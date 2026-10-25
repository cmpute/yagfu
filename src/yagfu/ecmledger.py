"""ECM work ledger — YAFU's cumulative work accounting, ported.

Three ideas are kept from YAFU (bbuhrow/yafu @ c7d9028, factor/autofactor.c
and include/yafu_ecm.h):

1. A fixed ladder of B1 values, one per 5-digit factor-size level t15..t65.
2. A work ledger: every completed curve at level-j parameters contributes
   ``c / M[i][j]`` of normalized work toward *every* depth level i, where
   ``M[i][j]`` is the number of curves at B1[j] needed to fully reach
   t_level[i].  Work done at small B1 therefore still counts (a little)
   toward deeper levels, and the ledger survives cofactor splits because a
   new composite simply gets a fresh target while the curves already spent
   on its ancestor are accounted per-composite by the scheduler.
3. A depth formula: find the first level i whose accumulated normalized
   work s_i < 1; the achieved depth is t[i-1] + 5*s_i (0 when i = 0).

The curve table is YAFU's ``ecm_data_param3`` — yagfu always runs GPU stage
1 (gmp-ecm -gpu), which forces parametrization 3, and gpuecm2 with
GMP-ECM's default B2.  9e99 marks "impossible / never" combinations.
"""

from __future__ import annotations

NUM_LEVELS = 11

#: factor size in digits for each ECM level
T_LEVELS = (15, 20, 25, 30, 35, 40, 45, 50, 55, 60, 65)

#: B1 for each level (GMP-ECM's standard table, = yafu's ecm_std_b1)
B1_LADDER = (
    2_000, 11_000, 50_000, 250_000, 1_000_000, 3_000_000, 11_000_000,
    43_000_000, 110_000_000, 260_000_000, 850_000_000,
)

_INVALID = 9e99

#: ecm_data_param3 from yafu c7d9028 include/yafu_ecm.h:
#: CURVES[i][j] = curves at B1[j] that complete t_level[i].
CURVES = (
    # curves: B1=   2e3   11e3   50e3   25e4    1e6     3e6    11e6    43e6   110e6   260e6   850e7
    (43,       8,     4,     2,     2,      1,      1,      1,      1,      1,      1,      1),   # t15
    (1743,     107,   24,    9,     5,      4,      3,      2,      2,      2,      1),           # t20
    (134769,   2402,  261,   58,    22,     13,     8,      5,      4,      3,      3),           # t25
    (1.7e7,    82576, 4108,  513,   137,    61,     29,     16,     11,     8,      6),           # t30
    (3.3e9,    3967858, 88265, 6022, 1071,   374,    138,    61,     38,     25,     17),         # t35
    (9e99,     2.5e8, 2402639, 87544, 10283, 2753,   788,    278,    151,    91,     52),         # t40
    (9e99,     2e10,  8.1e7, 1534319, 118226, 24017,  5208,   1459,   692,    373,    185),       # t45
    (9e99,     9e99,  3.3e9, 3.1e7,  1565171, 241048, 39497,  8704,   3583,   1709,   737),       # t50
    (9e99,     9e99,  2.3e11, 7.5e8, 2.4e7,   2713723, 336066, 57844,  20479,  8656,   3220),     # t55
    (9e99,     9e99,  1.5e17, 2e10,  3.9e8,   3.4e7,   3167410, 419970, 128305, 47888,  15391),   # t60
    (9e99,     9e99,  2.1e22, 6.7e13, 7.7e9,  4.8e8,   3.2e7,   3346252, 872747, 288516, 78923),  # t65
)

#: pretest target as a fraction of the current cofactor's digit count
#: (yafu: light 2/9, normal 4/13, deep 1/3)
PLAN_RATIOS = {"light": 2.0 / 9.0, "normal": 4.0 / 13.0, "deep": 1.0 / 3.0}


def target_depth(digits: int, plan: str = "normal") -> float:
    """ECM target t-level for a `digits`-digit cofactor under `plan`."""
    return PLAN_RATIOS[plan] * digits


class EcmLedger:
    """Cumulative normalized ECM work across all depth levels."""

    def __init__(self) -> None:
        self.work = [0.0] * NUM_LEVELS  # s_i in the module docstring
        self.done_curves = [0] * NUM_LEVELS

    def add(self, level_j: int, curves: int) -> None:
        for i in range(NUM_LEVELS):
            m = CURVES[i][level_j]
            if m < _INVALID:  # impossible combos contribute ~0
                self.work[i] += curves / m
        self.done_curves[level_j] += curves

    def first_incomplete(self) -> int:
        for i, s in enumerate(self.work):
            if s < 1.0:
                return i
        return NUM_LEVELS

    def depth(self) -> float:
        """Achieved t-level under YAFU's formula."""
        i = self.first_incomplete()
        if i == 0:
            return 0.0
        if i == NUM_LEVELS:
            return float(T_LEVELS[-1])
        return T_LEVELS[i - 1] + 5.0 * self.work[i]

    def curves_for_target(self, level_j: int, target: float) -> int:
        """Curves to run at B1[level_j] so that `depth()` reaches `target`.

        Binary search, mirroring yafu's curve-count estimation; the result
        is capped at the level's own diagonal budget (M[j][j]), as in yafu.
        """
        lo, hi = 0, int(CURVES[level_j][level_j])
        if self._depth_with(level_j, hi) < target:
            return hi
        while lo < hi:
            mid = (lo + hi) // 2
            if self._depth_with(level_j, mid) >= target:
                hi = mid
            else:
                lo = mid + 1
        return lo

    def _depth_with(self, level_j: int, extra: int) -> float:
        work = [
            self.work[i] + (extra / CURVES[i][level_j]
                            if CURVES[i][level_j] < _INVALID else 0.0)
            for i in range(NUM_LEVELS)
        ]
        for i, s in enumerate(work):
            if s < 1.0:
                if i == 0:
                    return 0.0
                return T_LEVELS[i - 1] + 5.0 * s
        return float(T_LEVELS[-1])

    def level_budget_left(self, level_j: int) -> int:
        return max(0, int(CURVES[level_j][level_j]) - self.done_curves[level_j])
