import pytest

from yagfu.ecmledger import (
    B1_LADDER,
    CURVES,
    T_LEVELS,
    EcmLedger,
    target_depth,
)


def test_tables():
    assert len(T_LEVELS) == len(B1_LADDER) == len(CURVES) == 11
    assert T_LEVELS[0] == 15 and T_LEVELS[-1] == 65
    assert B1_LADDER[0] == 2000 and B1_LADDER[-1] == 850_000_000
    # diagonal is the per-level budget and must increase
    diag = [CURVES[i][i] for i in range(11)]
    assert diag == [43, 107, 261, 513, 1071, 2753, 5208, 8704, 20479, 47888, 78923]
    # each row non-decreasing difficulty: going up one level at same B1
    # needs (weakly) more curves
    for i in range(10):
        assert CURVES[i][i] <= CURVES[i + 1][i]


def test_target_depth_matches_yafu_plan_ratios():
    assert target_depth(80, "normal") == pytest.approx(4 * 80 / 13)
    assert target_depth(60, "normal") == pytest.approx(18.4615, abs=1e-3)
    assert target_depth(150, "normal") == pytest.approx(46.1538, abs=1e-3)
    assert target_depth(90, "light") == pytest.approx(20.0)
    assert target_depth(90, "deep") == pytest.approx(30.0)


def test_ledger_80_digit_normal_run():
    """The appendix's worked example, param-3 edition.

    normal plan, 80-digit cofactor: target = t24.6154.  The scheduler runs
    one budget at each level until the depth crosses the target:
    43 @ B1=2e3 -> 107 @ B1=11e3 -> ~230 @ B1=5e4.
    """
    led = EcmLedger()
    target = target_depth(80)
    assert target == pytest.approx(24.61538, abs=1e-4)

    # level 0: full diagonal budget of 43 curves
    c = led.curves_for_target(0, target)
    assert c == 43
    led.add(0, c)
    assert led.depth() == pytest.approx(15.1234, abs=1e-3)

    # level 1: 107 curves complete t20 well before the target
    c = led.curves_for_target(1, target)
    assert c == 107
    led.add(1, c)
    assert led.depth() == pytest.approx(20.2243, abs=1e-3)

    # level 2: 230 curves (not the full 261 budget) reach t24.6154
    c = led.curves_for_target(2, target)
    assert c == 230
    led.add(2, c)
    assert led.depth() >= target
    assert led.depth() == pytest.approx(24.63, abs=0.02)


def test_budget_cap_when_level_cannot_reach_target():
    led = EcmLedger()
    # target t30 from level 1 alone is impossible: cap at the diagonal budget
    assert led.curves_for_target(1, 30.0) == 107


def test_all_complete_saturates():
    led = EcmLedger()
    for j in range(11):
        led.add(j, int(CURVES[j][j]) + 50)
    assert led.depth() == 65.0
    assert led.first_incomplete() == 11


def test_partial_first_level_depth_is_zero():
    # yafu: when the first incomplete level is t15, reported depth is 0
    led = EcmLedger()
    led.add(0, 10)
    assert led.first_incomplete() == 0
    assert led.depth() == 0.0


def test_level_budget_left():
    led = EcmLedger()
    led.add(0, 20)
    assert led.level_budget_left(0) == 23
    led.add(0, 100)
    assert led.level_budget_left(0) == 0


def test_seed_to_depth():
    """--ecm-t-level start: the ledger is credited to the requested depth
    without curves, completed stages show no budget left, and the stage
    carrying the start keeps exactly its remainder."""
    led = EcmLedger()
    led.seed_to_depth(38.0)
    assert led.depth() == pytest.approx(38.0, abs=1e-2)
    assert led.done_curves[:5] == [43, 107, 261, 513, 1071]
    assert led.done_curves[5] == 1652          # round(0.6 * 2753)
    assert led.level_budget_left(5) == 2753 - 1652
    assert led.level_budget_left(4) == 0       # completed: never re-run

    # seeding only ever raises: real work is not taken back
    d = led.depth()
    led.seed_to_depth(20.0)
    assert led.depth() == pytest.approx(d)
    fresh = EcmLedger()
    fresh.add(5, 2000)                         # real curves past the seed
    d = fresh.depth()
    fresh.seed_to_depth(38.0)
    assert fresh.depth() == pytest.approx(d)


def test_seed_edges():
    led = EcmLedger()
    # sub-t15 depth is not representable: no-op (the always-run first
    # batch covers it)
    led.seed_to_depth(12.0)
    assert led.depth() == 0.0
    assert led.done_curves == [0] * 11
    # the ladder top: everything complete, nothing left to run
    led.seed_to_depth(65.0)
    assert led.depth() == 65.0
    assert led.done_curves == [int(CURVES[i][i]) for i in range(11)]
    assert all(led.level_budget_left(i) == 0 for i in range(11))


def test_t_level_walk_38_to_44():
    """The --ecm-t-level 38.0-44.0 walk: the ladder's own
    min(curves_for_target, level_budget_left) loop on the seeded ledger
    runs 2/5 of the t40 diagonal at B1=3e6, then the minimal curves at
    B1=11e6 that land on t44 — fewer than the naive 4/5 * 5208 because
    the 3e6 curves' cross-level credit (1101/24017 of the t45 row) is
    counted, so the walk lands exactly on the requested depth."""
    led = EcmLedger()
    led.seed_to_depth(38.0)
    target = 44.0
    ran = []
    for j in range(11):
        want = min(led.curves_for_target(j, target),
                   led.level_budget_left(j))
        if j == 0 and led.done_curves[0] == 0:
            want = max(want, CURVES[0][0])     # always-run first batch
        if want <= 0:
            continue
        led.add(j, want)
        ran.append((B1_LADDER[j], want))
    assert ran == [(3_000_000, 1101), (11_000_000, 3928)]
    assert led.depth() == pytest.approx(44.0, abs=1e-2)
