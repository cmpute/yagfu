"""Wave sizing: relation target, wall cap, and geometry-change rescale.

These are the two hardening rules added after a logI change left the state's
relations/q describing the old geometry and a 142M-wide wave ran into
SIEVE_TIMEOUT_S: the first wave after a geometry change is rescaled, and
every wave is capped by the last completed wave's measured wall/width ratio.
"""

from yagfu.backends.nfs import (
    WAVE_TARGET_S,
    _geom_area_scale,
    _load_nfs_state,
    _save_nfs_state,
    _wave_width,
)

# the state a logI15 wave leaves behind: 24M relations over a 142.3M-wide
# band in ~7.1h (the first, mis-sized wave after the switch)
CNT, WID, WALL = 24_000_000, 142_336_530, 7.1 * 3600


def test_relation_target_prices_from_last_wave():
    w = _wave_width(16_000_000, CNT, WID, 0.0)
    assert 99_000_000 < w < 100_500_000        # ~16M / (24M/142.3M)


def test_wall_cap_predicts_target():
    w = _wave_width(16_000_000, CNT, WID, WALL)
    assert w == int(WID * (WAVE_TARGET_S / WALL))
    assert w < _wave_width(16_000_000, CNT, WID, 0.0)


def test_geometry_scale_shrinks_stale_yield():
    stale = _wave_width(16_000_000, CNT, WID, 0.0)
    scaled = _wave_width(16_000_000, CNT, WID, 0.0, 0.66)
    assert abs(scaled - int(stale * 0.66)) <= 2


def test_wall_cap_does_not_override_a_smaller_geometry_scale():
    scaled_only = _wave_width(16_000_000, CNT, WID, 0.0, 0.66)
    assert _wave_width(16_000_000, CNT, WID, WALL, 0.66) == scaled_only


def test_no_measurement_falls_back_to_probe():
    assert _wave_width(16_000_000, 0, 0, 0.0) == 30_000


def test_floor_is_never_below_probe_scale():
    assert _wave_width(1, CNT, WID, 0.0) >= 2000


def test_area_scale_numbers():
    # logI 14->15 at J pinned: 2x area, relations/q x1.5 -> width x0.66
    assert abs(_geom_area_scale(14, 8192, 15, 8192) - 0.66) < 0.01
    assert abs(_geom_area_scale(15, 8192, 15, 16384) - 0.66) < 0.01
    assert abs(_geom_area_scale(15, 16384, 15, 8192) - 2 ** 0.6) < 0.02
    assert _geom_area_scale(15, 8192, 15, 8192) == 1.0
    assert _geom_area_scale(20, 1 << 19, 2, 2) == 2.0       # clamp


def test_state_roundtrips_geometry_and_wall(tmp_path):
    n = 10 ** 70 + 3
    _save_nfs_state(tmp_path, n, q_next=123, q_cap=456, appended=7,
                    filters=[[1, 2, 3, None, None]], filter_seq=1,
                    wave_seq=2, last_cnt=CNT, last_width=WID,
                    last_wall=WALL, timed_wall=21600.0,
                    geometry=[15, 8192])
    s = _load_nfs_state(tmp_path, n)
    assert s["geometry"] == [15, 8192]
    assert s["last_wall"] == WALL
    assert s["timed_wall"] == 21600.0
