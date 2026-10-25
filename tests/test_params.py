import pytest

from yagfu.backends import find_bin
from yagfu.backends.siqs import _narrow_bug_params
from yagfu.params import (
    NFS_PARAMS_ANCHORS,
    mpqs_lp1_bound,
    nfs_params,
    poly_budget,
    sieve_time_estimate,
)


def test_mpqs_lp1():
    assert mpqs_lp1_bound(70) is None
    assert mpqs_lp1_bound(75) is None
    assert mpqs_lp1_bound(85) == 150_000_000
    assert mpqs_lp1_bound(95) == 300_000_000
    assert mpqs_lp1_bound(110) == 300_000_000
    v = mpqs_lp1_bound(90)
    assert 150_000_000 < v <= 300_000_000


def test_nfs_params_anchors_exact():
    assert nfs_params(70) == (350_000, 350_000, 23, 12, 800_000)
    assert nfs_params(100) == (3_000_000, 3_000_000, 27, 14, 15_000_000)
    assert nfs_params(150) == (100_000_000, 100_000_000, 31, 14, 2_000_000_000)


def test_nfs_params_clamps():
    # below/above the table: clamp to the end rows
    assert nfs_params(40) == nfs_params(70)
    assert nfs_params(200) == nfs_params(150)


def test_nfs_params_monotone_interpolation():
    prev = 0
    for d in range(70, 150, 5):
        rlim, alim, lpb, logI, raw = nfs_params(d)
        assert rlim == alim
        assert 12 <= logI <= 14
        assert lpb >= 23
        assert raw >= prev
        prev = raw


def test_poly_budget():
    assert poly_budget(70) == 60.0            # floor binds up to ~c123
    assert poly_budget(90) == 60.0
    assert poly_budget(120) == 60.0           # 0.08 * T(120) = 42s < floor
    b130 = poly_budget(130)
    assert b130 == pytest.approx(0.08 * sieve_time_estimate(130))
    assert poly_budget(100, override=0) == 0  # msieve's own deadline
    assert poly_budget(100, override=999) == 999


def test_sieve_time_estimate_anchors():
    assert sieve_time_estimate(100) == pytest.approx(59.0)
    assert sieve_time_estimate(90) == pytest.approx(19.67, abs=0.5)


_NARROW_LOG = """\
[Stage 500: Tuning] [Tuning] Small-N adaptive config: shc_dim=6 totalPolys=32 totalIntervals=16 metaB=32
[Stage 1000: Sieve] [Sieve] [Config] VALIDATION ERROR: NARROW interval coverage must equal 2M. numIntervals(16) * sievingBlockSize(65536) = 1048576 >
                                                    2M(524288). Sieving past [-M,M) is fast per position and barren in relations. Required: numIntervals = 8 (--params field 2).
"""


def test_narrow_bug_params():
    # tuple in the shape the internal Small-N path would have used
    assert _narrow_bug_params(_NARROW_LOG) == "32,8,1,8,32,256,32,256"


def test_narrow_bug_params_needs_both_matches():
    assert _narrow_bug_params("Required: numIntervals = 8") is None
    assert _narrow_bug_params("totalPolys=32") is None
    assert _narrow_bug_params("unrelated") is None


def test_find_bin_order(tmp_path, monkeypatch):
    # 1. explicit dir wins
    a = tmp_path / "a"; a.mkdir()
    (a / "ecm").write_text("#!/bin/sh\n")
    b = tmp_path / "b"; b.mkdir()
    (b / "ecm").write_text("#!/bin/sh\n")
    monkeypatch.delenv("YAGFU_BIN", raising=False)
    assert find_bin("ecm", a) == a / "ecm"
    # 2. YAGFU_BIN
    monkeypatch.setenv("YAGFU_BIN", str(b))
    assert find_bin("ecm", None) == b / "ecm"
    # 3. missing binary raises with candidates listed
    monkeypatch.setenv("YAGFU_BIN", str(tmp_path / "missing"))
    with pytest.raises(FileNotFoundError, match="no-such-bin"):
        find_bin("no-such-bin", None)
