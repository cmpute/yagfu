import pytest

from yagfu.backends import find_bin
from yagfu.backends.siqs import _narrow_bug_params
from yagfu.params import (
    NFS_PARAMS_ANCHORS,
    mpqs_lp1_bound,
    nfs_params,
    poly_budget,
    sieve_time_estimate,
    snfs_gnfs_equiv,
    snfs_params,
    snfs_sieve_params,
)


def test_mpqs_lp1():
    assert mpqs_lp1_bound(70) is None
    assert mpqs_lp1_bound(79) is None
    assert mpqs_lp1_bound(80) == 150_000_000
    assert mpqs_lp1_bound(85) == 150_000_000
    assert mpqs_lp1_bound(89) == 150_000_000
    assert mpqs_lp1_bound(90) == 300_000_000
    assert mpqs_lp1_bound(110) == 300_000_000


def test_nfs_params_anchors_exact():
    assert nfs_params(70) == (350_000, 350_000, 23, 12)
    assert nfs_params(100) == (3_000_000, 3_000_000, 27, 14)
    assert nfs_params(150) == (100_000_000, 100_000_000, 31, 14)


def test_nfs_params_clamps_above():
    # above the table: clamp to the last row
    assert nfs_params(200) == nfs_params(150)


def test_nfs_params_below_table_raises():
    # clamping a tiny composite to the c70 row is not a valid sieve job
    # (norms under the bounds, cofactor-queue overflow at the first q)
    with pytest.raises(ValueError, match="from c70"):
        nfs_params(40)
    with pytest.raises(ValueError, match="from c70"):
        nfs_params(6)


def test_run_nfs_refuses_below_table(tmp_path):
    """The backend must not even set a workdir up for a tiny composite."""
    from yagfu.backends.nfs import run_nfs
    from yagfu.config import RunContext

    wd = tmp_path / "nfs"
    res = run_nfs(409 * 449, RunContext(quiet=True,
                                        log_file=tmp_path / "run.log"), wd)
    assert not res.ok and "c70" in res.error
    assert not wd.exists()


def test_nfs_params_monotone_interpolation():
    prev = 0
    for d in range(70, 150, 5):
        rlim, alim, lpb, logI = nfs_params(d)
        assert rlim == alim
        assert 12 <= logI <= 14
        assert lpb >= 23
        assert rlim >= prev
        prev = rlim


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


def test_snfs_params_anchors_exact():
    assert snfs_params(100) == (500_000, 500_000, 25, 12)
    assert snfs_params(300) == (200_000_000, 200_000_000, 32, 14)


def test_snfs_params_clamps():
    assert snfs_params(50) == snfs_params(100)
    assert snfs_params(400) == snfs_params(300)


def test_snfs_params_track_factordb_jobs():
    # the two factordb-generated snfs jobs pinned in the anchor comments:
    # d~170 -> ~7M / lpb 27 / I13, d~270 -> ~85.6M / lpb 31 (I15 capped 14)
    rlim, alim, lpb, logI = snfs_params(170.84)
    assert rlim == alim and 6_000_000 <= rlim <= 9_000_000
    assert lpb == 27 and logI == 13
    rlim, alim, lpb, logI = snfs_params(269.75)
    assert rlim == alim and 80_000_000 <= rlim <= 110_000_000
    assert lpb == 31 and logI == 14


def test_snfs_gnfs_equiv():
    # yafu's empirical rule; factordb's own page text pins the example:
    # difficulty 169.75 "costs about as much as GNFS on a 125-digit number"
    assert snfs_gnfs_equiv(169.75) == 125.06
    assert snfs_gnfs_equiv(269.75) == 181.06


def test_snfs_sieve_params_calibrated_point():
    # Kyle Askine's 15^253-1 review (Nov 2026) was of the factordb
    # suggestion at SCALED difficulty 274.74: "600M/400M at least" and
    # "95/63 for mfb" with the special-q on the rational side.  The anchor
    # sits on factordb's scale, not the halved form's true norm difficulty.
    (rlim, alim, lpbr, lpba, mfbr, mfba, rlbd, albd, logI) = \
        snfs_sieve_params(275.0, "r")
    assert rlim == 600_000_000 and alim == 400_000_000
    assert lpbr == 32 and mfbr == 95          # 3LP on the special-q side
    assert lpba == 32 and mfba == 63          # 2LP on the other side
    assert rlbd > albd                        # CADO lambda rule follows mfb
    assert logI == 14
    # the real 15^253-1 suggestion interpolates just under the anchor
    near = snfs_sieve_params(274.74, "r")
    assert 590_000_000 <= near[0] <= 600_000_000
    # mirroring the sides flips the mfb split, keeps the limits
    flipped = snfs_sieve_params(275.0, "a")
    assert flipped[:2] == (rlim, alim)
    assert (flipped[3], flipped[5]) == (32, 95)
    assert (flipped[2], flipped[4]) == (32, 63)


def test_snfs_sieve_params_2lp_below_threshold():
    # below SNFS_3LP_MIN_DIFFICULTY both sides stay 2LP
    for side in ("a", "r"):
        p = snfs_sieve_params(110.0, side)
        assert p[4] == 2 * p[2] and p[5] == 2 * p[3]


def test_snfs_sieve_params_monotone_limits():
    prev = snfs_sieve_params(100.0, "r")
    for d in (120, 135, 160, 200, 260, 300):
        cur = snfs_sieve_params(float(d), "r")
        assert cur[0] >= prev[0] and cur[1] >= prev[1]
        prev = cur
