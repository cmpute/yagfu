import pytest

from yagfu.backends import find_bin
from yagfu.backends.siqs import _narrow_bug_params
from yagfu.params import (
    NFS_PARAMS_ANCHORS,
    SNFS_2LP_UNMEASURED_ABOVE,
    SNFS_3LP_PROMOTE_SLACK,
    mpqs_lp1_bound,
    nfs_params,
    nfs_relations_estimate,
    poly_budget,
    sieve_time_estimate,
    snfs_gnfs_equiv,
    snfs_lp_structure,
    snfs_params,
    snfs_relations_estimate,
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
    # sits on factordb's scale; the criterion reproduces it from a measured
    # rational-side median (that family surveys ~400 bits at d275).
    (rlim, alim, lpbr, lpba, mfbr, mfba, rlbd, albd, logI) = \
        snfs_sieve_params(275.0, "r", [1, 0, 0, 0, 0, 0, 1], -(15 ** 46), 1,
                          sq_norm_bits=400.0)
    assert rlim == 600_000_000 and alim == 400_000_000
    assert lpbr == 32 and mfbr == 95          # 3LP on the special-q side
    assert lpba == 32 and mfba == 63          # 2LP on the other side
    assert rlbd > albd                        # CADO lambda rule follows mfb
    assert logI == 14
    # the real 15^253-1 suggestion interpolates just under the anchor
    near = snfs_sieve_params(274.74, "r", [1, 0, 0, 0, 0, 0, 1],
                             -(15 ** 46), 1, sq_norm_bits=400.0)
    assert 590_000_000 <= near[0] <= 600_000_000
    # mirroring the sides flips the mfb split, keeps the limits; the
    # +-1-coefficient algebraic side's measured median (~115 at this
    # geometry) stays under the threshold, so side a runs 2LP at 32 bits
    flipped = snfs_sieve_params(275.0, "a", [1, 0, 0, 0, 0, 0, 1],
                                -(15 ** 46), 1, sq_norm_bits=115.0)
    assert flipped[:2] == (rlim, alim)
    assert (flipped[3], flipped[5]) == (32, 63)
    assert (flipped[2], flipped[4]) == (32, 63)


def test_snfs_sieve_params_2lp_below_threshold():
    # a d110 job without norm info stays 2LP at the classic step
    for side in ("a", "r"):
        p = snfs_sieve_params(110.0, side)
        assert p[4] == 2 * p[2] - 1 and p[5] == 2 * p[3] - 1


def test_snfs_sieve_params_monotone_limits():
    prev = snfs_sieve_params(100.0, "r")
    for d in (120, 135, 160, 200, 260, 300):
        cur = snfs_sieve_params(float(d), "r")
        assert cur[0] >= prev[0] and cur[1] >= prev[1]
        prev = cur


# --------------------------------------------------------------------------
# graded LP ladder + norm promotion (ws-15* calibration, Nov 2026)
# --------------------------------------------------------------------------

#: y^6 - y^3 + 1 with base m: the halved-cyclotomic degree-6 family the
#: ws-15c522/558/642 jobs sieved; +-1 coefficients keep the algebraic
#: norms box-limited (~2^80 at logI 14) whatever the difficulty
_DEG6 = [1, 0, 0, -1, 0, 0, 1]


def test_snfs_lp_ladder_2lp_algebraic_sq():
    # ws-15c522 family, special-q on the algebraic side: cofactor ~2^52 is
    # far under the 2LP room, so no promotion — and the ladder steps lpb
    # with difficulty instead of pinning 32 from d125 (the A/B: c522's
    # 3LP@32 needed 329.5M where c261's 2LP@28 decided at 45.1M)
    lp = snfs_lp_structure(205.38, "a", _DEG6, -(15 ** 29), 1)
    assert (lp.sq_lpb, lp.other_lpb) == (30, 29)
    assert not lp.three_lp_sq
    assert (lp.sq_mfb, lp.other_mfb) == (59, 57)   # 2LP: mfb = 2*lpb - 1
    p = snfs_sieve_params(205.38, "a", _DEG6, -(15 ** 29), 1)
    # side "a": the sq bounds land on the algebraic side
    assert (p[2], p[3], p[4], p[5]) == (29, 30, 57, 59)
    assert p[6] == 2.27 and p[7] == 2.27          # CADO lambda rule
    lp = snfs_lp_structure(239.93, "a", _DEG6, -(15 ** 53), 1)
    assert (lp.sq_lpb, lp.other_lpb) == (31, 30)
    assert not lp.three_lp_sq


def test_snfs_lp_promotion_rational_sq():
    # same family, special-q on the rational side: |Y0|*B norms grow with
    # the base and the typical cofactor clears 2*lpb + slack, promoting
    # the sq side to 3LP at the ladder's own lpb
    lp = snfs_lp_structure(205.38, "r", [1, 0, 0, 1, 0, 0, 1],
                           -(15 ** 29), 1)
    assert lp.three_lp_sq
    assert lp.cofactor_bits > 2 * lp.sq_lpb + SNFS_3LP_PROMOTE_SLACK
    assert (lp.sq_lpb, lp.sq_mfb) == (30, 89)      # 3LP: mfb = 3*lpb - 1
    p = snfs_sieve_params(205.38, "r", [1, 0, 0, 1, 0, 0, 1],
                          -(15 ** 29), 1)
    assert (p[2], p[4]) == (30, 89)
    # difficulty-consistent small bases stay 2LP: a d120 job's |Y0|*B norms
    # leave the cofactor under the threshold
    lp = snfs_lp_structure(120.0, "r", [1, 0, 0, 1, 0, 0, 1],
                           -(10 ** 20), 1)
    assert not lp.three_lp_sq


def test_snfs_2lp_unmeasured_at_high_difficulty():
    # the d>=245 forced-3LP band is gone: the criterion decides everywhere,
    # and a 2LP verdict at d>=245 is legal but unmeasured
    # (SNFS_2LP_UNMEASURED_ABOVE -- run_nfs warns; every deciding point at
    # d>=245 ran 3LP under the old blanket rule)
    assert SNFS_2LP_UNMEASURED_ABOVE == 245.0
    # side a at d254 with a measured median typical of the +-1-coeff family
    # (the q-lattice pushes the d205 median up by a few bits): 2LP at the
    # ladder's 32-bit step
    lp = snfs_lp_structure(254.04, "a", _DEG6, -(15 ** 29), 1,
                           sq_norm_bits=172.0)
    assert not lp.three_lp_sq
    assert (lp.sq_lpb, lp.sq_mfb, lp.other_lpb, lp.other_mfb) == \
        (32, 63, 32, 63)
    # ... while a rational-side median of the c253 class still promotes
    lp = snfs_lp_structure(254.04, "r", _DEG6, -(15 ** 29), 1,
                           sq_norm_bits=400.0)
    assert lp.three_lp_sq and (lp.sq_lpb, lp.sq_mfb) == (32, 95)
    # without a polynomial the criterion cannot fire: 2LP at the ladder's
    # LPB step (the warned population when it lands at d>=245)
    lp = snfs_lp_structure(254.04, "r")
    assert not lp.three_lp_sq and (lp.sq_lpb, lp.sq_mfb) == (32, 63)


def test_snfs_lp_normscan_measured_promotion():
    # normscan-measured input drives the promotion with its own slack
    # (SNFS_3LP_PROMOTE_SLACK_MEASURED = 110): the c261 family's measured
    # side-a median (165.7 bits at d205) must stay 2LP -- c261 decided
    # 45.1M raw there at 2LP lpb 28 -- while a measured rational-side
    # median of 236 (the ws-15c265 level) clears the higher threshold
    lp = snfs_lp_structure(205.38, "a", _DEG6, -(15 ** 29), 1,
                           sq_norm_bits=165.7)
    assert lp.norm_source == "normscan" and lp.promote_slack == 110.0
    assert not lp.three_lp_sq and lp.sq_mfb == 59
    lp = snfs_lp_structure(205.38, "r", [1, 0, 0, 1, 0, 0, 1],
                           -(15 ** 29), 1, sq_norm_bits=236.0)
    assert lp.three_lp_sq and (lp.sq_lpb, lp.sq_mfb) == (30, 89)
    # the REAL Phi6 measurement (side-0 median 141.2) DECLINES where the
    # proxy would promote: the proxy's static box sits 87 bits below the
    # q-lattice reality on this side and over-promotes the family
    lp = snfs_lp_structure(205.38, "r", [1, 0, 0, 1, 0, 0, 1],
                           -(15 ** 29), 1, sq_norm_bits=141.2)
    assert not lp.three_lp_sq and lp.sq_mfb == 59


def test_snfs_relations_estimate_by_profile():
    # the measured 3LP deciding counts anchor their table directly
    assert snfs_relations_estimate(205.38, True) == 330_000_000
    assert 329_000_000 <= snfs_relations_estimate(219.03, True) <= 331_000_000
    est = snfs_relations_estimate(274.74, True)
    assert 395_000_000 <= est <= 410_000_000
    # the 2LP table floors at c261's deciding 45.1M, padded for the
    # ladder's heavier 30/29 profile
    assert 59_000_000 <= snfs_relations_estimate(205.38, False) <= 62_000_000
    assert snfs_relations_estimate(100.0, False) == 10_000_000
    # the 7x profile spread at one difficulty is the point
    assert (snfs_relations_estimate(205.38, True)
            / snfs_relations_estimate(205.38, False) > 4)
    # the GNFS table is untouched (c152 decided at 235.8M)
    assert 220_000_000 <= nfs_relations_estimate(152) <= 240_000_000
