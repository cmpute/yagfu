"""SNFS polynomials yagfu finds itself, and the difficulty YAFU's rule
gives them: Phi Polynomial Lab's enumeration (snfsforms), the raw and the
scaled difficulty with the special-q side (backends.nfs), and the poly
blocks --bench-nfs-poly writes and --poly reads back.  Pure Python: no
scorer, no msieve."""

import math

import pytest

from yagfu import snfsforms as S
from yagfu.backends import nfs
from yagfu.cli import (_form_from_record, _form_text, _poly_from_file,
                       _seed_gnfs_poly)


def _phi30(t):
    """Phi_30(t) = t^8 + t^7 - t^5 - t^4 - t^3 + t + 1."""
    return t**8 + t**7 - t**5 - t**4 - t**3 + t + 1


# the open C166 of C(570,14) = Phi_570(14) = Phi_30(14^19) / Phi_30(14),
# a factor of 14^285 + 1; and its degree-4 cyclotomic form (the example
# phi.kyleaskine.com's author checked against FactorDB: difficulty 174.21,
# scaled 178.95 at skew 2)
N166 = _phi30(14**19) // _phi30(14)
C166 = [1, -4, -4, 1, 1]
Y0_166, Y1_166 = 14**38 + 1, -14**19


def test_phi_forms_of_the_c166():
    """Phi's enumeration for 14^285 + 1 and its cofactor: 25 forms over the
    degrees 4 to 9, every one with a root modulo the cofactor and no poly
    twice; the degree-4 cyclotomic form is the t + 1/t form of Phi_15."""
    assert len(str(N166)) == 166 and (14**285 + 1) % N166 == 0
    cands, skipped = S.enumerate_candidates(14, 1, 285, 1, N166)
    assert len(cands) == 25 and skipped == []
    assert all(c.has_root(N166) for c in cands)
    keys = [(c.coefficients, c.y0, c.y1) for c in cands]
    assert len(set(keys)) == 25
    assert {c.construction for c in cands} == {"generic rounding", "cyclotomic"}
    cyc = next(c for c in cands if c.degree == 4 and c.construction == "cyclotomic")
    assert (cyc.coefficients, cyc.y0, cyc.y1) == (tuple(C166), Y0_166, Y1_166)
    # degrees score-poly cannot score are left out when asked
    c48, _ = S.enumerate_candidates(14, 1, 285, 1, N166, degrees=(4, 5, 6, 7, 8))
    assert not any(c.degree == 9 for c in c48) and len(c48) < 25


def test_form_parsing():
    assert S.parse_form("13^263-1") == (13, 1, 263, -1)
    assert S.parse_form("14^308+1") == (14, 1, 308, 1)
    assert S.parse_form("5^12-3^12") == (5, 3, 12, -1)
    assert S.parse_form(" 2^15 + 1 ") == (2, 1, 15, 1)
    for bad in ("13^263", "13^-1", "M127", ""):
        with pytest.raises(ValueError):
            S.parse_form(bad)
    # Phi_a(b) divides b^a - 1 for odd a, b^(a/2) + 1 for even a
    assert S.form_cyclotomic(13, 381) == (13, 1, 381, -1)
    assert S.form_cyclotomic(14, 570) == (14, 1, 285, 1)
    assert _form_text((14, 1, 285, 1)) == "14^285+1"
    assert _form_text((5, 3, 12, -1)) == "5^12-3^12"


def test_enumeration_rejects_what_the_target_does_not_divide():
    with pytest.raises(ValueError):
        S.enumerate_candidates(14, 1, 285, 1, 14**285 - 1)    # N !| the form
    with pytest.raises(ValueError):
        S.enumerate_candidates(14, 7, 285, 1, N166)           # bases not coprime
    with pytest.raises(ValueError):
        S.enumerate_candidates(14, 1, 285, 1, N166, degrees=(3,))   # off 4..9


def test_form_read_off_a_record_term():
    """C(a,b), a cofactor's C(a,b)/..., the repunit alias R(b,n) and a term
    that is itself a power form all name the form; anything else, none."""
    f = lambda term: _form_from_record({"term": term})   # noqa: E731
    assert f("C(381,13)") == (13, 1, 381, -1)
    assert f("C(381,13)/1011937") == (13, 1, 381, -1)
    assert f("C(570,14)") == (14, 1, 285, 1)
    assert f("R(15,491)") == (15, 1, 491, -1)
    assert f("13^32+1") == (13, 1, 32, 1)
    assert f("1100000000218514376") is None and f("") is None


def test_base_difficulty_is_the_forms_digits():
    """D = log10 |F(-Y0, Y1)|: 8*19*log10(14) for the C166 form, with the
    homogeneous value exact however large the coefficients are."""
    assert nfs.snfs_base_difficulty(C166, Y0_166, Y1_166) == \
        pytest.approx(152 * math.log10(14), abs=1e-9)
    assert nfs.snfs_base_difficulty([14, 0, 0, 0, 1], 14**38, 1) == \
        pytest.approx(4 * 38 * math.log10(14), abs=1e-9)
    with pytest.raises(ValueError):
        nfs.snfs_base_difficulty([0, 0, 1], 0, 1)


def test_norms_at_siever_13_reproduce_the_checked_example():
    """Over YAFU's reference rectangle for siever 13 (a, b) = (8192000,
    4096000) at skew 2, the two log norms and the rule's answer."""
    la, lr = nfs.snfs_log_norms(C166, Y0_166, Y1_166, 2.0, 13)
    assert (la, lr) == pytest.approx((26.44944, 50.16523), abs=2e-5)
    assert (lr - la) / 5.0 == pytest.approx(4.74316, abs=2e-5)
    assert 152 * math.log10(14) + (lr - la) / 5.0 == \
        pytest.approx(178.95462, abs=2e-4)             # the rational side


def test_norms_at_a_corner_the_form_vanishes_at():
    """A binomial z^d - c at the skew c^(1/d) is zero on the rectangle's
    corner, and with the corner in integers the value there is only
    truncation noise; the norms are read 1% inside instead.  The same for
    a rational side read at the slope of its own root."""
    area = 2.0 ** 25 * 10.0 ** 6                      # siever 13's rectangle
    skew = 5 ** 0.25
    a, b = int(math.sqrt(area * skew)), int(math.sqrt(area / skew))
    assert a**4 - 5 * b**4 != 0                       # noise, not zero
    assert abs(a**4 - 5 * b**4) * max(a, b) <= sum(
        abs(c) * a**i * b**(4 - i) for i, c in enumerate([-5, 0, 0, 0, 1])) * 4
    la, lr = nfs.snfs_log_norms([-5, 0, 0, 0, 1], -1, 1, skew, 13)
    inside = b + max(1, b // 100)
    assert (la, lr) == pytest.approx((math.log10(abs(a**4 - 5 * inside**4)),
                                      math.log10(abs(a - inside))))
    # off that skew the corner itself is read, YAFU's way
    la, lr = nfs.snfs_log_norms([-5, 0, 0, 0, 1], -1, 1, 3.0, 13)
    a, b = int(math.sqrt(area * 3.0)), int(math.sqrt(area / 3.0))
    assert (la, lr) == pytest.approx((math.log10(abs(a**4 - 5 * b**4)),
                                      math.log10(abs(a - b))))


def test_norm_siever_follows_yafus_table():
    """The siever of the table row nearest the difficulty, YAFU's own 13
    off the table — which every SNFS difficulty above 185 is."""
    assert nfs.snfs_norm_siever(174.21) == 15         # nearer 175 than 170
    assert nfs.snfs_norm_siever(172) == 14
    assert nfs.snfs_norm_siever(100) == 12
    assert nfs.snfs_norm_siever(80) == 11
    assert nfs.snfs_norm_siever(185) == 16
    assert nfs.snfs_norm_siever(186) == 13
    assert nfs.snfs_norm_siever(302.58) == 13
    assert nfs.snfs_norm_siever(50) == 13


def test_scale_difficulty_by_yafu_rule():
    """YAFU's rule as its source has it: the siever from the table, so the
    checked example lands on 15's rectangle, not the 13 it was checked at.
    The rational side only when it is larger by more than five orders."""
    job = nfs.SnfsJob(n=N166, coeffs=C166, y0=Y0_166, y1=Y1_166, skew=2.0,
                      difficulty=0.0, scaled=0.0)
    nfs.snfs_scale_difficulty(job)
    assert job.difficulty == pytest.approx(152 * math.log10(14), abs=1e-9)
    assert job.scaled == pytest.approx(177.99338, abs=2e-4)
    assert job.side == "r"
    # the algebraic side keeps a difficulty it is given, and a side it is told to keep
    job = nfs.SnfsJob(n=N166, coeffs=C166, y0=Y0_166, y1=Y1_166, skew=2.0,
                      difficulty=200.0, scaled=0.0, side="a")
    nfs.snfs_scale_difficulty(job, keep_side=True)
    assert job.difficulty == 200.0 and job.side == "a"
    assert job.scaled == pytest.approx(200.0 + (nfs.snfs_log_norms(
        C166, Y0_166, Y1_166, 2.0, nfs.snfs_norm_siever(200.0))[1]
        - nfs.snfs_log_norms(C166, Y0_166, Y1_166, 2.0,
                             nfs.snfs_norm_siever(200.0))[0]) / 5.0)
    # a balanced poly keeps its difficulty: with the rational side sized to
    # the algebraic norm at the reference corner, neither is the larger
    area = 2.0 ** 25 * 10.0 ** 6                      # siever 13's rectangle
    corner = int(math.sqrt(area))
    job = nfs.SnfsJob(n=N166, coeffs=[1, 0, 0, 0, 0, 0, 1], y0=-2 * corner ** 5,
                      y1=1, skew=1.0, difficulty=0.0, scaled=0.0)
    nfs.snfs_scale_difficulty(job)
    assert job.side == "a"
    assert job.scaled == pytest.approx(job.difficulty, abs=1e-9)


def test_jobs_from_form():
    """The C166's forms as jobs: degrees 4 to 8, every poly with its root,
    sized by YAFU's rule; the cyclotomic form carries its 174.21."""
    jobs = nfs.snfs_jobs_from_form(N166, 14, 1, 285, 1)
    expected, _ = S.enumerate_candidates(14, 1, 285, 1, N166, degrees=(4, 5, 6, 7, 8))
    assert len(jobs) == len(expected) == 20
    assert all(nfs.poly_has_root_mod_n(j) for j in jobs)
    assert all(j.difficulty > 0 and j.scaled >= j.difficulty for j in jobs)
    assert all(j.form == "14^285+1" and j.side in ("a", "r") for j in jobs)
    cyc = next(j for j in jobs if j.coeffs == C166)
    assert (cyc.y0, cyc.y1) == (Y0_166, Y1_166)
    assert cyc.difficulty == pytest.approx(152 * math.log10(14), abs=1e-9)
    assert all(j.skew == 1.0 for j in jobs)     # until score-poly re-skews


def test_snfs_poly_block_round_trip():
    """snfs_poly_text is what snfs_job_from_file reads: the poly and the
    difficulties come back; a bare block gets the difficulties from the
    rule; one with them keeps them."""
    job = nfs.SnfsJob(n=N166, coeffs=C166, y0=Y0_166, y1=Y1_166, skew=2.0,
                      difficulty=174.21, scaled=178.95, side="r", form="14^285+1")
    back = nfs.snfs_job_from_file(nfs.snfs_poly_text(job))
    assert (back.n, back.coeffs, back.y0, back.y1, back.side, back.form) == \
        (job.n, job.coeffs, job.y0, job.y1, "r", job.form)
    assert (back.difficulty, back.scaled) == (174.21, 178.95)
    bare = f"n: {N166}\nc4: 1\nc3: 1\nc2: -4\nc1: -4\nc0: 1\nY1: {Y1_166}\nY0: {Y0_166}\nskew: 2\n"
    back = nfs.snfs_job_from_file(bare)
    assert back is not None and back.coeffs == C166
    assert back.difficulty == pytest.approx(152 * math.log10(14), abs=1e-9)
    assert back.scaled == pytest.approx(177.99338, abs=2e-4) and back.side == "r"
    # a job file that names a rational side keeps it when the rest is derived
    back = nfs.snfs_job_from_file(bare + "# special-q on the rational side\n")
    assert back.side == "r" and back.scaled == pytest.approx(177.99338, abs=2e-4)
    back = nfs.snfs_job_from_file(bare + "# special-q on the algebraic side\n")
    assert back.side == "a" and back.scaled == pytest.approx(177.99338, abs=2e-4)


FB = {"N": "123", "SKEW": "3.0", "R0": "-100000", "R1": "1", "A0": "1", "A4": "5"}


def test_gnfs_poly_blocks():
    """A GNFS poly travels between msieve.fb key/values, the GGNFS poly
    block that carries its type, and msieve.fb text — zero coefficients
    dropped, any of the three read back."""
    assert nfs.gnfs_fb_from_text(nfs.gnfs_poly_text(123, FB)) == FB
    assert nfs.gnfs_fb_from_text(nfs.gnfs_fb_text(FB)) == FB
    with pytest.raises(ValueError):
        nfs.gnfs_poly_text(124, FB)              # another number's poly
    assert nfs.gnfs_fb_from_text("# nothing here\n") is None


def test_poly_from_file(tmp_path):
    """--poly reads an SNFS block, a GNFS block and an msieve.fb, and
    refuses a block that does not say which it is."""
    snfs = tmp_path / "p.snfs.job"
    snfs.write_text(nfs.snfs_poly_text(nfs.SnfsJob(
        n=331, coeffs=[1, -4, -4, 1, 1], y0=5, y1=-2, skew=2.0, difficulty=0.0,
        scaled=0.0, form="2^15+1")))
    kind, job, fb = _poly_from_file(str(snfs))
    assert kind == "snfs" and fb is None
    assert (job.n, job.coeffs, job.y0, job.y1) == (331, [1, -4, -4, 1, 1], 5, -2)
    assert job.difficulty > 0 and job.scaled >= job.difficulty
    gnfs = tmp_path / "p.gnfs.job"
    gnfs.write_text(nfs.gnfs_poly_text(123, FB))
    assert _poly_from_file(str(gnfs)) == ("gnfs", None, FB)
    msieve = tmp_path / "msieve.fb"
    msieve.write_text(nfs.gnfs_fb_text(FB))
    assert _poly_from_file(str(msieve)) == ("gnfs", None, FB)
    bare = tmp_path / "bare.job"
    bare.write_text("n: 123\nc0: 1\nc4: 5\nY0: -100000\nY1: 1\n")
    with pytest.raises(SystemExit):
        _poly_from_file(str(bare))
    with pytest.raises(SystemExit):
        _poly_from_file(str(tmp_path / "missing"))


def test_seed_gnfs_poly(tmp_path):
    """The --poly GNFS poly lands as the msieve.fb of the nfs workdir, in
    the shape run_nfs reuses."""
    wd = tmp_path / "nfs"
    _seed_gnfs_poly(wd, FB)
    back = nfs._parse_msieve_fb((wd / "msieve.fb").read_text())
    assert back["N"] == "123" and back["A4"] == "5" and back["R0"] == "-100000"
    _seed_gnfs_poly(wd, FB)                      # idempotent
    assert (wd / "msieve.fb").exists()
