import pytest

from yagfu.cli import (_db_factor_list, bench_nfs, build_parser,
                       factor_with_db, load_numbers)
from yagfu.config import RunContext
from yagfu.engine import FactorResult, FoundFactor


def test_ecm_only_flag():
    ap = build_parser()
    args = ap.parse_args(["--ecm-only", "12345"])
    assert args.ecm_only is True
    args = ap.parse_args(["12345"])
    assert args.ecm_only is False


def test_ctx_defaults():
    ctx = RunContext()
    assert ctx.ecm_only is False
    assert ctx.quiet is False
    assert ctx.verbose == 0


def test_log_levels(capsys):
    ctx = RunContext()
    ctx.log("key")                 # default: shown
    ctx.logv("detail")             # needs -v: hidden
    err = capsys.readouterr().err
    assert "key" in err and "detail" not in err
    ctx.verbose = 1
    ctx.logv("detail2")
    assert "detail2" in capsys.readouterr().err
    quiet = RunContext(quiet=True)
    quiet.log("hidden")
    assert capsys.readouterr().err == ""


def test_load_numbers_file(tmp_path):
    f = tmp_path / "nums.txt"
    f.write_text("# comment\n\n123456789\n  98765 \n")
    args = build_parser().parse_args(["-f", str(f)])
    assert load_numbers(args) == [123456789, 98765]


def test_load_numbers_mutually_exclusive(tmp_path, capsys):
    f = tmp_path / "n.txt"
    f.write_text("6\n")
    args = build_parser().parse_args(["-f", str(f), "10"])
    with pytest.raises(SystemExit):
        load_numbers(args)


def test_load_numbers_id_needs_db():
    args = build_parser().parse_args(["id:1100000000631376229"])
    with pytest.raises(SystemExit):
        load_numbers(args)


def test_load_numbers_id_form():
    args = build_parser().parse_args(["-db", "id:1100000000631376229"])
    assert load_numbers(args) == [("id", "1100000000631376229")]


def test_load_numbers_file_mixed(tmp_path):
    f = tmp_path / "nums.txt"
    f.write_text("id:123\n# composite\n\n98765\nid: 456 \n")
    args = build_parser().parse_args(["-db", "-f", str(f)])
    assert load_numbers(args) == [("id", "123"), 98765, ("id", "456")]


def test_load_numbers_bad_id():
    args = build_parser().parse_args(["-db", "id:12x3"])
    with pytest.raises(SystemExit):
        load_numbers(args)
    args = build_parser().parse_args(["-db", "id:"])
    with pytest.raises(SystemExit):
        load_numbers(args)


# ---------------------------------------------------------------- factor-db

class FakeFdb:
    """get_number/get_number_by_id/get_factor_decimals canned; records
    reports and id lookups."""

    def __init__(self, status, factors, rec=None):
        self.status = status
        self.factors = factors
        self.rec = rec
        self.reported = None
        self.by_id = []

    def get_number(self, n, snfs=False):
        return {"fid": "fid1", "status": self.status}

    def get_number_by_id(self, fid, snfs=False):
        self.by_id.append((fid, snfs))
        return self.rec

    def get_factor_decimals(self, fid):
        return [str(f) for f in self.factors]

    def report_factors(self, fid, n, factors):
        self.reported = list(factors)
        return {}

    def factor_fid_map(self, fid):
        return {}

    def report_ecm(self, *a, **k):
        return {}


class _QuietCtx:
    def log(self, msg):
        pass

    def logv(self, msg):
        pass


class FakeEngine:
    """Records the (number, snfs) it was asked to factor; finds canned
    primes."""

    def __init__(self, new_factors):
        self.new_factors = new_factors
        self.asked = []
        self.ctx = _QuietCtx()

    def factor(self, n, wd, snfs=None):
        self.asked.append((n, snfs))
        res = FactorResult(n=n)
        res.factors = [FoundFactor(f, "nfs", True) for f in self.new_factors]
        prod = 1
        for f in res.factors:
            prod *= f.value
        res.complete = prod == n
        return res


def test_db_factor_list_skips_composites():
    # a CF record may store composite factors: n = p * c131 (the real
    # 716207641227... record).  Only the primes may count as known.
    c = 1000003 * 1000033                      # composite
    n = 2 * c
    fdb = FakeFdb("CF", [2, c])
    assert _db_factor_list(fdb, n, "fid1") == [2]


def test_db_cf_composite_factor_goes_to_engine():
    # regression: factordb knows n = 4043353 * c (c composite, as in the
    # real 716207641227... record).  The composite must stay in the
    # remainder for the engine, and only the locally-found primes are
    # reported back.
    p = 4043353
    c = 1000003 * 1000033                      # the "c131" of the record
    n = p * c
    fdb = FakeFdb("CF", [p, c])
    eng = FakeEngine([1000003, 1000033])
    res, merged, fdb_info = factor_with_db(n, fdb, eng, None, report=True)
    assert eng.asked == [(c, None)]            # engine got the composite part
    assert sorted(f.value for f in merged) == [1000003, 1000033, 4043353]
    prod = 1
    for f in merged:
        prod *= f.value
    assert prod == n
    assert fdb.reported == [1000003, 1000033]


def test_db_ff_never_runs_engine():
    p, q = 4043353, 2305843009213693951
    n = p * q
    fdb = FakeFdb("FF", [p, q])
    eng = FakeEngine([])
    res, merged, fdb_info = factor_with_db(n, fdb, eng, None, report=True)
    assert eng.asked == []
    assert res.complete
    assert sorted(f.value for f in merged) == [p, q]
    assert fdb.reported is None


# ------------------------------------------------------------------- snfs

def _snfs_rec(n, scaled=40.0, status="C"):
    """A factordb record carrying an SNFS suggestion (probe shape).

    The poly x^2-49 with y0=-7/y1=1 has its root exactly (f(-y0/y1)=0 for
    every n), so it survives parse_snfs_suggestion's root-mod-n filter
    regardless of the fixture's n."""
    return {"fid": "fid1", "status": status,
            "snfs": {"n": str(n), "status": status,
                     "polys": [{"c": ["-49", "0", "1"],
                                "difficulty": scaled, "scaled": scaled,
                                "side": "a", "skew": 1.0,
                                "y0": "-7", "y1": "1",
                                "form": "x^2-49", "e": 1e-4}]}}


def test_db_snfs_suggestion_reaches_engine():
    n = 1000003 * 1000033
    fdb = FakeFdb("C", [], rec=_snfs_rec(n))
    eng = FakeEngine([1000003, 1000033])
    res, merged, fdb_info = factor_with_db(n, fdb, eng, None, report=False,
                                           rec=fdb.rec, use_snfs=True)
    assert len(eng.asked) == 1 and eng.asked[0][0] == n
    assert eng.asked[0][1] is not None and eng.asked[0][1].scaled == 40.0
    assert fdb_info["snfs"] == 40.0


def test_db_snfs_skipped_when_db_knows_factors():
    # a CF record: the remainder lost db-known factors, and the
    # suggestion maps the whole record's number — it does not apply
    p = 4043353
    c = 1000003 * 1000033
    n = p * c
    fdb = FakeFdb("CF", [p], rec=_snfs_rec(n, status="CF"))
    eng = FakeEngine([1000003, 1000033])
    res, merged, fdb_info = factor_with_db(n, fdb, eng, None, report=False,
                                           rec=fdb.rec, use_snfs=True)
    assert eng.asked == [(c, None)]
    assert "snfs" not in fdb_info


def test_db_snfs_off_by_default():
    n = 1000003 * 1000033
    fdb = FakeFdb("C", [], rec=_snfs_rec(n))
    eng = FakeEngine([1000003, 1000033])
    res, merged, fdb_info = factor_with_db(n, fdb, eng, None, report=False,
                                           rec=fdb.rec)
    assert eng.asked == [(n, None)]
    assert "snfs" not in fdb_info


# ------------------------------------------------------------- --bench-nfs

class BenchClient:
    """Canned get_number/get_number_by_id for bench_nfs."""

    def __init__(self, rec):
        self.rec = rec

    def get_number(self, n, snfs=False):
        return self.rec

    def get_number_by_id(self, fid, snfs=False):
        return self.rec


# x^6 + 2 at m = 10^5 divides n: the suggestion poly has a root mod n;
# x^5 + 1 does not (m^5 + 1 != n) and parse_snfs_suggestion drops it
BENCH_N = 10 ** 30 + 2
BENCH_REC = {
    "fid": "1100000008650831857",
    "status": "C",
    "decimal": str(BENCH_N),
    "snfs": {
        "n": str(BENCH_N),
        "polys": [
            {"c": [2, 0, 0, 0, 0, 0, 1], "y0": -10 ** 5, "y1": 1,
             "difficulty": 30.0, "scaled": 30.0, "skew": 1.0,
             "side": "a", "e": 3.0e-11, "form": "10^30+2"},
            {"c": [1, 0, 0, 0, 0, 1], "y0": -10 ** 5, "y1": 1,
             "difficulty": 25.0, "scaled": 25.0, "skew": 1.0,
             "side": "a"},
        ],
    },
}


def _bench_args(tmp_path):
    return build_parser().parse_args(
        ["--bench-nfs", "id:1100000008650831857", "--bench-nfs-time", "1",
         "--workdir", str(tmp_path / "wd")])


def test_bench_nfs_recommends_snfs(tmp_path, capsys):
    """score-poly and the timed GNFS polyselect run (stubbed binaries in
    bin_dir); the deg-6 SNFS candidate's normalized E wins and the bogus
    no-root candidate never reaches the scorer."""
    bindir = tmp_path / "bins"
    bindir.mkdir()
    (bindir / "score-poly").write_text(
        '#!/bin/sh\n'
        'echo "skew 12.5, murphyE 3.0e-11, size 1.0e-15, alpha 0.5, '
        'rroots 2"\n')
    (bindir / "msieve").write_text(
        '#!/bin/sh\n'
        'trap \'printf "combined = 2.5e-12\\n"\n'
        '      printf "N %s\\nSKEW 3.0\\nR0 -100000\\nR1 1\\nA0 1\\nA4 5\\n" '
        '> msieve.fb\n'
        '      exit 0\' INT\n'
        'echo "expecting poly E from 9.0e-11 to > 1.2e-10"\n'
        'echo "polyselect best E = 2.0e-12"\n'
        'sleep 30 &\n'
        'wait\n')
    for p in (bindir / "score-poly", bindir / "msieve"):
        p.chmod(0o755)
    ctx = RunContext(device="0", devices=("0",), threads=2,
                     bin_dir=str(bindir), log_file=tmp_path / "b.log",
                     quiet=True)
    args = _bench_args(tmp_path)
    rc = bench_nfs(args, ctx, BenchClient(BENCH_REC))
    out = capsys.readouterr().out
    assert rc == 0
    assert f"bench-nfs: c{len(str(BENCH_N))} (id:1100000008650831857)" in out
    # only the root-holding degree-6 candidate is scored
    assert "degree-6 candidate: normalized Murphy E 3.000e-11" in out
    assert "degree-5" not in out
    # GNFS side: degree 4 from the stub fb, normalized by 1.6^(6-4)
    assert "best SNFS Murphy E: 3.000e-11 (normalized, degree 6, " \
           "score-poly)" in out
    assert "best GNFS Murphy E: 6.400e-12 (normalized, degree 4, " \
           "msieve -np)" in out
    assert "recommendation: SNFS (normalized Murphy E ratio" in out
    # the normal flow's ECM pretest targets, per route: the suggestion's
    # scaled difficulty (gnfs-equiv 0.56*30+30) for SNFS, the digit count
    # for GNFS; plan=normal is 4/13 of either
    assert "ecm pretest target (plan=normal): SNFS route t14.4 " \
           "(gnfs c46.8), GNFS route t9.5" in out
    assert "SNFS route: the pretest would run to t14.4" in out
    assert "--bench-nfs-time" in out            # the undershoot note


def test_bench_nfs_gnfs_when_no_snfs(tmp_path, capsys):
    """No factordb suggestion: the timed GNFS polyselect alone decides."""
    bindir = tmp_path / "bins"
    bindir.mkdir()
    (bindir / "msieve").write_text(
        '#!/bin/sh\n'
        'trap \'printf "combined = 2.5e-12\\n"\n'
        '      printf "N %s\\nSKEW 3.0\\nR0 -100000\\nR1 1\\nA0 1\\nA4 4\\n" '
        '> msieve.fb\n'
        '      exit 0\' INT\n'
        'echo "polyselect best E = 2.0e-12"\n'
        'sleep 30 &\n'
        'wait\n')
    (bindir / "msieve").chmod(0o755)
    ctx = RunContext(device="0", devices=("0",), threads=2,
                     bin_dir=str(bindir), log_file=tmp_path / "b.log",
                     quiet=True)
    args = build_parser().parse_args(
        ["--bench-nfs", str(BENCH_N), "--bench-nfs-time", "1",
         "--workdir", str(tmp_path / "wd")])
    rc = bench_nfs(args, ctx, BenchClient({"fid": None, "status": "C",
                                           "decimal": str(BENCH_N),
                                           "snfs": None}))
    out = capsys.readouterr().out
    assert rc == 0
    assert "snfs: no factordb suggestion" in out
    assert "recommendation: GNFS (no SNFS polynomial available)" in out
    # no suggestion: only the GNFS route's target, priced by digits
    assert f"ecm pretest target (plan=normal): " \
           f"t{4 * len(str(BENCH_N)) / 13:.1f} (c{len(str(BENCH_N))})" in out


def test_bench_nfs_prime_record(tmp_path, capsys):
    ctx = RunContext(device="0", devices=("0",), log_file=tmp_path / "b.log",
                     quiet=True)
    args = _bench_args(tmp_path)
    rc = bench_nfs(args, ctx,
                   BenchClient({"fid": "1", "status": "P",
                                "decimal": "7919", "snfs": None}))
    out = capsys.readouterr().out
    assert rc == 1
    assert "already prime" in out


def test_bench_nfs_bad_input(tmp_path):
    ctx = RunContext(device="0", devices=("0",), log_file=tmp_path / "b.log",
                     quiet=True)
    args = build_parser().parse_args(["--bench-nfs", "id:12x3"])
    assert bench_nfs(args, ctx, BenchClient(None)) == 2
    args = build_parser().parse_args(["--bench-nfs", "xyz"])
    assert bench_nfs(args, ctx, BenchClient(None)) == 2
