import pytest

from yagfu.cli import _db_factor_list, build_parser, factor_with_db, load_numbers
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
    """A factordb record carrying an SNFS suggestion (probe shape)."""
    return {"fid": "fid1", "status": status,
            "snfs": {"n": str(n), "status": status,
                     "polys": [{"c": ["1", "0", "0", "1"],
                                "difficulty": scaled, "scaled": scaled,
                                "side": "a", "skew": 1.0,
                                "y0": "-7", "y1": "1",
                                "form": "x^3+1", "e": 1e-4}]}}


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
