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


# ---------------------------------------------------------------- factor-db

class FakeFdb:
    """get_number/get_factor_decimals canned; records reports."""

    def __init__(self, status, factors):
        self.status = status
        self.factors = factors
        self.reported = None

    def get_number(self, n):
        return {"fid": "fid1", "status": self.status}

    def get_factor_decimals(self, fid):
        return [str(f) for f in self.factors]

    def report_factors(self, fid, n, factors):
        self.reported = list(factors)
        return {}

    def factor_fid_map(self, fid):
        return {}

    def report_ecm(self, *a, **k):
        return {}


class FakeEngine:
    """Records the number it was asked to factor; finds canned primes."""

    def __init__(self, new_factors):
        self.new_factors = new_factors
        self.asked = []

    def factor(self, n, wd):
        self.asked.append(n)
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
    assert eng.asked == [c]                    # engine got the composite part
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
