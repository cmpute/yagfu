import json

import pytest

import pytest

from yagfu.backends import nfs
from yagfu.cli import (_db_factor_list, _make_ctx, bench_nfs, build_parser,
                       emit, factor_with_db, load_numbers)
from yagfu.config import RunContext
from yagfu.engine import FactorResult, FoundFactor


def test_ecm_only_flag():
    ap = build_parser()
    args = ap.parse_args(["--ecm-only", "12345"])
    assert args.ecm_only is True
    args = ap.parse_args(["12345"])
    assert args.ecm_only is False


def _make_ctx_without_gpu(monkeypatch, argv):
    """_make_ctx as the CLI runs it, minus the GPU lookup."""
    monkeypatch.setattr("yagfu.cli.resolve_devices", lambda specs: ("0",))
    monkeypatch.setattr("yagfu.cli._verify_devices", lambda devices: None)
    return _make_ctx(build_parser().parse_args(argv))


def test_ecm_t_level_combines_with_ecm_only(monkeypatch):
    ctx = _make_ctx_without_gpu(monkeypatch,
                                ["--ecm-only", "--ecm-t-level", "25-30", "42"])
    assert ctx.ecm_only is True
    assert ctx.ecm_t_level == (25.0, 30.0)


def test_ecm_t_level_bad_ranges_are_usage_errors(monkeypatch):
    # these checks named `ap`, a local of main(), and raised NameError
    for argv, message in [(["--no-ecm", "--ecm-t-level", "25-30", "42"],
                           "does not combine with --no-ecm"),
                          (["--ecm-t-level", "abc", "42"], "wants LO-HI"),
                          (["--ecm-t-level", "40-70", "42"], "tops out at t65"),
                          (["--ecm-t-level", "44-38", "42"],
                           "LO must not exceed HI")]:
        with pytest.raises(SystemExit) as e:
            _make_ctx_without_gpu(monkeypatch, argv)
        assert message in str(e.value), argv


def test_emit_json_carries_ecm_ledger(capsys):
    res = FactorResult(n=42, ecm_ledger=[(42, 17.5), (21, 22.0)])
    emit(42, res, None, None, 1.0, True, None, False)
    obj = json.loads(capsys.readouterr().out)
    assert obj["ecm_ledger"] == [{"n": "42", "t_level": 17.5},
                                 {"n": "21", "t_level": 22.0}]


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


MSIEVE_SLOW = (
    '#!/bin/sh\n'
    # the first polynomial only after 0.5 s; stopped before, saves nothing
    "trap '[ -f poly.found ] || exit 0\n"
    '      printf "combined = 2.5e-12\\n"\n'
    '      printf "N 123\\nSKEW 3.0\\nR0 -100000\\nR1 1\\nA0 1\\nA4 5\\n" '
    '> msieve.fb\n'
    "      exit 0' INT\n"
    'echo "expecting poly E from 9.0e-11 to > 1.2e-10"\n'
    '( sleep 0.5\n'
    '  echo "polyselect best E = 2.0e-12"\n'
    '  : > poly.found ) &\n'
    'sleep 60 &\n'
    'wait\n')


def _bench_until_poly(tmp_path, capsys, msieve_body, until_poly):
    bindir = tmp_path / "bins"
    bindir.mkdir(parents=True)
    p = bindir / "msieve"
    p.write_text(msieve_body)
    p.chmod(0o755)
    ctx = RunContext(device="0", devices=("0",), threads=2,
                     bin_dir=str(bindir), log_file=tmp_path / "b.log",
                     quiet=True)
    argv = ["--bench-nfs", str(BENCH_N), "--bench-nfs-time", "0.3"]
    if until_poly:
        argv.append("--bench-nfs-until-poly")
    args = build_parser().parse_args(argv + ["--workdir", str(tmp_path / "wd")])
    rc = bench_nfs(args, ctx, BenchClient({"fid": None, "status": "C",
                                           "decimal": str(BENCH_N),
                                           "snfs": None}))
    return rc, capsys.readouterr().out


def test_bench_nfs_until_poly_reports_overrun(tmp_path, capsys):
    """--bench-nfs-until-poly: a GNFS search with no polynomial when the
    limit is up runs on until the first one appears; the report says by how
    much the limit was overrun, and prices the undershoot note with the
    time actually spent.  Without the flag the same search is cut at the
    limit and reports no polynomial."""
    rc, out = _bench_until_poly(tmp_path, capsys, MSIEVE_SLOW, until_poly=False)
    assert rc == 1                              # nothing on either side
    assert "gnfs: msieve -np polyselect, 0.3s limit: no polynomial in the limit" in out
    assert "--bench-nfs-until-poly:" not in out

    rc, out = _bench_until_poly(tmp_path / "again", capsys, MSIEVE_SLOW,
                                until_poly=True)
    assert rc == 0
    assert "--bench-nfs-until-poly: no polynomial within the 0.3s limit, " \
           "ran on for" in out and "s more until one appeared" in out
    assert "best GNFS Murphy E: 6.400e-12 (normalized, degree 4, " \
           "msieve -np)" in out
    assert "GNFS polyselect undershoots a full search" in out
    assert "a 0.3s GNFS polyselect" not in out      # priced by the real time


def test_bench_nfs_until_poly_search_ends_without_one(tmp_path, capsys):
    """A search msieve ends by itself without a polynomial: the flag cannot
    wait past that, and the report says so."""
    rc, out = _bench_until_poly(
        tmp_path, capsys,
        '#!/bin/sh\n'
        'echo "expecting poly E from 9.0e-11 to > 1.2e-10"\n'
        'sleep 0.5\n'
        'exit 0\n',
        until_poly=True)
    assert rc == 1                                  # no E on either side
    assert "--bench-nfs-until-poly: no polynomial within the 0.3s limit, " \
           "ran on for" in out
    assert "s more, then the search ended without one" in out
    assert "no usable E on either side; cannot recommend" in out


def test_bench_nfs_gnfs_polyselect_runs_the_cpu_engine(tmp_path, capsys):
    """The bench GNFS polyselect spawns msieve with the CPU stage-1 engine
    (the fork's GPU one dies on inputs from ~252 digits up), and a search
    that ends without saving an msieve.fb still yields its E — the degree
    read back from the log it streamed."""
    bindir = tmp_path / "bins"
    bindir.mkdir()
    (bindir / "msieve").write_text(
        '#!/bin/sh\n'
        'echo "$@" > msieve.args\n'
        'echo "polynomial degree: 6"\n'
        'echo "polyselect best E = 2.0e-12"\n'
        'sleep 0.1\n'                       # ends on its own, saves no fb
        'exit 0\n')
    (bindir / "msieve").chmod(0o755)
    ctx = RunContext(device="0", devices=("0",), threads=2,
                     bin_dir=str(bindir), log_file=tmp_path / "b.log",
                     quiet=True)
    args = build_parser().parse_args(
        ["--bench-nfs", str(BENCH_N), "--bench-nfs-time", "1", "--keep",
         "--workdir", str(tmp_path / "wd")])
    rc = bench_nfs(args, ctx, BenchClient({"fid": None, "status": "C",
                                           "decimal": str(BENCH_N),
                                           "snfs": None}))
    out = capsys.readouterr().out
    assert rc == 0
    wd = tmp_path / "wd" / "bench-nfs" / "gnfs"
    spawned = (wd / "msieve.args").read_text()
    assert "stage1_engine=cpu_gerbicz" in spawned
    assert "gpu_cubsort" not in spawned
    # degree 6 from the log (no fb): normalization 1.6^0, the E unchanged
    assert "best GNFS Murphy E: 2.000e-12 (normalized, degree 6, " \
           "msieve -np)" in out
    assert "recommendation: GNFS (no SNFS polynomial available)" in out


def test_bench_nfs_gnfs_death_gets_a_diagnosis(tmp_path, capsys):
    """msieve dying well short of the limit names its exit code and last
    log line, instead of just 'no polynomial'."""
    bindir = _stub_score_poly(tmp_path / "bins")
    p = bindir / "msieve"
    p.write_text(
        '#!/bin/sh\n'
        'echo "expecting poly E from 9.0e-11 to > 1.2e-10"\n'
        'echo "corrupted double-linked list"\n'
        'exit 1\n')
    p.chmod(0o755)
    ctx = RunContext(device="0", devices=("0",), threads=2,
                     bin_dir=str(bindir), log_file=tmp_path / "b.log",
                     quiet=True)
    args = build_parser().parse_args(
        ["--bench-nfs", str(BENCH_N), "--bench-nfs-time", "1",
         "--workdir", str(tmp_path / "wd")])
    rc = bench_nfs(args, ctx, BenchClient(BENCH_REC))
    assert rc == 0
    log = (tmp_path / "b.log").read_text()
    assert "msieve ended on its own, rc [1]" in log
    assert "last log line: corrupted double-linked list" in log
    assert "recommendation: SNFS (GNFS polyselect produced no polynomial)" \
        in capsys.readouterr().out


def test_bench_nfs_estimates_gnfs_past_msieve_bound(tmp_path, capsys):
    """A number past msieve's own stage-1 norm bound (309+ digits) skips
    the GNFS polyselect and compares routes against the digit-count
    estimate instead; msieve never runs."""
    bindir = tmp_path / "bins"
    bindir.mkdir()
    marker = tmp_path / "spawned"
    p = bindir / "msieve"
    p.write_text(f'#!/bin/sh\ntouch {marker}\n')
    p.chmod(0o755)
    ctx = RunContext(device="0", devices=("0",), threads=2,
                     bin_dir=str(bindir), log_file=tmp_path / "b.log",
                     quiet=True)
    n = 3 * 10 ** 308 + 9                       # 309 digits
    args = build_parser().parse_args(
        ["--bench-nfs", str(n), "--bench-nfs-time", "1",
         "--workdir", str(tmp_path / "wd")])
    rc = bench_nfs(args, ctx, BenchClient({"fid": None, "status": "C",
                                           "decimal": str(n), "snfs": None}))
    out = capsys.readouterr().out
    est = 10.0 ** (-0.058 * 309 - 2.65136)
    assert 2.5e-21 < est < 2.9e-21              # the site's 2.67e-21 for C309
    assert rc == 0
    assert not marker.exists()                  # msieve never ran
    assert "gnfs: skipped (msieve derives no stage-1 norm bound past " \
           "308 digits)" in out
    assert f"gnfs: estimate from the digit count: E {est:.3e} (degree 6)" \
        in out
    assert "best GNFS Murphy E: unavailable" in out
    assert "best GNFS Murphy E (estimate, from the digit count): " \
        f"{est:.3e} (degree 6)" in out
    # no SNFS side at all: the estimate alone still recommends
    assert "recommendation: GNFS (no SNFS polynomial available)" in out


def test_bench_nfs_route_compares_snfs_with_the_gnfs_estimate(
        tmp_path, capsys, monkeypatch):
    """Past the polyselect bound the route comparison runs on the
    estimate, and says so — the measured-vs-estimated asymmetry must not
    hide behind a bare ratio."""
    import yagfu.cli as cli_mod

    monkeypatch.setattr(cli_mod, "GNFS_POLYSELECT_MAX_DIGITS", 2)
    bindir = _stub_score_poly(tmp_path / "bins")
    ctx = RunContext(device="0", devices=("0",), threads=2,
                     bin_dir=str(bindir), log_file=tmp_path / "b.log",
                     quiet=True)
    args = build_parser().parse_args(
        ["--bench-nfs", str(BENCH_N), "--bench-nfs-time", "1",
         "--workdir", str(tmp_path / "wd")])
    rc = bench_nfs(args, ctx, BenchClient(BENCH_REC))
    out = capsys.readouterr().out
    assert rc == 0
    est = 10.0 ** (-0.058 * len(str(BENCH_N)) - 2.65136)
    assert "gnfs: skipped (msieve derives no stage-1 norm bound past " \
           "2 digits)" in out
    assert f"gnfs: estimate from the digit count: E {est:.3e} (degree 6)" \
        in out
    # at 31 digits the estimate dwarfs the stub's SNFS E, so GNFS wins —
    # the point is that the reason names the estimate
    assert "recommendation: GNFS (normalized Murphy E ratio " \
        in out and "(GNFS estimated from the digit count)" in out


def _stub_score_poly(bindir, murphy="3.0e-11"):
    """score-poly stub: the same E and skew (2) for every candidate."""
    bindir.mkdir(parents=True, exist_ok=True)
    p = bindir / "score-poly"
    p.write_text('#!/bin/sh\n'
                 'echo "skew 2.0, murphyE %s, size 1.0e-15, alpha 0.5, '
                 'rroots 2"\n' % murphy)
    p.chmod(0o755)
    return bindir


C331_REC = {"fid": "1100000000000000331", "status": "C", "decimal": "331",
            "term": "C(30,2)", "snfs": None}    # 331 = Phi_30(2) | 2^15 + 1


def test_bench_nfs_local_forms_without_a_suggestion(tmp_path, capsys):
    """No factordb suggestion, but the record's term names C(30,2), a
    factor of 2^15 + 1: the SNFS side is the forms enumerated locally for
    that power sum, scored and ranked.  With no GNFS side asked for there
    is no recommendation; the best poly is printed and written, and reads
    back as the same poly."""
    bindir = _stub_score_poly(tmp_path / "bins")
    ctx = RunContext(device="0", devices=("0",), threads=2,
                     bin_dir=str(bindir), log_file=tmp_path / "b.log",
                     quiet=True)
    out_poly = tmp_path / "out" / "c331"
    args = build_parser().parse_args(
        ["--bench-nfs", "331", "--bench-nfs-time", "1", "--bench-nfs-no-gnfs",
         "--bench-nfs-poly", str(out_poly), "--workdir", str(tmp_path / "wd")])
    rc = bench_nfs(args, ctx, BenchClient(dict(C331_REC)))
    out = capsys.readouterr().out
    assert rc == 0
    assert "polynomials enumerated locally from 2^15+1" in out
    assert "degree-4 candidate: normalized Murphy E " in out and "(local)" in out
    assert "gnfs: skipped (--bench-nfs-no-gnfs)" in out
    assert "recommendation:" not in out
    assert "ecm pretest target (plan=normal): SNFS route t" in out
    # every candidate scores the same raw E, so the lowest degree wins on
    # the normalization and the first of them, the cyclotomic form, stays
    assert "best SNFS polynomial: degree 4, difficulty " in out
    assert ", from 2^15+1" in out
    for line in ("n: 331", "skew: 2", "c4: 1", "c3: 1", "c2: -4", "c1: -4",
                 "c0: 1", "Y1: -2", "Y0: 5", "type: snfs"):
        assert f"    {line}\n" in out, line
    job = nfs.snfs_job_from_file(
        (out_poly.parent / "c331.snfs.job").read_text())
    assert (job.n, job.coeffs, job.y0, job.y1) == (331, [1, -4, -4, 1, 1], 5, -2)
    assert job.difficulty > 0 and job.scaled >= job.difficulty
    assert job.side in ("a", "r") and job.form == "2^15+1"
    assert not (out_poly.parent / "c331.gnfs.job").exists()


def test_bench_nfs_form_option(tmp_path, capsys):
    """--bench-nfs-form names the power form itself (a record without a
    term), and has to be one the number divides."""
    bindir = _stub_score_poly(tmp_path / "bins")
    ctx = RunContext(device="0", devices=("0",), threads=2,
                     bin_dir=str(bindir), log_file=tmp_path / "b.log",
                     quiet=True)
    args = build_parser().parse_args(
        ["--bench-nfs", "331", "--bench-nfs-form", "3^7-1",
         "--workdir", str(tmp_path / "wd")])
    assert bench_nfs(args, ctx, BenchClient(dict(C331_REC))) == 2
    assert "does not divide 3^7-1" in capsys.readouterr().err
    args = build_parser().parse_args(
        ["--bench-nfs", "331", "--bench-nfs-form", "2^15+1",
         "--bench-nfs-no-gnfs", "--workdir", str(tmp_path / "wd")])
    assert bench_nfs(args, ctx, BenchClient(
        {"fid": None, "status": "C", "decimal": "331", "snfs": None})) == 0
    assert "polynomials enumerated locally from 2^15+1" in capsys.readouterr().out


def test_poly_option_seeds_the_nfs_workdir(tmp_path, monkeypatch):
    """--poly with a GNFS poly: main drops it in as the nfs workdir's
    msieve.fb, which run_nfs reuses, and factors its number; with an SNFS
    poly the job goes to the engine and nothing is seeded.  --poly is not
    for another number, nor together with -db."""
    import yagfu.cli as cli
    calls = []

    class FakeEngine:
        def __init__(self, ctx):
            pass

        def factor(self, n, wd, snfs=None):
            fb = wd / "nfs" / "msieve.fb"
            calls.append((n, snfs, fb.read_text() if fb.exists() else None))
            from yagfu.engine import FactorResult
            return FactorResult(n=n)

    quiet = lambda _args: RunContext(device="0", devices=("0",), quiet=True)  # noqa: E731
    monkeypatch.setattr(cli, "_make_ctx", quiet)
    monkeypatch.setattr(cli, "Engine", FakeEngine)
    fb = {"N": "123", "SKEW": "3.0", "R0": "-100000", "R1": "1", "A0": "1",
          "A4": "5"}
    gnfs = tmp_path / "p.gnfs.job"
    gnfs.write_text(nfs.gnfs_poly_text(123, fb))
    cli.main(["123", "--poly", str(gnfs), "--workdir", str(tmp_path / "wd"), "-q"])
    assert len(calls) == 1 and calls[0][0] == 123 and calls[0][1] is None
    assert calls[0][2] == nfs.gnfs_fb_text(fb)

    snfs = tmp_path / "p.snfs.job"
    snfs.write_text(nfs.snfs_poly_text(nfs.SnfsJob(
        n=331, coeffs=[1, -4, -4, 1, 1], y0=5, y1=-2, skew=2.0,
        difficulty=0.0, scaled=0.0, form="2^15+1")))
    calls.clear()
    cli.main(["331", "--poly", str(snfs), "--workdir", str(tmp_path / "wd2"), "-q"])
    assert len(calls) == 1 and calls[0][0] == 331
    assert calls[0][1].coeffs == [1, -4, -4, 1, 1] and calls[0][2] is None

    calls.clear()
    with pytest.raises(SystemExit):
        cli.main(["124", "--poly", str(gnfs), "--workdir", str(tmp_path / "wd3")])
    with pytest.raises(SystemExit):
        cli.main(["123", "--poly", str(gnfs), "-db", "--workdir", str(tmp_path / "wd4")])
    assert calls == []


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
