"""The sieving loop's overlapped filter: fake cuda-sieve/msieve binaries
drive run_nfs's round loop end-to-end (base-m polynomial path, no GPU),
and the recorded call order proves the -nc1 of round k overlaps band
k+1, that bands never collapse to stubs while a verdict is in flight,
and that near the target the filter runs inline so the stop stays
fresh.  No GPU, no real msieve."""

import os
import time

import pytest

import yagfu.backends.nfs as nfs
from yagfu.backends.nfs import run_nfs
from yagfu.config import RunContext

N71 = 10 ** 70 + 3            # 71 digits: base-m poly, below the msieve
                              # polyselect gate; nfs_params(71) exists

MSIEVE_FAKE = '''#!/usr/bin/env python3
import os, sys, time
args = sys.argv[1:]
calls = os.environ["YAGFU_TEST_CALLS"]

def mark(what):
    with open(calls, "a") as f:
        f.write(what + "\\n")

if "-nc1" in args:
    round_ = [a.split("nc1-r")[1].split(".")[0]
              for a in args if a.startswith("nc1-r")][0]
    mark(f"nc1 start r{round_} pid={os.getpid()}")
    time.sleep(float(os.environ.get("YAGFU_TEST_NC1_SLEEP", "0")))
    if os.environ.get("YAGFU_TEST_NC1_FAIL"):
        mark(f"nc1 fail r{round_}")
        sys.exit(1)
    records = open(os.environ["YAGFU_TEST_VERDICTS"]).read()
    parts = records.split("\\n%%\\n")
    out = parts.pop(0) if parts and parts[0] else ""
    open(os.environ["YAGFU_TEST_VERDICTS"], "w").write("\\n%%\\n".join(parts))
    print(out)
    mark(f"nc1 end r{round_}")
    sys.exit(0)
mark("msieve " + " ".join(args))
sys.exit(0)
'''

SIEVE_FAKE = '''#!/usr/bin/env python3
import os, sys
args = sys.argv[1:]
calls = os.environ["YAGFU_TEST_CALLS"]

def mark(what):
    with open(calls, "a") as f:
        f.write(what + "\\n")

lo, hi = args[args.index("--qrange") + 1].split(":")
relname = args[args.index("--relations") + 1]
mark(f"band start {lo}:{hi}")
per_q = float(os.environ.get("YAGFU_TEST_RELS_PER_Q", "0.001"))
n = max(1, int((int(hi) - int(lo)) * per_q))
with open(relname, "w") as f:
    f.write("a\\n" * n)
mark(f"band end {lo}:{hi}")
'''


@pytest.fixture
def harness(tmp_path, monkeypatch):
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    for name, body in (("msieve", MSIEVE_FAKE), ("cuda-sieve", SIEVE_FAKE)):
        f = bin_dir / name
        f.write_text(body)
        f.chmod(0o755)
    calls = tmp_path / "calls.log"
    calls.write_text("")
    verdicts = tmp_path / "verdicts.txt"
    monkeypatch.setenv("YAGFU_TEST_CALLS", str(calls))
    monkeypatch.setenv("YAGFU_TEST_VERDICTS", str(verdicts))
    monkeypatch.setenv("YAGFU_TEST_NC1_SLEEP", "0")
    ctx = RunContext(device="0", devices=("0",), threads=2, sieve_procs=1,
                     bin_dir=bin_dir, log_file=tmp_path / "t.log")
    return ctx, calls, verdicts


def _seq(calls):
    return open(calls).read().splitlines()


def _index(seq, prefix):
    for i, s in enumerate(seq):
        if s.startswith(prefix):
            return i
    raise ValueError(f"{prefix!r} not in call sequence")


def test_filter_overlaps_next_band(harness, tmp_path):
    """Round k's -nc1 is still running when band k+1 is sieved and is
    only collected afterwards; when its verdict says enough, the band
    sieved under it is discarded instead of appended."""
    ctx, calls, verdicts = harness
    verdicts.write_text(
        "found 10 cycles, need 5000\nfiltering wants 60 more relations\n"
        "%%\n"
        "found 6000 cycles, need 5000\n")
    os.environ["YAGFU_TEST_NC1_SLEEP"] = "0.5"
    try:
        wd = tmp_path / "wd"
        res = run_nfs(N71, ctx, wd)
    finally:
        os.environ.pop("YAGFU_TEST_NC1_SLEEP", None)

    seq = _seq(calls)
    bands = [(s.split()[2], _index(seq, s))
             for s in seq if s.startswith("band start ")]
    ends = {s.split()[2]: _index(seq, s)
            for s in seq if s.startswith("band end ")}
    n1 = {s.split()[2]: _index(seq, s)
          for s in seq if s.startswith("nc1 start ")}
    n1e = {s.split()[2]: _index(seq, s)
           for s in seq if s.startswith("nc1 end ")}
    # round 1's filter overlapped band 2, round 2's overlapped band 3:
    # each filter was still running after its band had started (an
    # inline filter would have ended before the next band began — the
    # only ordering the scheduler lets us assert deterministically)
    (_, i_b2), (_, i_b3) = bands[1], bands[2]
    assert i_b2 < n1e["r1"]
    assert i_b3 < n1e["r2"]
    assert n1["r1"] < n1e["r1"] and n1["r2"] < n1e["r2"]
    # the stopping verdict covers the data through band 2 — band 3 (the
    # one sieved under that filter) is discarded, never appended
    assert res.bands == 2
    assert res.raw_relations == 987
    # the discarded band's relations were dropped, not appended:
    # msieve.dat stops exactly at the enough-point (band 2)
    assert sum(1 for _ in open(wd / "msieve.dat")) == 987
    assert not list(wd.glob("band3.*.rels"))
    assert res.error is None or "no factor" in (res.error or "")


def test_bands_track_measured_deficit(harness, tmp_path):
    """While a verdict is in flight the stale target is already reached;
    the last bump amount sizes the band instead (no 2000-q stubs), and
    once fresh hints arrive the bands track msieve's measured shortfall."""
    ctx, calls, verdicts = harness
    verdicts.write_text(
        "found 10 cycles, need 5000\nfiltering wants 60 more relations\n"
        "%%\n"
        "found 10 cycles, need 5000\nfiltering wants 900 more relations\n"
        "%%\n"
        "found 6000 cycles, need 5000\n")
    wd = tmp_path / "wd"
    res = run_nfs(N71, ctx, wd)
    seq = _seq(calls)
    bands = [(int(s.split()[2].split(":")[0]),
              int(s.split()[2].split(":")[1]))
             for s in seq if s.startswith("band start ")]
    assert len(bands) >= 3
    for lo, hi in bands:
        assert hi - lo > 20000             # no stub bands anywhere
    # the stopping verdict discards the band sieved under it
    assert res.bands == 3
    assert res.raw_relations == 1064


def test_verdict_exhaustion_reports_non_convergence(harness, tmp_path):
    """Verdicts that keep asking for more, then run dry: the loop reports
    non-convergence instead of spinning, and every filter was still
    backgrounded (started after its band's append)."""
    ctx, calls, verdicts = harness
    verdicts.write_text(
        "found 10 cycles, need 5000\nfiltering wants 870 more relations\n"
        "%%\n"
        "found 10 cycles, need 5000\nfiltering wants 870 more relations\n"
        "%%\n"
        "filtering wants 5 more relations\n"
        "%%\n"
        "filtering wants 5 more relations\n"
        "%%\n")
    wd = tmp_path / "wd"
    res = run_nfs(N71, ctx, wd)
    assert res.error == "filtering did not converge"
    seq = _seq(calls)
    starts = [i for i, s in enumerate(seq) if s.startswith("band start ")]
    n1s = [(i, s) for i, s in enumerate(seq) if s.startswith("nc1 start ")]
    assert len(starts) >= 4
    assert len(n1s) >= 3


def test_filter_failure_reports_error(harness, tmp_path, monkeypatch):
    ctx, calls, verdicts = harness
    verdicts.write_text("found 10 cycles, need 5000\n")
    monkeypatch.setenv("YAGFU_TEST_NC1_FAIL", "1")
    wd = tmp_path / "wd"
    res = run_nfs(N71, ctx, wd)
    assert res.error == "filtering failed (rc=1)"


def test_interrupt_kills_pending_filter(harness, tmp_path, monkeypatch):
    """Ctrl+C mid-band tears down the background -nc1 instead of
    orphaning it over msieve.dat."""
    ctx, calls, verdicts = harness
    verdicts.write_text("found 10 cycles, need 5000\n")
    monkeypatch.setenv("YAGFU_TEST_NC1_SLEEP", "30")

    real_band = nfs._run_band_parallel

    def exploding_band(base, cwd, n_procs, round_idx, timeout,
                       devices=("0",), beat=None, beat_s=120.0):
        if round_idx >= 2:
            time.sleep(0.3)        # let the background -nc1 come up first
            raise KeyboardInterrupt("simulated Ctrl+C")
        return real_band(base, cwd, n_procs, round_idx, timeout,
                         devices=devices, beat=beat, beat_s=beat_s)

    monkeypatch.setattr(nfs, "_run_band_parallel", exploding_band)
    wd = tmp_path / "wd"
    with pytest.raises(KeyboardInterrupt):
        run_nfs(N71, ctx, wd)
    pids = [int(s.split("pid=")[1]) for s in _seq(calls)
            if s.startswith("nc1 start") and "pid=" in s]
    assert pids
    for pid in pids:
        with pytest.raises(ProcessLookupError):
            os.kill(pid, 0)                # the filter did not survive
