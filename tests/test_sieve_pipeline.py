"""The sieving loop's overlapped filter: fake cuda-sieve/msieve binaries
drive run_nfs's round loop end-to-end (base-m polynomial path, no GPU),
and the recorded call order proves the probe round's -nc1 is collected
inline (its hint sizes band 2), that round k>=2's -nc1 overlaps band
k+1, and that bands never collapse to stubs while a verdict is in
flight.  No GPU, no real msieve."""

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
    sleep_s = float(os.environ.get("YAGFU_TEST_NC1_SLEEP", "0"))
    if round_ == "1":
        sleep_s = 0.0    # the probe filter is collected inline; the sleep
                         # exists to keep a later round's filter in flight
    time.sleep(sleep_s)
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
    """The probe round's -nc1 is collected inline before band 2 is sized
    (its hint replaces the generous initial target); round k>=2's -nc1
    still runs while band k+1 is sieved and is only collected afterwards;
    when its verdict says enough, the band sieved under it is discarded
    instead of appended."""
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
    n1 = {s.split()[2]: _index(seq, s)
          for s in seq if s.startswith("nc1 start ")}
    n1e = {s.split()[2]: _index(seq, s)
           for s in seq if s.startswith("nc1 end ")}
    (_, i_b2), (_, i_b3) = bands[1], bands[2]
    # the probe verdict was in hand before band 2 was sized, and band 2
    # carries the hint's size, not the (much larger) initial target
    assert n1e["r1"] < i_b2
    lo, hi = (int(x) for x in bands[1][0].split(":"))
    assert hi - lo < 100_000
    # rounds past the probe overlap as before: round 2's filter was still
    # running after band 3 had started (the only ordering the scheduler
    # lets us assert deterministically)
    assert i_b3 < n1e["r2"]
    assert n1["r1"] < n1e["r1"] and n1["r2"] < n1e["r2"]
    # the stopping verdict covers the data through band 2 — band 3 (the
    # one sieved under that filter) is discarded, never appended
    assert res.bands == 2
    assert res.raw_relations == 107
    # the discarded band's relations were dropped, not appended:
    # msieve.dat stops exactly at the enough-point (band 2)
    assert sum(1 for _ in open(wd / "msieve.dat")) == 107
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
    assert len(bands) >= 4
    for lo, hi in bands:
        assert hi - lo > 20000             # no stub bands anywhere
    # the stopping verdict discards the band sieved under it (band 4);
    # bands 1-3 are the appended ones
    assert res.bands == 3
    assert res.raw_relations == 185


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
    orphaning it over msieve.dat.  The probe's filter is collected
    inline, so the interrupt lands on round 3 with round 2's filter
    (a 30s sleeper) still in flight."""
    ctx, calls, verdicts = harness
    verdicts.write_text("filtering wants 500 more relations\n")
    monkeypatch.setenv("YAGFU_TEST_NC1_SLEEP", "30")

    real_band = nfs._run_band_parallel

    def exploding_band(base, cwd, n_procs, round_idx, timeout,
                       devices=("0",), beat=None, beat_s=120.0):
        if round_idx >= 3:
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
