"""The reactive sieving pipeline: fake cuda-sieve/msieve binaries drive
run_nfs's wave loop end-to-end (base-m polynomial path, no GPU).  The
recorded call order proves the append gate (msieve.dat only grows while
no -nc1 reads it), verdict reactivity (the next wave is aimed at the
refreshed rung, "enough" kills the wave in flight and drops the queue),
and that the rung estimator homes in on the measured excess curve.  No
GPU, no real msieve."""

import os
import time

import pytest

import yagfu.backends.nfs as nfs
from yagfu.backends.nfs import SnfsJob, _filter_rung, run_nfs
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
    dat = sum(1 for _ in open(os.path.join(os.getcwd(), "msieve.dat")))
    mark(f"nc1 start r{round_} dat={dat} pid={os.getpid()}")
    sleep_s = float(os.environ.get("YAGFU_TEST_NC1_SLEEP", "0"))
    if round_ == "1":
        sleep_s = 0.0    # the probe filter is instant; the sleep exists
                         # to keep a later round's filter in flight
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
import os, sys, time
args = sys.argv[1:]
calls = os.environ["YAGFU_TEST_CALLS"]

def mark(what):
    with open(calls, "a") as f:
        f.write(what + "\\n")

lo, hi = args[args.index("--qrange") + 1].split(":")
relname = args[args.index("--relations") + 1]
sq = " sq=0" if "--sq-side" in args else ""
mark(f"band start {lo}:{hi}{sq} pid={os.getpid()}")
time.sleep(float(os.environ.get("YAGFU_TEST_SIEVE_SLEEP", "0")))
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


def _bands(seq):
    """(lo, hi, call index) per sieve wave, in launch order."""
    return [(int(s.split()[2].split(":")[0]), int(s.split()[2].split(":")[1]),
             i) for i, s in enumerate(seq) if s.startswith("band start ")]


def test_verdict_reacts_midwave_and_stop_kills(harness, tmp_path):
    """A stop verdict that lands while a wave is still sieving kills the
    wave outright and drops the queue: the loop never starts another
    wave, and msieve.dat stops exactly at the enough-point."""
    ctx, calls, verdicts = harness
    # the probe's filter is instant; give the second filter a stop
    # verdict so it lands while wave 2 (1s sleeper) is still running
    verdicts.write_text(
        "found 10 cycles, need 5000\nfiltering wants 60 more relations\n"
        "%%\n"
        "found 6000 cycles, need 5000\n")
    os.environ["YAGFU_TEST_NC1_SLEEP"] = "0.2"
    os.environ["YAGFU_TEST_SIEVE_SLEEP"] = "1.0"
    try:
        wd = tmp_path / "wd"
        res = run_nfs(N71, ctx, wd)
    finally:
        os.environ.pop("YAGFU_TEST_NC1_SLEEP", None)
        os.environ.pop("YAGFU_TEST_SIEVE_SLEEP", None)

    seq = _seq(calls)
    bands = _bands(seq)
    # wave 1 (probe) + wave 2 (appended) + wave 3, which started before
    # the stop verdict landed and was killed mid-run
    assert len(bands) == 3, bands
    assert not any(s.startswith("band end 261000:") for s in seq)
    # only waves 1-2 were appended: the stop covers exactly their data
    assert res.bands == 2
    assert res.raw_relations == 63
    assert sum(1 for _ in open(wd / "msieve.dat")) == 63
    # the killed wave's partial output and the queue are gone
    assert not list(wd.glob("w*.rels"))
    # the wave processes did not survive the kill
    pids = [int(s.split("pid=")[1]) for s in seq
            if s.startswith("band start") and "pid=" in s]
    for pid in pids:
        with pytest.raises(ProcessLookupError):
            os.kill(pid, 0)
    assert res.error is None or "no factor" in (res.error or "")


def test_waves_track_refreshed_rung(harness, tmp_path):
    """The full measurement loop: two filters measure the excess curve,
    the rung switches from the ladder to the extrapolated estimate, and
    the stop lands one rung earlier than the ladder would have sieved."""
    ctx, calls, verdicts = harness
    ctx.wave_max = 100          # small waves: deterministic, no speculation
    verdicts.write_text(
        # r1 (30 raw): 28 unique, excess -9 / 5
        "found 2 duplicates and 28 unique relations\n"
        "keeping 37 ideals with weight <= 200, target excess is 5\n"
        "filtering wants 1000000 more relations\n"
        "%%\n"
        # r2 (63 raw): 58 unique, excess 0 / 5: slope 0.3 turned the
        # curve; extrapolation lands at 93, under the ladder's 126
        "found 5 duplicates and 58 unique relations\n"
        "keeping 58 ideals with weight <= 200, target excess is 5\n"
        "filtering wants 1000000 more relations\n"
        "%%\n"
        # r3: enough
        "found 90 cycles, need 5\n")
    # r2 sleeps long enough that wave 3 (speculative) is guaranteed in
    # flight when its verdict lands, and gets killed by it
    os.environ["YAGFU_TEST_NC1_SLEEP"] = "0.55"
    os.environ["YAGFU_TEST_SIEVE_SLEEP"] = "0.3"
    try:
        wd = tmp_path / "wd"
        res = run_nfs(N71, ctx, wd)
    finally:
        os.environ.pop("YAGFU_TEST_NC1_SLEEP", None)
        os.environ.pop("YAGFU_TEST_SIEVE_SLEEP", None)
    seq = _seq(calls)
    bands = _bands(seq)
    for lo, hi, _ in bands:
        assert hi - lo > 20_000            # no stub waves anywhere
    # the estimate rung (93) is live and under the ladder's 126
    rung = [s for s in open(tmp_path / "t.log") if "(estimate)" in s]
    assert any("next filter at 93" in s for s in rung)
    # r3 fired the moment the drain crossed the estimate rung (appended
    # 132), and its stop verdict killed the speculative wave in flight:
    # total sieving stays at 132 where the pure ladder would have run to
    # 2 x 63 = 126 -> 252 before the next verdict
    assert any(s.startswith("nc1 start r3 dat=132") for s in seq)
    assert res.bands == 3
    assert res.raw_relations == 132
    assert sum(1 for _ in open(wd / "msieve.dat")) == 132
    assert not list(wd.glob("w*.rels"))
    assert "queue dropped" in (tmp_path / "t.log").read_text()
    assert res.error is None or "no factor" in (res.error or "")


def test_append_gate_holds_while_filter_reads(harness, tmp_path):
    """msieve.dat never grows while a filter is running: waves completed
    in that window queue up and are appended wholesale when it exits
    (the r2 nc1 covers everything the queue had accumulated)."""
    ctx, calls, verdicts = harness
    ctx.wave_max = 100
    verdicts.write_text(
        "found 2 duplicates and 28 unique relations\n"
        "keeping 50 ideals with weight <= 200, target excess is 5\n"
        "filtering wants 1000000 more relations\n"
        "%%\n"
        "found 6000 cycles, need 5\n")
    os.environ["YAGFU_TEST_NC1_SLEEP"] = "0.5"
    os.environ["YAGFU_TEST_SIEVE_SLEEP"] = "0.05"
    try:
        wd = tmp_path / "wd"
        res = run_nfs(N71, ctx, wd)
    finally:
        os.environ.pop("YAGFU_TEST_NC1_SLEEP", None)
        os.environ.pop("YAGFU_TEST_SIEVE_SLEEP", None)

    seq = _seq(calls)
    n1s = [s for s in seq if s.startswith("nc1 start ")]
    assert len(n1s) == 2
    # r1 launched right after the probe append (dat = 30)
    assert "dat=30" in n1s[0]
    # r2 launched only after r1 exited, with every wave queued in the
    # meantime appended first: its dat exceeds r1's snapshot
    dat2 = int(n1s[1].split("dat=")[1].split()[0])
    assert dat2 >= 60
    assert res.raw_relations == dat2
    assert res.error is None or "no factor" in (res.error or "")


def test_old_style_verdicts_still_converge(harness, tmp_path):
    """Logs with only the legacy verdict lines (no unique/keeping
    measurements) degrade to the pure ladder and still reach the stop."""
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
    bands = _bands(seq)
    assert len(bands) >= 4
    for lo, hi, _ in bands:
        assert hi - lo > 20_000             # no stub waves anywhere
    assert res.bands >= 3
    assert res.raw_relations == sum(1 for _ in open(wd / "msieve.dat"))
    assert res.error is None or "no factor" in (res.error or "")


def test_verdict_exhaustion_reports_non_convergence(harness, tmp_path):
    """Verdicts that keep asking for more, then run dry: the loop reports
    non-convergence instead of spinning, and every filter was still
    backgrounded (started after its data's append)."""
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
    starts = [s for s in seq if s.startswith("band start ")]
    n1s = [s for s in seq if s.startswith("nc1 start ")]
    assert len(starts) >= 4
    assert len(n1s) >= 4


def test_filter_failure_reports_error(harness, tmp_path, monkeypatch):
    ctx, calls, verdicts = harness
    verdicts.write_text("found 10 cycles, need 5000\n")
    monkeypatch.setenv("YAGFU_TEST_NC1_FAIL", "1")
    wd = tmp_path / "wd"
    res = run_nfs(N71, ctx, wd)
    assert res.error == "filtering failed (rc=1)"


def test_interrupt_kills_wave_and_filter(harness, tmp_path, monkeypatch):
    """Ctrl+C mid-wave tears down the background -nc1 and the wave's
    sieve chunks instead of orphaning them over msieve.dat."""
    ctx, calls, verdicts = harness
    verdicts.write_text("filtering wants 500 more relations\n")
    monkeypatch.setenv("YAGFU_TEST_NC1_SLEEP", "30")

    real_band = nfs._run_band_parallel

    def exploding_band(base, cwd, n_procs, tag, timeout,
                       devices=("0",), beat=None, beat_s=120.0,
                       stop=None, poll_s=2.0):
        if int(tag[1:]) >= 2:
            time.sleep(0.3)        # let the background -nc1 come up first
            raise KeyboardInterrupt("simulated Ctrl+C")
        return real_band(base, cwd, n_procs, tag, timeout, devices=devices,
                         beat=beat, beat_s=beat_s, stop=stop, poll_s=poll_s)

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


def test_resume_reuses_appended_data(harness, tmp_path, monkeypatch):
    """A failed run's msieve.dat, filter points and q cursor carry over:
    the resumed run does not re-sieve the probe and its first -nc1
    covers exactly the data the first run appended plus the queued
    waves it left on disk."""
    ctx, calls, verdicts = harness
    ctx.wave_max = 100
    verdicts.write_text(
        "found 2 duplicates and 28 unique relations\n"
        "keeping 50 ideals with weight <= 200, target excess is 5\n"
        "filtering wants 1000000 more relations\n")
    monkeypatch.setenv("YAGFU_TEST_NC1_FAIL", "1")
    wd = tmp_path / "wd"
    res1 = run_nfs(N71, ctx, wd)
    assert res1.error == "filtering failed (rc=1)"
    dat1 = sum(1 for _ in open(wd / "msieve.dat"))
    queued = sum(sum(1 for _ in open(wd / f)) for f in wd.glob("w*.rels"))
    q_next = __import__("json").loads((wd / "nfs-state.json").read_text())["q_next"]
    monkeypatch.delenv("YAGFU_TEST_NC1_FAIL")
    calls2 = tmp_path / "calls2.log"
    calls2.write_text("")
    monkeypatch.setenv("YAGFU_TEST_CALLS", str(calls2))
    verdicts.write_text("found 6000 cycles, need 5\n")
    res2 = run_nfs(N71, ctx, wd)
    seq = _seq(calls2)
    n1 = [s for s in seq if s.startswith("nc1 start ")]
    assert n1 and f"dat={dat1 + queued}" in n1[0]   # nothing lost, redone
    first_band = _bands(seq)[0]
    assert first_band[0] >= q_next                  # the q cursor carried over
    assert res2.raw_relations >= dat1


def test_filter_rung_estimator():
    """The rung: ladder below two measured points and on non-positive
    slopes; the secant extrapolation caps the ladder once the excess
    curve turns over; a near-flat positive slope extrapolates past the
    ladder and is capped by it."""
    assert _filter_rung([], 1.0, 1.15) == (None, "ladder")
    assert _filter_rung([(1, 100, 90, -10, 5)], 1.0, 1.15) == (200, "ladder")
    turning = [(1, 100, 90, -10, 5), (2, 200, 180, -25, 5)]
    assert _filter_rung(turning, 1.0, 1.15) == (400, "ladder")   # slope < 0
    # slope 0.125, u_need = 380 + 5/0.125 = 420, ratio 0.95 -> 508 < ladder
    turned = turning + [(3, 400, 380, 0, 5)]
    assert _filter_rung(turned, 1.0, 1.15) == (508, "estimate")
    # slope 0.002 -> extrapolates far past the ladder; capped by it
    flat = [(1, 100, 90, -10, 5), (2, 200, 180, -9.8, 5)]
    assert _filter_rung(flat, 1.0, 1.15) == (400, "estimate")
    # unparseable measurements (older logs) keep the ladder
    blind = [(1, 100, None, None, None), (2, 200, 180, -25, 5)]
    assert _filter_rung(blind, 1.0, 1.15) == (400, "ladder")
    # a tiny topup still leaves a minimum growth step (no filter spam)
    assert _filter_rung([(1, 100, 90, -10, 5)], 0.0, 1.15) == (102, "ladder")


# ------------------------------------------------------------------- snfs

def _snfs_job(n, side="a"):
    return SnfsJob(n=n, coeffs=[1, 0, 0, 0, 0, 0, 3], y0=-100, y1=1,
                   skew=1.0, difficulty=70.1, scaled=70.6, side=side,
                   form="x^6+3", e=1e-4)


def test_snfs_run_skips_polyselect_and_writes_job(harness, tmp_path):
    """With a supplied polynomial the job file is written directly (no
    msieve -np anywhere), the params come from the SNFS table, and the
    algebraic-side default leaves --sq-side off the sieve command."""
    ctx, calls, verdicts = harness
    verdicts.write_text("found 6000 cycles, need 5\n")
    wd = tmp_path / "wd"
    res = run_nfs(N71, ctx, wd, snfs=_snfs_job(N71))
    seq = _seq(calls)
    msieve_calls = [s for s in seq if s.startswith("msieve ")]
    assert msieve_calls and all("-np" not in s for s in msieve_calls)
    txt = (wd / f"c{len(str(N71))}.job").read_text()
    assert f"n: {N71}" in txt and "type: snfs" in txt
    assert "c6: 3" in txt and "Y0: -100" in txt
    assert res.poly_method == "snfs (factordb)"
    assert res.stages["polyselect_s"] == 0.0
    assert not any("sq=0" in s for s in seq)


def test_snfs_rational_side_sieves_with_sq_side_0(harness, tmp_path):
    ctx, calls, verdicts = harness
    verdicts.write_text("found 6000 cycles, need 5\n")
    wd = tmp_path / "wd"
    run_nfs(N71, ctx, wd, snfs=_snfs_job(N71, side="r"))
    seq = _seq(calls)
    assert any(s.startswith("band start ") and " sq=0 " in f"{s} "
               for s in seq)


def test_snfs_job_reused_from_workdir(harness, tmp_path):
    """The second run on the same workdir picks the written job back up
    instead of rewriting it: poly_method says resumed."""
    ctx, calls, verdicts = harness
    verdicts.write_text("filtering wants 50 more relations\n")
    wd = tmp_path / "wd"
    res1 = run_nfs(N71, ctx, wd, snfs=_snfs_job(N71))
    assert res1.poly_method == "snfs (factordb)"
    res2 = run_nfs(N71, ctx, wd, snfs=_snfs_job(N71))
    assert res2.poly_method == "snfs (resumed)"


def test_snfs_job_file_roundtrip():
    """_snfs_job_text -> snfs_job_from_file preserves the polynomial and
    the difficulty/side comments; a factordb snfs.php-style file parses
    with its params ignored (yagfu picks them from the difficulty)."""
    from yagfu.backends.nfs import snfs_job_from_file
    from yagfu.params import snfs_params

    job = _snfs_job(N71, side="r")
    rlim, alim, lpb, _ = snfs_params(job.scaled)
    text = nfs._snfs_job_text(job, rlim, alim, lpb)
    back = snfs_job_from_file(text)
    assert back.n == job.n and back.coeffs == job.coeffs
    assert back.y0 == job.y0 and back.y1 == job.y1
    assert back.side == "r" and back.scaled == job.scaled

    php = f"""n: {N71}
# x^66+1, difficulty: 70.10, scaled difficulty: 70.60
# special-q on the algebraic side, siever gnfs-lasieve4I13e, start at q = 1000 in ranges of 4000
type: snfs
size: 70
skew: 1.0000
c6: 3
c0: 1
Y1: 1
Y0: -100
rlim: 7000000
alim: 7000000
lpbr: 27
lpba: 27
mfbr: 54
mfba: 54
rlambda: 2.5
alambda: 2.5
"""
    j2 = snfs_job_from_file(php)
    assert j2 is not None
    assert j2.n == N71 and j2.coeffs == [1, 0, 0, 0, 0, 0, 3]
    assert j2.difficulty == 70.10 and j2.scaled == 70.60
    assert j2.side == "a"
    assert snfs_job_from_file("n: 123\n") is None     # no polynomial
