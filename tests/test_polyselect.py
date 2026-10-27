"""PolyselectJob tests: multi-GPU range sharding, spawn pinning, result
merge, interrupt salvage, retry policy — plus the quality-gated block
ladder and the legacy wall-cap watcher.  No GPU, no real msieve."""

import math
import signal
import subprocess
import time
from pathlib import Path

import pytest

from yagfu.backends import nfs
from yagfu.config import RunContext

C110 = 18680501626534293594781018822882202042069695039599176584471642868419555422528756559182034596825447693688538183
C137 = 45533138366837631613836555826169837773902778359947838840845761213856615752148244696044438294775069794018243413637217191807580253436610113
# what msieve itself prints ("searching leading coefficients from 1 to N")
MSIEVE_C110_TOP = 96927
MSIEVE_C137_TOP = 2268794


# ------------------------------------------------------------ range math

def test_poly_coeff_max_matches_msieve():
    assert nfs._poly_coeff_max(C110) == MSIEVE_C110_TOP
    assert nfs._poly_coeff_max(C137) == MSIEVE_C137_TOP
    n4 = 10 ** 94 + 3                      # 95 digits -> degree 4 -> 1/12
    assert nfs._poly_coeff_max(n4) == int(float(n4) ** (1 / 12) / 3.0)


def test_poly_ranges_split_disjoint_covering():
    assert nfs._poly_ranges(C110, 1) == [None]
    rs = nfs._poly_ranges(C110, 4)
    assert len(rs) == 4
    assert rs[0][0] == 1 and rs[-1][1] == MSIEVE_C110_TOP
    for (_, hi), (lo, _) in zip(rs, rs[1:]):
        assert lo == hi + 1                # contiguous, no gap, no overlap


def test_poly_deadline_table_matches_msieve():
    """The replicated stage-1 deadline must agree with what msieve prints
    (c152 on this box: "poly select deadline: 609820")."""
    n152 = 37917182519133761458994004180287393522121573678735988165161922758472807134459667342498067501509551723560171548116096739466016888383316598100704114893997
    assert nfs._poly_deadline_s(math.log10(n152)) == 609820
    # exact rows and interpolation (220.0 selects degree 6 in msieve's
    # rule, whose interpolated truncation prints 1079999 — so compare
    # the degree-5 flat tail by equality of two of its points instead)
    assert nfs._poly_deadline_s(110.0) == 45 * 60
    assert nfs._poly_deadline_s(190.0) == nfs._poly_deadline_s(215.0)
    assert 23 * 60 < nfs._poly_deadline_s(108.5) < 45 * 60
    assert nfs._poly_deadline_s(80.0) == 12 * 60


def _ladder_block_ranges(n, k, threads):
    """Round-1 block ranges the ladder must spawn for these params."""
    wall_full = nfs._poly_deadline_s(math.log10(n)) / threads
    budget = nfs.poly_budget(len(str(n)), None)
    width = max(nfs.POLY_MIN_BLOCK,
                int(nfs._poly_coeff_max(n) * budget / wall_full))
    return [(i * width + 1, min((i + 1) * width, nfs._poly_coeff_max(n)))
            for i in range(k)]


# ------------------------------------------------------------- spawning

class FakeP:
    def __init__(self, cmd, **kw):
        self.cmd, self.kw = cmd, kw

    def poll(self):
        return 0

    def wait(self, timeout=None):
        return 0

    def send_signal(self, sig):
        pass


def test_sharded_spawn_pins_and_splits(tmp_path, monkeypatch, capsys):
    """Four cards -> four processes, one per private poly-{i} subdir with
    its own CUDA_VISIBLE_DEVICES pin, a comma-free range arg, -t split
    across the cards, and no stale msieve.fb anywhere (msieve would read
    one and skip the search).  The default mode opens with a round-1
    block, not the whole range."""
    def record_spawn(cmd, **kw):
        p = FakeP(cmd, **kw)
        spawns.append(p)
        return p

    spawns = []
    monkeypatch.setattr(nfs, "find_bin", lambda name, bd: "/bin/msieve")
    monkeypatch.setattr(subprocess, "Popen", record_spawn)
    (tmp_path / "msieve.fb").write_text("N stale\n")          # top-level
    stale = tmp_path / "poly-0"
    stale.mkdir()
    (stale / "msieve.fb").write_text("N stale\n")             # shard dir

    ctx = RunContext(device="0", devices=("0", "1", "2", "3"), threads=32,
                     log_file=tmp_path / "t.log")
    nfs.PolyselectJob(C110, ctx, tmp_path)

    assert len(spawns) == 4
    assert not (tmp_path / "msieve.fb").exists()              # stale gone
    got = []
    for i, p in enumerate(spawns):
        assert p.kw["cwd"] == str(tmp_path / f"poly-{i}")
        assert p.kw["env"]["CUDA_VISIBLE_DEVICES"] == str(i)
        args = p.cmd[2]
        assert args.startswith("stage1_engine=gpu_cubsort")
        assert "," not in args          # legacy X,Y range parse must not fire
        lo = int(args.split("min_coeff=")[1].split()[0])
        hi = int(args.split("max_coeff=")[1].split()[0])
        got.append((lo, hi))
        assert p.cmd[p.cmd.index("-t") + 1] == "8"          # 32 // 4
        assert p.cmd[-1] == str(C110)
    assert got == _ladder_block_ranges(C110, 4, 8)
    assert "polyselect round 1 across 4 card(s)" in capsys.readouterr().err


def test_single_device_spawn_is_block_in_subdir(tmp_path, monkeypatch):
    """One card, default mode: the round-1 block still runs in a private
    subdir (the ladder promotes the winner to the root itself) with the
    full -t and the card pin."""
    spawns = []

    def record_spawn(cmd, **kw):
        p = FakeP(cmd, **kw)
        spawns.append(p)
        return p

    monkeypatch.setattr(nfs, "find_bin", lambda name, bd: "/bin/msieve")
    monkeypatch.setattr(subprocess, "Popen", record_spawn)
    ctx = RunContext(device="2", devices=("2",), threads=32,
                     log_file=tmp_path / "t.log")
    nfs.PolyselectJob(C110, ctx, tmp_path)
    assert len(spawns) == 1
    p = spawns[0]
    assert p.kw["cwd"] == str(tmp_path / "poly-0")
    (lo, hi), = _ladder_block_ranges(C110, 1, 32)
    assert f"min_coeff={lo} max_coeff={hi}" in p.cmd[2]
    assert p.cmd[p.cmd.index("-t") + 1] == "32"
    assert p.kw["env"]["CUDA_VISIBLE_DEVICES"] == "2"


def test_explicit_budget_keeps_legacy_spawn(tmp_path, monkeypatch, capsys):
    """An explicit --poly-budget reverts to the legacy shape: one
    full-range search (no range args on a lone card), hard cap."""
    spawns = []

    def record_spawn(cmd, **kw):
        p = FakeP(cmd, **kw)
        spawns.append(p)
        return p

    monkeypatch.setattr(nfs, "find_bin", lambda name, bd: "/bin/msieve")
    monkeypatch.setattr(subprocess, "Popen", record_spawn)
    ctx = RunContext(device="2", devices=("2",), threads=32,
                     poly_budget_s=3600.0, log_file=tmp_path / "t.log")
    nfs.PolyselectJob(C110, ctx, tmp_path)
    assert len(spawns) == 1
    p = spawns[0]
    assert p.kw["cwd"] == str(tmp_path)
    assert p.cmd[1:4] == ["-np", "stage1_engine=gpu_cubsort", "-t"]
    assert p.cmd[4] == "32"
    assert "min_coeff" not in p.cmd[2]
    assert p.kw["env"]["CUDA_VISIBLE_DEVICES"] == "2"
    assert "expected, cap" in capsys.readouterr().err

    spawns.clear()
    ctx4 = RunContext(device="0", devices=("0", "1", "2", "3"), threads=32,
                      poly_budget_s=3600.0, log_file=tmp_path / "t.log")
    nfs.PolyselectJob(C110, ctx4, tmp_path)
    got = []
    for p in spawns:
        args = p.cmd[2]
        got.append((int(args.split("min_coeff=")[1].split()[0]),
                    int(args.split("max_coeff=")[1].split()[0])))
    assert got == nfs._poly_ranges(C110, 4)
    assert "sharded across 4 GPUs" in capsys.readouterr().err


# ---------------------------------------------------------------- merge

def test_merge_promotes_best_shard(tmp_path, capsys):
    job = object.__new__(nfs.PolyselectJob)
    job.wd = tmp_path
    job.ctx = RunContext(log_file=tmp_path / "t.log")
    job.sharded = True
    for name, e in (("poly-0", 1.0e-9), ("poly-1", 1.5e-9)):
        d = tmp_path / name
        d.mkdir()
        (d / "msieve.fb").write_text(f"N {C110}\nR1 {name}\n")
        (d / "np.log").write_text(f"skew 1.00, combined = {e}\n")
    (tmp_path / "poly-2").mkdir()              # empty shard: no fb
    job._procs = [(tmp_path / "poly-0", None),
                  (tmp_path / "poly-1", None),
                  (tmp_path / "poly-2", None)]
    job._merge()
    assert "R1 poly-1" in (tmp_path / "msieve.fb").read_text()
    assert "combined = 1.5e-09" in (tmp_path / "np.log").read_text()
    assert "best from poly-1" in capsys.readouterr().err
    job._t0 = time.monotonic()
    job._finish()
    assert job.poly_e == 1.5e-9 and job.fb["N"] == str(C110)


def test_merge_never_displaces_better_incumbent(tmp_path, capsys):
    """The ladder promotes round winners; a later, worse round must not
    clobber the better polynomial already at the workdir root."""
    job = object.__new__(nfs.PolyselectJob)
    job.wd = tmp_path
    job.ctx = RunContext(log_file=tmp_path / "t.log")
    job.sharded = True
    (tmp_path / "msieve.fb").write_text(f"N {C110}\nR1 incumbent\n")
    (tmp_path / "np.log").write_text("skew 1.00, combined = 1.5e-09\n")
    d = tmp_path / "poly-0"
    d.mkdir()
    (d / "msieve.fb").write_text(f"N {C110}\nR1 worse\n")
    (d / "np.log").write_text("skew 1.00, combined = 1.2e-09\n")
    job._procs = [(d, None)]
    job._merge()
    assert "R1 incumbent" in (tmp_path / "msieve.fb").read_text()
    assert "keeps the incumbent E=1.5e-09" in capsys.readouterr().err


# ------------------------------------------------- quality-gated ladder

class LogP:
    """Fake msieve -np: exits immediately and leaves the shard dir with
    an np.log carrying msieve's E-expectation header plus one combined-E
    line (values popped per spawn), like a completed search."""

    values = []
    spawns = []

    def __init__(self, cmd, **kw):
        self.cmd, self.kw = cmd, kw
        e = LogP.values.pop(0)
        d = Path(kw["cwd"])
        (d / "np.log").write_text(
            "expecting poly E from 1e-8 to > 2e-8\n"
            f"skew 1.00, combined = {e}\n")
        (d / "msieve.fb").write_text(f"N {C110}\nR1 {d.name}\n")
        LogP.spawns.append((d, cmd))

    def poll(self):
        return 0

    def wait(self, timeout=None):
        return 0

    def send_signal(self, sig):
        pass


def _ladder_ctx(tmp_path, **kw):
    return RunContext(device="0", devices=("0", "1"), threads=8,
                      log_file=tmp_path / "t.log", **kw)


def test_ladder_runs_until_bar_met(tmp_path, monkeypatch, capsys):
    """Round 1 lands below msieve's expectation: the ladder continues
    over fresh, disjoint ground; round 2 meets the bar and stops early."""
    LogP.values = [5e-9, 5e-9, 2e-8, 2e-8]
    LogP.spawns = []
    monkeypatch.setattr(nfs, "find_bin", lambda name, bd: "/bin/msieve")
    monkeypatch.setattr(subprocess, "Popen",
                        lambda cmd, **kw: LogP(cmd, **kw))
    job = nfs.PolyselectJob(C110, _ladder_ctx(tmp_path), tmp_path)
    job.wait()
    assert len(LogP.spawns) == 4                 # 2 cards x 2 rounds
    r1_hi = max(int(cmd[2].split("max_coeff=")[1].split()[0])
                for d, cmd in LogP.spawns[:2])
    r2_lo = min(int(cmd[2].split("min_coeff=")[1].split()[0])
                for d, cmd in LogP.spawns[2:])
    assert r2_lo == r1_hi + 1                    # fresh ground, no overlap
    assert job.poly_e == 2e-8
    assert job.fb["N"] == str(C110)
    err = capsys.readouterr().err
    assert "met msieve's expectation" in err
    assert "keeps the incumbent" not in err


def test_ladder_keeps_incumbent_and_exhausts_range(tmp_path, monkeypatch,
                                                   capsys):
    """No round ever meets the bar: the ladder keeps the best incumbent,
    never regresses to a worse polynomial, and stops when the range is
    exhausted."""
    LogP.values = [5e-9, 4.9e-9, 4e-9, 4e-9, 1e-9, 1e-9]
    LogP.spawns = []
    monkeypatch.setattr(nfs, "find_bin", lambda name, bd: "/bin/msieve")
    monkeypatch.setattr(subprocess, "Popen",
                        lambda cmd, **kw: LogP(cmd, **kw))
    job = nfs.PolyselectJob(C110, _ladder_ctx(tmp_path), tmp_path)
    job.wait()
    assert len(LogP.spawns) == 6                 # 3 rounds to exhaustion
    err = capsys.readouterr().err
    assert "keeps the incumbent E=5e-09" in err  # no regression
    assert "searched the whole leading-coefficient range" in err
    assert "R1 poly-0" in (tmp_path / "msieve.fb").read_text()
    assert job.poly_e == 5e-9
    # the union of all spawned ranges covers [1, top] with no overlap
    seen = []
    for d, cmd in LogP.spawns:
        args = cmd[2]
        seen.append((int(args.split("min_coeff=")[1].split()[0]),
                     int(args.split("max_coeff=")[1].split()[0])))
    seen.sort()
    assert seen[0][0] == 1 and seen[-1][1] == MSIEVE_C110_TOP
    for (_, hi), (lo, _) in zip(seen, seen[1:]):
        assert lo == hi + 1


# ------------------------------------------------------ interrupt salvage

class AliveP:
    """Never exits on its own; dies on terminate()."""

    def __init__(self):
        self.dead = False

    def poll(self):
        return 0 if self.dead else None

    def wait(self, timeout=None):
        if self.dead:
            return 0
        raise KeyboardInterrupt("simulated Ctrl+C")

    def terminate(self):
        self.dead = True

    def kill(self):
        self.dead = True


def test_wait_interrupt_kills_all_and_salvages(tmp_path):
    """Ctrl+C during the search: every shard process is torn down, the
    best shard so far is still promoted to msieve.fb (a rerun reuses it),
    and the KeyboardInterrupt propagates."""
    job = object.__new__(nfs.PolyselectJob)
    job.wd = tmp_path
    job.ctx = RunContext(log_file=tmp_path / "t.log")
    job._hard_cap = True
    job.sharded = True
    good = tmp_path / "poly-0"
    good.mkdir()
    (good / "msieve.fb").write_text(f"N {C110}\nR1 salvaged\n")
    (good / "np.log").write_text("skew 1.00, combined = 1e-9\n")
    (tmp_path / "poly-1").mkdir()
    procs = [AliveP(), AliveP()]
    job._procs = [(good, procs[0]), (tmp_path / "poly-1", procs[1])]
    with pytest.raises(KeyboardInterrupt):
        job.wait()
    assert all(p.dead for p in procs)
    assert "R1 salvaged" in (tmp_path / "msieve.fb").read_text()


# --------------------------------------------------------- retry policy

class DeadP:
    def poll(self):
        return 1

    def wait(self, timeout=None):
        return 0


def _bare_job(tmp_path, sharded):
    job = object.__new__(nfs.PolyselectJob)
    job.wd = tmp_path
    job.ctx = RunContext(log_file=tmp_path / "t.log")
    job.n = C110
    job.engine = "gpu_cubsort"
    job.sharded = sharded
    job._engine_retried = False
    job._cap_logged = False
    job._t0 = time.monotonic()
    job._deadline_at = time.monotonic() + 3600
    return job


def test_engine_retry_to_cpu(tmp_path, monkeypatch, capsys):
    """A single-process polynomial-less GPU search respawns once on the
    CPU Gerbicz engine; a polynomial on disk or a prior retry stops it."""
    job = _bare_job(tmp_path, sharded=False)
    job._procs = [(tmp_path, DeadP())]
    spawns, watches = [], []
    monkeypatch.setattr(nfs, "find_bin", lambda name, bd: "/bin/msieve")
    monkeypatch.setattr(job, "_shard", lambda ms: spawns.append(ms),
                        raising=False)
    monkeypatch.setattr(job, "_watch", lambda: watches.append(1),
                        raising=False)
    assert job.engine_retry() is True          # gpu -> cpu_gerbicz, once
    assert job.engine == "cpu_gerbicz"
    assert spawns == ["/bin/msieve"] and watches == [1]
    assert "CPU Gerbicz engine" in capsys.readouterr().err
    assert job.engine_retry() is False
    ok = _bare_job(tmp_path, sharded=False)    # a polynomial exists
    ok._procs = [(tmp_path, DeadP())]
    (tmp_path / "msieve.fb").write_text("N 1\n")
    assert ok.engine_retry() is False


def test_all_shards_empty_retries_full_range_cpu(tmp_path, monkeypatch):
    """Every shard came back polynomial-less: one cpu_gerbicz respawn in
    the bare workdir over the whole default range (no min/max args)."""
    job = _bare_job(tmp_path, sharded=True)
    for name in ("poly-0", "poly-1"):
        (tmp_path / name).mkdir()
    job._procs = [(tmp_path / "poly-0", DeadP()),
                  (tmp_path / "poly-1", DeadP())]
    spawns = []

    def record_spawn(cmd, **kw):
        p = FakeP(cmd, **kw)
        spawns.append(p)
        return p

    monkeypatch.setattr(nfs, "find_bin", lambda name, bd: "/bin/msieve")
    monkeypatch.setattr(subprocess, "Popen", record_spawn)
    assert job.engine_retry() is True
    assert job.engine == "cpu_gerbicz"
    assert len(spawns) == 1
    p = spawns[0]
    assert p.kw["cwd"] == str(tmp_path)
    assert p.cmd[2] == "stage1_engine=cpu_gerbicz"     # no range args


def test_some_shards_have_poly_no_retry(tmp_path, monkeypatch, capsys):
    """An empty shard alongside ones that found polynomials: no CPU
    hours on the empty slice — the merge already kept the others."""
    job = _bare_job(tmp_path, sharded=True)
    (tmp_path / "poly-0").mkdir()
    (tmp_path / "poly-0" / "msieve.fb").write_text("N 1\n")
    (tmp_path / "poly-1").mkdir()
    job._procs = [(tmp_path / "poly-0", DeadP()),
                  (tmp_path / "poly-1", DeadP())]
    spawns = []
    monkeypatch.setattr(nfs, "find_bin", lambda name, bd: "/bin/msieve")
    monkeypatch.setattr(subprocess, "Popen",
                        lambda cmd, **kw: spawns.append(FakeP(cmd, **kw)))
    assert job.engine_retry() is False
    assert spawns == []
    assert "keeping the others" in capsys.readouterr().err


# -------------------------------------------------------- wall-cap watch

def test_deadline_watch_sigints_at_cap(tmp_path, capsys):
    """At the total wall cap the watcher SIGINTs msieve (so it saves its
    best-so-far polynomial) instead of killing it outright."""

    class SleepingP(AliveP):
        def wait(self, timeout=None):
            if self.dead:
                return 0
            raise subprocess.TimeoutExpired("msieve", timeout)

        def send_signal(self, sig):
            self.signals.append(sig)
            self.dead = True

    job = object.__new__(nfs.PolyselectJob)
    job.wd = tmp_path
    job.ctx = RunContext(log_file=tmp_path / "t.log")
    job._t0 = time.monotonic() - 10.0
    job._deadline_at = time.monotonic() - 1.0   # cap already passed
    job._cap_logged = False
    p = SleepingP()
    p.signals = []
    job._term_kill = lambda p: None
    job._deadline_watch(p)
    assert p.signals == [signal.SIGINT]
    assert "wall cap" in capsys.readouterr().err
