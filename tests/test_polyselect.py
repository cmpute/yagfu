"""PolyselectJob tests: multi-GPU range sharding, spawn pinning, result
merge, interrupt salvage, retry policy — plus the wall-cap watcher.
No GPU, no real msieve."""

import signal
import subprocess
import time

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
    one and skip the search)."""
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
    assert got == nfs._poly_ranges(C110, 4)
    assert "sharded across 4 GPUs" in capsys.readouterr().err


def test_single_device_spawn_is_unchanged(tmp_path, monkeypatch):
    """One card: bare workdir, no range args, full -t — byte-identical
    command line to the pre-sharding single-process search."""
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
    assert p.kw["cwd"] == str(tmp_path)
    assert p.cmd[1:4] == ["-np", "stage1_engine=gpu_cubsort", "-t"]
    assert p.cmd[4] == "32"
    assert p.kw["env"]["CUDA_VISIBLE_DEVICES"] == "2"


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
