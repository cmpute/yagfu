"""PolyselectJob unit tests (no GPU, no real msieve)."""

import signal
import subprocess
import time

from yagfu.backends import nfs
from yagfu.config import RunContext


def test_engine_retry_to_cpu(tmp_path, monkeypatch, capsys):
    """A polynomial-less GPU search respawns once on the CPU Gerbicz
    engine; a polynomial on disk or a prior retry stops it."""
    class DeadP:
        def poll(self):
            return 1

        def wait(self, timeout=None):
            return 0

    job = object.__new__(nfs.PolyselectJob)
    job.wd = tmp_path
    job.ctx = RunContext(log_file=tmp_path / "t.log")
    job.engine = "gpu_cubsort"
    job._engine_retried = False
    job._t0 = time.monotonic()
    job._deadline_at = time.monotonic() + 3600
    job._p = DeadP()
    spawns, watches = [], []
    monkeypatch.setattr(nfs, "find_bin", lambda name, bd: "/bin/msieve")
    monkeypatch.setattr(job, "_spawn", lambda ms: spawns.append(ms),
                        raising=False)
    monkeypatch.setattr(job, "_watch", lambda: watches.append(1),
                        raising=False)
    assert job.engine_retry() is True          # gpu -> cpu_gerbicz, once
    assert job.engine == "cpu_gerbicz"
    assert spawns == ["/bin/msieve"] and watches == [1]
    assert "CPU Gerbicz engine" in capsys.readouterr().err
    assert job.engine_retry() is False
    ok = object.__new__(nfs.PolyselectJob)     # a polynomial exists
    ok.wd, ok.ctx = tmp_path, job.ctx
    ok.engine, ok._engine_retried = "gpu_cubsort", False
    ok._t0, ok._deadline_at = time.monotonic(), time.monotonic() + 60
    ok._p = DeadP()
    (tmp_path / "msieve.fb").write_text("N 1\n")
    assert ok.engine_retry() is False


def test_deadline_watch_sigints_at_cap(tmp_path, capsys):
    """At the total wall cap the watcher SIGINTs msieve (so it saves its
    best-so-far polynomial) instead of killing it outright."""
    class AliveP:
        def __init__(self):
            self.signals = []

        def poll(self):
            return None

        def send_signal(self, sig):
            self.signals.append(sig)
            self.dead = True

        def wait(self, timeout=None):
            if getattr(self, "dead", False):
                return 0
            raise subprocess.TimeoutExpired("msieve", timeout)

    job = object.__new__(nfs.PolyselectJob)
    job.wd = tmp_path
    job.ctx = RunContext(log_file=tmp_path / "t.log")
    job._t0 = time.monotonic() - 10.0
    job._deadline_at = time.monotonic() - 1.0   # cap already passed
    job._p = AliveP()
    job._term_kill = lambda: None
    job._deadline_watch()
    assert job._p.signals == [signal.SIGINT]
    assert "wall cap" in capsys.readouterr().err
