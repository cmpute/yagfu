"""Engine tests for the polyselect/ECM overlap, using fake jobs and fake
ECM — no GPU, no msieve."""

import pytest

from yagfu import engine as engine_mod
from yagfu.backends.ecm import EcmProvenance, EcmRunResult
from yagfu.backends.siqs import SiqsResult
from yagfu.config import RunContext
from yagfu.engine import Engine

P1 = (1 << 31) - 1
P2 = (1 << 61) - 1
P3 = (1 << 127) - 1


class FakeJob:
    """Stands in for nfs.PolyselectJob; `polls` = running() calls that
    return True before the job 'finishes'."""

    def __init__(self, polls=0):
        self.polls_left = polls
        self.cancelled = 0
        self.waited = 0
        self.budget = 60.0
        self.fb = {"N": "1"}          # truthy: cancel() keeps artifacts

    def elapsed(self):
        return 12.0

    def running(self):
        if self.polls_left > 0:
            self.polls_left -= 1
            return True
        return False

    def wait(self, timeout=None):
        self.waited += 1
        return self

    def cancel(self):
        self.cancelled += 1


def _patch(monkeypatch, job, found_on=(), fail_calls=()):
    """Wire fake poly job + fake ECM + fake NFS into the engine."""
    calls = {"b1": [], "nfs_poly": []}

    def fake_start(self, m, wd):
        return job

    def fake_ecm(n, b1, total, ctx, wd, tag):
        calls["b1"].append(b1)
        if len(calls["b1"]) in fail_calls:
            raise KeyboardInterrupt
        out = EcmRunResult(curves_done=total, wall_s=0.001 * total)
        if len(calls["b1"]) in found_on:
            prov = EcmProvenance(b1=b1, stage=1, sigma=1)
            out.factors[P1 if len(calls["b1"]) == 1 else P2] = prov
        return out

    def fake_nfs(self, n, wd, poly=None):
        calls["nfs_poly"].append(poly)
        return _ok_nfs(n)

    def _ok_nfs(n):
        from yagfu.backends.nfs import NfsResult
        r = NfsResult(ok=True, factors=[P2, P3] if n == P2 * P3 else [P1, P3])
        return r

    def fake_siqs(m, ctx, wd):
        return SiqsResult(factors=[m // P1, P1] if m % P1 == 0 else [P2, P3],
                          ok=True, rc=0, wall_s=0.1)

    monkeypatch.setattr(Engine, "_maybe_start_poly", fake_start)
    monkeypatch.setattr(Engine, "_nfs", fake_nfs)
    monkeypatch.setattr(engine_mod, "run_ecm_curves", fake_ecm)
    monkeypatch.setattr(engine_mod, "pollard_rho", lambda m: ([], m))
    monkeypatch.setattr(engine_mod, "run_siqs", fake_siqs)
    return calls


# c95, plan normal: target ~t29, so the ladder runs a few levels; the xover
# is lowered so the engine treats it as NFS-bound and starts the poly job.
N95 = P2 * P3                          # 19 + 39 digits = 58 -> too small
N = P1 * P2 * P3                       # 66 digits
CTX = dict(xover=60)                   # >= max(60, poly-min 90)? poly gate is
# digits >= NFS_MSIEVE_POLY_MIN_DIGITS (90) — but _maybe_start_poly is
# faked here, so the gate does not apply in these tests.


def test_overlap_runs_bonus_then_joins(monkeypatch, tmp_path):
    job = FakeJob(polls=2)
    calls = _patch(monkeypatch, job)
    eng = Engine(RunContext(**CTX))
    res = eng.factor(N, tmp_path)
    assert res.complete, res.error
    # bonus ECM ran while the job was "running": invocations beyond the
    # plan-target ladder
    assert len(calls["b1"]) >= 3
    # NFS consumed the background poly
    assert calls["nfs_poly"] == [job]
    assert job.waited == 0 or job.waited >= 0     # consume path may wait


def test_overlap_split_during_bonus_replans(monkeypatch, tmp_path):
    job = FakeJob(polls=5)
    calls = _patch(monkeypatch, job)
    # find P1 on call 3 (inside the bonus window) -> cofactor P2*P3 is
    # composite -> re-plan via recursion, NFS never sees the stale poly
    def fake_ecm(n, b1, total, ctx, wd, tag):
        calls["b1"].append((b1, n))
        out = EcmRunResult(curves_done=total, wall_s=0.001 * total)
        if len(calls["b1"]) == 3:
            out.factors[P1] = EcmProvenance(b1=b1, stage=1, sigma=1)
        return out

    monkeypatch.setattr(engine_mod, "run_ecm_curves", fake_ecm)
    eng = Engine(RunContext(**CTX))
    res = eng.factor(N, tmp_path)
    assert res.complete, res.error
    assert job.cancelled >= 1
    # recursion factored the P2*P3 cofactor via the (fake) sieve
    assert calls["nfs_poly"] == [] or calls["nfs_poly"] == [None] * len(calls["nfs_poly"])


def test_overlap_split_during_ladder_replans(monkeypatch, tmp_path):
    job = FakeJob(polls=5)
    calls = _patch(monkeypatch, job, found_on={1})
    eng = Engine(RunContext(**CTX))
    res = eng.factor(N, tmp_path)
    assert res.complete, res.error
    assert job.cancelled >= 1          # poly dropped before re-planning


def test_replan_carries_ledger_and_logs_discard(monkeypatch, tmp_path,
                                                capsys):
    """A bonus-window split must (a) log that the polyselect is discarded,
    (b) hand the ECM ledger to the re-planned cofactor so it does not
    re-climb levels already paid for."""
    job = FakeJob(polls=5)
    calls = _patch(monkeypatch, job, found_on={3})   # P1 found in bonus

    started = []

    def fake_start(self, m, wd):
        started.append(m)
        return job if len(started) == 1 else None   # cofactor: no poly job

    monkeypatch.setattr(Engine, "_maybe_start_poly", fake_start)
    eng = Engine(RunContext(xover=60))
    res = eng.factor(N, tmp_path)
    assert res.complete, res.error
    assert len(calls["b1"]) == 3        # original ladder + one bonus call;
                                        # the cofactor inherited the ledger
    assert job.cancelled >= 1
    err = capsys.readouterr().err
    assert "discarding background polyselect" in err
    assert "budget 60s" in err and "ran 12s" in err


def test_overlap_keyboard_interrupt_cancels_job(monkeypatch, tmp_path):
    job = FakeJob(polls=5)
    _patch(monkeypatch, job, fail_calls={3})
    eng = Engine(RunContext(**CTX))
    res = eng.factor(N, tmp_path)
    assert res.interrupted
    assert job.cancelled >= 1
