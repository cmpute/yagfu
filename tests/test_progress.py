"""Tests for the progress/ETA/cleanup changes: fmt helpers, stage-2 B2
parsing, the ECM ladder surviving cofactor splits, workdir cleanup policy,
and the polled run_cmd_progress."""

import pytest

from yagfu import engine as engine_mod
from yagfu.backends.ecm import EcmProvenance, EcmRunResult, parse_stage2_log
from yagfu.backends.runner import run_cmd_progress
from yagfu.config import RunContext, cleanup_workdir
from yagfu.engine import Engine
from yagfu.fmt import fmt_count, fmt_dur


# ------------------------------------------------------------------- fmt

def test_fmt_count():
    assert fmt_count(3540) == "3,540"
    assert fmt_count(1_873_422) == "1.87M"
    assert fmt_count(12_746_592) == "12.7M"
    assert fmt_count(240_490_660_426) == "240.5G"
    assert fmt_count(15_892_628_251_516) == "15.9T"


def test_fmt_dur():
    assert fmt_dur(0.0026) == "2.6ms"
    assert fmt_dur(40) == "40s"
    assert fmt_dur(90) == "1.5m"
    assert fmt_dur(3.2 * 3600) == "3.2h"


# ------------------------------------------------------------- stage 2 B2

def test_parse_stage2_log_factor_and_b2():
    log = (
        "Using B1=50000, B2=12746592, polynomial Dickson(6), sigma=3:1234567\n"
        "Step 2 took 123ms\n"
        "Using B1=50000, B2=12746592, polynomial Dickson(6), sigma=3:7654321\n"
        "********** Factor found in step 2: 123456789012345678901234567891\n"
        "Factor found in step 2: 999\n"          # the input number: not a factor
    )
    factors, b2 = parse_stage2_log(log, 50000, 999)
    assert b2 == 12746592
    f = factors[123456789012345678901234567891]
    assert (f.b1, f.stage, f.sigma, f.b2) == (50000, 2, 7654321, 12746592)
    assert 999 not in factors


def test_parse_stage2_log_no_factor():
    log = "Using B1=11000, B2=1873422, polynomial Dickson(6), sigma=3:42\n"
    factors, b2 = parse_stage2_log(log, 11000, 123)
    assert factors == {}
    assert b2 == 1873422


# ------------------------------------------- ECM ladder survives splits

P1 = (1 << 31) - 1     # 10-digit prime
P2 = (1 << 61) - 1     # 19-digit prime
P3 = (1 << 127) - 1    # 39-digit prime


def _fake_ecm(monkeypatch, found_on_call):
    calls = []

    def fake(n, b1, total, ctx, wd, tag):
        calls.append((b1, n))
        prov = EcmProvenance(b1=b1, stage=2, sigma=1, b2=123)
        out = EcmRunResult(curves_done=total, wall_s=0.001 * total)
        if len(calls) in found_on_call:
            out.factors[P1 if len(calls) == 1 else P2] = prov
        return out

    monkeypatch.setattr(engine_mod, "run_ecm_curves", fake)
    monkeypatch.setattr(engine_mod, "pollard_rho", lambda m: ([], m))
    return calls


def test_ecm_ladder_survives_split(monkeypatch, tmp_path):
    """After ECM splits N = P1*P2*P3, the ladder must continue on the
    cofactor at the next level (B1 2000 -> 11000), not restart at B1=2000."""
    calls = _fake_ecm(monkeypatch, found_on_call={1, 2})
    eng = Engine(RunContext())
    res = eng.factor(P1 * P2 * P3, tmp_path)
    assert res.complete, res.error
    assert [b1 for b1, _ in calls] == [2000, 11000]
    assert calls[1][1] == P2 * P3          # factoring the cofactor now


def test_ecm_ladder_survives_split_ecm_only(monkeypatch, tmp_path):
    calls = _fake_ecm(monkeypatch, found_on_call={1, 2})
    eng = Engine(RunContext(ecm_only=True))
    res = eng.factor(P1 * P2 * P3, tmp_path)
    assert res.complete, res.error
    assert [b1 for b1, _ in calls] == [2000, 11000]
    assert calls[1][1] == P2 * P3


# ------------------------------------------------------------ workdir

def test_cleanup_workdir(tmp_path):
    def mk():
        wd = tmp_path / "wd"
        wd.mkdir(exist_ok=True)
        (wd / "x").write_text("x")
        return wd

    ctx = RunContext()
    wd = mk()
    cleanup_workdir(ctx, wd, keep=True)     # failed/interrupted: kept
    assert wd.exists()
    wd = mk()
    cleanup_workdir(ctx, wd, keep=False)    # complete success: removed
    assert not wd.exists()
    keepall = RunContext(keep_workdir=True)
    wd = mk()
    cleanup_workdir(keepall, wd, keep=False)
    assert wd.exists()                      # --keep always wins


def test_cli_keeps_workdir_on_failure(tmp_path, monkeypatch, capsys):
    from yagfu import cli
    from yagfu.engine import Engine, FactorResult

    def fake_factor(self, n, wd):
        (wd / "scratch.txt").write_text("x")
        return FactorResult(n=n, complete=False, error="boom")

    monkeypatch.setattr(Engine, "factor", fake_factor)
    rc = cli.main(["--workdir", str(tmp_path), "1000000007"])
    assert rc == 1
    assert list(tmp_path.glob("000-*/scratch.txt")), "failure keeps logs"


def test_cli_removes_workdir_on_success(tmp_path, monkeypatch, capsys):
    from yagfu import cli
    from yagfu.engine import Engine, FactorResult, FoundFactor

    def fake_factor(self, n, wd):
        (wd / "scratch.txt").write_text("x")
        res = FactorResult(n=n, complete=True)
        res.factors.append(FoundFactor(n, "td", True))
        return res

    monkeypatch.setattr(Engine, "factor", fake_factor)
    rc = cli.main(["--workdir", str(tmp_path), "1000000007"])
    assert rc == 0
    assert not list(tmp_path.glob("000-*")), "success cleans up"


# ------------------------------------------------------ run_cmd_progress

def test_run_cmd_progress_forwards_lines(tmp_path):
    seen = []
    script = "echo one; echo two; sleep 0.4; echo three"
    rc, wall = run_cmd_progress(
        ["bash", "-c", script], tmp_path, tmp_path / "l.log",
        every_s=0.1, on_lines=seen.extend)
    assert rc == 0
    assert seen == ["one", "two", "three"]


def test_run_cmd_progress_timeout(tmp_path):
    rc, _ = run_cmd_progress(["bash", "-c", "sleep 30"], tmp_path,
                             tmp_path / "l.log", timeout=0.5, every_s=0.2)
    assert rc == -9


def test_feed_lines_splits_carriage_returns(tmp_path):
    """msieve paces progress updates with bare \\r; each must arrive as its
    own line, and a trailing \\r-terminated segment must not be lost."""
    from yagfu.backends.runner import _feed_lines
    log = tmp_path / "l.log"
    log.write_bytes(b"commencing linear algebra\nupdate 1\rupdate 2\r")
    seen = []
    pos, pending = _feed_lines(log, 0, "", seen.extend)
    assert seen == ["commencing linear algebra", "update 1", "update 2"]
    assert pending == ""
    with open(log, "ab") as f:
        f.write(b"update 3\rdone\n")
    pos, pending = _feed_lines(log, pos, pending, seen.extend)
    assert seen[-2:] == ["update 3", "done"]
