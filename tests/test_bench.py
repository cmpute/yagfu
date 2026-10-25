"""Unit tests for scripts/bench_factor.py (crossover math)."""

import importlib.util
import math
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
_spec = importlib.util.spec_from_file_location(
    "bench_factor", ROOT / "scripts" / "bench_factor.py")
bf = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(bf)


def test_fit_exp_exact():
    pts = [(70, 10.0), (80, 30.0), (90, 90.0)]  # T = 10 * 3^((d-70)/10)
    ln_a, k = bf._fit_exp(pts)
    for d, t in pts:
        assert math.exp(ln_a + k * d) == pytest.approx(t)


def test_crossover_known_crossing(capsys, monkeypatch):
    # SIQS: T = exp(-15 + 0.20 d), NFS: T = exp(-3.6 + 0.08 d)
    # equal at d* = (lnA_nfs - lnA_siqs)/(k_siqs - k_nfs) = 11.4/0.12 = 95
    ds = [70, 80, 90, 100]
    siqs = [math.exp(-15 + 0.20 * d) for d in ds]
    nfs = [math.exp(-3.6 + 0.08 * d) for d in ds]
    fake = ([{"method": "siqs", "digits": d, "ok": True, "wall_s": t}
             for d, t in zip(ds, siqs)]
            + [{"method": "nfs", "digits": d, "ok": True, "wall_s": t}
               for d, t in zip(ds, nfs)])
    monkeypatch.setattr(bf, "load_results", lambda: fake)
    d_star = bf.cmd_crossover()
    assert d_star == pytest.approx(95.0, rel=1e-4)
    out = capsys.readouterr().out
    assert "suggested: yagfu --xover 95" in out


def test_crossover_needs_two_points(capsys, monkeypatch):
    monkeypatch.setattr(bf, "load_results",
                        lambda: [{"method": "siqs", "digits": 70,
                                  "ok": True, "wall_s": 1.6},
                                 {"method": "nfs", "digits": 70,
                                  "ok": True, "wall_s": 60.0}])
    assert bf.cmd_crossover() is None
    assert "need >= 2" in capsys.readouterr().out


def test_crossover_ignores_failed_runs(capsys, monkeypatch):
    ds = [70, 80, 90, 100]
    fake = ([{"method": "siqs", "digits": d, "ok": d < 80,  # only one ok
              "wall_s": 1.0} for d in ds]
            + [{"method": "nfs", "digits": d, "ok": True, "wall_s": 2.0}
               for d in ds])
    monkeypatch.setattr(bf, "load_results", lambda: fake)
    assert bf.cmd_crossover() is None
    assert "need >= 2 successful siqs" in capsys.readouterr().out
