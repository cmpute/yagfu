"""SNFS flow: a supplied SnfsJob is consumed by the first sieve stage,
which sieves the polynomial's own number (not the possibly smaller
post-ECM cofactor); factors of that number are absorbed after stripping
what the ECM pretest already found; the pretest itself is sized by the
SNFS difficulty.  All fakes: no GPU, no msieve."""

from yagfu import engine as engine_mod
from yagfu.backends.ecm import EcmProvenance, EcmRunResult
from yagfu.backends.nfs import NfsResult, SnfsJob
from yagfu.backends.siqs import SiqsResult
from yagfu.config import RunContext
from yagfu.engine import Engine
from yagfu.primes import is_probable_prime as _ipp


def _prime(digits: int) -> int:
    n = 10 ** (digits - 1) + 27
    while not _ipp(n):
        n += 2
    return n


PA = _prime(30)
QB, QC, QD = _prime(30), _prime(31), _prime(30)
N3 = PA * QB * QC                      # c91: ECM split leaves c61
N4 = PA * QB * QC * QD                 # c121


def _job(n, scaled=60.0) -> SnfsJob:
    return SnfsJob(n=n, coeffs=[1, 0, 0, 1], y0=-12345, y1=1, skew=1.0,
                   difficulty=scaled, scaled=scaled, side="a",
                   form="x^3+1", e=1e-4)


def _patch_stdlib_fakes(monkeypatch):
    """No rho splits, ECM finds nothing unless the test's fake adds to it."""
    monkeypatch.setattr(engine_mod, "pollard_rho", lambda m: ([], m))
    monkeypatch.setattr(engine_mod, "run_ecm_curves",
                        lambda n, b1, total, ctx, wd, tag:
                        EcmRunResult(curves_done=total,
                                     wall_s=0.001 * total))


def test_snfs_suggestion_reaches_run_nfs(monkeypatch, tmp_path):
    """no-ECM run: the sieve stage calls run_nfs with the suggestion's own
    number and the job, not the (here equal) cofactor."""
    calls = []

    def fake_nfs(self, n, wd, snfs=None):
        calls.append((n, snfs))
        return NfsResult(ok=True, factors=[PA, QB, QC])

    monkeypatch.setattr(Engine, "_nfs", fake_nfs)
    _patch_stdlib_fakes(monkeypatch)
    eng = Engine(RunContext(no_ecm=True))
    res = eng.factor(N3, tmp_path, snfs=_job(N3))
    assert res.complete, res.error
    assert calls == [(N3, calls[0][1])]
    assert calls[0][1].scaled == 60.0
    assert sorted(f.value for f in res.factors) == sorted([PA, QB, QC])


def test_snfs_absorbs_after_ecm_split(monkeypatch, tmp_path):
    """The pretest split PA off; the SNFS run still sieves the original
    number N3, and the composite factor PA*QB is absorbed by stripping
    the already-collected PA."""
    ecm_calls, nfs_calls = [], []

    def fake_ecm(n, b1, total, ctx, wd, tag):
        ecm_calls.append(n)
        out = EcmRunResult(curves_done=total, wall_s=0.001 * total)
        if len(ecm_calls) == 1:
            out.factors[PA] = EcmProvenance(b1=b1, stage=1, sigma=1)
        return out

    def fake_nfs(self, n, wd, snfs=None):
        nfs_calls.append((n, snfs))
        return NfsResult(ok=True, factors=[PA * QB, QC])

    monkeypatch.setattr(engine_mod, "run_ecm_curves", fake_ecm)
    monkeypatch.setattr(engine_mod, "pollard_rho", lambda m: ([], m))
    monkeypatch.setattr(Engine, "_nfs", fake_nfs)
    eng = Engine(RunContext(xover=1))
    res = eng.factor(N3, tmp_path, snfs=_job(N3))
    assert res.complete, res.error
    assert nfs_calls == [(N3, nfs_calls[0][1])]
    assert sorted(f.value for f in res.factors) == sorted([PA, QB, QC])


def test_snfs_consumed_once(monkeypatch, tmp_path):
    """A partial SNFS split recurses on the surviving cofactor with the
    usual selection: the suggestion never reaches a second sieve stage."""
    calls = []

    def fake_nfs(self, n, wd, snfs=None):
        calls.append((n, snfs))
        if n == N4:
            return NfsResult(ok=True, factors=[PA * QB])
        assert n == QC * QD
        return NfsResult(ok=True, factors=[QC, QD])

    monkeypatch.setattr(Engine, "_nfs", fake_nfs)
    _patch_stdlib_fakes(monkeypatch)
    eng = Engine(RunContext(xover=1))
    res = eng.factor(N4, tmp_path, snfs=_job(N4))
    assert res.complete, res.error
    assert [(n, s is not None) for n, s in calls] == [(N4, True),
                                                      (QC * QD, False)]
    assert sorted(f.value for f in res.factors) == sorted([PA, QB, QC, QD])


def test_snfs_failure_falls_back_to_siqs(monkeypatch, tmp_path):
    """A failed SNFS run is consumed: the usual selection takes the
    post-ECM cofactor (SIQS here — the split left c61)."""
    ecm_calls, calls = [], {"nfs": [], "siqs": []}

    def fake_ecm(n, b1, total, ctx, wd, tag):
        ecm_calls.append(n)
        out = EcmRunResult(curves_done=total, wall_s=0.001 * total)
        if len(ecm_calls) == 1:
            out.factors[PA] = EcmProvenance(b1=b1, stage=1, sigma=1)
        return out

    def fake_nfs(self, n, wd, snfs=None):
        calls["nfs"].append(n)
        return NfsResult(ok=False, error="boom")

    def fake_siqs(m, ctx, wd):
        calls["siqs"].append(m)
        return SiqsResult(factors=[QB, QC], ok=True, rc=0, wall_s=0.1)

    monkeypatch.setattr(engine_mod, "run_ecm_curves", fake_ecm)
    monkeypatch.setattr(engine_mod, "pollard_rho", lambda m: ([], m))
    monkeypatch.setattr(Engine, "_nfs", fake_nfs)
    monkeypatch.setattr(engine_mod, "run_siqs", fake_siqs)
    eng = Engine(RunContext(xover=1))
    res = eng.factor(N3, tmp_path, snfs=_job(N3))
    assert res.complete, res.error
    # the SNFS attempt on the attach number failed and was consumed; the
    # fallback then re-selected on the cofactor (nfs first at xover=1,
    # failing again, then SIQS)
    assert calls["nfs"] == [N3, QB * QC]
    assert calls["siqs"] == [QB * QC]


def test_ecm_pretest_sized_by_gnfs_equivalent(monkeypatch, tmp_path):
    """The ladder's target is priced by the difficulty's yafu GNFS
    equivalent (0.56 D + 30), not the raw digit count and not the
    difficulty itself: scaled 60 -> gnfs c63.6 -> t19.6 at normal plan."""
    log = tmp_path / "t.log"

    def fake_nfs(self, n, wd, snfs=None):
        return NfsResult(ok=True, factors=[PA, QB, QC])

    monkeypatch.setattr(Engine, "_nfs", fake_nfs)
    _patch_stdlib_fakes(monkeypatch)
    eng = Engine(RunContext(xover=1, log_file=log))
    res = eng.factor(N3, tmp_path, snfs=_job(N3, scaled=60.0))
    assert res.complete, res.error
    text = log.read_text()
    assert "snfs 60 ~ gnfs c63.6" in text
    assert "target=t19.6" in text
