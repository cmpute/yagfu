"""Engine routing tests: SIQS-vs-NFS is decided after the ECM ladder, on
the post-ECM cofactor's size — nothing starts a polyselect before that
decision.  All fakes: no GPU, no msieve."""

from yagfu import engine as engine_mod
from yagfu.backends.ecm import EcmProvenance, EcmRunResult
from yagfu.backends.siqs import SiqsResult
from yagfu.config import RunContext
from yagfu.engine import Engine
from yagfu.primes import is_probable_prime as _ipp


def _prime(digits: int) -> int:
    n = 10 ** (digits - 1) + 27
    while not _ipp(n):
        n += 2
    return n


PA = _prime(35)           # the factor ECM "finds" in the first ladder call
QS1, QS2 = _prime(30), _prime(31)   # 96-digit N -> c61 cofactor (< xover)
QN1, QN2 = _prime(48), _prime(47)   # 130-digit N -> c94 cofactor (>= xover)


def _patch_routing(monkeypatch, siqs_factors, nfs_factors):
    """Fake ECM finds PA on the first invocation (mid-ladder split); the
    fake sieves return the given factors.  PolyselectJob is replaced by a
    tripwire: constructing one before the sieve stage is a routing bug."""
    from yagfu.backends import nfs as nfs_mod

    calls = {"ecm": [], "siqs": [], "nfs": [], "poly": []}

    def tripwire_poly(*a, **k):
        calls["poly"].append(a[0] if a else None)
        raise AssertionError("polyselect must not start before the "
                             "post-ECM routing decision")

    def fake_ecm(n, b1, total, ctx, wd, tag):
        calls["ecm"].append((b1, n))
        out = EcmRunResult(curves_done=total, wall_s=0.001 * total)
        if len(calls["ecm"]) == 1:
            out.factors[PA] = EcmProvenance(b1=b1, stage=1, sigma=1)
        return out

    def fake_siqs(m, ctx, wd):
        calls["siqs"].append(m)
        return SiqsResult(factors=list(siqs_factors), ok=True, rc=0,
                          wall_s=0.1)

    def fake_nfs(self, n, wd):
        calls["nfs"].append(n)
        from yagfu.backends.nfs import NfsResult
        return NfsResult(ok=True, factors=list(nfs_factors))

    monkeypatch.setattr(nfs_mod, "PolyselectJob", tripwire_poly)
    monkeypatch.setattr(engine_mod, "run_ecm_curves", fake_ecm)
    monkeypatch.setattr(engine_mod, "pollard_rho", lambda m: ([], m))
    monkeypatch.setattr(engine_mod, "run_siqs", fake_siqs)
    monkeypatch.setattr(Engine, "_nfs", fake_nfs)
    return calls


def test_ladder_split_routes_cofactor_to_siqs(monkeypatch, tmp_path):
    """Mid-ladder split with a below-xover cofactor: no polyselect ever
    starts, the cofactor is routed to SIQS by its own size, and the ladder
    keeps climbing on the cofactor to the original target."""
    calls = _patch_routing(monkeypatch, (QS1, QS2), ())
    eng = Engine(RunContext(xover=90, device="0", devices=("0", "1")))
    res = eng.factor(PA * QS1 * QS2, tmp_path)
    assert res.complete, res.error
    assert calls["poly"] == []
    assert calls["siqs"] == [QS1 * QS2]         # routed by cofactor size
    assert calls["nfs"] == []
    # the ladder itself keeps climbing the original target after the split
    assert len(calls["ecm"]) >= 2
    assert calls["ecm"][0][1] == PA * QS1 * QS2
    assert calls["ecm"][1][1] == QS1 * QS2


def test_ladder_split_routes_nfs_cofactor_to_nfs(monkeypatch, tmp_path):
    """Mid-ladder split with an above-xover cofactor: NFS receives exactly
    the cofactor — its polyselect runs inside run_nfs, after the routing."""
    calls = _patch_routing(monkeypatch, (), (QN1, QN2))
    eng = Engine(RunContext(xover=90, device="0", devices=("0", "1")))
    res = eng.factor(PA * QN1 * QN2, tmp_path)
    assert res.complete, res.error
    assert calls["poly"] == []
    assert calls["nfs"] == [QN1 * QN2]
    assert calls["siqs"] == []
