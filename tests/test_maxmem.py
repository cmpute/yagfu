from yagfu.backends.ecm import (
    DEFAULT_MAXMEM_FRAC,
    gpu_total_mem_mb,
    resolve_maxmem,
)



def test_resolve_explicit_wins(monkeypatch):
    monkeypatch.setattr("yagfu.backends.ecm.gpu_total_mem_mb", lambda d: 24564)
    assert resolve_maxmem(4096, "0") == 4096
    assert resolve_maxmem(1, "0") == 1


def test_resolve_zero_disables(monkeypatch):
    monkeypatch.setattr("yagfu.backends.ecm.gpu_total_mem_mb", lambda d: 24564)
    assert resolve_maxmem(0, "0") is None


def test_resolve_default_90pct(monkeypatch):
    monkeypatch.setattr("yagfu.backends.ecm.gpu_total_mem_mb", lambda d: 24564)
    assert resolve_maxmem(None, "0") == int(24564 * DEFAULT_MAXMEM_FRAC)
    assert DEFAULT_MAXMEM_FRAC == 0.9


def test_resolve_unknown_capacity(monkeypatch):
    monkeypatch.setattr("yagfu.backends.ecm.gpu_total_mem_mb", lambda d: None)
    assert resolve_maxmem(None, "0") is None


def test_gpu_total_mem_parses_nvidia_smi(monkeypatch):
    class Fake:
        stdout = "24564\n"
    monkeypatch.setattr("yagfu.backends.ecm.subprocess.run",
                        lambda *a, **k: Fake())
    import yagfu.backends.ecm as m
    monkeypatch.setattr(m, "_GPU_MEM_CACHE", {})
    assert gpu_total_mem_mb("9") == 24564
    assert m._GPU_MEM_CACHE["9"] == 24564


def test_gpu_total_mem_handles_failure(monkeypatch):
    def boom(*a, **k):
        raise OSError("no nvidia-smi")
    import yagfu.backends.ecm as m
    monkeypatch.setattr(m, "_GPU_MEM_CACHE", {})
    monkeypatch.setattr(m.subprocess, "run", boom)
    assert gpu_total_mem_mb("3") is None
