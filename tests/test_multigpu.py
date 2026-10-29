"""Multi-GPU wiring: device resolution, subprocess pinning, NFS band
fan-out and the SIQS localhost cluster — all GPU-free."""

import os
import time

import pytest

from yagfu.config import RunContext, resolve_devices


# ------------------------------------------------------------- resolve / pin

def test_resolve_devices_no_ambient(monkeypatch):
    monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising=False)
    assert resolve_devices(("0", "1")) == ("0", "1")
    assert resolve_devices(("3", "3")) == ("3", "3")  # duplicates are legal


def test_resolve_devices_translates_ambient(monkeypatch):
    # --device ordinals mean what they mean to CUDA in this shell: indices
    # into CUDA_VISIBLE_DEVICES — the stored id must be the physical one
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "2,3")
    assert resolve_devices(("0", "1")) == ("2", "3")


def test_resolve_devices_uuid_entries(monkeypatch):
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "GPU-abc-123,1")
    assert resolve_devices(("0",)) == ("GPU-abc-123",)
    assert resolve_devices(("1",)) == ("1",)


def test_resolve_devices_out_of_range(monkeypatch):
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "2,3")
    with pytest.raises(ValueError, match="out of range"):
        resolve_devices(("2",))


def test_resolve_devices_rejects_non_int(monkeypatch):
    monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising=False)
    with pytest.raises(ValueError):
        resolve_devices(("x",))
    with pytest.raises(ValueError):
        resolve_devices(("-1",))


def test_resolve_devices_empty(monkeypatch):
    monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising=False)
    with pytest.raises(ValueError):
        resolve_devices(())


def test_resolve_devices_hidden_all(monkeypatch):
    # CUDA_VISIBLE_DEVICES="" means zero visible GPUs, not "unset"
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "")
    with pytest.raises(ValueError, match="hides every GPU"):
        resolve_devices(("0",))


def test_resolve_devices_all_ambient(monkeypatch):
    # "all" inside a restricted shell = the visible entries, in order
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "2,3")
    assert resolve_devices(("all",)) == ("2", "3")
    assert resolve_devices(("ALL",)) == ("2", "3")  # case-insensitive


def test_resolve_devices_all_enumerates(monkeypatch):
    import yagfu.config as config

    monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising=False)
    monkeypatch.setattr(config, "_list_gpu_ords", lambda: ["0", "1", "2", "3"])
    assert resolve_devices(("all",)) == ("0", "1", "2", "3")


def test_resolve_devices_all_no_gpus(monkeypatch):
    import yagfu.config as config

    monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising=False)
    monkeypatch.setattr(config, "_list_gpu_ords", lambda: [])
    with pytest.raises(ValueError, match="enumerate"):
        resolve_devices(("all",))


def test_resolve_devices_all_mixed_rejected(monkeypatch):
    monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising=False)
    with pytest.raises(ValueError, match="mixed"):
        resolve_devices(("0", "all"))
    with pytest.raises(ValueError, match="mixed"):
        resolve_devices(("all", "all"))


def test_pin_env():
    ctx = RunContext(device="3")
    assert ctx.pin_env() == {"CUDA_VISIBLE_DEVICES": "3"}
    assert ctx.pin_env("7") == {"CUDA_VISIBLE_DEVICES": "7"}


def test_verify_devices(monkeypatch):
    from yagfu import cli

    class R:
        def __init__(self, rc, out):
            self.returncode, self.stdout = rc, out

    ok = R(0, "GPU 0: RTX 4090 (UUID: GPU-aa11)\n"
              "GPU 1: RTX 4090 (UUID: GPU-bb22)\n")
    monkeypatch.setattr(cli.subprocess, "run", lambda cmd, **kw: ok)
    cli._verify_devices(("0", "1", "GPU-aa11", "GPU-bb22"))
    with pytest.raises(SystemExit):
        cli._verify_devices(("5",))
    with pytest.raises(SystemExit):
        cli._verify_devices(("2",))          # index beyond the last card

    # an unusable nvidia-smi must never block a run (GPU-less machines)
    monkeypatch.setattr(cli.subprocess, "run", lambda cmd, **kw: R(1, ""))
    cli._verify_devices(("9",))


# ------------------------------------------------------------ runner envs

def test_run_procs_parallel_envs(tmp_path):
    from yagfu.backends.runner import run_procs_parallel
    jobs = [(["bash", "-c", 'echo "$CUDA_VISIBLE_DEVICES"'],
             tmp_path / "a.log", None),
            (["bash", "-c", 'echo "$CUDA_VISIBLE_DEVICES"'],
             tmp_path / "b.log", None)]
    envs = [{"CUDA_VISIBLE_DEVICES": "3"}, {"CUDA_VISIBLE_DEVICES": "5"}]
    run_procs_parallel(jobs, tmp_path, 30, poll_s=0.05, envs=envs)
    assert (tmp_path / "a.log").read_text().strip() == "3"
    assert (tmp_path / "b.log").read_text().strip() == "5"


def test_run_procs_parallel_envs_none_inherits(tmp_path):
    from yagfu.backends.runner import run_procs_parallel
    ambient = os.environ.get("CUDA_VISIBLE_DEVICES", "")
    jobs = [(["bash", "-c", 'echo "$CUDA_VISIBLE_DEVICES"'],
             tmp_path / "a.log", None)]
    run_procs_parallel(jobs, tmp_path, 30, poll_s=0.05, envs=[None])
    assert (tmp_path / "a.log").read_text().strip() == ambient


# --------------------------------------------------------------- ECM pinning

def test_stage1_shards_pin_each_card(monkeypatch, tmp_path):
    import yagfu.backends.ecm as ecm
    ctx = RunContext(devices=("3", "5"))
    cap = {}

    def fake_rpp(jobs, cwd, timeout, **kw):
        cap["jobs"], cap["kw"] = jobs, kw
        return [(0, 0.1)] * len(jobs)

    monkeypatch.setattr(ecm, "run_procs_parallel", fake_rpp)
    monkeypatch.setattr(ecm, "find_bin", lambda name, bd: "/bin/true")
    shards = ecm._stage1_shards(101 * 103, 1000, [64, 64], ("3", "5"),
                                ctx, tmp_path, "t")
    assert len(shards) == 2
    # every process addresses its pinned card as ordinal 0...
    for cmd, *_ in cap["jobs"]:
        assert cmd[cmd.index("-gpudevice") + 1] == 0
    # ...and each is pinned to a different physical card
    assert cap["kw"]["envs"] == [{"CUDA_VISIBLE_DEVICES": "3"},
                                 {"CUDA_VISIBLE_DEVICES": "5"}]
    assert [s.res_path.name for s in shards] == \
        ["ecm-t.s0.residues", "ecm-t.s1.residues"]


def test_single_device_stage2_pinned(monkeypatch, tmp_path):
    """--device 1 with one card: gpuecm2 (no device flag of its own) must
    still be pinned instead of falling back to the ambient default card."""
    import yagfu.backends.ecm as ecm
    ctx = RunContext(device="1", devices=("1",))
    seen = {}

    def fake_stage1(n, b1, curves, ctx, wd, tag, device, stop_check=None):
        (wd / f"ecm-{tag}.residues").write_bytes(b"SIGMA=1\n")
        return 0, {}, 128, 0.1, None

    def fake_stage2(n, b1, res_path, ctx, wd, tag, maxmem, env_extra=None):
        seen["env"] = env_extra
        return {}, 0.1, None, 999_982

    monkeypatch.setattr(ecm, "_stage1", fake_stage1)
    monkeypatch.setattr(ecm, "_stage2", fake_stage2)
    out = ecm.run_ecm_curves(101 * 103, 11_000, 128, ctx, tmp_path, "t")
    assert not out.error and out.curves_done == 128
    assert seen["env"] == {"CUDA_VISIBLE_DEVICES": "1"}


# ------------------------------------------------------------- NFS band split

def _capture_rpp(monkeypatch, mod):
    cap = {}

    def fake_rpp(jobs, cwd, timeout, **kw):
        cap["jobs"], cap["kw"] = jobs, kw
        return [(0, 0.1)] * len(jobs)

    monkeypatch.setattr(mod, "run_procs_parallel", fake_rpp)
    return cap


def test_band_round_robin_devices(monkeypatch, tmp_path):
    import yagfu.backends.nfs as nfs
    cap = _capture_rpp(monkeypatch, nfs)
    base = ["/bin/cuda-sieve", "--pipeline", "--device", 0, (100_000, 109_000)]
    rc, wall, names = nfs._run_band_parallel(base, tmp_path, 2, 1, 30,
                                             ("3", "5", "7"))
    assert rc == 0 and len(names) == 6          # 2 procs x 3 cards
    assert [e["CUDA_VISIBLE_DEVICES"] for e in cap["kw"]["envs"]] == \
        ["3", "5", "7", "3", "5", "7"]          # adjacent chunks, different cards
    qr = [c[c.index("--qrange") + 1] for c, _, _ in cap["jobs"]]
    assert qr[0] == "100000:101500"
    assert qr[-1] == "107500:109000"
    for cmd, _, _ in cap["jobs"]:
        assert cmd[cmd.index("--device") + 1] == "0"


def test_band_width_caps_total_chunks(monkeypatch, tmp_path):
    import yagfu.backends.nfs as nfs
    cap = _capture_rpp(monkeypatch, nfs)
    base = ["/bin/cuda-sieve", "--device", 0, (100_000, 103_000)]  # width 3000
    nfs._run_band_parallel(base, tmp_path, 2, 1, 30, ("0", "1"))
    assert len(cap["jobs"]) == 2                # 1500 per chunk floor


def test_band_single_device_unchanged(monkeypatch, tmp_path):
    import yagfu.backends.nfs as nfs
    cap = _capture_rpp(monkeypatch, nfs)
    base = ["/bin/cuda-sieve", "--device", 0, (100_000, 130_000)]
    rc, _, names = nfs._run_band_parallel(base, tmp_path, 1, 1, 30, ("0",))
    assert rc == 0 and len(names) == 1
    assert cap["kw"]["envs"] == [{"CUDA_VISIBLE_DEVICES": "0"}]


# ------------------------------------------------------------- SIQS cluster

def _stub_bin(tmp_path, body):
    d = tmp_path / "bin"
    d.mkdir(exist_ok=True)
    p = d / "cuda-mpqs"
    p.write_text("#!/bin/bash\n" + body)
    p.chmod(0o755)
    return d


#: success on every path; echoes the pinned device so tests can assert it
_OK_BODY = """
echo "dev=$CUDA_VISIBLE_DEVICES"
if [[ "$*" == *"--cluster_mode coordinator"* ]]; then
  echo "[CUDA-MPQS]   Factor: 2"
  echo "[CUDA-MPQS]   Factor: 5"
  echo "[CUDA-MPQS] SUCCESS: Factorization Verified."
  exit 0
elif [[ "$*" == *"--cluster_mode worker"* ]]; then
  exit 1
fi
echo "[CUDA-MPQS]   Factor: 2"
echo "[CUDA-MPQS] SUCCESS: Factorization Verified."
exit 0
"""

#: coordinator fails (cluster as a whole fails) but the solo path succeeds
_FALLBACK_BODY = """
echo "dev=$CUDA_VISIBLE_DEVICES"
if [[ "$*" == *"--cluster_mode"* ]]; then
  echo "Error: cluster boom"
  exit 1
fi
echo "[CUDA-MPQS]   Factor: 2"
echo "[CUDA-MPQS] SUCCESS: Factorization Verified."
exit 0
"""


def test_cluster_specs():
    from yagfu.backends.siqs import _cluster_specs
    ctx = RunContext(device="0", devices=("0", "1", "2"))
    specs = _cluster_specs("/bin/true", 10 ** 79, ctx, 4242)
    assert len(specs) == 3
    coord = specs[0][0]
    assert coord[coord.index("--cluster_mode") + 1] == "coordinator"
    assert coord[coord.index("--listen_port") + 1] == 4242
    assert coord[coord.index("--expected_workers") + 1] == 2
    assert specs[0][1] == {"CUDA_VISIBLE_DEVICES": "0"}
    assert specs[0][3] == "cluster-coord.log"
    for j, (cmd, env, d, log) in enumerate(specs[1:], 1):
        assert cmd[cmd.index("--cluster_mode") + 1] == "worker"
        assert cmd[cmd.index("--coordinator_host") + 1] == "127.0.0.1"
        assert cmd[cmd.index("--coordinator_port") + 1] == 4242
        assert env == {"CUDA_VISIBLE_DEVICES": str(j)}
        assert d == f"work-w{j}" and log == f"cluster-w{j}.log"
    for cmd, *_ in specs:
        assert cmd[cmd.index("--device") + 1] == 0


def test_summarize():
    from yagfu.backends.siqs import _summarize
    log = ("[CUDA-MPQS] Verifying Product...\n"
           "[CUDA-MPQS]   Factor: 3\n"
           "[CUDA-MPQS] SUCCESS: Factorization Verified.\n")
    res = _summarize(log, 0, 1.0, 9)
    assert res.ok and res.factors == [3]
    res = _summarize(log, 1, 1.0, 9)
    assert not res.ok and res.error


def test_free_port():
    import socket
    from yagfu.backends.siqs import _free_port
    port = _free_port()
    assert 1024 <= port <= 65535
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.bind(("127.0.0.1", port))                 # must actually be free
    s.close()


def test_siqs_cluster_success(tmp_path):
    from yagfu.backends.siqs import run_siqs
    bin_dir = _stub_bin(tmp_path, _OK_BODY)
    ctx = RunContext(device="0", devices=("0", "1"), bin_dir=bin_dir,
                     quiet=True)
    wd = tmp_path / "wd"
    res = run_siqs(10 ** 79, ctx, wd)           # c80: cluster territory
    assert res.ok and res.factors == [2, 5]
    mpqs = wd / "mpqs"
    coord = (mpqs / "cluster-coord.log").read_text()
    assert "dev=0" in coord                      # coordinator pinned to card 0
    assert "dev=1" in (mpqs / "cluster-w1.log").read_text()
    assert not (mpqs / "run.log").exists()       # never went solo


def test_siqs_small_number_stays_solo(tmp_path):
    from yagfu.backends.siqs import run_siqs
    bin_dir = _stub_bin(tmp_path, _OK_BODY)
    ctx = RunContext(device="0", devices=("0", "1"), bin_dir=bin_dir,
                     quiet=True)
    wd = tmp_path / "wd"
    res = run_siqs(10 ** 60, ctx, wd)           # c61: below the threshold
    assert res.ok and res.factors == [2]
    mpqs = wd / "mpqs"
    assert "dev=0" in (mpqs / "run.log").read_text()
    assert not (mpqs / "cluster-coord.log").exists()


def test_siqs_single_device_solo_pinned(tmp_path):
    from yagfu.backends.siqs import run_siqs
    bin_dir = _stub_bin(tmp_path, _OK_BODY)
    ctx = RunContext(device="4", devices=("4",), bin_dir=bin_dir, quiet=True)
    wd = tmp_path / "wd"
    res = run_siqs(10 ** 79, ctx, wd)           # big, but only one card
    assert res.ok
    assert "dev=4" in (wd / "mpqs" / "run.log").read_text()


def test_siqs_cluster_falls_back_to_solo(tmp_path):
    from yagfu.backends.siqs import run_siqs
    bin_dir = _stub_bin(tmp_path, _FALLBACK_BODY)
    ctx = RunContext(device="0", devices=("0", "1"), bin_dir=bin_dir,
                     quiet=True)
    wd = tmp_path / "wd"
    res = run_siqs(10 ** 79, ctx, wd)
    assert res.ok and res.factors == [2]        # recovered via the solo run
    mpqs = wd / "mpqs"
    assert (mpqs / "cluster-coord.log").exists()
    assert "dev=0" in (mpqs / "run.log").read_text()


def test_cluster_procs_tear_down_workers(tmp_path, monkeypatch):
    import yagfu.backends.siqs as siqs
    monkeypatch.setattr(siqs, "CLUSTER_STOP_GRACE_S", 1.0)
    work = tmp_path / "w"
    work.mkdir()
    flag = tmp_path / "killed.flag"
    coord = ["bash", "-c", "sleep 0.3; exit 0"]
    # sleep in the background: a foreground sleep would defer bash's TERM
    # trap until it exits, and the point here is that TERM lands at once
    worker = ["bash", "-c",
              f'trap "echo x > {flag}; exit 3" TERM; sleep 30 & wait']
    specs = [(coord, {}, "d0", "c.log"), (worker, {}, "d1", "w.log")]
    t0 = time.monotonic()
    rc = siqs._run_cluster_procs(work, specs, 30)
    assert rc == 0                               # the coordinator's rc
    assert flag.exists()                         # straggler worker was killed
    assert time.monotonic() - t0 < 10


def test_cluster_procs_timeout_kills_all(tmp_path, monkeypatch):
    import yagfu.backends.siqs as siqs
    work = tmp_path / "w"
    work.mkdir()
    flag = tmp_path / "killed.flag"
    specs = [(["bash", "-c", f'trap "echo x > {flag}" TERM; sleep 30 & wait'],
              {}, "d0", "c.log")]
    rc = siqs._run_cluster_procs(work, specs, 1.0)
    assert rc == -9
    assert flag.exists()


def test_stdout_capture_is_per_attempt(tmp_path):
    import yagfu.backends.nfs as nfs
    first = nfs._attempt_stdout(tmp_path, "w0019.0.stdout")
    assert first == tmp_path / "w0019.0.stdout"
    first.write_text("first attempt's error lines")    # must survive a relaunch
    assert nfs._attempt_stdout(tmp_path, "w0019.0.stdout") == \
        tmp_path / "w0019.0.stdout.attempt2"
    (tmp_path / "w0019.0.stdout.attempt2").write_text("x")
    assert nfs._attempt_stdout(tmp_path, "w0019.0.stdout") == \
        tmp_path / "w0019.0.stdout.attempt3"
