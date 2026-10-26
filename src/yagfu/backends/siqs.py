"""cuda-mpqs backend: whole SIQS/MPQS pipeline in one GPU process.

With several devices (and a number large enough to be worth the setup),
the backend instead runs cuda-mpqs's own cluster mode on localhost TCP:
a coordinator on the first card plus one worker per remaining card.  The
coordinator sieves, runs the matrix/linalg/sqrt stages and prints the
result; workers only sieve and stream relations through host memory
(no GPU peer-to-peer / NVLink path is involved anywhere).
"""

from __future__ import annotations

import os
import re
import socket
import subprocess
import time
from dataclasses import dataclass, field
from pathlib import Path

from ..config import RunContext
from ..params import mpqs_lp1_bound
from . import find_bin
from .runner import _term_kill_ladder, read_log, run_cmd

MPQS_TIMEOUT_S = 6 * 3600

#: below this size a localhost cluster's handshake and setup cost more
#: than the sieving itself — solo on the first card finishes sooner
CLUSTER_MIN_DIGITS = 80

#: how long workers get to self-stop on the coordinator's STOP barrier
#: before they are torn down
CLUSTER_STOP_GRACE_S = 60.0

_RE_FACTOR = re.compile(r"Factor:\s+(\d+)")
_RE_SUCCESS = "SUCCESS: Factorization Verified"

#: cuda-mpqs 1.0.8's Small-N auto-config derives double the legal
#: numIntervals and dies with this validation error before sieving.
#: The message itself states the required value; the retry rebuilds the
#: tuple in the shape the internal Small-N path would have used
#: (totalPolys, intervals, polyBlockSize=1, blocksPerCycle=intervals,
#:  metaB=totalPolys, metaT=256, sasB=min(256,totalPolys), sasT=256).
_RE_NARROW_BUG = re.compile(r"NARROW interval coverage must equal 2M")
_RE_NEED_INTERVALS = re.compile(r"Required: numIntervals\s*=\s*(\d+)")
_RE_TOTAL_POLYS = re.compile(r"Small-N adaptive config:.*totalPolys=(\d+)")


@dataclass
class SiqsResult:
    factors: list[int] = field(default_factory=list)
    ok: bool = False
    wall_s: float = 0.0
    rc: int = -1
    error: str | None = None
    internal: dict = field(default_factory=dict)


def _narrow_bug_params(log: str) -> str | None:
    """--params tuple fixing the Small-N interval bug, or None."""
    m_i = _RE_NEED_INTERVALS.search(log)
    m_p = _RE_TOTAL_POLYS.search(log)
    if not (m_i and m_p):
        return None
    intervals, total_polys = int(m_i.group(1)), int(m_p.group(1))
    if not intervals or not total_polys:
        return None
    sas_b = min(256, total_polys)
    return f"{total_polys},{intervals},1,{intervals},{total_polys},256,{sas_b},256"


def _summarize(log: str, rc: int, wall: float, n: int) -> SiqsResult:
    """Verified factors + internals scraped from a cuda-mpqs log."""
    factors = sorted(int(m) for m in _RE_FACTOR.findall(log))
    ok = (rc == 0 and _RE_SUCCESS in log and factors
          and all(n % f == 0 for f in factors))

    internal: dict = {}
    for key, pat in (
        ("sieve_s", r"Duration:\s+([\d.]+)ms"),
        ("matrix_s", r"Construction Completed in ([\d.]+)ms"),
        ("linalg_s", r"Total \(All\)\s*:\s*([\d.]+)ms"),
        ("pipeline_total_s", r"Total:\s+([\d.]+)\s*s"),
    ):
        m = re.search(pat, log)
        if m:
            internal[key] = float(m.group(1)) / (1000 if key != "pipeline_total_s" else 1)
    m = re.search(r"Selected F=(\d+), M=(\d+)", log)
    if m:
        internal["F"], internal["M"] = int(m.group(1)), int(m.group(2))

    err = None
    if not ok:
        err = f"cuda-mpqs rc={rc}, {len(factors)} factor line(s)"
        if "Error" in log or "ERROR" in log:
            for line in log.splitlines():
                if "rror" in line:
                    err += f" | {line.strip()[:200]}"
                    break
    return SiqsResult(factors=factors, ok=ok, wall_s=wall, rc=rc, error=err,
                      internal=internal)


def _run_solo(n: int, ctx: RunContext, binary, work: Path,
              digits: int) -> SiqsResult:
    """Classic single-process cuda-mpqs run on the first device."""
    base_cmd = [binary, "--N", n, "--device", 0, "--dir", "work",
                "--verbose"]
    lp1 = mpqs_lp1_bound(digits)
    if lp1:
        base_cmd += ["--lp1_bound", lp1]
    pin = ctx.pin_env(ctx.device)
    log_path = work / "run.log"
    ctx.log(f"siqs: cuda-mpqs on c{digits} (lp1={lp1})")
    t0 = time.monotonic()
    try:
        rc, wall = run_cmd(base_cmd, work, log_path, env_extra=pin,
                           timeout=MPQS_TIMEOUT_S)
    except Exception as e:  # noqa: BLE001
        return SiqsResult(error=f"cuda-mpqs failed to run: {e!r}",
                          wall_s=time.monotonic() - t0)

    # retry once with a corrected pinned tuple on the Small-N bug
    log = read_log(log_path)
    if rc != 0 and _RE_NARROW_BUG.search(log):
        fixed = _narrow_bug_params(log)
        if fixed:
            ctx.log(f"siqs: cuda-mpqs Small-N interval bug, retrying with "
                    f"--params {fixed}")
            t1 = time.monotonic()
            try:
                rc, w2 = run_cmd(base_cmd + ["--params", fixed],
                                 work, work / "run2.log", env_extra=pin,
                                 timeout=MPQS_TIMEOUT_S)
            except Exception as e:  # noqa: BLE001
                return SiqsResult(error=f"cuda-mpqs retry failed: {e!r}",
                                  wall_s=time.monotonic() - t0)
            wall += time.monotonic() - t1
            log = read_log(work / "run2.log")
    return _summarize(log, rc, wall, n)


# ------------------------------------------------------------------- cluster

def _free_port() -> int:
    """An unused localhost TCP port for the cluster to meet on."""
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.bind(("127.0.0.1", 0))
    port = s.getsockname()[1]
    s.close()
    return port


def _cluster_specs(binary, n: int, ctx: RunContext, port: int):
    """(cmd, pin_env, dir_name, log_name) per process: the coordinator on
    devices[0] plus one worker per remaining card.

    Every process is pinned to its card via CUDA_VISIBLE_DEVICES and
    addresses it as ordinal 0.  A worker's normal exit code is 1 ("no
    factors" — it never runs the sqrt): only the coordinator's rc and
    log decide the outcome.
    """
    common = [binary, "--N", n, "--device", 0, "--verbose"]
    lp1 = mpqs_lp1_bound(len(str(n)))
    if lp1:
        common += ["--lp1_bound", lp1]
    specs = [(common + ["--cluster_mode", "coordinator",
                        "--listen_port", port,
                        "--expected_workers", len(ctx.devices) - 1,
                        "--dir", "work-c"],
              ctx.pin_env(ctx.devices[0]), "work-c", "cluster-coord.log")]
    for i, dev in enumerate(ctx.devices[1:], 1):
        specs.append((common + ["--cluster_mode", "worker",
                                "--coordinator_host", "127.0.0.1",
                                "--coordinator_port", port,
                                "--dir", f"work-w{i}"],
                      ctx.pin_env(dev), f"work-w{i}", f"cluster-w{i}.log"))
    return specs


def _kill_all(procs) -> None:
    for p in procs:
        if p.poll() is None:
            p.terminate()
    for p in procs:
        _term_kill_ladder(p)


def _run_cluster_procs(work: Path, specs, timeout: float) -> int:
    """Run the cluster until the coordinator exits; return its rc.

    Workers are expected to self-stop on the coordinator's STOP barrier;
    anything still alive after the grace window is torn down.  Ctrl+C
    and the overall timeout never leave a process behind.
    """
    t0 = time.monotonic()
    procs = []
    try:
        for cmd, pin, dir_name, log_name in specs:
            (work / dir_name).mkdir(parents=True, exist_ok=True)
            f = open(work / log_name, "wb")
            try:
                procs.append(subprocess.Popen(
                    [str(c) for c in cmd], cwd=str(work),
                    stdout=f, stderr=subprocess.STDOUT,
                    env={**os.environ, **pin}))
            finally:
                f.close()
        coord = procs[0]
        while coord.poll() is None:
            if time.monotonic() - t0 > timeout:
                _kill_all(procs)
                return -9
            time.sleep(2.0)
        grace_until = time.monotonic() + CLUSTER_STOP_GRACE_S
        while time.monotonic() < grace_until \
                and any(p.poll() is None for p in procs[1:]):
            time.sleep(2.0)
        _kill_all(procs)
        return coord.returncode
    except BaseException:  # incl. KeyboardInterrupt: never orphan workers
        _kill_all(procs)
        raise


def _run_cluster(n: int, ctx: RunContext, binary, work: Path,
                 digits: int) -> SiqsResult | None:
    """One localhost cluster run; None = it failed (caller falls back)."""
    for stale in work.glob("cluster-*.log"):
        # a retry in the same workdir must not read the previous attempt's log
        stale.unlink(missing_ok=True)
    port = _free_port()
    specs = _cluster_specs(binary, n, ctx, port)
    ctx.log(f"siqs: cuda-mpqs cluster on c{digits} "
            f"({len(ctx.devices)} GPUs, localhost:{port}, "
            f"lp1={mpqs_lp1_bound(digits)})")
    t0 = time.monotonic()
    try:
        rc = _run_cluster_procs(work, specs, MPQS_TIMEOUT_S)
    except Exception as e:  # noqa: BLE001
        ctx.log(f"siqs: cluster failed to run: {e!r}")
        return None
    res = _summarize(read_log(work / "cluster-coord.log"), rc,
                     time.monotonic() - t0, n)
    if not res.ok:
        ctx.log(f"siqs: cluster failed ({res.error})")
        return None
    return res


def run_siqs(n: int, ctx: RunContext, wd: Path) -> SiqsResult:
    """Run cuda-mpqs on n inside wd; returns verified prime factors."""
    binary = find_bin("cuda-mpqs", ctx.bin_dir)
    work = wd / "mpqs"
    work.mkdir(parents=True, exist_ok=True)
    digits = len(str(n))

    if len(ctx.devices) > 1 and digits >= CLUSTER_MIN_DIGITS:
        res = _run_cluster(n, ctx, binary, work, digits)
        if res is not None:
            return res
        ctx.log(f"siqs: retrying solo on device {ctx.devices[0]}")
    return _run_solo(n, ctx, binary, work, digits)
