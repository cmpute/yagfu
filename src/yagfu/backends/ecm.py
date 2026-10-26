"""GPU ECM backend: gmp-ecm GPU stage 1 + gpuecm2 GPU stage 2.

Validated contract (RTX 4090, gmp-ecm 7.0.6 --enable-gpu + gpuecm2 0.1.0):

* stage 1:  ``echo N | ecm -gpu -gpudevice D [-gpucurves C] -save res B1 1``
  — B2=1 skips CPU stage 2 entirely; ~0.3 s fixed overhead.  A full run
  computes one GPU batch of curves; the default batch is card-dependent
  (8192 on an RTX 4090), ``-gpucurves`` is rounded up to the card's
  granularity (128 there).  If any curve finds a factor, the run ends and
  the residue file is EMPTY; otherwise it holds one line per curve with
  ``SIGMA=`` recorded.
* stage 2:  ``gpuecm2 -v -resume res B1`` — GMP-ECM's default B2, batches
  residues automatically; with -v every curve prints a
  ``Using B1=..., B2=..., sigma=3:X`` header immediately before its
  result, so a factor line's sigma is the most recent header's.
"""

from __future__ import annotations

import re
import subprocess
import time
from dataclasses import dataclass, field
from pathlib import Path

from ..config import RunContext
from ..fmt import fmt_count, fmt_dur
from . import find_bin
from .runner import (
    KILLED_RC,
    read_log,
    run_cmd,
    run_cmd_progress,
    run_procs_parallel,
)

#: gpuecm2 default build limit
GPUECM2_MAX_BITS = 8192

#: gmp-ecm rounds -gpucurves up to the per-SM multiple; this bound keeps us
#: close to the requested count while staying a valid multiple everywhere
_CURVES_GRANULARITY = 128

_RE_CURVES_HDR = re.compile(r"Using B1=.*\((\d+) curves\)")
_RE_COMPUTING = re.compile(r"Computing (\d+) Step 1")
_RE_STEP1_FACTOR = re.compile(
    r"GPU: factor (\d+) found in Step 1 with curve \d+ \(-sigma (\d+):(\d+)\)")
_RE_STEP2_HDR = re.compile(
    r"Using B1=([\d.]+)(?:-[\d.]+)?, B2=(\d+)(?:-\d+)?, polynomial [^,]+(?:, sigma=(\d+):(\d+))?")
_RE_STEP2_FACTOR = re.compile(r"Factor found in step (\d+): (\d+)")
_RE_UNSUPPORTED = re.compile(
    r"input number should be stricly lower|Bits exceeded|exceeds the .* limit|"
    r"no kernel|kernel size")


class EcmUnsupported(Exception):
    """The number cannot run on the GPU ECM pipeline (too large for kernels)."""


# --------------------------------------------------------------------- maxmem

#: fraction of total VRAM granted to gpuecm2 stage 2 by default
DEFAULT_MAXMEM_FRAC = 0.9

_GPU_MEM_CACHE: dict[str, int | None] = {}


def gpu_total_mem_mb(device: str = "0") -> int | None:
    """Total VRAM of `device` in MB via nvidia-smi (cached; None if unknown)."""
    if device in _GPU_MEM_CACHE:
        return _GPU_MEM_CACHE[device]
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.total",
             "--format=csv,noheader,nounits", "-i", str(device)],
            capture_output=True, text=True, timeout=30)
        mb = int(out.stdout.strip().splitlines()[-1])
    except Exception:  # noqa: BLE001
        mb = None
    _GPU_MEM_CACHE[device] = mb
    return mb


def resolve_maxmem(explicit: int | None, device: str = "0") -> int | None:
    """gpuecm2 -maxmem value in MB.

    explicit wins (0 disables the cap); otherwise 90% of the card's total
    VRAM; None if the capacity cannot be determined (gpuecm2 then falls
    back to its own free-memory budgeting).
    """
    if explicit is not None:
        return explicit or None
    total = gpu_total_mem_mb(device)
    if not total:
        return None
    return int(total * DEFAULT_MAXMEM_FRAC)


@dataclass
class EcmProvenance:
    """How an ECM factor was found — kept for factordb report_ecm."""
    b1: int
    stage: int                 # 1 or 2
    sigma: int | None = None   # param-3 sigma
    b2: int | None = None      # effective default B2 (stage 2 only)


@dataclass
class EcmRunResult:
    factors: dict[int, EcmProvenance] = field(default_factory=dict)
    curves_done: int = 0
    wall_s: float = 0.0
    stage1_s: float = 0.0
    stage2_s: float = 0.0
    b2: int | None = None            # effective stage-2 B2 (GMP-ECM default)
    aborted: bool = False            # stopped early on request (not an error)
    error: str | None = None


def _parse_stage1_log(log: str, b1: int, requested: int
                      ) -> tuple[dict[int, EcmProvenance], int]:
    """(factors, curves actually run) from an ``ecm -gpu`` log."""
    if _RE_UNSUPPORTED.search(log):
        raise EcmUnsupported(log.strip().splitlines()[-1])
    factors: dict[int, EcmProvenance] = {}
    for m in _RE_STEP1_FACTOR.finditer(log):
        f, _param, sigma = int(m.group(1)), m.group(2), int(m.group(3))
        factors.setdefault(f, EcmProvenance(b1=b1, stage=1, sigma=sigma))
    m = _RE_COMPUTING.search(log) or _RE_CURVES_HDR.search(log)
    ran = int(m.group(1)) if m else requested
    return factors, ran


def _stage1(n: int, b1: int, curves: int, ctx: RunContext, wd: Path,
            tag: str, device: str, stop_check=None
            ) -> tuple[int, dict[int, EcmProvenance], int, float, str | None]:
    """One GPU stage-1 invocation; returns (rc, factors, curves, wall, error).

    `device` is a physical id: the process is pinned to it via
    CUDA_VISIBLE_DEVICES and addresses it as ordinal 0.

    With stop_check, the process is polled and killed when it fires; a
    killed batch never wrote its residues, so it reports ran=0 (any factor
    lines already printed are still kept — a free factor is a free factor).
    """
    ecm = find_bin("ecm", ctx.bin_dir)
    res_path = wd / f"ecm-{tag}.residues"
    log_path = wd / f"ecm-{tag}.stage1.log"
    cmd = [ecm, "-gpu", "-gpudevice", 0, "-save", res_path.name,
           "-gpucurves", curves, str(b1), "1"]
    pin = ctx.pin_env(device)
    t0 = time.monotonic()
    try:
        if stop_check is None:
            rc, _ = run_cmd(cmd, wd, log_path, env_extra=pin,
                            input_bytes=f"{n}\n".encode(),
                            timeout=max(600, 4 * 86400))
        else:
            rc, _ = run_cmd_progress(
                cmd, wd, log_path, timeout=max(600, 4 * 86400),
                every_s=1.0, input_bytes=f"{n}\n".encode(),
                env_extra=pin, stop=stop_check)
    except Exception as e:  # noqa: BLE001
        return -1, {}, 0, time.monotonic() - t0, f"ecm failed to run: {e!r}"
    wall = time.monotonic() - t0
    factors, ran = _parse_stage1_log(read_log(log_path), b1, curves)
    if rc == KILLED_RC:
        ran = 0
    return rc, factors, ran, wall, None


@dataclass
class _Shard:
    """One stage-1 device shard of a round."""
    device: str
    factors: dict[int, EcmProvenance]
    ran: int
    wall_s: float
    res_path: Path


def _split_curves(total: int, weights: list[float]) -> list[int]:
    """Split `total` across shards proportionally to `weights`
    (largest-remainder method; each shard gets an exact integer share)."""
    if total <= 0 or not weights:
        return [0] * len(weights)
    w_sum = sum(weights)
    if w_sum <= 0:
        weights = [1.0] * len(weights)
        w_sum = float(len(weights))
    raw = [total * w / w_sum for w in weights]
    counts = [int(x) for x in raw]
    for i in sorted(range(len(weights)),
                    key=lambda i: raw[i] - counts[i], reverse=True)[:total - sum(counts)]:
        counts[i] += 1
    return counts


def _bump_rates(ctx: RunContext, shards: list[_Shard]) -> None:
    """EWMA of per-device curves/sec — next round's split weights."""
    for s in shards:
        if s.ran > 0 and s.wall_s > 0:
            inst = s.ran / s.wall_s
            old = ctx.ecm_device_rates.get(s.device)
            ctx.ecm_device_rates[s.device] = \
                inst if old is None else 0.7 * old + 0.3 * inst


def _stage1_shards(n: int, b1: int, counts: list[int], devices: tuple,
                   ctx: RunContext, wd: Path, tag: str,
                   stop_check=None) -> list[_Shard]:
    """One stage-1 process per device with a nonzero count, all concurrent.

    Counts are requested as-is: gmp-ecm rounds each process's -gpucurves
    up to that card's own granularity, and the parsed per-log `ran` keeps
    the accounting exact on heterogeneous cards.  With stop_check, all
    shards are killed when it fires and report ran=0 (their factors, if
    any, are kept)."""
    ecm = find_bin("ecm", ctx.bin_dir)
    jobs, envs, metas = [], [], []
    for i, (dev, cnt) in enumerate(zip(devices, counts)):
        if cnt <= 0:
            continue
        stag = f"{tag}.s{i}"          # index, not device: --device 0,0 is legal
        res_path = wd / f"ecm-{stag}.residues"
        cmd = [ecm, "-gpu", "-gpudevice", 0, "-save", res_path.name,
               "-gpucurves", cnt, str(b1), "1"]
        jobs.append((cmd, wd / f"ecm-{stag}.stage1.log", f"{n}\n".encode()))
        envs.append({"CUDA_VISIBLE_DEVICES": str(dev)})   # dev is physical
        metas.append((i, str(dev), cnt, res_path))
    results = run_procs_parallel(jobs, wd, max(600, 4 * 86400),
                                 poll_s=1.0, stop=stop_check, envs=envs)
    shards = []
    for (rc, wall), (i, dev, cnt, res_path) in zip(results, metas):
        try:
            log = read_log(wd / f"ecm-{tag}.s{i}.stage1.log")
        except OSError:
            log = ""
        factors, ran = _parse_stage1_log(log, b1, cnt)
        if rc == KILLED_RC:
            ran = 0
        shards.append(_Shard(dev, factors, ran, wall, res_path))
        ctx.logv(f"ecm: shard {i} (device {dev}): {ran:,} curves in "
                 f"{fmt_dur(wall)} "
                 f"({fmt_count(ran / wall) if wall else 0}/s)")
    return shards


def _merge_residues(paths: list[Path], out_path: Path) -> int:
    """Concatenate the nonempty residue files into one for a single stage-2
    run (gpuecm2 batches internally; bigger file = better batching).
    Returns the merged line count."""
    total = 0
    with open(out_path, "wb") as out:
        for p in paths:
            try:
                data = p.read_bytes()
            except OSError:
                continue
            if data:
                out.write(data)
                total += data.count(b"\n")
    return total


def parse_stage2_log(log: str, b1: int, n: int
                     ) -> tuple[dict[int, EcmProvenance], int | None]:
    """(factors, effective B2) from a gpuecm2 -v log for n at B1.

    Every curve prints a ``Using B1=..., B2=..., sigma=3:X`` header before
    its result, so a factor's sigma is the most recent header's.  The first
    header's B2 is the effective default B2 for this B1 (all curves share
    it unless -maxmem forced different parameters).
    """
    factors: dict[int, EcmProvenance] = {}
    sigma = b2 = None
    b2_used = None
    for line in log.splitlines():
        m = _RE_STEP2_HDR.search(line)
        if m:
            b2 = int(m.group(2))
            if b2_used is None:
                b2_used = b2
            sigma = int(m.group(4)) if m.group(4) else None
            continue
        m = _RE_STEP2_FACTOR.search(line)
        if m:
            f = int(m.group(2))
            if f != n:  # "found the input number" is not a factorization
                factors.setdefault(f, EcmProvenance(
                    b1=b1, stage=int(m.group(1)), sigma=sigma, b2=b2))
    return factors, b2_used


def _stage2(n: int, b1: int, res_path: Path, ctx: RunContext, wd: Path,
            tag: str, maxmem: int | None, env_extra: dict | None = None
            ) -> tuple[dict[int, EcmProvenance], float, str | None, int | None]:
    """gpuecm2 on the saved residues; returns (factors, wall, error, b2)."""
    g2 = find_bin("gpuecm2", ctx.bin_dir)
    log_path = wd / f"ecm-{tag}.stage2.log"
    cmd = [g2, "-v"]
    if maxmem:
        cmd += ["-maxmem", maxmem]
    cmd += ["-resume", res_path.name, str(b1)]
    t0 = time.monotonic()
    try:
        rc, _ = run_cmd(cmd, wd, log_path, timeout=max(600, 4 * 86400),
                        env_extra=env_extra)
    except Exception as e:  # noqa: BLE001
        return {}, time.monotonic() - t0, f"gpuecm2 failed to run: {e!r}", None
    wall = time.monotonic() - t0
    log = read_log(log_path)

    if _RE_UNSUPPORTED.search(log):
        raise EcmUnsupported(log.strip().splitlines()[-1])

    factors, b2 = parse_stage2_log(log, b1, n)
    err = None if rc in (0, 2, 6, 8, 10, 14) else f"gpuecm2 rc={rc}"
    return factors, wall, err, b2


def run_ecm_curves(n: int, b1: int, total_curves: int, ctx: RunContext,
                   wd: Path, tag: str, stop_check=None,
                   devices: tuple | None = None) -> EcmRunResult:
    """Run up to total_curves curves at B1 in GPU batches; stop at first
    round that finds a factor (subsequent curves are pointless then).

    With one device this is the classic single-process path.  With several
    devices each round fans stage 1 out one process per card (counts split
    by learned per-card throughput), the residues are merged, and a
    single gpuecm2 stage 2 runs on the first device.  `devices` overrides
    ctx.devices (the polyselect overlap passes the cards the polyselect is
    not using).

    stop_check () -> bool is polled while stage 1 runs; when it fires the
    running stage-1 batch is killed and discarded (aborted=True, its
    curves count as 0) — used by the polyselect overlap so bonus ECM
    yields instantly when the background polyselect finishes.  A stage 2
    that already has residues always runs to completion so its round's
    work is kept.
    """
    if n.bit_length() > GPUECM2_MAX_BITS:
        raise EcmUnsupported(f"{n.bit_length()} bits > gpuecm2 limit of {GPUECM2_MAX_BITS}")

    devices = ctx.devices if devices is None else devices
    single = len(devices) == 1
    s2_dev = devices[0]
    maxmem = resolve_maxmem(ctx.ecm_stage2_maxmem_mb, s2_dev)
    if maxmem:
        ctx.logv(f"ecm: gpuecm2 -maxmem {maxmem} MB"
                 + ("" if ctx.ecm_stage2_maxmem_mb else " (90% of VRAM)"))
    out = EcmRunResult()
    done = 0
    inv = 0
    while done < total_curves and not out.factors:
        if stop_check is not None and stop_check():
            out.aborted = True
            return out
        inv += 1
        remaining = total_curves - done
        round_ran = 0
        if single:
            want = min(remaining, ctx.ecm_maxcurves)
            want = min(((want + _CURVES_GRANULARITY - 1) // _CURVES_GRANULARITY)
                       * _CURVES_GRANULARITY, ctx.ecm_maxcurves)
            rc, facts, ran, wall, err = _stage1(n, b1, want, ctx, wd,
                                                f"{tag}.{inv}", s2_dev,
                                                stop_check=stop_check)
            out.stage1_s += wall
            out.curves_done += ran
            round_ran = ran
            out.factors.update(facts)  # first provenance wins
            if err:
                out.error = err
                return out
            if out.factors:
                # keep only factors of n (paranoia: verify everything)
                out.factors = {f: p for f, p in out.factors.items() if n % f == 0}
                return out
            res_paths = [wd / f"ecm-{tag}.{inv}.residues"]
            round_wall = wall
        else:
            weights = [ctx.ecm_device_rates.get(d) or 1.0 for d in devices]
            counts = [min(c, ctx.ecm_maxcurves)
                      for c in _split_curves(remaining, weights)]
            shards = _stage1_shards(n, b1, counts, devices, ctx, wd,
                                    f"{tag}.{inv}", stop_check=stop_check)
            if not shards:
                return out
            _bump_rates(ctx, shards)
            round_ran = sum(s.ran for s in shards)
            out.curves_done += round_ran
            round_wall = max(s.wall_s for s in shards)
            out.stage1_s += round_wall
            for s in shards:
                out.factors.update(s.factors)
            if out.factors:
                out.factors = {f: p for f, p in out.factors.items() if n % f == 0}
                return out
            res_paths = [s.res_path for s in shards]
        merged = wd / f"ecm-{tag}.{inv}.merged"
        n_res = _merge_residues(res_paths, merged)
        if n_res == 0:
            if stop_check is not None and stop_check():
                out.aborted = True       # killed batch: discard its curves
                out.curves_done -= round_ran
            return out  # no residues: nothing to resume
        # gpuecm2's CLI has no device flag; pinning is the only way it
        # lands anywhere but the ambient default card
        env2 = ctx.pin_env(s2_dev)
        facts2, wall2, err2, b2 = _stage2(n, b1, merged, ctx, wd,
                                          f"{tag}.{inv}", maxmem,
                                          env_extra=env2)
        out.stage2_s += wall2
        if err2:
            out.error = err2
            return out
        if b2 is not None and out.b2 is None:
            out.b2 = b2
        out.factors.update({f: p for f, p in facts2.items() if n % f == 0})
        done += round_ran
        if stop_check is not None and stop_check():
            out.aborted = True           # round kept (stage 2 finished)
            return out
    out.wall_s = out.stage1_s + out.stage2_s
    return out
