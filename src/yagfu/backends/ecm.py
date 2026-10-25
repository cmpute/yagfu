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
from . import find_bin
from .runner import count_lines, read_log, run_cmd

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
    error: str | None = None


def _stage1(n: int, b1: int, curves: int, ctx: RunContext, wd: Path,
            tag: str) -> tuple[int, dict[int, EcmProvenance], int, float, str | None]:
    """One GPU stage-1 invocation; returns (rc, factors, curves, wall, error)."""
    ecm = find_bin("ecm", ctx.bin_dir)
    res_path = wd / f"ecm-{tag}.residues"
    log_path = wd / f"ecm-{tag}.stage1.log"
    cmd = [ecm, "-gpu", "-gpudevice", ctx.device, "-save", res_path.name,
           "-gpucurves", curves, str(b1), "1"]
    t0 = time.monotonic()
    try:
        rc, _ = run_cmd(cmd, wd, log_path, input_bytes=f"{n}\n".encode(),
                        timeout=max(600, 4 * 86400))
    except Exception as e:  # noqa: BLE001
        return -1, {}, 0, time.monotonic() - t0, f"ecm failed to run: {e!r}"
    wall = time.monotonic() - t0
    log = read_log(log_path)

    if _RE_UNSUPPORTED.search(log):
        raise EcmUnsupported(log.strip().splitlines()[-1])

    factors: dict[int, EcmProvenance] = {}
    for m in _RE_STEP1_FACTOR.finditer(log):
        f, _param, sigma = int(m.group(1)), m.group(2), int(m.group(3))
        factors.setdefault(f, EcmProvenance(b1=b1, stage=1, sigma=sigma))

    m = _RE_COMPUTING.search(log) or _RE_CURVES_HDR.search(log)
    ran = int(m.group(1)) if m else curves
    return rc, factors, ran, wall, None


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
            tag: str, maxmem: int | None
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
        rc, _ = run_cmd(cmd, wd, log_path, timeout=max(600, 4 * 86400))
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
                   wd: Path, tag: str) -> EcmRunResult:
    """Run up to total_curves curves at B1 in GPU batches; stop at first
    batch that finds a factor (subsequent curves are pointless then)."""
    if n.bit_length() > GPUECM2_MAX_BITS:
        raise EcmUnsupported(f"{n.bit_length()} bits > gpuecm2 limit of {GPUECM2_MAX_BITS}")

    maxmem = resolve_maxmem(ctx.ecm_stage2_maxmem_mb, ctx.device)
    if maxmem:
        ctx.logv(f"ecm: gpuecm2 -maxmem {maxmem} MB"
                 + ("" if ctx.ecm_stage2_maxmem_mb else " (90% of VRAM)"))
    out = EcmRunResult()
    done = 0
    inv = 0
    while done < total_curves and not out.factors:
        inv += 1
        want = min(total_curves - done, ctx.ecm_maxcurves)
        want = min(((want + _CURVES_GRANULARITY - 1) // _CURVES_GRANULARITY)
                   * _CURVES_GRANULARITY, ctx.ecm_maxcurves)
        rc, facts, ran, wall, err = _stage1(n, b1, want, ctx, wd, f"{tag}.{inv}")
        out.stage1_s += wall
        out.curves_done += ran
        out.factors.update(facts)  # first provenance wins
        if err:
            out.error = err
            return out
        if out.factors:
            # keep only factors of n (paranoia: verify everything)
            out.factors = {f: p for f, p in out.factors.items() if n % f == 0}
            return out
        res_path = wd / f"ecm-{tag}.{inv}.residues"
        if not res_path.exists() or count_lines(res_path) == 0:
            return out  # no residues: nothing to resume
        facts2, wall2, err2, b2 = _stage2(n, b1, res_path, ctx, wd,
                                          f"{tag}.{inv}", maxmem)
        out.stage2_s += wall2
        if err2:
            out.error = err2
            return out
        if b2 is not None and out.b2 is None:
            out.b2 = b2
        out.factors.update({f: p for f, p in facts2.items() if n % f == 0})
        done += ran
    out.wall_s = out.stage1_s + out.stage2_s
    return out
