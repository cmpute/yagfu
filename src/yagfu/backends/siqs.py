"""cuda-mpqs backend: whole SIQS/MPQS pipeline in one GPU process."""

from __future__ import annotations

import re
import time
from dataclasses import dataclass, field
from pathlib import Path

from ..config import RunContext
from ..params import mpqs_lp1_bound
from . import find_bin
from .runner import read_log, run_cmd

MPQS_TIMEOUT_S = 6 * 3600

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


def run_siqs(n: int, ctx: RunContext, wd: Path) -> SiqsResult:
    """Run cuda-mpqs on n inside wd; returns verified prime factors."""
    binary = find_bin("cuda-mpqs", ctx.bin_dir)
    work = wd / "mpqs"
    work.mkdir(parents=True, exist_ok=True)
    digits = len(str(n))

    base_cmd = [binary, "--N", n, "--device", ctx.device, "--dir", "work",
                "--verbose"]
    lp1 = mpqs_lp1_bound(digits)
    if lp1:
        base_cmd += ["--lp1_bound", lp1]
    log_path = work / "run.log"
    ctx.log(f"siqs: cuda-mpqs on c{digits} (lp1={lp1})")
    t0 = time.monotonic()
    try:
        rc, wall = run_cmd(base_cmd, work, log_path, timeout=MPQS_TIMEOUT_S)
    except Exception as e:  # noqa: BLE001
        return SiqsResult(error=f"cuda-mpqs failed to run: {e!r}",
                          wall_s=time.monotonic() - t0)
    log = read_log(log_path)

    # retry once with a corrected pinned tuple on the Small-N bug
    if rc != 0 and _RE_NARROW_BUG.search(log):
        fixed = _narrow_bug_params(log)
        if fixed:
            ctx.log(f"siqs: cuda-mpqs Small-N interval bug, retrying with "
                    f"--params {fixed}")
            t1 = time.monotonic()
            try:
                rc, w2 = run_cmd(base_cmd + ["--params", fixed],
                                 work, work / "run2.log", timeout=MPQS_TIMEOUT_S)
            except Exception as e:  # noqa: BLE001
                return SiqsResult(error=f"cuda-mpqs retry failed: {e!r}",
                                  wall_s=time.monotonic() - t0)
            wall += time.monotonic() - t1
            log = read_log(work / "run2.log")

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
