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


@dataclass
class SiqsResult:
    factors: list[int] = field(default_factory=list)
    ok: bool = False
    wall_s: float = 0.0
    rc: int = -1
    error: str | None = None
    internal: dict = field(default_factory=dict)


def run_siqs(n: int, ctx: RunContext, wd: Path) -> SiqsResult:
    """Run cuda-mpqs on n inside wd; returns verified prime factors."""
    binary = find_bin("cuda-mpqs", ctx.bin_dir)
    work = wd / "mpqs"
    work.mkdir(parents=True, exist_ok=True)
    digits = len(str(n))

    cmd = [binary, "--N", n, "--device", ctx.device, "--dir", "work", "--verbose"]
    lp1 = mpqs_lp1_bound(digits)
    if lp1:
        cmd += ["--lp1_bound", lp1]
    log_path = work / "run.log"
    ctx.log(f"siqs: cuda-mpqs on c{digits} (lp1={lp1})")
    t0 = time.monotonic()
    try:
        rc, wall = run_cmd(cmd, work, log_path, timeout=MPQS_TIMEOUT_S)
    except Exception as e:  # noqa: BLE001
        return SiqsResult(error=f"cuda-mpqs failed to run: {e!r}",
                          wall_s=time.monotonic() - t0)
    log = read_log(log_path)

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
