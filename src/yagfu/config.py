"""Shared run context passed through the engine and backends."""

from __future__ import annotations

import os
import shutil
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path


def _nproc() -> int:
    try:
        return max(1, len(os.sched_getaffinity(0)))
    except AttributeError:  # pragma: no cover
        return max(1, os.cpu_count() or 1)


@dataclass
class RunContext:
    bin_dir: Path | None = None          # explicit --bin-dir
    device: str = "0"                    # CUDA device ordinal
    threads: int = field(default_factory=lambda: max(1, _nproc() - 2))
    sieve_procs: int = 1                 # cuda-sieve processes per band
    plan: str = "normal"                 # light | normal | deep
    xover: int = 105                     # SIQS -> NFS crossover (digits)
    verbose: int = 0
    keep_workdir: bool = False
    timeout_s: float | None = None       # per-number wall budget
    force_method: str | None = None      # debug: "siqs" | "nfs"
    ecm_maxcurves: int = 8192            # curves per GPU stage-1 invocation
    ecm_maxmem_mb: int | None = None     # gpuecm2 -maxmem; None = 90% of VRAM,
                                         # 0 = no cap (gpuecm2's own free-mem check)
    poly_budget_s: float | None = None   # NFS polyselect override
    topup_frac: float = 0.5

    # per-number state (reset by engine)
    workdir: Path = field(default_factory=lambda: Path("."))
    deadline: float | None = None

    def log(self, msg: str) -> None:
        if self.verbose > 0:
            print(f"[{time.strftime('%H:%M:%S')}] {msg}", file=sys.stderr, flush=True)

    def check_deadline(self) -> None:
        if self.deadline is not None and time.monotonic() > self.deadline:
            raise TimeoutError(f"per-number timeout of {self.timeout_s:.0f}s exceeded")


def cleanup_workdir(ctx: RunContext, wd: Path, ok: bool) -> None:
    """Remove a per-number workdir unless kept or a failure needs it."""
    if ctx.keep_workdir or not ok:
        return
    if wd.exists():
        shutil.rmtree(wd, ignore_errors=True)
