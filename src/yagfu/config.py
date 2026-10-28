"""Shared run context passed through the engine and backends."""

from __future__ import annotations

import os
import re
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path


def _nproc() -> int:
    try:
        return max(1, len(os.sched_getaffinity(0)))
    except AttributeError:  # pragma: no cover
        return max(1, os.cpu_count() or 1)


def _list_gpu_ords() -> list[str]:
    """Physical ordinals of every GPU nvidia-smi sees, ascending.

    Empty when nvidia-smi is missing, fails, or lists nothing — callers
    decide whether that is fatal.
    """
    try:
        out = subprocess.run(["nvidia-smi", "-L"], capture_output=True,
                             text=True, timeout=15).stdout
    except (OSError, subprocess.SubprocessError):
        return []
    return sorted(set(re.findall(r"^GPU (\d+):", out, re.MULTILINE)),
                  key=int)


def resolve_devices(specs) -> tuple[str, ...]:
    """Translate --device ordinals into physical device ids.

    The ordinals the user passes mean what they mean to every CUDA program
    in this shell: indices into CUDA_VISIBLE_DEVICES when it is set, plain
    physical ids otherwise.  Every backend subprocess is instead pinned with
    CUDA_VISIBLE_DEVICES=<physical id>, which needs the physical id; UUID
    entries pass through untouched (both CUDA and nvidia-smi -i accept
    them).  The special spec "all" expands to every card this shell can
    see (the CUDA_VISIBLE_DEVICES entries when set, else nvidia-smi's
    list) and must be the only spec.  Raises ValueError on anything that
    cannot be resolved.
    """
    ambient = os.environ.get("CUDA_VISIBLE_DEVICES")
    entries = ([e.strip() for e in ambient.split(",") if e.strip()]
               if ambient is not None else [])
    if ambient is not None and not entries:
        raise ValueError("CUDA_VISIBLE_DEVICES is set but hides every GPU")

    specs = [str(s).strip() for s in specs]
    if any(s.lower() == "all" for s in specs):
        if len(specs) != 1:
            raise ValueError('"all" cannot be mixed with explicit devices')
        if ambient is not None:
            return tuple(entries)
        all_ords = _list_gpu_ords()
        if not all_ords:
            raise ValueError(
                '--device all: could not enumerate GPUs (nvidia-smi missing '
                'or lists none); pass explicit ordinals instead')
        return tuple(all_ords)

    out = []
    for spec in specs:
        s = str(spec).strip()
        if not s.isdigit():
            raise ValueError(f"device {spec!r} is not a device ordinal")
        if not entries:
            out.append(s)
            continue
        idx = int(s)
        if idx >= len(entries):
            raise ValueError(
                f"device {idx} out of range for CUDA_VISIBLE_DEVICES="
                f"{ambient!r} ({len(entries)} device(s) visible)")
        out.append(entries[idx])
    if not out:
        raise ValueError("no devices given")
    return tuple(out)


@dataclass
class RunContext:
    bin_dir: Path | None = None          # explicit --bin-dir
    device: str = "0"                    # primary CUDA device, physical id
                                         # (polyselect, Lanczos, ECM stage 2)
    devices: tuple[str, ...] = ("0",)    # fan-out set, physical ids: ECM
                                         # stage 1 shards + sieve chunks
    ecm_device_rates: dict = field(default_factory=dict)
                                         # learned curves/s per device (scratch)
    threads: int = field(default_factory=lambda: max(1, _nproc() - 2))
    sieve_procs: int = 1                 # cuda-sieve processes per band
    plan: str = "normal"                 # light | normal | deep
    xover: int = 105                     # SIQS -> NFS crossover (digits)
    verbose: int = 0                     # -v count: extra detail level
    quiet: bool = False                  # -q: silence all progress logs
    keep_workdir: bool = False
    timeout_s: float | None = None       # per-number wall budget
    force_method: str | None = None      # debug: "siqs" | "nfs"
    ecm_only: bool = False               # no sieves: ECM ladder forever
    no_ecm: bool = False                 # skip the ECM stage entirely (reruns)
    ecm_maxcurves: int = 8192            # curves per GPU stage-1 invocation
    ecm_stage2_maxmem_mb: int | None = None     # gpuecm2 -maxmem; None = 90% of VRAM,
                                         # 0 = no cap (gpuecm2's own free-mem check)
    poly_budget_s: float | None = None   # NFS polyselect override
    poly_frac: float = 0.65              # quality gate: stop polyselect when
                                         # the best E reaches this fraction of
                                         # msieve's printed expected E
    topup_frac: float = 1.0          # NFS ladder step: the next -nc1 waits at
                                     # most one topup step past the last
                                     # filter's data; caps every rung (GPU
                                     # sieving is cheap, -nc1 passes are not)
    target_safety: float = 1.15      # margin over the requirement extrapolated
                                     # from the measured excess-vs-unique curve
    nfs_disk_margin: float = 0.8     # abort the NFS path after ECM when the
                                     # workdir's free space is below this
                                     # fraction of the pre-flight estimate
    wave_max: int = 16_000_000       # max relations per sieve wave: the
                                     # verdict reaction granularity (~2 min
                                     # of 4-GPU sieving)
    log_file: Path = field(default_factory=lambda: Path("yagfu.log"))
                                         # complete run log, appended (cwd)

    # per-number state (reset by engine)
    workdir: Path = field(default_factory=lambda: Path("."))
    deadline: float | None = None

    def pin_env(self, device: str | None = None) -> dict[str, str]:
        """CUDA_VISIBLE_DEVICES override pinning one subprocess to a device.

        ctx.devices holds physical ids; a pinned process sees exactly its
        card as ordinal 0, so every backend's own device flag says 0.  The
        return value is an env_extra-style override dict, merged over the
        inherited environment by the run_cmd family.
        """
        return {"CUDA_VISIBLE_DEVICES": device if device is not None
                else self.device}

    def _tee(self, line: str) -> None:
        """Append the line to the run log (cwd/yagfu.log)."""
        fh = self.__dict__.get("_log_fh")
        if fh is None:
            try:
                fh = self.log_file.open("a", encoding="utf-8")
            except OSError:
                fh = False        # unusable log file must never kill a run
            self.__dict__["_log_fh"] = fh
        if fh:
            try:
                fh.write(line + "\n")
                fh.flush()
            except OSError:
                pass

    def log(self, msg: str) -> None:
        """Key progress info — shown by default (YAFU-style), silenced by -q.
        Always appended to the run log file."""
        line = f"[{time.strftime('%H:%M:%S')}] {msg}"
        self._tee(line)
        if not self.quiet:
            print(line, file=sys.stderr, flush=True)

    def logv(self, msg: str) -> None:
        """Extra detail — only with -v."""
        if self.verbose <= 0:
            return
        line = f"[{time.strftime('%H:%M:%S')}] {msg}"
        self._tee(line)
        print(line, file=sys.stderr, flush=True)

    def check_deadline(self) -> None:
        if self.deadline is not None and time.monotonic() > self.deadline:
            raise TimeoutError(f"per-number timeout of {self.timeout_s:.0f}s exceeded")


def cleanup_workdir(ctx: RunContext, wd: Path, keep: bool) -> None:
    """Remove a per-number workdir.

    Only a number that factored completely cleans up after itself;
    a failure keeps the workdir for post-mortem, as does an interrupt
    and --keep.
    """
    if ctx.keep_workdir or keep:
        return
    if wd.exists():
        shutil.rmtree(wd, ignore_errors=True)
