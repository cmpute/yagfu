"""Backend binary location and shared helpers.

Lookup order for each binary name: --bin-dir, $YAGFU_BIN, the copy bundled
inside the wheel (yagfu/bin/), then $PATH.
"""

from __future__ import annotations

import os
import shutil
from pathlib import Path

#: binaries bundled by scripts/build_wheel.sh
BUNDLED = ("ecm", "gpuecm2", "cuda-mpqs", "cuda-sieve", "fbgen", "fbgen_gpu", "msieve")


class BinNotFound(FileNotFoundError):
    pass


def _package_bin_dir() -> Path:
    return Path(__file__).resolve().parent.parent / "bin"


def find_bin(name: str, override_dir: Path | str | None = None) -> Path:
    candidates: list[Path] = []
    if override_dir:
        candidates.append(Path(override_dir) / name)
    env = os.environ.get("YAGFU_BIN")
    if env:
        candidates.append(Path(env) / name)
    candidates.append(_package_bin_dir() / name)
    which = shutil.which(name)
    if which:
        candidates.append(Path(which))
    for c in candidates:
        if c.is_file():
            if not os.access(c, os.X_OK):
                os.chmod(c, 0o755)  # wheel installs occasionally drop the bit
            return c
    tried = "\n  ".join(str(c) for c in candidates) or "(no candidates)"
    raise BinNotFound(
        f"backend binary '{name}' not found; tried:\n  {tried}\n"
        f"pass --bin-dir, set YAGFU_BIN, or use a wheel built with bundled binaries"
    )
