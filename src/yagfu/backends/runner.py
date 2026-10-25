"""Subprocess plumbing shared by all backends (ported from bench_factor.py)."""

from __future__ import annotations

import os
import signal
import subprocess
import time
from pathlib import Path


def run_cmd(cmd, cwd, log_path, env_extra=None, timeout=None, input_bytes=None):
    """Run a subprocess, merged output to log_path; return (rc, wall)."""
    env = dict(os.environ)
    if env_extra:
        env.update(env_extra)
    t0 = time.monotonic()
    with open(log_path, "wb") as f:
        p = subprocess.run(
            [str(c) for c in cmd], cwd=str(cwd), stdout=f,
            stderr=subprocess.STDOUT, env=env, timeout=timeout,
            input=input_bytes,
        )
    return p.returncode, time.monotonic() - t0


def run_cmd_wall_cap(cmd, cwd, log_path, cap_s, grace_s=90, timeout=3600,
                     input_bytes=None):
    """Run a subprocess under a cooperative wall-clock cap (SIGINT first)."""
    t0 = time.monotonic()
    with open(log_path, "wb") as f:
        p = subprocess.Popen([str(c) for c in cmd], cwd=str(cwd),
                             stdout=f, stderr=subprocess.STDOUT,
                             stdin=subprocess.PIPE if input_bytes is not None else None)
        if input_bytes is not None:
            assert p.stdin is not None
            p.stdin.write(input_bytes)
            p.stdin.close()
        try:
            p.wait(timeout=cap_s)
        except subprocess.TimeoutExpired:
            p.send_signal(signal.SIGINT)
            try:
                p.wait(timeout=grace_s)
            except subprocess.TimeoutExpired:
                p.terminate()
                try:
                    p.wait(timeout=15)
                except subprocess.TimeoutExpired:
                    p.kill()
                    p.wait()
    if time.monotonic() - t0 > timeout:
        return -9, time.monotonic() - t0
    return p.returncode, time.monotonic() - t0


def count_lines(path: Path) -> int:
    """Fast line count (relation/residue files reach gigabytes)."""
    out = subprocess.run(["wc", "-l", str(path)], capture_output=True, text=True)
    return int(out.stdout.split()[0])


def read_log(path: Path) -> str:
    return path.read_text(errors="replace")
