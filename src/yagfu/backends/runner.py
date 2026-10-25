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


def _feed_lines(log_path: Path, pos: int, pending: str, on_lines):
    """Hand the complete new lines of log_path to on_lines.

    Reads from byte offset `pos`; returns (new_pos, pending_partial_line).
    Both \\n and \\r terminate a line — msieve paces its "linear algebra
    completed ..." updates with bare carriage returns."""
    try:
        with open(log_path, "rb") as f:
            f.seek(pos)
            chunk = f.read()
    except OSError:
        return pos, pending
    if not chunk:
        return pos, pending
    text = pending + chunk.decode(errors="replace")
    lines = text.replace("\r", "\n").split("\n")
    pending = lines.pop()
    if lines:
        on_lines([l for l in lines if l.strip()])
    return pos + len(chunk), pending


def run_cmd_progress(cmd, cwd, log_path, timeout=3600, env_extra=None,
                     every_s=30.0, on_lines=None):
    """Run a subprocess like run_cmd, but poll it instead of blocking: every
    `every_s` the log lines appended so far are handed to on_lines(list[str]),
    so long stages can surface their own progress.

    Returns (rc, wall); rc is -9 when `timeout` expires (child killed).
    KeyboardInterrupt kills the child and propagates, like run_cmd.
    """
    env = dict(os.environ)
    if env_extra:
        env.update(env_extra)
    t0 = time.monotonic()
    with open(log_path, "wb") as f:
        p = subprocess.Popen([str(c) for c in cmd], cwd=str(cwd),
                             stdout=f, stderr=subprocess.STDOUT, env=env)
        pos, pending = 0, ""
        try:
            while p.poll() is None:
                if on_lines:
                    pos, pending = _feed_lines(log_path, pos, pending, on_lines)
                try:
                    p.wait(timeout=every_s)
                except subprocess.TimeoutExpired:
                    pass
                if time.monotonic() - t0 > timeout:
                    p.kill()
                    p.wait()
                    return -9, time.monotonic() - t0
            if on_lines:
                _feed_lines(log_path, pos, pending, on_lines)
        except BaseException:  # incl. KeyboardInterrupt: never orphan children
            p.terminate()
            try:
                p.wait(timeout=10)
            except subprocess.TimeoutExpired:
                p.kill()
                p.wait()
            raise
    return p.returncode, time.monotonic() - t0


def count_lines(path: Path) -> int:
    """Fast line count (relation/residue files reach gigabytes)."""
    out = subprocess.run(["wc", "-l", str(path)], capture_output=True, text=True)
    return int(out.stdout.split()[0])


def read_log(path: Path) -> str:
    return path.read_text(errors="replace")
