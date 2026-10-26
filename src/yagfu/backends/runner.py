"""Subprocess plumbing shared by all backends (ported from bench_factor.py)."""

from __future__ import annotations

import os
import signal
import subprocess
import time
from pathlib import Path

#: rc reported when a process was killed on request (stop callback),
#: as opposed to timing out (-9) or dying on its own
KILLED_RC = -1000


def _term_kill_ladder(p):
    """SIGTERM -> SIGKILL, never raises."""
    if p.poll() is None:
        p.terminate()
        try:
            p.wait(timeout=10)
        except subprocess.TimeoutExpired:
            p.kill()
            p.wait()


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
                     input_bytes=None, env_extra=None):
    """Run a subprocess under a cooperative wall-clock cap (SIGINT first)."""
    env = dict(os.environ)
    if env_extra:
        env.update(env_extra)
    t0 = time.monotonic()
    with open(log_path, "wb") as f:
        p = subprocess.Popen([str(c) for c in cmd], cwd=str(cwd),
                             stdout=f, stderr=subprocess.STDOUT,
                             stdin=subprocess.PIPE if input_bytes is not None else None,
                             env=env)
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
                     every_s=30.0, on_lines=None, input_bytes=None,
                     stop=None):
    """Run a subprocess like run_cmd, but poll it instead of blocking: every
    `every_s` the log lines appended so far are handed to on_lines(list[str]),
    so long stages can surface their own progress.

    `stop` () -> bool is polled too: when it fires the child is killed and
    (KILLED_RC, wall) returns — an early-yield request, not a failure.
    Returns (rc, wall); rc is -9 when `timeout` expires.
    KeyboardInterrupt kills the child and propagates, like run_cmd.
    """
    env = dict(os.environ)
    if env_extra:
        env.update(env_extra)
    t0 = time.monotonic()
    with open(log_path, "wb") as f:
        p = subprocess.Popen([str(c) for c in cmd], cwd=str(cwd),
                             stdout=f, stderr=subprocess.STDOUT, env=env,
                             stdin=subprocess.PIPE if input_bytes is not None
                             else None)
        if input_bytes is not None:
            try:
                p.stdin.write(input_bytes)
                p.stdin.close()
            except OSError:
                pass  # child died before reading its input
        pos, pending = 0, ""
        try:
            while p.poll() is None:
                if on_lines:
                    pos, pending = _feed_lines(log_path, pos, pending, on_lines)
                try:
                    p.wait(timeout=every_s)
                except subprocess.TimeoutExpired:
                    pass
                if stop is not None and stop():
                    _term_kill_ladder(p)
                    return KILLED_RC, time.monotonic() - t0
                if time.monotonic() - t0 > timeout:
                    p.kill()
                    p.wait()
                    return -9, time.monotonic() - t0
            if on_lines:
                _feed_lines(log_path, pos, pending, on_lines)
        except BaseException:  # incl. KeyboardInterrupt: never orphan children
            _term_kill_ladder(p)
            raise
    return p.returncode, time.monotonic() - t0


def run_procs_parallel(jobs, cwd, timeout, beat=None, beat_s=120.0,
                       poll_s=2.0, stop=None, envs=None):
    """Run subprocesses concurrently, polled from the calling thread.

    jobs: list of (cmd, stdout_log_path, input_bytes_or_None) tuples.
    envs: optional list of per-job env overrides (dict or None), merged
    over os.environ the same way run_cmd merges env_extra — how device
    pinning reaches each shard of a fan-out.
    Returns one (rc, wall_s) per job, in order (wall measured from the
    common start to that process's exit).  `beat(elapsed_s)` fires every
    beat_s while anything still runs.  `stop` () -> bool kills the set
    early: processes we terminated report KILLED_RC.  Raises
    subprocess.TimeoutExpired after `timeout` wall seconds.  Ctrl+C and
    timeouts never orphan children: both paths terminate the whole set
    before propagating.
    """
    t0 = time.monotonic()
    procs = []
    for j, (cmd, log_path, input_bytes) in enumerate(jobs):
        env = None
        if envs is not None and envs[j] is not None:
            env = dict(os.environ)
            env.update(envs[j])
        f = open(log_path, "wb")
        p = subprocess.Popen([str(c) for c in cmd], cwd=str(cwd),
                             stdout=f, stderr=subprocess.STDOUT,
                             stdin=subprocess.PIPE if input_bytes is not None
                             else None, env=env)
        if input_bytes is not None:
            try:
                p.stdin.write(input_bytes)
                p.stdin.close()
            except OSError:
                pass  # child died before reading its input
        procs.append((p, f))
    ends = {}
    stopped = False
    deadline = t0 + timeout
    next_beat = t0 + beat_s if beat else None
    try:
        while True:
            alive = False
            for i, (p, _) in enumerate(procs):
                if i in ends:
                    continue
                if p.poll() is None:
                    alive = True
                else:
                    ends[i] = time.monotonic()
            if not alive:
                break
            if stop is not None and stop():
                stopped = True
            if time.monotonic() > deadline:
                raise subprocess.TimeoutExpired(
                    str(jobs[0][0][0]) if jobs else "run_procs_parallel",
                    timeout)
            if stopped:
                break
            if next_beat is not None and time.monotonic() >= next_beat:
                beat(time.monotonic() - t0)
                next_beat += beat_s
            time.sleep(poll_s)
    except BaseException:  # incl. KeyboardInterrupt: never orphan children
        for p, _ in procs:
            try:
                p.terminate()
            except OSError:
                pass
        for p, _ in procs:
            _term_kill_ladder(p)
        raise
    finally:
        for _, f in procs:
            f.close()
    if stopped:
        for p, _ in procs:
            if p.poll() is None:
                _term_kill_ladder(p)
        return [(KILLED_RC, time.monotonic() - t0) for _ in procs]
    return [(p.returncode, ends.get(i, time.monotonic()) - t0)
            for i, (p, _) in enumerate(procs)]


def count_lines(path: Path) -> int:
    """Fast line count (relation/residue files reach gigabytes)."""
    out = subprocess.run(["wc", "-l", str(path)], capture_output=True, text=True)
    return int(out.stdout.split()[0])


def read_log(path: Path) -> str:
    return path.read_text(errors="replace")
