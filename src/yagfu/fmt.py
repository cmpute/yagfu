"""Tiny formatters for progress logs (counts, durations)."""


def fmt_count(x: float) -> str:
    """Compact count: 3540 -> '3,540'; 12.7e6 -> '12.7M'; 3.2e12 -> '3.2T'."""
    x = float(x)
    for div, suf in ((1e12, "T"), (1e9, "G"), (1e6, "M")):
        if x >= div:
            v = x / div
            return f"{v:.1f}{suf}" if v >= 10 else f"{v:.2f}{suf}"
    return f"{int(x):,}"


def fmt_dur(s: float) -> str:
    """Compact duration: 2.2ms, 40s, 1.5m, 3.2h."""
    if s < 1:
        return f"{s * 1000:.1f}ms"
    if s < 90:
        return f"{s:.0f}s"
    if s < 5400:
        return f"{s / 60:.1f}m"
    return f"{s / 3600:.1f}h"
