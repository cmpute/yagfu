"""The yagfu orchestrator: one composite in, verified prime factors out.

Pipeline per composite (YAFU's three retained modules — fixed prework
budget, cumulative ECM work ledger, independent final algorithm selector):

    trial division (< 256) -> perfect-power strip -> Pollard rho
        (x^2+3, x^2+2, x^2+1; 1000 batch-GCD rounds each)
    -> GPU ECM ladder (t15..t65, param-3 ledger, always at least the
       B1=2000 batch — our prework covers less than YAFU's, so that first
       batch is cheap insurance; afterwards YAFU's rule applies: stop when
       the ledger reaches plan*digits, skip higher levels entirely when
       digits <= 45 and target < t15)
    -> final selector: cuda-mpqs (SIQS) below the crossover digit count,
       msieve+cuda-sieve (GNFS) at/above it; each is the other's fallback.

Every factor found anywhere is re-verified for divisibility, classified by
BPSW, and recursed on if composite.  The result is only `complete` when the
collected factors multiply back to n and all of them are P/PRP.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path

from .arith import perfect_power, pollard_rho, trial_divide
from .backends.ecm import EcmProvenance, EcmUnsupported, run_ecm_curves
from .backends.siqs import run_siqs
from .config import RunContext
from .ecmledger import B1_LADDER, CURVES, NUM_LEVELS, EcmLedger, target_depth
from .primes import SPRP_2761_LIMIT, is_probable_prime


@dataclass
class FoundFactor:
    value: int
    source: str                    # td | power | rho | ecm | siqs | nfs | trivial
    proven: bool                   # primality proven (TD or < SPRP(2,7,61) bound)
    ecm: EcmProvenance | None = None

    @property
    def tag(self) -> str:
        return f"{'P' if self.proven else 'PRP'}{len(str(self.value))}"


@dataclass
class FactorResult:
    n: int
    factors: list[FoundFactor] = field(default_factory=list)
    complete: bool = False
    stages: dict[str, float] = field(default_factory=dict)
    counts: dict[str, int] = field(default_factory=dict)
    error: str | None = None

    def bump(self, key: str, n: int = 1) -> None:
        self.counts[key] = self.counts.get(key, 0) + n

    def spend(self, stage: str, seconds: float) -> None:
        self.stages[stage] = self.stages.get(stage, 0.0) + seconds


def _digits(n: int) -> int:
    return len(str(n))


class FactorError(Exception):
    pass


class Engine:
    def __init__(self, ctx: RunContext):
        self.ctx = ctx
        self._run_nfs = None  # lazy import; keeps SIQS-only runs light

    # ------------------------------------------------------------------ api

    def factor(self, n: int, wd: Path) -> FactorResult:
        res = FactorResult(n=n)
        if n < 2:
            res.error = "n must be >= 2"
            return res
        self.ctx.workdir = wd
        self.ctx.deadline = (time.monotonic() + self.ctx.timeout_s
                             if self.ctx.timeout_s else None)
        self.ctx.log(f"fac: factoring {n} (c{_digits(n)})")
        try:
            res.factors = self._factor_rec(n, wd, res, depth=0)
            res.complete = self._verify(n, res.factors)
            if not res.complete:
                res.error = res.error or "factors do not multiply back to n"
        except (FactorError, EcmUnsupported, TimeoutError) as e:
            res.error = str(e)
        return res

    # -------------------------------------------------------------- helpers

    def _collect(self, f: int, source: str, wd: Path, res: FactorResult,
                 depth: int, ecm: EcmProvenance | None = None) -> list[FoundFactor]:
        """Classify f; recurse if composite. Returns its prime factors."""
        if is_probable_prime(f):
            proven = source in ("td", "power") or f < SPRP_2761_LIMIT
            ff = FoundFactor(f, source, proven, ecm)
            self._note_factor(ff)
            return [ff]
        self.ctx.log(f"fac: c{_digits(f)} composite factor from {source}, "
                     f"recursing")
        return self._factor_rec(f, wd, res, depth + 1)

    def _note_factor(self, ff: FoundFactor, note: str = "") -> None:
        """YAFU-style factor announcement (skips trivial TD/small primes)."""
        if ff.source in ("td", "power", "trivial") or ff.value < 2**16:
            return
        extra = {"ecm": " (ecm)", "rho": " (rho)", "siqs": " (siqs)",
                 "nfs": " (nfs)"}.get(ff.source, "")
        self.ctx.log(f"fac: {ff.tag} = {ff.value}{extra}{note}")

    @staticmethod
    def _verify(n: int, factors: list[FoundFactor]) -> bool:
        prod = 1
        for f in factors:
            if not is_probable_prime(f.value):
                return False
            prod *= f.value
        return prod == n

    # ------------------------------------------------------------- pipeline

    def _factor_rec(self, m: int, wd: Path, res: FactorResult,
                    depth: int) -> list[FoundFactor]:
        """Factor composite m (> 1) completely into primes."""
        ctx = self.ctx
        if m == 1:
            return []
        if is_probable_prime(m):
            return [FoundFactor(m, "trivial", m < SPRP_2761_LIMIT)]

        out: list[FoundFactor] = []

        # -- trial division (all primes < 256) -----------------------------
        t = time.monotonic()
        small, m = trial_divide(m)
        for p in small:
            out.append(FoundFactor(p, "td", True))
            res.bump("td_primes")
        if small:
            parts = []
            for p in sorted(set(small)):
                e = small.count(p)
                parts.append(f"{p}^{e}" if e > 1 else f"{p}")
            ctx.log("fac: td: " + " * ".join(parts))
        res.spend("td", time.monotonic() - t)
        if m == 1:
            return out
        if is_probable_prime(m):
            ff = FoundFactor(m, "td", m < SPRP_2761_LIMIT)
            self._note_factor(ff, " (td cofactor)")
            out.append(ff)
            return out
        ctx.check_deadline()

        # -- perfect powers --------------------------------------------------
        t = time.monotonic()
        pp = perfect_power(m)
        if pp is not None:
            base, e = pp
            res.spend("power", time.monotonic() - t)
            ctx.log(f"power: {m} == base^{e}, recursing")
            sub = self._collect(base, "power", wd, res, depth)
            for _ in range(e):
                out.extend(sub)
            return out
        ctx.check_deadline()

        # -- Pollard rho ------------------------------------------------------
        t = time.monotonic()
        fs, m = pollard_rho(m)
        res.spend("rho", time.monotonic() - t)
        for f in fs:
            ctx.log(f"rho: factor {f} ({len(str(f))} digits)")
            res.bump("rho_factors")
            out.extend(self._collect(f, "rho", wd, res, depth))
        if m == 1:
            return out
        if is_probable_prime(m):
            ff = FoundFactor(m, "rho", m < SPRP_2761_LIMIT)
            self._note_factor(ff, " (rho cofactor)")
            out.append(ff)
            return out
        ctx.check_deadline()

        # -- GPU ECM ladder ----------------------------------------------------
        t = time.monotonic()
        groups, m = self._ecm_stage(m, wd, res, depth)
        res.spend("ecm", time.monotonic() - t)
        for g in groups:
            out.extend(g)
        if m == 1:
            return out
        if is_probable_prime(m):
            ff = FoundFactor(m, "ecm", m < SPRP_2761_LIMIT)
            self._note_factor(ff, " (ecm cofactor)")
            out.append(ff)
            return out
        ctx.check_deadline()

        # -- sieves --------------------------------------------------------------
        return out + self._sieve_stage(m, wd, res, depth)

    # --------------------------------------------------------------- stages

    def _ecm_stage(self, m: int, wd: Path, res: FactorResult,
                   depth: int) -> tuple[list[list[FoundFactor]], int]:
        """GPU ECM ladder with the YAFU work ledger.

        Returns (factor groups, remaining cofactor).  The ledger survives
        cofactor splits: after dividing out a found factor the target is
        recomputed for the smaller cofactor while banked work still counts.
        """
        ctx = self.ctx
        collected: list[list[FoundFactor]] = []
        ledger = EcmLedger()
        target = min(target_depth(_digits(m), ctx.plan), 65.0)
        ctx.log(f"ecm: c{_digits(m)}, plan={ctx.plan}, target=t{target:.1f}")

        j = 0
        inv = 0
        while j < NUM_LEVELS:
            if j > 0 and (ledger.depth() >= target
                          or (_digits(m) <= 45 and target < 15.0)):
                break  # target reached, or YAFU's trivial-pretest skip
            want = min(ledger.curves_for_target(j, target),
                       ledger.level_budget_left(j))
            if j == 0:
                want = max(want, CURVES[0][0])  # the always-run first batch
            if want > 0:
                inv += 1
                ctx.check_deadline()
                r = run_ecm_curves(m, B1_LADDER[j], want, ctx, wd,
                                   tag=f"l{j}-{inv}")
                res.bump("ecm_invocations")
                if r.error:
                    raise FactorError(f"ecm: {r.error}")
                ledger.add(j, r.curves_done)
                ctx.log(f"ecm: B1={B1_LADDER[j]:,}, {r.curves_done} curves "
                        f"-> t{ledger.depth():.2f} (target t{target:.2f})"
                        + (f"; {len(r.factors)} factor(s)!" if r.factors else ""))
                if r.curves_done == 0 and not r.factors:
                    raise FactorError("ecm made no progress")
                if r.factors:
                    for f in sorted(r.factors, key=lambda x: -len(str(x))):
                        if m % f:
                            continue
                        res.bump("ecm_factors")
                        collected.append(self._collect(
                            f, "ecm", wd, res, depth, ecm=r.factors[f]))
                        m //= f
                        if m == 1 or is_probable_prime(m):
                            return collected, m
                    target = min(target_depth(_digits(m), ctx.plan), 65.0)
                    ctx.log(f"ecm: cofactor now c{_digits(m)}, "
                            f"target=t{target:.1f}")
                    continue  # same level, fresh target and budget
            j += 1
        return collected, m

    def _sieve_stage(self, m: int, wd: Path, res: FactorResult,
                     depth: int) -> list[FoundFactor]:
        ctx = self.ctx
        digits = _digits(m)
        want_nfs = ctx.force_method == "nfs" or (
            ctx.force_method != "siqs" and digits >= ctx.xover)
        order = ("nfs", "siqs") if want_nfs else ("siqs", "nfs")

        for method in order:
            ctx.check_deadline()
            try:
                if method == "siqs":
                    t = time.monotonic()
                    r = run_siqs(m, ctx, wd)
                    res.spend("siqs", time.monotonic() - t)
                    res.bump("siqs_runs")
                    ctx.log(f"siqs: rc={r.rc} in {r.wall_s:.1f}s "
                            f"({r.error or 'ok'})")
                    if r.ok:
                        return self._absorb(r.factors, m, "siqs", wd, res, depth)
                else:
                    t = time.monotonic()
                    factors = self._nfs(m, wd)
                    res.spend("nfs", time.monotonic() - t)
                    res.bump("nfs_runs")
                    if factors:
                        return self._absorb(factors, m, "nfs", wd, res, depth)
                    ctx.log("nfs: no factors")
            except TimeoutError:
                raise
            except (FactorError, EcmUnsupported):
                raise
            except Exception as e:  # noqa: BLE001
                ctx.log(f"{method} crashed: {e!r}")
            other = order[1] if method == order[0] else None
            if other:
                ctx.log(f"{method} failed on c{digits}, trying {other}")
        raise FactorError(f"both sieves failed on c{digits} (logs in {wd})")

    def _absorb(self, raw_factors: list[int], m: int, source: str, wd: Path,
                res: FactorResult, depth: int) -> list[FoundFactor]:
        """Turn a sieve's factor dump into verified prime FoundFactors."""
        out: list[FoundFactor] = []
        cof = m
        for f in sorted(set(raw_factors)):
            while cof % f == 0:
                out.extend(self._collect(f, source, wd, res, depth))
                cof //= f
        if cof > 1:
            out.extend(self._factor_rec(cof, wd, res, depth + 1))
        return out

    def _nfs(self, n: int, wd: Path) -> list[int]:
        if self._run_nfs is None:
            from .backends.nfs import run_nfs
            self._run_nfs = run_nfs
        return self._run_nfs(n, self.ctx, wd / "nfs")
