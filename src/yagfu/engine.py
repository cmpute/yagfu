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
       In ecm-only mode the sieves are never reached: the ladder climbs
       every level to its full budget and then keeps sampling the t65
       level forever, until the number is factored or the user interrupts.

Factors accumulate into the FactorResult *as they are found*, so any
abort — Ctrl+C, --timeout, a backend failure — still yields the partial
factorization.  A result is only `complete` when the collected factors
multiply back to n and all of them are P/PRP.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path

from .arith import perfect_power, pollard_rho, trial_divide
from .backends.ecm import EcmProvenance, EcmUnsupported, run_ecm_curves
from .backends.siqs import run_siqs
from .config import RunContext
from .ecmledger import (
    B1_LADDER,
    CURVES,
    NUM_LEVELS,
    T_LEVELS,
    EcmLedger,
    target_depth,
)
from .fmt import fmt_count, fmt_dur
from .params import NFS_MSIEVE_POLY_MIN_DIGITS
from .primes import SPRP_2761_LIMIT, is_probable_prime

#: curve count per endless top-level batch in ecm-only mode
ECM_ONLY_TOPUP_CURVES = 1024


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
    interrupted: bool = False
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
            self._factor_rec(n, wd, res, depth=0)
        except KeyboardInterrupt:
            # the backend subprocess was killed by subprocess.run's
            # exception path; report whatever has been collected so far
            res.error = "interrupted (Ctrl+C)"
            res.interrupted = True
        except (FactorError, EcmUnsupported, TimeoutError) as e:
            res.error = str(e)
        res.complete = self._verify(n, res.factors)
        if not res.complete and not res.error:
            res.error = "factors do not multiply back to n"
        return res

    # -------------------------------------------------------------- helpers

    def _collect(self, f: int, source: str, wd: Path, res: FactorResult,
                 depth: int, ecm: EcmProvenance | None = None) -> None:
        """Classify f and record it; recurse if composite."""
        if is_probable_prime(f):
            proven = source in ("td", "power") or f < SPRP_2761_LIMIT
            ff = FoundFactor(f, source, proven, ecm)
            res.factors.append(ff)
            self._note_factor(ff)
            return
        self.ctx.log(f"fac: c{_digits(f)} composite factor from {source}, "
                     f"recursing")
        self._factor_rec(f, wd, res, depth + 1)

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

    def _factor_rec(self, m: int, wd: Path, res: FactorResult, depth: int,
                    ledger: EcmLedger | None = None) -> None:
        """Factor composite m (> 1); primes accumulate into res.factors.

        `ledger` pre-seeds the ECM work ledger — curves run on an ancestor
        number count for its cofactors, so a re-plan after an overlap split
        continues from the work already spent instead of re-climbing."""
        ctx = self.ctx
        if m == 1:
            return
        if is_probable_prime(m):
            res.factors.append(FoundFactor(m, "trivial", m < SPRP_2761_LIMIT))
            return

        # -- trial division (all primes < 256) -----------------------------
        t = time.monotonic()
        small, m = trial_divide(m)
        for p in small:
            res.factors.append(FoundFactor(p, "td", True))
            res.bump("td_primes")
        if small:
            parts = []
            for p in sorted(set(small)):
                e = small.count(p)
                parts.append(f"{p}^{e}" if e > 1 else f"{p}")
            ctx.log("fac: td: " + " * ".join(parts))
        res.spend("td", time.monotonic() - t)
        if m == 1:
            return
        if is_probable_prime(m):
            ff = FoundFactor(m, "td", m < SPRP_2761_LIMIT)
            res.factors.append(ff)
            self._note_factor(ff, " (td cofactor)")
            return
        ctx.check_deadline()

        # -- perfect powers --------------------------------------------------
        t = time.monotonic()
        pp = perfect_power(m)
        if pp is not None:
            base, e = pp
            res.spend("power", time.monotonic() - t)
            ctx.log(f"power: {m} == base^{e}, recursing")
            mark = len(res.factors)
            self._collect(base, "power", wd, res, depth)
            sub = res.factors[mark:]
            for _ in range(e - 1):
                res.factors.extend(sub)
            return
        ctx.check_deadline()

        # -- Pollard rho ------------------------------------------------------
        t = time.monotonic()
        fs, m = pollard_rho(m)
        res.spend("rho", time.monotonic() - t)
        for f in fs:
            ctx.log(f"rho: factor {f} ({len(str(f))} digits)")
            res.bump("rho_factors")
            self._collect(f, "rho", wd, res, depth)
        if m == 1:
            return
        if is_probable_prime(m):
            ff = FoundFactor(m, "rho", m < SPRP_2761_LIMIT)
            res.factors.append(ff)
            self._note_factor(ff, " (rho cofactor)")
            return
        ctx.check_deadline()

        # -- GPU ECM ladder (+ background NFS polyselect when NFS-bound) --
        if ledger is None:
            ledger = EcmLedger()
        job = self._maybe_start_poly(m, wd)
        try:
            if ctx.no_ecm:
                ctx.log("ecm: skipped (--no-ecm)")
            else:
                orig = m
                t = time.monotonic()
                m = self._ecm_stage(m, wd, res, depth, ledger)
                res.spend("ecm", time.monotonic() - t)
                if self._ecm_done(m, res):
                    return
                if m != orig:
                    # split during the pretest: the polyselect (if any) was
                    # for the original number — drop it, re-plan the cofactor
                    self._replan_after_split(job, m, wd, res, depth, ledger)
                    return
                # enter the bonus window even if the budgeted attempt just
                # died: the overlap loop retries it uncapped in place
                if job is not None:
                    t = time.monotonic()
                    m = self._poly_overlap_ecm(m, wd, res, depth, ledger, job)
                    res.spend("ecm", time.monotonic() - t)
                    if self._ecm_done(m, res):
                        return
                    if m != orig:
                        self._replan_after_split(job, m, wd, res, depth, ledger)
                        return
            ctx.check_deadline()

            # -- sieves ----------------------------------------------------------
            if ctx.ecm_only:
                raise FactorError(
                    f"ecm-only: c{_digits(m)} not fully factored by ECM")
            self._sieve_stage(m, wd, res, depth, nfs_poly=job)
        finally:
            if job is not None:
                job.cancel()

    # --------------------------------------------------------------- stages

    def _maybe_start_poly(self, m: int, wd: Path):
        """Start background NFS polyselect if this number is NFS-bound:
        CPU msieve -np and GPU ECM do not compete, so the polyselect budget
        becomes free wall time (and free bonus ECM depth)."""
        ctx = self.ctx
        if ctx.ecm_only or ctx.no_ecm or ctx.no_poly_overlap:
            return None
        digits = _digits(m)
        if digits < NFS_MSIEVE_POLY_MIN_DIGITS:
            return None
        if not (ctx.force_method == "nfs"
                or (ctx.force_method != "siqs" and digits >= ctx.xover)):
            return None
        from .backends.nfs import PolyselectJob, nfs_resume_pending
        if nfs_resume_pending(wd / "nfs", m):
            return None    # a previous run's polynomial is already there
        return PolyselectJob(m, ctx, wd / "nfs")

    def _ecm_done(self, m: int, res: FactorResult) -> bool:
        """Post-ECM exit check: fully split, or a prime cofactor."""
        if m == 1:
            return True
        if is_probable_prime(m):
            ff = FoundFactor(m, "ecm", m < SPRP_2761_LIMIT)
            res.factors.append(ff)
            self._note_factor(ff, " (ecm cofactor)")
            return True
        return False

    def _replan_after_split(self, job, cofactor: int, wd: Path,
                            res: FactorResult, depth: int,
                            ledger: EcmLedger) -> None:
        """ECM split the number while background polyselect was running:
        that polynomial was for the original number, so drop it and re-plan
        the (smaller) cofactor — keeping the ECM ledger, since curves run
        on a number count for its cofactors, and letting the cofactor's own
        (smaller, digit-based) polyselect budget apply."""
        ctx = self.ctx
        if job is not None:
            ctx.log(f"nfs: discarding background polyselect (number was "
                    f"split by ECM; budget {fmt_dur(job.budget)}, "
                    f"ran {fmt_dur(job.elapsed())})")
            job.cancel()
        if cofactor > 1:
            self._factor_rec(cofactor, wd, res, depth + 1, ledger)

    def _poly_overlap_ecm(self, m: int, wd: Path, res: FactorResult,
                          depth: int, ledger: EcmLedger, job) -> int:
        """Bonus ECM while background polyselect holds the CPUs: keep
        filling level budgets above the plan target.

        Only a polyselect that finished WITH a polynomial yields the GPUs
        back.  A budgeted attempt that dies polynomial-less is retried
        uncapped right here — the CPUs stay busy and the overlap runs on
        through the retry; killing the bonus batch for it would just
        waste free curves (a 4-GPU stage-1 batch at B1=11M is a minute of
        work).  Only a retry that also fails ends the bonus early, and
        NFS then reports the polyselect failure.  A factor found here
        (even in a killed batch's log) ends the overlap — the caller
        re-plans the cofactor."""
        ctx = self.ctx
        ctx.log("ecm: polyselect running in background, climbing bonus "
                "levels")

        j = 0
        inv = 0
        while True:
            ctx.check_deadline()
            if job.poly_ready():
                break               # polynomial in hand: NFS can start
            if not job.running():
                if not job.retry_uncapped():
                    break           # failed for good: let NFS report it
                continue
            while j < NUM_LEVELS and ledger.level_budget_left(j) <= 0:
                j += 1
            if j >= NUM_LEVELS:
                break               # whole ladder exhausted
            want = min(ledger.level_budget_left(j), ctx.ecm_maxcurves)
            inv += 1
            r = run_ecm_curves(m, B1_LADDER[j], want, ctx, wd,
                               tag=f"o{j}-{inv}", stop_check=job.poly_ready)
            res.bump("ecm_invocations")
            if r.error:
                raise FactorError(f"ecm: {r.error}")
            ledger.add(j, r.curves_done)
            ctx.log(f"ecm: B1={B1_LADDER[j]:,}"
                    + (f", B2={fmt_count(r.b2)}" if r.b2 else "")
                    + f", {r.curves_done:,} curves (polyselect overlap) "
                    f"-> t{ledger.depth():.2f}"
                    + (f"; {len(r.factors)} factor(s)!" if r.factors else ""))
            if r.curves_done == 0 and not r.factors and not r.aborted:
                raise FactorError("ecm made no progress")
            if r.factors:
                for f in sorted(r.factors, key=lambda x: -len(str(x))):
                    if m % f:
                        continue
                    res.bump("ecm_factors")
                    self._collect(f, "ecm", wd, res, depth, ecm=r.factors[f])
                    m //= f
                return m
            if r.aborted:
                # stop only fires on success now, so this batch's kill
                # really did buy the polynomial — drop it and join NFS
                if r.curves_done == 0:
                    ctx.log("ecm: polyselect finished mid-batch; dropping "
                            "the partial bonus batch")
                break
        return m

    def _ecm_stage(self, m: int, wd: Path, res: FactorResult, depth: int,
                   ledger: EcmLedger) -> int:
        """GPU ECM ladder with the YAFU work ledger; returns the cofactor.

        The ledger and the ladder position survive cofactor splits: a curve
        run on a number is equally a curve on each of its divisors, so after
        dividing out a found factor the ladder simply continues on the
        cofactor instead of starting over.  In ecm-only mode there is no
        target: every level runs to its full budget and the top level is
        then sampled forever until the number factors or the user
        interrupts.
        """
        ctx = self.ctx
        top = NUM_LEVELS - 1
        if ctx.ecm_only:
            ctx.log(f"ecm: c{_digits(m)}, ecm-only: every level to full "
                    f"budget, then endless t65")
        else:
            target = min(target_depth(_digits(m), ctx.plan), 65.0)
            ctx.log(f"ecm: c{_digits(m)}, plan={ctx.plan}, "
                    f"target=t{target:.1f}")

        j = 0
        inv = 0
        while True:
            if j >= NUM_LEVELS:
                if not ctx.ecm_only:
                    break
                j = top                        # park at t65 forever
            elif not ctx.ecm_only and j > 0 and (
                    ledger.depth() >= target
                    or (_digits(m) <= 45 and target < 15.0)):
                break  # target reached, or YAFU's trivial-pretest skip
            if ctx.ecm_only:
                want = ledger.level_budget_left(j)
                if want <= 0 and j == top:
                    want = ECM_ONLY_TOPUP_CURVES  # budget spent: keep going
            else:
                want = min(ledger.curves_for_target(j, target),
                           ledger.level_budget_left(j))
                if j == 0 and ledger.done_curves[0] == 0:
                    want = max(want, CURVES[0][0])  # always-run first batch
            if want <= 0:
                j += 1
                continue
            inv += 1
            ctx.check_deadline()
            r = run_ecm_curves(m, B1_LADDER[j], want, ctx, wd,
                               tag=f"l{j}-{inv}")
            res.bump("ecm_invocations")
            if r.error:
                raise FactorError(f"ecm: {r.error}")
            ledger.add(j, r.curves_done)
            rate = r.wall_s / r.curves_done if r.curves_done else 0.0
            planned = (int(CURVES[j][j]) if ctx.ecm_only
                       else ledger.curves_for_target(j, target))
            remaining = planned - ledger.done_curves[j]
            if ctx.ecm_only and ledger.level_budget_left(j) <= 0 and j == top:
                status = f"(t65 sampling, {ledger.done_curves[j]:,} curves)"
            elif ctx.ecm_only:
                status = f"(level t{T_LEVELS[j]})"
            else:
                status = f"(target t{target:.2f})"
            ctx.log(f"ecm: B1={B1_LADDER[j]:,}"
                    + (f", B2={fmt_count(r.b2)}" if r.b2 else "")
                    + f", {r.curves_done:,} curves"
                    + (f", {fmt_dur(rate)}/curve" if rate else "")
                    + (f", ~{fmt_dur(remaining * rate)} left" if rate > 0
                       and remaining > 0 else "")
                    + f" -> t{ledger.depth():.2f} {status}"
                    + (f"; {len(r.factors)} factor(s)!" if r.factors else ""))
            if r.curves_done == 0 and not r.factors:
                raise FactorError("ecm made no progress")
            if r.factors:
                for f in sorted(r.factors, key=lambda x: -len(str(x))):
                    if m % f:
                        continue
                    res.bump("ecm_factors")
                    self._collect(f, "ecm", wd, res, depth, ecm=r.factors[f])
                    m //= f
                    if m == 1 or is_probable_prime(m):
                        return m
                # curves already run count for the cofactor: keep the
                # ledger and the ladder position, just switch numbers
                ctx.log(f"ecm: cofactor now c{_digits(m)}, "
                        f"continuing at t{T_LEVELS[j]}")
                continue
            j += 1
        return m

    def _sieve_stage(self, m: int, wd: Path, res: FactorResult, depth: int,
                     nfs_poly=None) -> None:
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
                        self._absorb(r.factors, m, "siqs", wd, res, depth)
                        return
                else:
                    t = time.monotonic()
                    r = self._nfs(m, wd, poly=nfs_poly if method == order[0]
                                  else None)
                    res.spend("nfs", time.monotonic() - t)
                    res.bump("nfs_runs")
                    if r.ok and r.factors:
                        self._absorb(r.factors, m, "nfs", wd, res, depth)
                        return
                    ctx.log(f"nfs: no factors ({r.error})")
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
                res: FactorResult, depth: int) -> None:
        """Turn a sieve's factor dump into verified prime FoundFactors."""
        cof = m
        for f in sorted(set(raw_factors)):
            while cof % f == 0:
                self._collect(f, source, wd, res, depth)
                cof //= f
        if cof > 1:
            self._factor_rec(cof, wd, res, depth + 1)

    def _nfs(self, n: int, wd: Path, poly=None):
        if self._run_nfs is None:
            from .backends.nfs import run_nfs
            self._run_nfs = run_nfs
        return self._run_nfs(n, self.ctx, wd / "nfs", poly=poly)
