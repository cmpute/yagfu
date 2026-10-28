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
       With a supplied SNFS polynomial (factordb's -snfs suggestion, or
       --snfs-job) the polynomial's number is sieved with it instead — the
       ECM pretest priced by the difficulty's GNFS equivalent — and only
       a failed SNFS run falls back to the usual selection.
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
from typing import TYPE_CHECKING

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
import shutil

from .params import nfs_disk_estimate, snfs_gnfs_equiv
from .primes import SPRP_2761_LIMIT, is_probable_prime

if TYPE_CHECKING:
    from .backends.nfs import SnfsJob

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

    def factor(self, n: int, wd: Path,
               snfs: "SnfsJob | None" = None) -> FactorResult:
        res = FactorResult(n=n)
        if n < 2:
            res.error = "n must be >= 2"
            return res
        self.ctx.workdir = wd
        self.ctx.deadline = (time.monotonic() + self.ctx.timeout_s
                             if self.ctx.timeout_s else None)
        self.ctx.log(f"fac: factoring {n} (c{_digits(n)})")
        if snfs is not None or _digits(n) >= self.ctx.xover:
            self._nfs_disk_report(snfs, n, wd, pre_flight=True)
        try:
            self._factor_rec(n, wd, res, depth=0, snfs=snfs)
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
                    ledger: EcmLedger | None = None,
                    snfs: "SnfsJob | None" = None) -> None:
        """Factor composite m (> 1); primes accumulate into res.factors.

        `ledger` pre-seeds the ECM work ledger — curves run on an ancestor
        number count for its cofactors, so a recursion after a mid-ladder
        split continues from the work already spent instead of re-climbing.
        `snfs` is an SNFS polynomial suggestion for this same number: it
        rides through the ECM stage (and a split's recursion) and is
        consumed by the first sieve stage reached; recursions on sibling
        factors (power/rho/sieve splits) never see it."""
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

        # -- GPU ECM ladder ---------------------------------------------------
        if ledger is None:
            ledger = EcmLedger()
        if ctx.no_ecm:
            ctx.log("ecm: skipped (--no-ecm)")
        else:
            orig = m
            t = time.monotonic()
            m = self._ecm_stage(m, wd, res, depth, ledger, snfs)
            res.spend("ecm", time.monotonic() - t)
            if self._ecm_done(m, res):
                return
            if m != orig:
                # split during the pretest: the ladder already continued on
                # the cofactor to the target; recurse keeps the ledger (the
                # re-entry's cheaper prework re-runs, the ladder skips)
                # and the snfs suggestion (it maps the original number)
                self._factor_rec(m, wd, res, depth + 1, ledger=ledger,
                                 snfs=snfs)
                return
        ctx.check_deadline()

        # -- sieves (routed on the post-ECM cofactor size) --------------------
        if ctx.ecm_only:
            raise FactorError(
                f"ecm-only: c{_digits(m)} not fully factored by ECM")
        self._sieve_stage(m, wd, res, depth, snfs=snfs)

    # --------------------------------------------------------------- stages

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

    def _ecm_stage(self, m: int, wd: Path, res: FactorResult, depth: int,
                   ledger: EcmLedger,
                   snfs: "SnfsJob | None" = None) -> int:
        """GPU ECM ladder with the YAFU work ledger; returns the cofactor.

        The ledger and the ladder position survive cofactor splits: a curve
        run on a number is equally a curve on each of its divisors, so after
        dividing out a found factor the ladder simply continues on the
        cofactor instead of starting over (the plan target deliberately
        stays the original number's — a cofactor this deep into the ladder
        is unlikely to still hold small factors).  In ecm-only mode there
        is no target: every level runs to its full budget and the top level
        is then sampled forever until the number factors or the user
        interrupts.

        With an SNFS suggestion the pretest is priced against what the
        sieving job actually costs: the yafu GNFS-equivalent of the
        scaled difficulty (params.snfs_gnfs_equiv, 0.56 D + 30), not the
        raw digit count and not the difficulty itself.  A c135 record
        with an SNFS-170 suggestion is a ~c125-equivalent job and gets a
        t38-class pretest, not t52.
        """
        ctx = self.ctx
        # the effective size: GNFS-equivalent of the difficulty when
        # sieving SNFS, the digit count otherwise
        sz = (snfs_gnfs_equiv(snfs.scaled) if snfs is not None
              else _digits(m))
        top = NUM_LEVELS - 1
        if ctx.ecm_only:
            ctx.log(f"ecm: c{_digits(m)}, ecm-only: every level to full "
                    f"budget, then endless t65")
        else:
            target = min(target_depth(sz, ctx.plan), 65.0)
            ctx.log(f"ecm: c{_digits(m)}"
                    + (f" (snfs {snfs.scaled:g} ~ gnfs c{sz:g})"
                       if snfs is not None
                       else "")
                    + f", plan={ctx.plan}, target=t{target:.1f}")

        j = 0
        inv = 0
        while True:
            if j >= NUM_LEVELS:
                if not ctx.ecm_only:
                    break
                j = top                        # park at t65 forever
            elif not ctx.ecm_only and j > 0 and (
                    ledger.depth() >= target
                    or (sz <= 45 and target < 15.0)):
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

    def _nfs_disk_report(self, snfs: "SnfsJob | None", m: int, wd: Path,
                         pre_flight: bool = False) -> None:
        """Log the NFS working-set estimate for this job and the workdir's
        free space.  SNFS estimates key on the polynomial's scaled
        difficulty (its sieving size is fixed by the poly, ECM splits
        cannot shrink it); GNFS on the current composite, so the post-ECM
        re-check sees the difficulty the pretest actually left."""
        import shutil
        if snfs is not None:
            equiv = snfs_gnfs_equiv(snfs.scaled)
            kind = f"SNFS d{snfs.scaled:g} (GNFS-equiv c{equiv:.0f})"
        else:
            equiv = float(_digits(m))
            kind = f"c{equiv:.0f}"
        est = nfs_disk_estimate(round(equiv))
        free = shutil.disk_usage(wd).free / 1e9
        margin = self.ctx.nfs_disk_margin
        self.ctx.log(
            f"nfs: disk estimate for {kind}: ~{est['total_gb']:.0f} GB "
            f"(~{est['raw_rels']/1e6:.0f}M raw relations -> dat "
            f"{est['dat_gb']:.0f} + dup {est['dup_gb']:.0f} + lp "
            f"{est['lp_gb']:.0f} + mat/cyc {est['mat_gb']:.0f}); workdir "
            f"has {free:.0f} GB free "
            f"({free/est['total_gb']:.0%} of the estimate, "
            f"margin {margin:.0%})")
        if pre_flight and free < margin * est["total_gb"]:
            self.ctx.log(
                f"nfs: BELOW the disk margin - the run will be aborted "
                f"when the ECM stage ends unless space is freed first "
                f"(ECM factors may shrink this estimate)")

    def _sieve_stage(self, m: int, wd: Path, res: FactorResult,
                     depth: int, snfs: "SnfsJob | None" = None) -> None:
        ctx = self.ctx
        digits = _digits(m)

        # disk pre-flight, recomputed on the post-ECM cofactor: an ECM
        # split can shrink a GNFS job (an SNFS job sieves its polynomial's
        # number, so its estimate cannot shrink).  Runs before anything
        # is sieved; failing here costs only the pretest.
        if snfs is not None or digits >= ctx.xover \
                or ctx.force_method == "nfs":
            self._nfs_disk_report(snfs, m, wd)
            equiv = (snfs_gnfs_equiv(snfs.scaled) if snfs is not None
                     else float(digits))
            est = nfs_disk_estimate(round(equiv))
            free = shutil.disk_usage(wd).free / 1e9
            if free < ctx.nfs_disk_margin * est["total_gb"]:
                raise FactorError(
                    f"disk: the NFS path needs ~{est['total_gb']:.0f} GB "
                    f"but the workdir has {free:.0f} GB free (<"
                    f"{ctx.nfs_disk_margin:.0%} of the estimate).  Free "
                    f"space (or point the workdir at a larger volume) and "
                    f"restart; the ECM pretest factors are in the log")

        if snfs is not None:
            # the suggestion maps its own number, which may be larger than
            # the post-ECM cofactor: sieve the polynomial's number (the
            # job cost is the same) and strip what the pretest already
            # found when absorbing.  A failed suggestion is consumed —
            # retrying it on the fallback path would fail the same way.
            attach = snfs.n
            ctx.log(f"snfs: sieving c{_digits(attach)} with the supplied "
                    f"polynomial (difficulty {snfs.scaled:g})")
            ctx.check_deadline()
            t = time.monotonic()
            r = self._nfs(attach, wd, snfs=snfs)
            res.spend("nfs", time.monotonic() - t)
            res.bump("nfs_runs")
            if r.ok and r.factors:
                res.bump("snfs_runs")
                self._absorb_snfs(r.factors, attach, m, wd, res, depth)
                return
            ctx.log(f"snfs: failed ({r.error}); falling back to the "
                    f"usual sieve selection")
            snfs = None

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
                    r = self._nfs(m, wd)
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

    def _absorb_snfs(self, raw_factors: list[int], attach_n: int, m: int,
                     wd: Path, res: FactorResult, depth: int) -> None:
        """Absorb factors of attach_n into the cofactor m.

        The SNFS run sieved attach_n, of which m is a divisor (the ECM
        pretest may have stripped primes along the way): every collected
        prime divides attach_n too, so stripping it from each returned
        factor leaves a divisor of m.  Composites recurse; whatever of m
        survives recurses with the usual sieve selection."""
        cof = m
        for f in sorted(set(raw_factors), reverse=True):
            v = f
            for ff in res.factors:
                while v % ff.value == 0:
                    v //= ff.value
            if v <= 1:
                continue
            if cof % v != 0:
                self.ctx.log(f"snfs: factor {f} does not split the "
                             "cofactor, ignoring")
                continue
            cof //= v
            self._collect(v, "nfs", wd, res, depth)
        if cof > 1:
            self.ctx.log(f"snfs: c{_digits(cof)} of the attach number "
                         f"still composite, recursing")
            self._factor_rec(cof, wd, res, depth + 1)

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

    def _nfs(self, n: int, wd: Path, snfs: "SnfsJob | None" = None):
        if self._run_nfs is None:
            from .backends.nfs import run_nfs
            self._run_nfs = run_nfs
        return self._run_nfs(n, self.ctx, wd / "nfs", snfs=snfs)
