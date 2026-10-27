# yagfu — yet another GPU factorization utility

`yagfu` factors integers on NVIDIA GPUs. It is a [yafu](https://github.com/bbuhrow/yafu)-style
driver — automatic method selection, work accounting, recursive cofactor handling —
except every heavy stage runs on the GPU. The driver itself is a pure-stdlib Python
package installed as the `yagfu` command; the seven backend binaries it drives are
built from the submodules of this repository and can be bundled into a single wheel
for copy-install on same-environment machines.

```
input composite
   │
   ├─ trial division           all 54 primes below 256
   ├─ perfect-power strip      b^e extracted, recursed
   ├─ Pollard rho (Brent)      x²+3, x²+2, x²+1 in order; rhomax=1000 batch-GCDs
   ├─ GPU ECM                  gmp-ecm -gpu (stage 1)  →  gpuecm2 (stage 2)
   │                           B1 ladder 2e3…85e7, YAFU work ledger, default B2
   └─ final selector           < 105 digits: cuda-mpqs (SIQS)
                               ≥ 105 digits: msieve polyselect + cuda-sieve
                                             sieving + msieve filter/LA/sqrt
                               (each is the other's fallback)
```

Every factor found at any stage is verified for divisibility, classified by a
self-implemented Baillie–PSW test, and recursed on if composite; a result is only
reported complete when the collected primes multiply back to the input.

## Algorithm strategy

The scheduler keeps the three ideas that make yafu's `factor()` work, adapted to a
GPU cost model:

**Fixed prework budget.** Trial division is deliberately shallow (256, not yafu's
10 000) and rho runs exactly yafu's three-polynomial sweep with `rhomax=1000`
(batch-GCD rounds, not squarings). On GPU hardware deeper CPU prework does not pay:
the first ECM batch costs ~0.4 s and covers more ground than any of it.

**Cumulative ECM work ledger.** Eleven levels t15…t65 at B1 = 2e3, 11e3, 5e4, 25e4,
1e6, 3e6, 11e6, 43e6, 11e7, 26e7, 85e7. Curves run at level j contribute
`c / M[i][j]` of normalized work toward *every* depth level i, where M is yafu's
`ecm_data_param3` curve table (the GPU forces parametrization 3, so that — not the
param-0 table — is the right one here). The achieved depth is `t[i-1] + 5·s[i]`
for the first incomplete level i. The target depth is `plan × digits`
(light 2/9, **normal 4/13**, deep 1/3). When ECM splits the number, the ledger
and the ladder position carry over to the cofactor unchanged — a curve run on a
number is equally a curve on each of its divisors — so only the number being
factored is swapped, never the work already spent. B2 is never pinned — gpuecm2
uses GMP-ECM's default (logged per batch), which maximizes the per-curve find
probability.

Two deliberate deviations from yafu: P−1 / P+1 / Fermat are skipped entirely, and
at least one B1=2000 batch always runs even when yafu would jump straight to the
sieve (`target < 15 && digits ≤ 45`) — cheap insurance given the shallower prework.

**Independent final selector.** Below the crossover digit count the job goes to
cuda-mpqs, at or above it to the GNFS pipeline; the default crossover is 105,
extrapolated from measured total-wall times on an RTX 4090 (mpqs wins through
c100: 117 s vs 167 s). NFS parameters come from a per-size anchor table
(c70…c150, piecewise-linear), polyselect is quality-gated (below), and the
sieving loop is closed by msieve's filter verdict ("found N cycles, need M") with
geometric relation top-ups. The SIQS-vs-NFS choice is made once, after the ECM
ladder, on the cofactor that actually reaches the sieves.

**Polyselect (multi-GPU, inside the NFS stage).** With several devices the
polynomial search runs one `msieve -np` process per card over disjoint
leading-coefficient ranges — msieve's default range (`1 .. N^(1/(d(d−1)))/3`,
replicated in `nfs.py`); only multiples of msieve's ad-sieve
multiplier are ever searched and each process aligns its begin upward, so the
slices partition the searched set exactly, with zero cross-card traffic. Each
process gets its own `poly-*` subdir and `CUDA_VISIBLE_DEVICES` pin, `-t`
divided across the cards (the GPU engine caps its producer pool at 4 anyway;
`-t` also sizes the stage-2/3 CPU pool). The search is **quality-gated**:
the bundled msieve fork streams every new overall-best Murphy E to its log
(`polyselect best E = … (ad = …)`) while the root optimizer keeps pace with
the sweep, and yagfu watches those lines against the bar msieve itself
prints ("expecting poly E from X to > Y") — the moment the best E reaches
the bar, the whole search is SIGINTed and msieve writes out its best
candidate. A search that never meets the bar ends with msieve's own range.
The winner's `msieve.fb`/`np.log` are promoted to the NFS workdir root,
which is exactly the single-process layout the downstream stages and resume
paths expect. A shard that finds nothing is only retried on the CPU Gerbicz
engine when *no* shard produced a polynomial (an unlucky empty slice is not
worth CPU hours when the others hold a polynomial). An interrupted search is
torn down cooperatively — SIGINT at the bar or cap, SIGTERM on Ctrl+C — and
msieve saves its best-so-far candidate on the way out, so a rerun with the
same workdir reuses the salvaged polynomial instead of re-searching.
(The pre-port GPU stage-1 emitted no candidates at c110–c119; the trunk-port
engine does — verified at c110 and c137.)
`--poly-budget` forces the legacy shape: one full-range search under a hard
wall cap, best-so-far polynomial at the cap.

**Multi-GPU.** `--device` takes a comma-separated list (`--device 0,1` — the
ordinals mean what they mean to CUDA in your shell, i.e. indices into
`CUDA_VISIBLE_DEVICES` when it is set) or the form `--device all`, which
expands to every card visible in this shell (the `CUDA_VISIBLE_DEVICES`
entries when set, else nvidia-smi's list). Every backend subprocess is pinned to
one physical card via `CUDA_VISIBLE_DEVICES` and addresses it as ordinal 0,
which is also what makes single-card `--device 3` land every stage —
including gpuecm2 stage 2 (its CLI has no device flag) and the GPU polyselect
(msieve `-np` has none either) — on that exact card. Per stage:

* **ECM stage 1** fans out one `ecm -gpu` process per card, counts split by
  an EWMA of each card's measured curves/s; residues are merged into one
  gpuecm2 stage-2 run pinned to the first card.
* **NFS sieving** splits each q-band into `--sieve-procs` × cards cuda-sieve
  chunks, chunk *i* round-robbed onto card *i % k* (adjacent q ranges land on
  different cards). `--sieve-procs` is per card. The msieve `-nc1` filter
  that closes each round runs in the background, overlapped with the next
  band's sieving — it only reads `msieve.dat` and a band is appended only
  after the verdict covering the data so far has been collected, so the
  stop check stays exact while the filter cost (25–35% of wall time on
  c150-class runs) hides under GPU time.
* **SIQS** switches to cuda-mpqs's built-in cluster mode from c80 up: a
  coordinator on the first card plus one localhost TCP worker per remaining
  card, work split by SM×clock weight; below c80 (or with one card) it stays
  a single process, and a failed cluster falls back to the solo run.
  All inter-card traffic is relations over TCP through host memory — no
  NVLink/peer-to-peer GPU path is used anywhere.
* **Polyselect** runs one `msieve -np` per card over disjoint, quality-gated
  leading-coefficient blocks (best polynomial wins; see above); **GPU Lanczos
  and NFS sqrt** are single-card (first device): this msieve build has no
  multi-GPU Lanczos without an MPI build.


Primality: Baillie–PSW (base-2 strong MR + strong Lucas, Selfridge method A);
below 2³² a deterministic SPRP(2, 7, 61) is used and such factors are reported as
proven primes (`P##`) rather than probable primes (`PRP##`).

## Quick start

```sh
# 1. build the backends (needs CUDA toolkit; auto-detects the GPU)
./build.sh                              # binaries land in bin/

# 2. build the wheel (~85 MB compressed, all binaries inside)
scripts/build_wheel.sh

# 3. install on this machine — or any machine with the same OS/GPU/driver
pip install dist/yagfu-0.1.0-py3-none-any.whl     # pipx works too
yagfu 123456789...
```

Without a wheel, run from the repository directly:

```sh
PYTHONPATH=src YAGFU_BIN=$PWD/bin python3 -m yagfu.cli <N>
```

Binary lookup order: `--bin-dir` > `$YAGFU_BIN` > the wheel-bundled `yagfu/bin/`
> `$PATH`.

## CLI

```
yagfu <number>                 factor one decimal number
yagfu -f numbers.txt           factor a file: one number per line (# comments OK)
yagfu -db <number>             consult factordb.com first, report results back
yagfu -f batch.txt -db --json  batch mode, JSONL output
```

| option | default | meaning |
|---|---|---|
| `-f, --file FILE` | — | input file, one number per line |
| `-db, --factor-db` | off | factordb fetch + report (see below) |
| `--plan {light,normal,deep}` | `normal` | ECM target depth (2/9, 4/13, 1/3 of digits) |
| `--ecm-only` | off | never run SIQS/NFS: climb every ECM level to its full budget, then sample t65 forever until the number factors or Ctrl+C |
| `--no-ecm` | off | skip the ECM stage entirely: TD → rho → SIQS/NFS (rerun convenience when the pretest is known useless) |
| `--xover N` | 105 | SIQS → NFS crossover in digits |
| `--device N[,N…]` | 0 | CUDA device(s); a list fans GPU ECM stage 1, polyselect and NFS sieving out across the cards (ECM counts/polyselect coefficient ranges/sieve chunks split per card), SIQS runs cuda-mpqs cluster mode from c80; ECM stage 2 and Lanczos use the first device |
| `--threads N` | nproc−2 | CPU threads for msieve stages |
| `--sieve-procs N` | 1 | concurrent cuda-sieve processes per q-band, per GPU |
| `--ecm-stage2-maxmem MB` | 90% of VRAM | gpuecm2 stage-2 memory cap — limits dF via GMP-ECM's memory model (smaller blocks, same B2 coverage); `0` = no cap |
| `--timeout S` | none | per-number wall budget |
| `--workdir DIR` | temp dir | scratch base directory; a number's workdir is removed only when it factors completely, kept on failure/Ctrl+C (`--keep` keeps always). An NFS workdir is a resume checkpoint: rerun the same number with the same `--workdir` to continue a killed job |
| `--bin-dir DIR` | — | backend binary directory |
| `--json` | off | one JSON object per input on stdout |
| `--db-token FILE` | `~/.factordb_token` | factordb token file (one line) |
| `--db-no-report` | off | with `-db`: query only, never report |
| `--db-dry-run` | off | with `-db`: print RPC payloads instead of sending |
| `--poly-budget S` | 2h | NFS polyselect total wall cap, all sharded processes together: the search runs to completion (cooperative SIGINT at the cap saves the best-so-far polynomial) |
| `--topup-frac F` | 1.0 | NFS relation target growth per filter round (GPU sieving is cheap, `-nc1` passes are not — grow fast, save filter rounds) |
| `-v / -q` | — | more detail in the stderr log / silence stderr logs |

Results go to stdout, one primary line per input; YAFU-style progress logs go
to stderr by default (banner, trial-division summary, every factor found with
its `P##`/`PRP##` tag and the stage that found it, ECM batch progress with
B1/B2, per-curve rate and a level-completion ETA, NFS sieve rounds with
relation rate and ETA, and msieve's own `linear algebra at X%, ETA` lines
forwarded live during the matrix step). `-v` adds detail (memory caps,
per-band sieve updates), `-q` silences stderr entirely. The complete log is
also appended to `yagfu.log` in the working directory (in append mode, kept
across runs):

```
$ yagfu 3946068388498480527218916724621611671856950960754487550101128852810403
[20:33:52] fac: factoring 3946068388498480527218916724621611671856950960754487550101128852810403 (c70)
[20:33:53] ecm: c70, plan=normal, target=t21.5
[20:33:53] ecm: B1=2,000, B2=147,396, 128 curves, 3.3ms/curve -> t15.37 (target t21.54)
[20:33:54] ecm: B1=11,000, B2=1.87M, 128 curves, 4.1ms/curve -> t20.27 (target t21.54)
[20:33:55] ecm: B1=50,000, B2=12.7M, 128 curves, 8.4ms/curve -> t22.72 (target t21.54)
[20:33:55] siqs: cuda-mpqs on c70 (lp1=None)
[20:33:56] siqs: rc=0 in 1.6s (ok)
[20:33:56] fac: PRP35 = 47360480608224695728619362875791753 (siqs)
[20:33:56] fac: PRP35 = 83319855242626065532276429111727051 (siqs)
3946068388498480527218916724621611671856950960754487550101128852810403 = 47360480608224695728619362875791753 * 83319855242626065532276429111727051
  [PRP35 PRP35]  4.3s
```

With `--json`, each input produces one line with `n`, `complete`, `factors`
(each with `value`/`tag`/`proven`/`source`, plus `ecm: {b1, b2, sigma, stage}`
provenance for ECM finds), stage timings, and counters. Exit code is 0 only if
every input factored completely.

Ctrl+C is handled gracefully everywhere (including inside GPU backends and
parallel cuda-sieve bands): the partial factorization found so far is printed,
the per-number workdir is kept, and the exit code is 130 — no traceback spam.

## factordb integration (`-db`)

Before factoring, the number's record is fetched via JSON-RPC
(`https://factordb.com:4059/rpc`, the same transport and politeness rules as the
reference `fdb_yafu_loop.py`: ≥0.5 s between calls, exponential backoff on
429/5xx):

* `FF` / `P` / `PR` — answered straight from the database, no GPU work;
* `CF` — known prime factors are divided out and only the remaining composite
  cofactor is factored locally;
* `C` / unknown — full local factorization.

After factoring, newly found prime/PRP factors are submitted
(`report_factors`, with `credit: true` when a token is configured). Factors that
ECM found (25–100 digits) additionally get their curve parameters submitted via
`report_ecm` — param 3, the sigma of the finding curve (parsed from the GPU
stage-1/stage-2 output), B1, and the effective default B2; factordb accepts
these as record orders. Reporting failures are logged and skipped; they never
affect the factorization result or the exit code.

The token is read from `--db-token FILE` (default `~/.factordb_token`, one line)
and sent in the `X-Fdb-User-Token` header; it is never printed. Without a token
everything still works, anonymously and without credit.

## Performance

RTX 4090, balanced semiprimes (`benchmarks/numbers.json`, fixed seed), single
GPU, measured wall times:

| digits | SIQS (cuda-mpqs) | GNFS (msieve + cuda-sieve + msieve) | yagfu auto (incl. pretest) |
|---:|---:|---:|---:|
| 70 | 1.6 s | 59.6 s | 4.3 s |
| 80 | 6.1 s | 95.9 s | ≈7 s * |
| 90 | 17.2 s | 132 s | 135 s (`--force nfs`) |
| 100 | 117 s | 167 s | ≈130 s * |

\* measured at c70 and c90; the c80/c100 entries are sums of the measured
pretest and method times, not separate end-to-end runs.

The ECM pretest itself is cheap: reaching the normal-plan target on a c100
(t30.8) costs ~17 s across four B1 levels. A 15–30 digit factor anywhere in the
pipeline is typically found in the first or second batch.

Reproduce and re-derive the crossover with the benchmark harness, which
drives the package backends directly and fits the YAFU-tune exponential
model `T(d) = A·e^(kd)` per method:

```sh
python3 scripts/bench_factor.py gen  --digits 70,80,90,100   # fixed-seed semiprimes
python3 scripts/bench_factor.py run  --method all            # ~10 min on a 4090
python3 scripts/bench_factor.py report
python3 scripts/bench_factor.py crossover   # fits both curves, prints --xover
```

## Repository layout

```
src/yagfu/            the Python package
  cli.py                CLI, per-number loop, factordb glue
  engine.py             scheduler: stage chain, recursion, final selector
  ecmledger.py          YAFU param-3 ECM work ledger (t15…t65)
  arith.py              trial division, Brent rho, perfect powers, iroot
  primes.py             BPSW + deterministic SPRP(2,7,61)
  params.py             NFS/SIQS parameter tables and interpolation
  fdb.py                factordb JSON-RPC client
  backends/
    ecm.py                ecm -gpu -save / gpuecm2 -resume driver, sigma parsing
    siqs.py               cuda-mpqs wrapper
    nfs.py                polyselect → sieve bands → filter/LA/sqrt pipeline
    runner.py             subprocess plumbing
scripts/build_wheel.sh  stages bin/ into the package and builds the wheel
scripts/bench_factor.py SIQS vs NFS benchmark harness and crossover fitter
                        (drives the package backends; stdlib only)
tests/                  112 unit tests (no GPU required; gmpy2 is only used
                        as a reference oracle for the primality tests)
bin/                    build output (gitignored)
```

Backends (git submodules):

| directory     | repository            | role                                |
|---------------|-----------------------|-------------------------------------|
| `cgbn`        | cmpute/cgbn (`perf-min`) | CUDA big-num library (shared)   |
| `gmp-ecm`     | cmpute/gmp-ecm (`perf-min`) | ECM stage 1 on GPU              |
| `gpuecm2`     | ebranger/gpuecm2 (`main`)   | ECM stage 2 on GPU              |
| `cuda-mpqs`   | drjanosch42/cuda-mpqs (`main`) | SIQS/MPQS on GPU            |
| `cuda-sieve`  | kyleaskine/cuda-sieve (`main`) | NFS relation collection     |
| `msieve-cuda` | cmpute/msieve-cuda (`msieve-lacuda-nfsathome`) | NFS polyselect/filter/LA/sqrt |

`gmp-ecm` and `gpuecm2` compile against the `cgbn` submodule; `gmp-ecm` is built
with `-DECM_CGBN_ADDCONV -DECM_CGBN_NORMCONV -DECM_CGBN_NORMDROP` (branch-free
warp-uniform CGBN optimisations, see the `perf-min` branch history).

## Building the backends

```sh
git clone --recurse-submodules <this repo>
./build.sh                                   # detect the GPU (sm_89 here), build all
./build.sh -a 89 -j 8                        # explicit arch, limited jobs
./build.sh --max-bits 32768                  # gpuecm2 kernels up to 32768-bit inputs
./build.sh --only gmp-ecm,gpuecm2            # rebuild just the ECM stage
```

All binaries are staged into `bin/`: `ecm`, `gpuecm2`, `cuda-mpqs`,
`cuda-sieve`, `fbgen`, `fbgen_gpu`, `msieve`.

## Development

```sh
PYTHONPATH=src python3 -m pytest tests/      # 112 unit tests, no GPU needed
PYTHONPATH=src YAGFU_BIN=$PWD/bin python3 -m yagfu.cli -v <N>   # live run
scripts/build_wheel.sh --no-stage            # rebuild wheel without re-staging
```

**Multi-GPU, deliberately not (yet).** Multi-GPU Lanczos needs an MPI build
of msieve (`build.sh` does not set `MPI=1`); and `-f` batch mode processes
numbers serially — an opt-in `--shard-gpus` (one pipeline per card over a
dynamic number queue) is the planned next step.

Per-number workdirs (ECM residue files, sieve bands, msieve logs) are removed
only when a number factors completely. A failed, crashed, or interrupted run
keeps its workdir for post-mortem, together with the base scratch dir (its
path is printed). `--keep` disables the removal entirely.

**NFS resume.** An NFS job checkpoints itself inside its per-number workdir:
the selected polynomial (`c{digits}.job`) and every sieved relation
(`msieve.dat`, appended incrementally — band files are deleted right after
appending, so relations occupy exactly one copy on disk) survive a kill, and
`nfs-state.json` pins the sieve position (next q, relation target/have). To
continue a killed job, rerun the same number with the same `--workdir`: the
polyselect and all sieved relations are reused and sieving continues at the
next unsieved q; only the interrupted band is redone. Resume needs an
explicit `--workdir` (the default temp dir has a fresh random name each run);
combine with `--no-ecm` to skip the ECM pretest on the rerun as well.
