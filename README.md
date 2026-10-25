# yagfu — yet another GPU factorization utility

A [yafu](https://github.com/bbuhrow/yafu)-style integer factorization driver whose
backend is entirely GPU programs. The driver is a pure-stdlib Python package
(`yagfu`, installed as the `yagfu` command); the heavy lifting is done by the
binaries built from the submodules, bundled into a wheel for copy-install on
same-environment machines.

Pipeline per composite (yafu's three retained modules: fixed prework budget,
cumulative ECM work ledger, independent final algorithm selector):

```
trial division      pollard rho (Brent)   GPU ECM                SIQS / NFS
all primes < 256    x²+3, x²+2, x²+1      ecm -gpu (stage 1)     cuda-mpqs
                                         gpuecm2   (stage 2)     msieve + cuda-sieve
```

Deviations from yafu (by design): no P−1 / P+1 / Fermat; trial division stops at
256 (not 10 000); ECM always uses GMP-ECM's default B2 (never pinned); ECM stage
runs param 3 with the yafu `ecm_data_param3` work table; at least one B1=2000
GPU batch always runs even when yafu would skip ECM (our prework is lighter);
primality is our own BPSW (MR base 2 + strong Lucas), with deterministic
SPRP(2, 7, 61) below 2³².

Scheduling summary:

* rho: three polynomials in order, `rhomax=1000` batch-GCD rounds each.
* ECM: target depth = plan × digits (light 2/9, normal 4/13, deep 1/3),
  t-levels t15..t65 at B1 2e3..85e7, work ledger accumulates across B1 levels
  and survives cofactor splits, each level capped at its diagonal curve budget.
* Final selector: cuda-mpqs below the crossover (default 105 digits, measured
  on RTX 4090), msieve+cuda-sieve GNFS at/above it; each is the other's
  fallback.

## Submodules

| directory     | repository                           | branch                   | role                          |
|---------------|--------------------------------------|--------------------------|-------------------------------|
| `cgbn`        | cmpute/cgbn                          | `perf-min`               | CUDA big-num library (shared) |
| `gmp-ecm`     | cmpute/gmp-ecm                       | `perf-min`               | ECM/P-1 stage 1 on GPU        |
| `gpuecm2`     | ebranger/gpuecm2                     | `main`                   | ECM stage 2 on GPU            |
| `cuda-mpqs`   | drjanosch42/cuda-mpqs                | `main`                   | SIQS/MPQS on GPU              |
| `cuda-sieve`  | kyleaskine/cuda-sieve                | `main`                   | NFS relation collection       |
| `msieve-cuda` | cmpute/msieve-cuda                   | `msieve-lacuda-nfsathome`| NFS filtering / LA / sqrt     |

> Note: `gpuecm2` and `cuda-mpqs` have no `master` branch; their default branch is
> `main` (recorded above).

## Building the backends

```sh
./build.sh                 # detect the native GPU (sm_89 here) and build everything
./build.sh -j 8            # limit parallel jobs
./build.sh -a 89           # explicit compute capability
./build.sh --max-bits 32768  # gpuecm2 stage-2 kernels for numbers up to 32768 bits
./build.sh --only gmp-ecm,gpuecm2   # build just the ECM stage
```

`gmp-ecm` and `gpuecm2` are both compiled against the `cgbn` submodule (`perf-min`).
`gmp-ecm` is built with `-DECM_CGBN_ADDCONV -DECM_CGBN_NORMCONV -DECM_CGBN_NORMDROP`,
which enable the branch-free warp-uniform CGBN optimisations in `cgbn_stage1.cu`
(see the `perf-min` branch commit history for details).

Every backend's final binary is staged into the unified `bin/` directory (gitignored):

```
bin/ecm          # gmp-ecm,  GPU stage 1
bin/gpuecm2      # gpuecm2,  GPU stage 2
bin/cuda-mpqs    # cuda-mpqs, SIQS/MPQS
bin/cuda-sieve   # cuda-sieve, NFS relation collection (renamed from bench/bench)
bin/fbgen        # cuda-sieve, CPU factor-base generator
bin/fbgen_gpu    # cuda-sieve, GPU factor-base generator
bin/msieve       # msieve-cuda, NFS post-processing
```

## The yagfu package

```sh
# dev: run from the repo without installing
PYTHONPATH=src YAGFU_BIN=$PWD/bin python3 -m yagfu.cli <N>

# tests (unit, no GPU needed; gmpy2 is used as the reference oracle only)
PYTHONPATH=src python3 -m pytest tests/

# wheel (~85 MB compressed, binaries inside; installs offline)
scripts/build_wheel.sh          # stages bin/ into src/yagfu/bin/ and builds dist/
pip install dist/yagfu-*.whl    # then `yagfu` is on PATH (pipx-compatible)
```

Binary lookup order: `--bin-dir` > `$YAGFU_BIN` > bundled `yagfu/bin/` > `$PATH`.

### CLI

```
yagfu 1234567890...                 # one number
yagfu -f numbers.txt                # one number per line (# comments OK)
yagfu -db -f batch.txt              # factordb.com: fetch, factor what's left, report back
```

Key options: `--plan {light,normal,deep}` (ECM depth), `--xover 105`
(SIQS→NFS crossover), `--device`, `--threads`, `--sieve-procs`, `--timeout`,
`--workdir/--keep`, `--json` (JSONL output), `-v` (progress log to stderr).

Output goes to stdout, one result line per input:

```
$ yagfu 3946068388498480527218916724621611671856950960754487550101128852810403
3946068388498480527218916724621611671856950960754487550101128852810403 = 47360480608224695728619362875791753 * 83319855242626065532276429111727051
  [PRP35 PRP35]  4.3s
```

Factors are tagged `P##` (primality proven: trial division or < 2³² SPRP) or
`PRP##` (BPSW). Exit code 0 only if every input factored completely.

### factordb integration (`-db` / `--factor-db`)

* Before factoring: `get_number` — `FF`/`P`/`PR` records are answered straight
  from the database; `CF` records have their known prime factors divided out and
  only the remaining composite cofactor is factored locally.
* After factoring: newly found prime/PRP factors are reported
  (`report_factors`, with credit when `~/.factordb_token` exists), and factors
  that ECM found (25–100 digits) get their curve parameters submitted
  (`report_ecm` with param 3, sigma, B1, and the effective default B2).
* Reporting never blocks the factorization: all failures are logged and skipped.
  `--db-no-report` queries only; `--db-dry-run` prints payloads instead of
  sending.

Transport: JSON-RPC 2.0 at `https://factordb.com:4059/rpc`, ≥0.5 s between
calls, exponential backoff on 429/5xx (same contract as the reference
`fdb_yafu_loop.py`).

## Cloning fresh

```sh
git clone --recurse-submodules <this repo>
```
