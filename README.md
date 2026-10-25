# yagfu — yet another GPU factorization utility

A [yafu](https://github.com/bbuhrow/yafu)-style integer factorization driver whose
backend is entirely GPU programs. The driver itself is Python (using `gmpy2` for big
integer arithmetic; no other third-party dependencies).

Pipeline (mirrors yafu, but every heavy stage is on the GPU):

```
trial division  ->  pollard rho  ->  ECM / P-1         ->  SIQS / NFS
(gmpy2)             (gmpy2)         gmp-ecm + gpuecm2     cuda-mpqs / cuda-sieve + msieve
```

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

## Building

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

Build outputs:

```
gmp-ecm/ecm
gpuecm2/build/gpuecm2
cuda-mpqs/build/tests/cuda-mpqs
cuda-sieve/bench/bench
msieve-cuda/msieve
```

## Cloning fresh

```sh
git clone --recurse-submodules <this repo>
```
