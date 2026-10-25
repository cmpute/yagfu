#!/usr/bin/env bash
#
# yagfu build script: compiles every GPU factoring backend used by yagfu.
#
#   cgbn        (header-only, nothing to compile)
#   gmp-ecm     ECM/P-1 stage 1 on the GPU (cmpute/gmp-ecm, perf-min)
#   gpuecm2     ECM stage 2 on the GPU (uses cmpute/cgbn)
#   cuda-mpqs   SIQS/MPQS on the GPU
#   cuda-sieve  NFS lattice sieving on the GPU
#   msieve-cuda NFS post-processing (filtering / linear algebra / sqrt)
#
# Usage: ./build.sh [options]
#   -a, --arch NN       CUDA compute capability (default: detect from the GPU
#                       in this machine, e.g. 89 for sm_89 / RTX 4090)
#   -j, --jobs N        parallel build jobs (default: CPU count - 2)
#   --max-bits B        largest modulus gpuecm2 compiles stage-2 kernels for
#                       (default 8192; 32768 for the full set, slower to build)
#   --only LIST         comma-separated subset, e.g. "gmp-ecm,gpuecm2"
#   -h, --help          show this help
#
# Every backend's final binary is staged into ./bin (gitignored) so the Python
# wheel can collect them from one place:
#   bin/ecm          gmp-ecm      GPU stage 1  (-gpu -save)
#   bin/gpuecm2      gpuecm2      GPU stage 2  (-resume)
#   bin/cuda-mpqs    cuda-mpqs    SIQS/MPQS
#   bin/cuda-sieve   cuda-sieve   NFS relation collection (built as bench/bench)
#   bin/fbgen        cuda-sieve   factor-base generator (CPU)
#   bin/fbgen_gpu    cuda-sieve   factor-base generator (GPU)
#   bin/msieve       msieve-cuda  NFS filtering / linear algebra / sqrt
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BIN="$ROOT/bin"       # unified output dir consumed by the Python wheel

# ---------------------------------------------------------------------------
# defaults / option parsing
# ---------------------------------------------------------------------------
JOBS="$(nproc 2>/dev/null || echo 4)"
JOBS=$(( JOBS > 2 ? JOBS - 2 : 1 ))   # leave two cores free for the host
SM=""                 # empty => auto-detect the native GPU
MAXBITS="8192"        # gpuecm2 largest modulus size
ONLY=""               # empty => build everything

die() { echo "ERROR: $*" >&2; exit 1; }
log() { printf '==> %s\n' "$*"; }

# copy a built binary into the unified bin/ dir and make it executable
install_bin() {  # <src> <name>
  mkdir -p "$BIN"
  cp -f "$1" "$BIN/$2"
  chmod +x "$BIN/$2"
  log "  staged $BIN/$2"
}

usage() { sed -n '2,27p' "$0"; }

while [ $# -gt 0 ]; do
  case "$1" in
    -a|--arch)     SM="$2"; shift 2 ;;
    -j|--jobs)     JOBS="$2"; shift 2 ;;
    --max-bits)    MAXBITS="$2"; shift 2 ;;
    --only)        ONLY="$2"; shift 2 ;;
    -h|--help)     usage; exit 0 ;;
    *)             die "unknown option: $1" ;;
  esac
done

# ---------------------------------------------------------------------------
# environment checks
# ---------------------------------------------------------------------------
for t in git nvcc cmake make gcc g++; do
  command -v "$t" >/dev/null 2>&1 || die "required tool not found: $t"
done

# make sure every submodule is populated (a fresh clone may not be)
log "initialising submodules"
git -C "$ROOT" submodule update --init --recursive

# CUDA install prefix, derived from the nvcc on PATH (e.g. /usr/local/cuda)
NVCC_BIN="$(command -v nvcc)"
CUDA_ROOT="$(cd "$(dirname "$NVCC_BIN")/.." && pwd)"

# native compute capability (bare digits, e.g. 89), unless the caller set one
if [ -z "$SM" ]; then
  SM="$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader 2>/dev/null \
        | head -1 | tr -d '.')"
  case "$SM" in
    ''|*[!0-9]*) die "could not detect GPU compute capability; pass --arch NN (e.g. --arch 89)" ;;
  esac
fi

log "building for compute capability sm_${SM}, CUDA at $CUDA_ROOT, -j$JOBS"

# ---------------------------------------------------------------------------
# per-component build steps
# ---------------------------------------------------------------------------

build_cgbn() {
  log "cgbn: header-only library, nothing to compile"
}

build_gmp_ecm() {
  log "building gmp-ecm (perf-min, GPU stage 1, CGBN opts ADDCONV/NORMCONV/NORMDROP)"
  cd "$ROOT/gmp-ecm"

  # The perf-min fork ships configure.ac but not a generated configure.
  if [ ! -x ./configure ]; then
    command -v autoreconf >/dev/null 2>&1 || die "autoreconf not found (needed to bootstrap gmp-ecm)"
    log "  running autoreconf -i"
    autoreconf -i
  fi

  local asm="--enable-asm-redc"
  command -v m4 >/dev/null 2>&1 || asm="--disable-asm-redc"

  # --with-cgbn-include points at the cmpute/cgbn submodule (perf-min).  The
  # three ECM_CGBN_* macros gate the branch-free / convergence optimisations in
  # cgbn_stage1.cu and need cgbn_add_conv/cgbn_sub_conv, which perf-min has.
  # They are -D defines that reach nvcc via CPPFLAGS in the .cu.lo rule.
  ./configure \
    --enable-gpu="$SM" \
    --with-cgbn-include="$ROOT/cgbn/include/cgbn" \
    --with-cuda="$CUDA_ROOT" \
    "$asm" \
    CPPFLAGS="-DECM_CGBN_ADDCONV -DECM_CGBN_NORMCONV -DECM_CGBN_NORMDROP"

  make -j"$JOBS"
  install_bin "$ROOT/gmp-ecm/ecm" ecm
}

build_gpuecm2() {
  log "building gpuecm2 (GPU stage 2, CGBN kernels up to ${MAXBITS} bits)"
  cd "$ROOT/gpuecm2"

  # gpuecm2 vendors an upstream NVlabs CGBN in third_party/cgbn/include; swap in
  # the cmpute/cgbn submodule (perf-min) so both stage-1 and stage-2 use it.
  local cgbn_inc="$ROOT/gpuecm2/third_party/cgbn/include"
  if [ ! -L "$cgbn_inc" ] || [ "$(readlink -f "$cgbn_inc")" != "$ROOT/cgbn/include" ]; then
    log "  linking third_party/cgbn/include -> $ROOT/cgbn/include"
    rm -rf "$cgbn_inc"
    ln -s "$ROOT/cgbn/include" "$cgbn_inc"
  fi

  cmake -S . -B build \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_CUDA_ARCHITECTURES="$SM" \
    -DGPUECM2_MAX_BITS="$MAXBITS"
  cmake --build build -j"$JOBS"

  # The gpuecm2 binary links gpuecm2core + cudart statically, so it is
  # self-contained (only system libgmp/libcuda are needed at run time).
  # libgpuecm2.so / gpuecm2.h are the C API and only needed for the -gpu2 build.
  install_bin "$ROOT/gpuecm2/build/gpuecm2" gpuecm2
}

build_cuda_mpqs() {
  log "building cuda-mpqs (SIQS/MPQS)"
  cd "$ROOT/cuda-mpqs"
  cmake -B build -DGPU_TARGET="$SM" -DCMAKE_BUILD_TYPE=Release
  cmake --build build -j"$JOBS"
  install_bin "$ROOT/cuda-mpqs/build/tests/cuda-mpqs" cuda-mpqs
}

build_cuda_sieve() {
  log "building cuda-sieve (NFS relation collection)"
  cd "$ROOT/cuda-sieve"
  make -C bench -j"$JOBS" GPU_ARCH="$SM"
  install_bin "$ROOT/cuda-sieve/bench/bench" cuda-sieve
  install_bin "$ROOT/cuda-sieve/bench/fbgen" fbgen
  install_bin "$ROOT/cuda-sieve/bench/fbgen_gpu" fbgen_gpu
}

build_msieve() {
  log "building msieve-cuda (NFS post-processing, sm_${SM})"
  cd "$ROOT/msieve-cuda"
  make -j"$JOBS" all CUDA="$SM"
  install_bin "$ROOT/msieve-cuda/msieve" msieve
}

# ---------------------------------------------------------------------------
# dispatch
# ---------------------------------------------------------------------------
if [ -n "$ONLY" ]; then
  IFS=',' read -r -a COMPONENTS <<< "$ONLY"
else
  COMPONENTS=(cgbn gmp-ecm gpuecm2 cuda-mpqs cuda-sieve msieve-cuda)
fi

for c in "${COMPONENTS[@]}"; do
  case "$c" in
    cgbn)                  build_cgbn ;;
    gmp-ecm)               build_gmp_ecm ;;
    gpuecm2)               build_gpuecm2 ;;
    cuda-mpqs)             build_cuda_mpqs ;;
    cuda-sieve)            build_cuda_sieve ;;
    msieve|msieve-cuda)    build_msieve ;;
    *)                     die "unknown component: $c" ;;
  esac
done

log "done; binaries staged in $BIN:"
[ -d "$BIN" ] && ls -1 "$BIN" | sed 's/^/  /'
