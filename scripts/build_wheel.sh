#!/usr/bin/env bash
# Build the yagfu wheel with the GPU backend binaries bundled inside.
#
#   scripts/build_wheel.sh              # stage bin/ (as built by ./build.sh)
#                                       # and build dist/yagfu-*.whl
#   scripts/build_wheel.sh --no-stage   # rebuild without refreshing binaries
#
# The wheel is py3-none-any (pure Python + data files) and installs offline;
# binaries land in <site-packages>/yagfu/bin/ and are found automatically.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BIN="$ROOT/bin"
PKG_BIN="$ROOT/src/yagfu/bin"

STAGE=1
[[ "${1:-}" == "--no-stage" ]] && STAGE=0

if [[ $STAGE == 1 ]]; then
    if [[ ! -x "$BIN/ecm" ]]; then
        echo "error: $BIN/ecm missing; run ./build.sh first (or pass --no-stage)" >&2
        exit 1
    fi
    echo "== staging binaries into $PKG_BIN =="
    rm -rf "$PKG_BIN"
    mkdir -p "$PKG_BIN"
    for f in "$BIN"/*; do
        if [ -d "$f" ]; then
            cp -r "$f" "$PKG_BIN/"        # e.g. cub/sort_engine.so for msieve
        else
            install -m 755 "$f" "$PKG_BIN/"
        fi
    done
    du -sh "$PKG_BIN"
else
    echo "== skipping staging (using existing $PKG_BIN) =="
fi

echo "== building wheel =="
cd "$ROOT"
rm -rf dist build src/yagfu.egg-info
pip wheel --no-deps --no-build-isolation -w dist .

echo "== done =="
ls -lh dist/
