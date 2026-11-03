#!/bin/bash
#
# Top-level build wrapper for the cutlass3 open-source PPU port.
#
# Usage:
#   ./build.sh 10000     # build for PPU SDK 10000 (arch ppu0010)
#   ./build.sh 10500     # build for PPU SDK 10500 (arch ppu0015)
#   ./build.sh           # build both arches (ppu0010;ppu0015)
#
# The PPU SDK location is taken from $PPU_SDK (preferred) or $PPU_HOME.
# The CMake toolchain bootstrap lives in cmake/PPUToolchain.cmake.

set -ex
set -o pipefail

# ---- locate PPU SDK ---------------------------------------------------------
PPU_SDK_ROOT="${PPU_SDK:-${PPU_HOME:-/usr/local/PPU_SDK}}"
if [ ! -x "${PPU_SDK_ROOT}/bin/hgcc" ]; then
  echo "ERROR: hgcc not found at ${PPU_SDK_ROOT}/bin/hgcc." >&2
  echo "       Set PPU_SDK=<path> (or PPU_HOME=<path>) and re-run." >&2
  exit 1
fi

export PATH="${PPU_SDK_ROOT}/bin:${PATH}"

# ---- resolve target arch list from positional arg ---------------------------
case "${1:-}" in
  10000) PPU_ARCHS="ppu0010" ;;
  10500) PPU_ARCHS="ppu0015" ;;
  "")
    echo "[build.sh] no PPU SDK version specified; building for both ppu0010 and ppu0015"
    PPU_ARCHS="ppu0010;ppu0015"
    ;;
  *)
    echo "ERROR: unsupported ppu_type '$1'. Supported: 10000 / 10500 / (empty for both)." >&2
    exit 1
    ;;
esac
echo "[build.sh] CUTLASS_PPU_ARCHS=${PPU_ARCHS}"
echo "[build.sh] device driver = hgcc (${PPU_SDK_ROOT}/bin/hgcc)"

# ---- configure & build ------------------------------------------------------
rm -rf build && mkdir -p build
cd build

cmake .. \
  -DPPU_SDK_ROOT="${PPU_SDK_ROOT}" \
  -DCUTLASS_PPU_ARCHS="${PPU_ARCHS}"

rm -f build.log
make -j8 2>&1 | tee -a build.log
