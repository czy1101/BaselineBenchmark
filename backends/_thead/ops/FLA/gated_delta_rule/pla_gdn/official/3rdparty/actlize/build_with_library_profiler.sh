#!/bin/bash
#
# Build wrapper for the cutlass3 library + profiler binaries on PPU.
#
# Usage:
#   ./build_with_library_profiler.sh 10000   # build for PPU SDK 10000 (ppu0010)
#   ./build_with_library_profiler.sh 10500   # build for PPU SDK 10500 (ppu0015)
#   ./build_with_library_profiler.sh         # build for both arches (ppu0010;ppu0015)
#
# Identical to build.sh except it also enables CUTLASS_ENABLE_LIBRARY and
# CUTLASS_ENABLE_PROFILER.

set -ex
set -o pipefail

PPU_SDK_ROOT="${PPU_SDK:-${PPU_HOME:-/usr/local/PPU_SDK}}"
if [ ! -x "${PPU_SDK_ROOT}/bin/hgcc" ]; then
  echo "ERROR: hgcc not found at ${PPU_SDK_ROOT}/bin/hgcc." >&2
  echo "       Set PPU_SDK=<path> (or PPU_HOME=<path>) and re-run." >&2
  exit 1
fi

export PATH="${PPU_SDK_ROOT}/bin:${PATH}"

case "${1:-}" in
  10000) PPU_ARCHS="ppu0010" ;;
  10500) PPU_ARCHS="ppu0015" ;;
  "")
    echo "[build_with_library_profiler.sh] no PPU SDK version specified; building for both ppu0010 and ppu0015"
    PPU_ARCHS="ppu0010;ppu0015"
    ;;
  *)
    echo "ERROR: unsupported ppu_type '$1'. Supported: 10000 / 10500 / (empty for both)." >&2
    exit 1
    ;;
esac
echo "[build_with_library_profiler.sh] CUTLASS_PPU_ARCHS=${PPU_ARCHS}"
echo "[build_with_library_profiler.sh] device driver = hgcc (${PPU_SDK_ROOT}/bin/hgcc)"

rm -rf build && mkdir -p build
cd build

cmake .. \
  -DPPU_SDK_ROOT="${PPU_SDK_ROOT}" \
  -DCUTLASS_ENABLE_LIBRARY=ON \
  -DCUTLASS_ENABLE_PROFILER=ON \
  -DCUTLASS_PPU_ARCHS="${PPU_ARCHS}"

rm -f build.log
make -j16 2>&1 | tee -a build.log
