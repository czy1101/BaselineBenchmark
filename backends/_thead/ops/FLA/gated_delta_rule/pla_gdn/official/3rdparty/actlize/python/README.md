# Library Generator

This directory contains the `cutlass_library` Python package — utilities for
procedurally enumerating and emitting C++ kernel instantiations consumed by the profiler and library targets.

## Contents

```
python/
  cutlass_library/   # kernel generator scripts (invoked by CMake)
    generator.py     # entry point — called by tools/library/CMakeLists.txt
    manifest.py      # kernel manifest & emission logic
    gemm_operation.py
    library.py
    ppu_utils.py     # PPU-specific tile configuration helpers
    __init__.py

  setup_library.py   # standalone pip-install script for the package
```

## Installation

```bash
cd python
python setup_library.py develop --user
```

Alternatively, you can invoke `generator.py` directly without installing:

```bash
python cutlass_library/generator.py --help
```

## Usage in the build system

CMake calls `generator.py` during the library build to produce kernel
instantiations under `build/tools/library/generated/`.  See
`tools/library/CMakeLists.txt` for the exact invocation.
