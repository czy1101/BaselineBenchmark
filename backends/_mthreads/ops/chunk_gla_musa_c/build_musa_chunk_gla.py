# Copyright 2026 FlagOS Contributors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.

import os
from pathlib import Path

SRC = Path(__file__).resolve().parent
# Build from the standalone baseline package directory.  Building from the
# repository root would make setuptools inherit the main project's src-layout
# and incorrectly copy the extension to src/BaselineBenchmark/....
os.chdir(SRC)

# The main FlagAttention project obtains its version from setuptools-scm.
# This standalone extension must also build from source archives without Git
# metadata, so keep it independent from the parent project's version lookup.
os.environ.setdefault("SETUPTOOLS_SCM_PRETEND_VERSION", "0.0.0")

from setuptools import setup  # noqa: E402
from torch_musa.utils.musa_extension import (  # noqa: E402
    BuildExtension,
    MUSAExtension,
)

mcc_args = ["-O3"]
if os.environ.get("MUSA_GLA_FAST_MATH", "0") == "1":
    # MUSA's fast math mode maps device transcendental functions such as
    # expf to the architecture's fast approximation.  It is opt-in because
    # the accuracy benchmark should be run again after enabling it.
    mcc_args.append("-use_fast_math")


setup(
    name="baselinebenchmark_mthreads_musa_chunk_gla",
    version="0.0.0",
    ext_modules=[
        MUSAExtension(
            # The shared object is placed next to musa_chunk_gla.py and is
            # loaded there through ``from . import _musa_chunk_gla``.
            name="_musa_chunk_gla",
            sources=[
                str(SRC / "musa_chunk_gla.cpp"),
                str(SRC / "musa_chunk_gla_kernel.mu"),
            ],
            libraries=["mublas"],
            extra_compile_args={"cxx": ["-O3"], "mcc": mcc_args},
        )
    ],
    cmdclass={"build_ext": BuildExtension},
)
