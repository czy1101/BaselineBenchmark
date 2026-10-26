"""Shared MetaX correctness-test configuration."""

import pytest


def pytest_sessionstart(session: pytest.Session) -> None:
    try:
        import triton.knobs
    except ImportError:
        return
    # FlagTree's autotuner reads this vendor knob unconditionally.
    triton.knobs.autotuning.adjust_block_size = False
