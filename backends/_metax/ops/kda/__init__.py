"""Native MACA KDA source record."""

from pathlib import Path

NATIVE_SOURCE = Path(__file__).with_name("native") / "chunk_kda_fwd.cu"

__all__ = ["NATIVE_SOURCE"]
