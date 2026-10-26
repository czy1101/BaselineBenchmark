"""MetaX TileOps NSA E2E working-tree snapshot."""

from pathlib import Path

SNAPSHOT_ROOT = Path(__file__).with_name("tileops_snapshot")

__all__ = ["SNAPSHOT_ROOT"]
