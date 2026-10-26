"""Helpers shared by MetaX correctness tests."""

import torch


def is_metax_available() -> bool:
    """Return whether this process uses a MetaX-enabled PyTorch build."""
    return torch.cuda.is_available() and (
        hasattr(torch.version, "metax") or "metax" in torch.__version__.lower()
    )
