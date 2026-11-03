"""Lazy access to the bundled PLA implementations used for GDN comparisons.

These entries preserve PLA's own call signatures; they do not replace the
registered FlagAttention operators. See README.md for supported inputs.
"""
from importlib import import_module

__all__ = ["chunk_forward", "recurrent_precomputed", "packed_decode"]


def __getattr__(name):
    entries = {
        "chunk_forward": (
            ".official.pla.prefill.flashqla", "chunk_gated_delta_rule_fwd"
        ),
        "recurrent_precomputed": (".precomputed.pla_precomputed", "run"),
        "packed_decode": (
            ".official.pla.decode",
            "fused_sigmoid_gating_delta_rule_forward_k_last_packed",
        ),
    }
    if name not in entries:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    # Load the framework first so its shared-library symbols are available.
    import torch  # noqa: F401

    module, attribute = entries[name]
    value = getattr(import_module(module, __name__), attribute)
    globals()[name] = value
    return value
