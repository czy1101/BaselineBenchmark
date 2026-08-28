"""Thin adapter for the externally installed MetaX SageAttention wheel."""

from importlib import import_module
from importlib.metadata import PackageNotFoundError, version

EXPECTED_VERSION = "2.0.1+metax3.7.2.0torch2.8"


def runtime_version() -> str:
    try:
        return version("sageattention")
    except PackageNotFoundError as exc:
        raise RuntimeError(
            "The MetaX SageAttention wheel is not installed"
        ) from exc


def sageattn(*args, **kwargs):
    actual = runtime_version()
    if actual != EXPECTED_VERSION:
        raise RuntimeError(
            f"Expected sageattention {EXPECTED_VERSION}, got {actual}"
        )
    return import_module("sageattention").sageattn(*args, **kwargs)
