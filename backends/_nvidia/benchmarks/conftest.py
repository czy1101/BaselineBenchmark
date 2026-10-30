"""NVIDIA benchmark options; prefixes avoid collisions with other backends."""


def pytest_addoption(parser):
    group = parser.getgroup("apply-min-p-benchmark")
    group.addoption("--min-p-warmup-ms", type=float, default=100)
    group.addoption("--min-p-repeat-ms", type=float, default=300)
    group.addoption("--min-p-device", type=int, default=0)
    group.addoption(
        "--min-p-dtypes", nargs="+", choices=("float16", "bfloat16", "float32")
    )
    group.addoption("--min-p-shape-file")
    group.addoption("--min-p-output")


def pytest_configure(config):
    config.addinivalue_line(
        "markers", "apply_min_p: native TileOPs/Triton Min-P performance comparison"
    )
