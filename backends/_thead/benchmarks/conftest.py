"""Markers shared by the T-Head baseline benchmarks."""


def pytest_configure(config):
    for marker, description in (
        ("benchmark", "accelerator benchmark"),
        ("chunk_gated_delta_rule", "T-Head chunk and recurrent GDN, including packed decode"),
    ):
        config.addinivalue_line("markers", marker + ": " + description)
