# MetaX correctness tests

All MetaX test sources live under this `test/` directory:

- Top-level `test_*.py` files exercise the adapted FlagAttention operators.
- `gla/`, `nsa/`, `kda/`, and `sage_attention/` contain the independent
  comparison-baseline tests. These baseline tests must not import adapted
  operators from FlagAttention.

Run the top-level adapted suites in the MetaX/FlagAttention environment:

```bash
pytest -q backends/_metax/test/test_*.py
```

Run the independent KDA and SageAttention suites separately, after installing
their respective baseline dependencies (`mcoplib._C` and the official
SageAttention wheel/runtime):

```bash
pytest -q backends/_metax/test/kda/test_chunk_kda_fwd.py
pytest -q backends/_metax/test/sage_attention/test_official_wheel.py
```

GLA and NSA retain selected original TileOps files under `upstream_snapshot/`.
Their internal `tests/` paths and imports are preserved as upstream source
layout, not as a second MetaX test root. Use the corresponding TileOps test
environment and support files to run those snapshots. See [GLA](gla/README.md)
and [NSA](nsa/README.md); backward support must not be inferred from a directory
move or a forward-only result.
