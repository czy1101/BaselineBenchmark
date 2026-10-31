"""Local paired benchmark; no backend registration or global function patching."""
import importlib.util
import json
import sys
from pathlib import Path
import torch
import torch_musa
from flag_attn import parallel_moba as pure_moba
from flag_attn.testing import backend

path = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location(
    "moba_baseline", path / "__init__.py", submodule_search_locations=[str(path)]
)
module = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = module
spec.loader.exec_module(module)

with torch.inference_mode():
    for length in (4096, 8192):
        torch.manual_seed(42)
        q, k, v = [torch.randn(2 * length, 16, 128, device="musa", dtype=torch.bfloat16) for _ in range(3)]
        cu = torch.tensor([0, length, 2 * length], device="musa", dtype=torch.int32)
        timings = {"pure_triton_ms": [], "experimental_baseline_ms": []}
        functions = {"pure_triton_ms": pure_moba, "experimental_baseline_ms": module.parallel_moba}
        for trial in range(4):
            names = list(functions) if trial % 2 == 0 else list(reversed(functions))
            for name in names:
                fn = functions[name]
                timings[name].append(backend.do_bench(
                    lambda: fn(q, k, v, cu, length, 128, 8),
                    warmup=25, rep=150, return_mode="median"
                ))
        print(json.dumps(dict(length=length, **timings)), flush=True)
