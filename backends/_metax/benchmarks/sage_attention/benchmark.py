import json
from pathlib import Path

import torch

from backends._metax.ops.sage_attention import sageattn


def main():
    shapes = json.loads(
        Path(__file__).with_name("shapes.json").read_text(encoding="utf-8")
    )

    for shape in shapes:
        q = torch.randn(
            (
                shape["batch"],
                shape["sequence_length"],
                shape["heads"],
                shape["head_dim"],
            ),
            device="cuda",
            dtype=torch.float16,
        )
        k = torch.randn_like(q)
        v = torch.randn_like(q)

        for _ in range(25):
            sageattn(q, k, v, tensor_layout="NHD", is_causal=True)
        torch.cuda.synchronize()

        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        for _ in range(100):
            sageattn(q, k, v, tensor_layout="NHD", is_causal=True)
        end.record()
        end.synchronize()

        print({**shape, "mean_ms": start.elapsed_time(end) / 100})


if __name__ == "__main__":
    main()
