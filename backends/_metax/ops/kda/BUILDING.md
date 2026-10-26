# Native KDA integration

`native/chunk_kda_fwd.cu` requires three mcoplib integration points:

1. Add the source to `VLLM_EXT_SRC` in `CMakeLists.txt`.
2. Declare `chunk_kda_fwd` in `op/vllm/ops.h`.
3. Register `torch.ops._C.chunk_kda_fwd` in
   `op/vllm/torch_bindings.cpp`.

The old prebuilt `_C.abi3.so` is deliberately excluded because it is
incompatible with the target PyTorch 2.8 runtime.
