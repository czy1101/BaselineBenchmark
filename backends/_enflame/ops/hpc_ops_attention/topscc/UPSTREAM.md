# Enflame TopsCC paged-attention source

The device kernel in this directory is vendored from the official
[`EnflameTechnology/Ubridge`](https://github.com/EnflameTechnology/Ubridge)
repository at commit `04c17896adbff718369b44cc56501d52d34e3662`:

- `kernels/attention.cpp`
- `kernels/third_party/attention_gqa_kernels.h`
- `kernels/utils/utils.h`

The files retain their upstream Apache-2.0 headers. `LICENSE.txt` is the
upstream project license. The Python and PyTorch extension wrappers next to
this directory are local integration code for BaselineBenchmark.

The S60 build target is `gcu300`. The Python wrapper invokes `topscc` lazily
and stores all generated objects outside the source directory.
