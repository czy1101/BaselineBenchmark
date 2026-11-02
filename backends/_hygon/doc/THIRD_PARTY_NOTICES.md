# InfLLM-V2 Native: third-party notices

This notice covers `backends/_hygon/ops/infllmv2/` and its test/benchmark.

HYTLASS headers are vendored unmodified from HYGON-AI/hytlass commit
47488c81e8b8222ae6bb64eeb4652bc5eb289be2. Each header contains the complete
BSD-3-Clause license and applicable Hygon/NVIDIA copyright notices.

https://github.com/HYGON-AI/hytlass/tree/47488c81e8b8222ae6bb64eeb4652bc5eb289be2

The generalized wave64 attention templates and C-to-A register mapping derive
from this project's frozen native v1 and infllmv2-hcu implementations
(Apache-2.0). The frozen Native v3 identity and lineage are documented in
`INFLLMV2_HYGON.md`; the original full manifests remain in the frozen archive.
Their register layout was informed by the MIT-licensed
HYGON-AI/composable_kernel-das commit 1bf5adb1b57941da0c16a0a823516986aab9bd14.
No CK source is vendored here.

The adapter and independent validation/benchmark harness are Apache-2.0.
The Torch-only mathematical references in the test file originate from
FlagAttention commit 54417c6e9e4f85cdca248981ed18dc5b9fa5eb5d (Apache-2.0).
