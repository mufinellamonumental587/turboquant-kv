# Suggested llama.cpp upstream edits (outline)

llama.cpp moves quickly; treat this as a **checklist of touch points**, not a ready-made patch.

## 1. KV cache type / buffer size

- Locate where `cache_type_k` / `cache_type_v` (or equivalent) select element size and stride.
- Introduce a mode such as `LLAMA_KV_TURBOQUANT` (name is your choice) whose **cell size** equals `TurboQuantPageLayout.page_bytes` for one paged block, or define per-layer raw byte buffers.
- Python: `from turboquant.vllm_pack import TurboQuantPageLayout, turboquant_paged_block_bytes` — use the same `block_size`, `num_kv_heads`, `head_dim`, and `aux_dtype` as in C++.

## 2. Loading quantizer state

- At model load, mmap or read the sidecar written by `turboquant.llama_cpp_pack.write_quantizer_metadata` (format documented in `turboquant/llama_cpp_pack.py` and `reference/turboquant_sidecar_loader.cpp`).
- Alternative: add optional GGUF tensors (e.g. `turboquant.pi.f32`, `turboquant.s.f32`, `turboquant.centroid.f32`, scalar `turboquant.bits`) and parse them in `llama_model_loader` (exact API depends on your fork).

## 3. Cache write path

- Hook the same place that currently copies K/V rows into the KV buffer (often after `ggml_mul_mat` for K/V projections + RoPE).
- For each new token, quantize **one row** `[n_kv_heads, head_dim]` per layer using the TurboQuant algorithm (mirror `TurboQuantProd._quantize_components` / `scatter_one_token` in Python).
- Write into the paged `uint8` layout using offsets from `TurboQuantPageLayout` (duplicate the field order in C++ or generate constants from a small script).

## 4. Attention / decode path

- **Baseline:** dequantize K (and V) to `fp16`/`fp32` with the reference decoder, then call existing `ggml_flash_attn` / matmul attention. Higher VRAM, simplest correctness test.
- **Fast path:** port fused attention over packed KV (see `turboquant/kernels/fused_attention.py` and Triton sources) into CUDA (`ggml-cuda`) or metal, following existing FP8/FP16 flash-attn patterns.

## 5. CLI / API

- Expose flags analogous to `--cache-type-k turboquant` and `--turboquant-meta path.tqmeta` (names illustrative).
- Thread `bits`, `head_dim`, and pointer/handle to loaded `Π`/`S`/centroids into `llama_context` or layer structs.

## 6. Tests

- Golden vectors: run Python `quantize_kv` + `decompress` on fixed RNG tensors; compare C++ dequant output within a tight tolerance.
- Page layout: fill one block with `scatter_one_token` in Python, dump `pages[0]` to a file, load in C++ and assert byte identity at each layout offset.

## Licensing

New files in llama.cpp should follow that project’s license (MIT). Reference code under `integrations/llama_cpp/reference/` in this repo is Apache-2.0; if you paste it into llama.cpp, relicense or rewrite to match upstream policy.
