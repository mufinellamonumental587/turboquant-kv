# TurboQuant Roadmap

## Чеклист: что уже есть

- [x] API `compress` / `decompress`, `quantize_kv`, `quantize_kv_cache`
- [x] `__version__` / `__all__`, дефолты `device` / `dtype`, `torch.no_grad`, опциональный `torch.compile`
- [x] Triton **attention scores** по сжатому K (`quantized_attention_scores_triton`)
- [x] Triton **fused** softmax×V (`quantized_attention_fused_triton`, **paged**: `quantized_attention_fused_triton_paged` + `pack_dense_kv_to_paged`)
- [x] **Causal**, **GQA/MQA** (`num_kv_heads`, `causal`), **additive `attention_mask`**
- [x] Простые бенчмарки `benchmarks/needle_in_a_haystack_simple.py`, `benchmarks/longbench_simple.py`
- [x] CI на CPU; Triton-тесты при CUDA
- [x] `TurboQuantModel` + квантование legacy `past_key_values` (tuple слоёв)
- [x] **Hugging Face `transformers`**: `TurboQuantCacheLayer` + `turboquant_dynamic_cache(config, quantizer)` — кэш в формате `Cache` со сжатым KV на полноразмерных слоях; экспорт в paged для совместимости с нашим Triton paged API; заметки по vLLM (см. `turboquant/hf_cache.py`)

## Чеклист: чего ещё нет (по приоритету)

- [x] README: HF Cache + Llama Triton fused (границы: маски, только Llama-стек)
- [x] Кастомный attention в HF (Llama): Triton fused на сжатом KV + инкремент в кэше (`strict_reencode=False`)
- [x] README: пошаговый `generate()` + заметки про деквант при росте контекста; пример `examples/hf_generate_turboquant_cache.py`
- [x] Нативный KV dtype / worker в **vLLM** (upstream) — overlay + `turboquant.vllm_pack`, гайд: `integrations/vllm_upstream/` (`README.md`, `UPSTREAM_EDITS.md`)
- [x] **llama.cpp**: sidecar `.tqmeta`, гайд + эталонный C++ (`integrations/llama_cpp/`); GGUF-тензоры — опционально по `UPSTREAM_EDITS.md`
- [x] Калибровка по датасету, режимы **1.58-bit** / расширенные **4-bit+** (`turboquant.calibration`, `codebook="ternary"` / `bits>=5`)
- [x] CI с GPU (регулярный прогон Triton) — `.github/workflows/triton-gpu.yml`, runner `turboquant-gpu` + `ENABLE_SELF_HOSTED_GPU_CI`
- [ ] **Релиз `0.1.0` на PyPI** — оставляем на самый последний момент перед публикацией

---

## Следующий фокус (после текущего спринта)

1. [x] Опциональное ускорение: `hybrid_float_cache` — материализованный float KV / гибрид (путь без fused; см. README про деквант).

## Качество и режимы (долгий хвост)

- Pre-computed centroids: частично (`preload_centroids`, внутренний кэш в `TurboQuantProd`).
