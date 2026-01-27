# Design: Config-Only KV Cache Spec Computation

**Author**: Xingyu Liu
**Status**: Implemented
**Created**: 2026-01-26
**Updated**: 2026-01-27

## Executive Summary

This document describes the refactored implementation that computes `KVCacheSpec` purely from configuration objects without instantiating the model. This enables earlier KV cache planning in the engine initialization pipeline and eliminates the dependency on model construction.

**Key Design Decision**: Layer indices (`int`) are used throughout the KV cache allocation path instead of layer names (`str`). This allows config-only computation since layer indices can be determined from configuration, while layer names require model instantiation.

---

## 1. Problem Statement

### 1.1 Previous Flow (Pain Points)

```
Engine Init
    │
    ▼
┌─────────────────────────────────┐
│  get_layers_from_vllm_config()  │  ← Requires model to be constructed
│  (iterates static_forward_ctx)  │
└─────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────┐
│  attn_module.get_kv_cache_spec()│  ← Instance method on each attention layer
│  (reads self.num_kv_heads, etc.)│
└─────────────────────────────────┘
    │
    ▼
    KVCacheConfig
```

**Problems:**
1. `get_layers_from_vllm_config()` requires the model to be fully constructed (reads from `static_forward_context`)
2. `get_kv_cache_spec()` reads from instance attributes (`self.num_kv_heads`, `self.head_size`, etc.) that were populated during model `__init__`
3. Layer names (e.g., `"model.layers.0.self_attn.attn"`) vary by model architecture and require model instantiation to discover
4. Tight coupling between attention modules and KV cache planning

### 1.2 Solution Overview

**Config-only KV cache spec computation using layer indices:**

```
Engine Init
    │
    ▼
┌─────────────────────────────────────────────┐
│  ModelArchConfigConvertor                    │  ← Reads only from HuggingFace config
│  (hf_config → LayerKVCacheConfig per layer)  │
└─────────────────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────────────────┐
│  compute_kv_cache_specs_from_config()        │  ← Creates KVCacheSpec per layer index
│  Returns: dict[int, KVCacheSpec]             │
└─────────────────────────────────────────────┘
    │
    ▼
    KVCacheConfig (uses layer indices throughout)
```

---

## 2. Implementation Details

### 2.1 Key Design Decision: Layer Indices over Layer Names

**Why layer indices instead of layer names?**

| Aspect | Layer Names (`str`) | Layer Indices (`int`) |
|--------|--------------------|-----------------------|
| Config-only? | ❌ No - names like `"model.layers.0.self_attn.attn"` vary by model | ✅ Yes - indices 0, 1, 2... are universal |
| PP Support | ❌ Complex - different stages have different name prefixes | ✅ Simple - just merge by index |
| Model Required? | ❌ Yes - need `static_forward_context` | ✅ No - indices from config |

**Binding Strategy:**
- KV cache allocation uses layer indices (config-only)
- At runtime, `forward_context` uses layer names (for model compatibility)
- Binding happens when the model is available via `extract_layer_index()` helper

### 2.2 Core Components

#### 2.2.1 Configuration Types (`vllm/config/kv_cache_spec_config.py`)

```python
class LayerAttentionType(str, Enum):
    """Attention mechanism type derived from HF layer_types."""
    FULL = "full_attention"
    SLIDING_WINDOW = "sliding_attention"
    CHUNKED_LOCAL = "chunked_attention"
    LINEAR = "linear_attention"


class LayerBlockType(str, Enum):
    """Block type derived from HF layers_block_type."""
    ATTENTION = "attention"
    MAMBA = "mamba"
    HYBRID = "hybrid"


@dataclass(frozen=True)
class LayerKVCacheConfig:
    """Per-layer configuration for KV cache spec computation."""
    layer_idx: int  # Layer identification (0-indexed)
    attention_type: LayerAttentionType = LayerAttentionType.FULL
    num_kv_heads: int | None = None
    head_size: int | None = None
    # ... other fields
    kv_sharing_target_layer_idx: int | None = None  # For KV sharing


@dataclass
class ModelKVCacheRequirements:
    """Complete KV cache requirements for a model."""
    layers: list[LayerKVCacheConfig]
    default_num_kv_heads: int
    default_head_size: int
    default_dtype: torch.dtype
```

#### 2.2.2 Model Architecture Config Convertor (`vllm/transformers_utils/model_arch_config_convertor.py`)

```python
class ModelArchConfigConvertorBase:
    """Base class for converting HF config to KV cache requirements."""

    def __init__(self, hf_config, hf_text_config):
        self.hf_config = hf_config
        self.hf_text_config = hf_text_config

    def get_kv_cache_requirements(self, kv_cache_dtype: torch.dtype) -> ModelKVCacheRequirements:
        """Build complete KV cache requirements from HF config."""
        layers = []
        for layer_idx in range(self.num_hidden_layers):
            layer_config = self._build_layer_config(layer_idx, kv_cache_dtype)
            layers.append(layer_config)

        return ModelKVCacheRequirements(
            layers=layers,
            default_num_kv_heads=self.num_kv_heads,
            default_head_size=self.head_size,
            default_dtype=kv_cache_dtype,
        )

# Model-specific convertors registered in MODEL_ARCH_CONFIG_CONVERTORS
MODEL_ARCH_CONFIG_CONVERTORS: dict[str, type[ModelArchConfigConvertorBase]] = {
    "llama": LlamaConfigConvertor,
    "gemma2": Gemma2ConfigConvertor,
    "deepseek_v3": DeepSeekV3ConfigConvertor,
    # ... other models
}
```

#### 2.2.3 KV Cache Spec Factory (`vllm/v1/worker/kv_cache_spec_utils.py`)

```python
def compute_kv_cache_specs_from_config(
    convertor: ModelArchConfigConvertorBase,
    cache_config: CacheConfig,
    kv_cache_dtype: torch.dtype,
) -> Iterable[tuple[int, KVCacheSpec]]:
    """Compute KV cache specs from config, without model instantiation.

    Returns:
        Iterable of (layer_idx, KVCacheSpec) tuples.
    """
    requirements = convertor.get_kv_cache_requirements(kv_cache_dtype)

    for layer_config in requirements.layers:
        spec = create_kv_cache_spec_from_config(
            layer_config, cache_config, requirements
        )
        if spec is not None:
            yield (layer_config.layer_idx, spec)
```

### 2.3 Updated Data Structures

All KV cache data structures now use layer indices:

```python
# vllm/v1/kv_cache_interface.py

@dataclass
class KVCacheGroupSpec:
    layer_indices: list[int]  # Changed from layer_names: list[str]
    kv_cache_spec: KVCacheSpec

@dataclass
class KVCacheTensor:
    size: int
    shared_by: list[int]  # Changed from list[str]
```

### 2.4 Model Runner Integration

Both V1 and V2 model runners use the same config-only approach:

```python
# vllm/v1/worker/gpu_model_runner.py (V1 - default)
# vllm/v1/worker/gpu/attn_utils.py (V2 - experimental)

def get_kv_cache_spec(vllm_config: VllmConfig) -> dict[int, KVCacheSpec]:
    """Config-only KV cache spec computation."""
    cache_config = vllm_config.cache_config
    model_config = vllm_config.model_config

    kv_cache_dtype = kv_cache_dtype_str_to_dtype(
        cache_config.cache_dtype, model_config
    )

    convertor_cls = MODEL_ARCH_CONFIG_CONVERTORS.get(
        model_config.hf_config.model_type, ModelArchConfigConvertorBase
    )
    convertor = convertor_cls(model_config.hf_config, model_config.hf_text_config)

    return dict(compute_kv_cache_specs_from_config(
        convertor=convertor,
        cache_config=cache_config,
        kv_cache_dtype=kv_cache_dtype,
    ))
```

---

## 3. Files Modified

### 3.1 Core Implementation

| File | Changes |
|------|---------|
| `vllm/config/kv_cache_spec_config.py` | New file: `LayerKVCacheConfig`, `ModelKVCacheRequirements`, enums |
| `vllm/transformers_utils/model_arch_config_convertor.py` | Extended with `get_kv_cache_requirements()` method |
| `vllm/v1/worker/kv_cache_spec_utils.py` | New file: `compute_kv_cache_specs_from_config()` factory |

### 3.2 Data Structure Updates

| File | Changes |
|------|---------|
| `vllm/v1/kv_cache_interface.py` | `KVCacheGroupSpec.layer_names` → `layer_indices`, `KVCacheTensor.shared_by` type |
| `vllm/v1/core/kv_cache_utils.py` | All `dict[str, KVCacheSpec]` → `dict[int, KVCacheSpec]` |
| `vllm/v1/worker/utils.py` | `AttentionGroup.layer_names` → `layer_indices` |

### 3.3 Model Runner Updates

| File | Changes |
|------|---------|
| `vllm/v1/worker/gpu_model_runner.py` | `get_kv_cache_spec()` uses config-only path, all dict types updated |
| `vllm/v1/worker/gpu/attn_utils.py` | `get_kv_cache_spec()` uses config-only path |
| `vllm/v1/worker/gpu/model_runner.py` | Uses updated `attn_utils.get_kv_cache_spec()` |

### 3.4 Type Annotation Updates

| File | Changes |
|------|---------|
| `vllm/v1/worker/worker_base.py` | `get_kv_cache_spec() -> dict[int, KVCacheSpec]` |
| `vllm/v1/worker/gpu_worker.py` | `get_kv_cache_spec() -> dict[int, KVCacheSpec]` |
| `vllm/v1/executor/abstract.py` | `get_kv_cache_specs() -> list[dict[int, KVCacheSpec]]` |

---

## 4. Supported Attention Types

| Attention Type | KVCacheSpec | Config Pattern |
|----------------|-------------|----------------|
| Full (decoder) | `FullAttentionSpec` | Default |
| Sliding window | `SlidingWindowSpec` | `layer_types: ["sliding_attention"]` or `sliding_window` config |
| Chunked local | `ChunkedLocalAttentionSpec` | `layer_types: ["chunked_attention"]` or `attention_chunk_size` |
| MLA | `MLAAttentionSpec` | `kv_lora_rank`, `is_mla` |
| Cross-attention | `CrossAttentionSpec` | Encoder-decoder architecture |
| Mamba/SSM | `MambaSpec` | `layers_block_type: ["mamba"]` |
| Hybrid | `FullAttentionSpec` + `MambaSpec` | `layers_block_type: ["hybrid"]` |

---

## 5. HF Config Fields Used

| Field | Type | Purpose |
|-------|------|---------|
| `num_hidden_layers` | `int` | Total layer count |
| `num_key_value_heads` | `int` | KV heads per layer |
| `head_dim` | `int` | Head dimension |
| `layer_types` | `list[str]` | Per-layer attention type |
| `layers_block_type` | `list[str]` | Per-layer block type (attention/mamba/hybrid) |
| `sliding_window` | `int` | Sliding window size |
| `attention_chunk_size` | `int` | Chunk size for local attention |
| `kv_lora_rank` | `int` | MLA latent dimension |
| `qk_rope_head_dim` | `int` | MLA RoPE dimension |
| `ssm_state_size` | `int` | Mamba state dimension |
| `conv_kernel_size` | `int` | Mamba conv kernel size |

---

## 6. Pipeline Parallelism Support

With layer indices, PP support is straightforward:

```python
def get_kv_cache_configs(
    vllm_config: VllmConfig,
    kv_cache_specs: list[dict[int, KVCacheSpec]],  # From each PP stage
    available_memory: list[int],
) -> list[KVCacheConfig]:
    """Merge KV cache specs from all PP stages."""

    # Merge by layer index - simple dict update
    merged_kv_cache_specs: dict[int, KVCacheSpec] = {}
    for kv_cache_spec_one_worker in kv_cache_specs:
        for layer_idx, layer_spec in kv_cache_spec_one_worker.items():
            if layer_idx not in merged_kv_cache_specs:
                merged_kv_cache_specs[layer_idx] = layer_spec
            else:
                assert merged_kv_cache_specs[layer_idx] == layer_spec

    # Generate unified config
    ...
```

---

## 7. Binding: Layer Indices to Layer Names

At runtime, when the model is available, we bind KV caches (keyed by layer index) to the forward context (keyed by layer name):

```python
# vllm/v1/worker/gpu/attn_utils.py

def init_kv_cache(
    runner_kv_caches: list[torch.Tensor],
    forward_context: dict[str, Any],  # Keyed by layer names
    kv_cache_config: KVCacheConfig,
    attn_backends: dict[int, AttentionBackend],  # Keyed by layer indices
    device: torch.device,
) -> dict[int, torch.Tensor]:
    # Allocate KV caches by layer index
    kv_caches = _reshape_kv_cache(kv_cache_config, kv_cache_raw_tensors, attn_backends)

    # Convert layer indices to layer names for binding
    from vllm.model_executor.models.utils import extract_layer_index

    kv_caches_by_name: dict[str, torch.Tensor] = {}
    for layer_name in forward_context:
        try:
            layer_idx = extract_layer_index(layer_name)
            if layer_idx in kv_caches:
                kv_caches_by_name[layer_name] = kv_caches[layer_idx]
        except (ValueError, IndexError):
            pass

    bind_kv_cache(kv_caches_by_name, forward_context, runner_kv_caches)
    return kv_caches
```

---

## 8. Testing

Tests are located in `tests/v1/worker/test_kv_cache_spec_from_config.py`.

```python
@pytest.mark.parametrize("model_name", [
    "meta-llama/Llama-2-7b-hf",
    "google/gemma-2-2b",
    "deepseek-ai/DeepSeek-V3",
])
def test_config_only_kv_cache_spec(model_name):
    """Test that config-only path produces correct KV cache specs."""
    vllm_config = VllmConfig(model=model_name)

    kv_cache_spec = get_kv_cache_spec(vllm_config)

    assert isinstance(kv_cache_spec, dict)
    assert all(isinstance(k, int) for k in kv_cache_spec.keys())
    assert all(isinstance(v, KVCacheSpec) for v in kv_cache_spec.values())
```

---

## 9. Migration Notes

### 9.1 For Downstream Code

If you have code that uses the old layer-name-based API:

**Before:**
```python
kv_cache_spec: dict[str, KVCacheSpec] = runner.get_kv_cache_spec()
for layer_name, spec in kv_cache_spec.items():
    ...
```

**After:**
```python
kv_cache_spec: dict[int, KVCacheSpec] = runner.get_kv_cache_spec()
for layer_idx, spec in kv_cache_spec.items():
    ...
```

### 9.2 For New Model Support

To add config-only support for a new model architecture:

1. Create a convertor class in `model_arch_config_convertor.py`:
   ```python
   class MyModelConfigConvertor(ModelArchConfigConvertorBase):
       def _build_layer_config(self, layer_idx, kv_cache_dtype):
           # Override to handle model-specific logic
           ...
   ```

2. Register in `MODEL_ARCH_CONFIG_CONVERTORS`:
   ```python
   MODEL_ARCH_CONFIG_CONVERTORS["my_model"] = MyModelConfigConvertor
   ```

---

## 10. References

- HuggingFace `layer_types` documentation
- vLLM attention layer: `vllm/attention/layer.py`
- Config-only entry point: `vllm/v1/worker/kv_cache_spec_utils.py`
- Model runner integration: `vllm/v1/worker/gpu_model_runner.py`
