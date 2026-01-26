# Design: Config-Only KV Cache Spec Computation

**Author**: Xingyu Liu
**Status**: Proposal
**Created**: 2026-01-26

## Executive Summary

This document proposes a refactor to compute `KVCacheSpec` purely from configuration objects (`ModelArchitectureConfig`, `CacheConfig`) without instantiating the model. This enables earlier KV cache planning in the engine initialization pipeline and eliminates the current dependency on model construction.

---

## 1. Current State Analysis

### 1.1 Current Flow (Pain Points)

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
3. No clear documentation of which config fields are needed for KV spec computation
4. Tight coupling between attention modules and KV cache planning

### 1.2 Attention Types Inventory

| Attention Type | KVCacheSpec Class | Instance Attrs Used | Config Source |
|----------------|-------------------|---------------------|---------------|
| Standard Full | `FullAttentionSpec` | `num_kv_heads`, `head_size`, `head_size_v`, `dtype` | `hf_config.num_key_value_heads`, `hf_config.head_dim` |
| Sliding Window | `SlidingWindowSpec` | `sliding_window`, `num_kv_heads`, `head_size`, `dtype` | `hf_config.sliding_window`, `layer_types[i]` |
| Chunked Local | `ChunkedLocalAttentionSpec` | `attention_chunk_size`, `num_kv_heads`, `head_size`, `dtype` | `hf_config.attention_chunk_size` |
| MLA | `MLAAttentionSpec` | `head_size` (latent), `dtype`, `cache_dtype_str` | `hf_config.kv_lora_rank`, `qk_rope_head_dim` |
| Cross-Attention | `CrossAttentionSpec` | `num_kv_heads`, `head_size`, `dtype` | `hf_config.decoder_attention_heads` |
| Encoder-Only | Returns `None` | - | - |
| Mamba/SSM | `MambaSpec` | `state_shapes`, `dtypes`, `mamba_type` | `hf_config.ssm_state_size`, `conv_kernel_size` |
| Static Sink | `SinkFullAttentionSpec` | `sink_len`, + FullAttentionSpec fields | Static sink config |

### 1.3 HF Config Structure

**HuggingFace provides two key fields for per-layer configuration:**

#### `layer_types: list[str]`
Specifies the attention mechanism for each layer:
| Value | Meaning |
|-------|---------|
| `"full_attention"` | Standard full attention |
| `"sliding_attention"` | Sliding window attention |
| `"chunked_attention"` | Chunked local attention |
| `"linear_attention"` | Linear/recurrent attention |

#### `layers_block_type: list[str]`
Specifies the block type for hybrid models:
| Value | Meaning |
|-------|---------|
| `"attention"` | Transformer attention block |
| `"mamba"` | Mamba/SSM block |
| `"hybrid"` | Parallel attention + Mamba (e.g., FalconH1) |

---

## 2. Proposed Design

### 2.1 Core Principle: Config-Driven Dispatch

Instead of relying on instantiated attention modules, we:
1. Extend `ModelArchitectureConfig` to carry per-layer KV cache requirements from HF config
2. Use direct dispatch (simple if/elif) to create `KVCacheSpec` based on layer type
3. Compute specs by iterating layers and dispatching to the appropriate factory function

**No plugin/registry pattern needed** since all attention implementations are in vLLM.

### 2.2 New Configuration Types

```python
# vllm/config/kv_cache_spec_config.py

from dataclasses import dataclass
from enum import Enum
from typing import Any

import torch


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
    HYBRID = "hybrid"  # Parallel attention + Mamba


class LayerRole(str, Enum):
    """Role of the layer in an encoder-decoder architecture."""
    DECODER = "decoder"           # Decoder self-attention (default)
    ENCODER = "encoder"           # Encoder self-attention
    CROSS_ATTENTION = "cross"     # Encoder-decoder cross-attention
    ENCODER_ONLY = "encoder_only" # Encoder-only model (BERT-style)


@dataclass(frozen=True)
class LayerKVCacheConfig:
    """Per-layer configuration for KV cache spec computation.

    This is the minimal info needed to compute a KVCacheSpec for one layer
    without instantiating the model.
    """
    # Layer identification
    layer_idx: int
    layer_name: str  # e.g., "model.layers.0.self_attn.attn"

    # From HF layer_types (attention mechanism within attention blocks)
    attention_type: LayerAttentionType = LayerAttentionType.FULL

    # From HF layers_block_type (block type for hybrid models)
    block_type: LayerBlockType = LayerBlockType.ATTENTION

    # Role (for encoder-decoder models)
    role: LayerRole = LayerRole.DECODER

    # Common attention parameters
    num_kv_heads: int | None = None
    head_size: int | None = None
    head_size_v: int | None = None  # For models with different K/V dims
    dtype: torch.dtype | None = None

    # Sliding window size (when attention_type == SLIDING_WINDOW)
    sliding_window: int | None = None

    # Chunk size (when attention_type == CHUNKED_LOCAL)
    attention_chunk_size: int | None = None

    # Sink attention
    sink_len: int | None = None

    # MLA-specific parameters
    kv_lora_rank: int | None = None
    qk_rope_head_dim: int | None = None
    cache_dtype_str: str | None = None

    # Mamba/SSM-specific parameters (when block_type == MAMBA)
    mamba_type: str | None = None  # "mamba1" or "mamba2"
    ssm_state_size: int | None = None
    conv_kernel_size: int | None = None
    intermediate_size: int | None = None
    n_groups: int | None = None
    mamba_num_heads: int | None = None
    mamba_head_dim: int | None = None

    # KV sharing
    kv_sharing_target_layer_name: str | None = None


@dataclass
class ModelKVCacheRequirements:
    """Complete KV cache requirements for a model, derived from config only.

    This replaces the need to instantiate the model just to get KV cache specs.
    """
    # Per-layer configurations
    layers: list[LayerKVCacheConfig]

    # Model-level defaults (applied when layer-specific values are None)
    default_num_kv_heads: int
    default_head_size: int
    default_dtype: torch.dtype

    # Architecture type hints
    is_encoder_decoder: bool = False
    is_hybrid: bool = False  # Has Mamba blocks
    is_mla: bool = False

    @property
    def layer_count(self) -> int:
        return len(self.layers)

    @property
    def attention_layer_count(self) -> int:
        return sum(1 for l in self.layers
                   if l.block_type in (LayerBlockType.ATTENTION, LayerBlockType.HYBRID))

    @property
    def mamba_layer_count(self) -> int:
        return sum(1 for l in self.layers
                   if l.block_type in (LayerBlockType.MAMBA, LayerBlockType.HYBRID))
```

### 2.3 Extended ModelArchitectureConfig

```python
# vllm/config/model_arch.py (additions)

from vllm.config.kv_cache_spec_config import (
    LayerAttentionType,
    LayerBlockType,
    LayerRole,
    LayerKVCacheConfig,
    ModelKVCacheRequirements,
)

@dataclass(config=ConfigDict(arbitrary_types_allowed=True))
class ModelArchitectureConfig:
    # ... existing fields ...

    # NEW: Per-layer attention types from HF layer_types
    layer_types: list[LayerAttentionType] | None = None
    """Per-layer attention type from HF config.layer_types.
    Values: full_attention, sliding_attention, chunked_attention, linear_attention"""

    # NEW: Per-layer block types from HF layers_block_type
    layers_block_type: list[LayerBlockType] | None = None
    """Per-layer block type from HF config.layers_block_type.
    Values: attention, mamba, hybrid"""

    # NEW: Sliding window size (applies to sliding_attention layers)
    sliding_window: int | None = None
    """Sliding window size for layers with attention_type == sliding_attention."""

    # NEW: Chunked local attention
    attention_chunk_size: int | None = None
    """Chunk size for layers with attention_type == chunked_attention."""

    # NEW: MLA-specific config
    kv_lora_rank: int | None = None
    qk_rope_head_dim: int | None = None

    # NEW: Mamba/SSM config (applies to mamba blocks)
    mamba_type: str | None = None  # "mamba1" or "mamba2"
    ssm_state_size: int | None = None
    conv_kernel_size: int | None = None
    mamba_expand: int | None = None  # For intermediate_size calculation

    # NEW: Encoder-decoder config
    encoder_num_layers: int | None = None
    decoder_num_layers: int | None = None
    encoder_num_kv_heads: int | None = None
    decoder_num_kv_heads: int | None = None

    # NEW: KV sharing config
    kv_sharing_pattern: dict[int, int] | None = None
    """Maps layer_idx -> target_layer_idx for KV cache sharing."""

    def get_kv_cache_requirements(
        self,
        kv_cache_dtype: torch.dtype,
        prefix: str = "model.layers",
    ) -> ModelKVCacheRequirements:
        """Build complete KV cache requirements from architecture config.

        This method enables KV cache spec computation without model instantiation.
        """
        layers = []

        for layer_idx in range(self.total_num_hidden_layers):
            layer_config = self._build_layer_kv_config(
                layer_idx=layer_idx,
                prefix=prefix,
                dtype=kv_cache_dtype,
            )
            layers.append(layer_config)

        # Determine if model has Mamba blocks
        is_hybrid = (
            self.layers_block_type is not None and
            any(bt in (LayerBlockType.MAMBA, LayerBlockType.HYBRID)
                for bt in self.layers_block_type)
        )

        return ModelKVCacheRequirements(
            layers=layers,
            default_num_kv_heads=self.total_num_kv_heads,
            default_head_size=self.head_size,
            default_dtype=kv_cache_dtype,
            is_encoder_decoder=self.encoder_num_layers is not None,
            is_hybrid=is_hybrid,
            is_mla=self.is_deepseek_mla,
        )

    def _build_layer_kv_config(
        self,
        layer_idx: int,
        prefix: str,
        dtype: torch.dtype,
    ) -> LayerKVCacheConfig:
        """Build KV cache config for a single layer."""
        # Get attention type from layer_types
        attention_type = LayerAttentionType.FULL
        if self.layer_types and layer_idx < len(self.layer_types):
            attention_type = self.layer_types[layer_idx]

        # Get block type from layers_block_type
        block_type = LayerBlockType.ATTENTION
        if self.layers_block_type and layer_idx < len(self.layers_block_type):
            block_type = self.layers_block_type[layer_idx]

        # Determine layer role
        role = self._get_layer_role(layer_idx)

        # Layer name follows vLLM convention
        layer_name = f"{prefix}.{layer_idx}.self_attn.attn"

        # Get sliding window for this layer (if sliding_attention)
        sliding_window = None
        if attention_type == LayerAttentionType.SLIDING_WINDOW:
            sliding_window = self.sliding_window

        # Get chunk size for this layer (if chunked_attention)
        chunk_size = None
        if attention_type == LayerAttentionType.CHUNKED_LOCAL:
            chunk_size = self.attention_chunk_size

        # Handle KV sharing
        kv_sharing_target = None
        if self.kv_sharing_pattern and layer_idx in self.kv_sharing_pattern:
            target_idx = self.kv_sharing_pattern[layer_idx]
            kv_sharing_target = f"{prefix}.{target_idx}.self_attn.attn"

        # Build the config
        return LayerKVCacheConfig(
            layer_idx=layer_idx,
            layer_name=layer_name,
            attention_type=attention_type,
            block_type=block_type,
            role=role,
            num_kv_heads=self._get_layer_num_kv_heads(layer_idx, role),
            head_size=self._get_layer_head_size(layer_idx),
            head_size_v=self.head_size,
            dtype=dtype,
            sliding_window=sliding_window,
            attention_chunk_size=chunk_size,
            kv_lora_rank=self.kv_lora_rank if self.is_deepseek_mla else None,
            qk_rope_head_dim=self.qk_rope_head_dim if self.is_deepseek_mla else None,
            mamba_type=self.mamba_type if block_type in (LayerBlockType.MAMBA, LayerBlockType.HYBRID) else None,
            ssm_state_size=self.ssm_state_size if block_type in (LayerBlockType.MAMBA, LayerBlockType.HYBRID) else None,
            conv_kernel_size=self.conv_kernel_size if block_type in (LayerBlockType.MAMBA, LayerBlockType.HYBRID) else None,
            kv_sharing_target_layer_name=kv_sharing_target,
        )

    def _get_layer_role(self, layer_idx: int) -> LayerRole:
        """Determine layer role for encoder-decoder models."""
        if self.encoder_num_layers is None:
            return LayerRole.DECODER

        if layer_idx < self.encoder_num_layers:
            return LayerRole.ENCODER
        else:
            return LayerRole.DECODER

    def _get_layer_num_kv_heads(self, layer_idx: int, role: LayerRole) -> int:
        """Get number of KV heads for a layer."""
        if role == LayerRole.ENCODER and self.encoder_num_kv_heads is not None:
            return self.encoder_num_kv_heads
        if role == LayerRole.DECODER and self.decoder_num_kv_heads is not None:
            return self.decoder_num_kv_heads
        return self.total_num_kv_heads

    def _get_layer_head_size(self, layer_idx: int) -> int:
        """Get head size for a layer."""
        if self.is_deepseek_mla:
            # MLA stores compressed latent: kv_lora_rank + qk_rope_head_dim
            return (self.kv_lora_rank or 0) + (self.qk_rope_head_dim or 0)
        return self.head_size
```

### 2.4 KV Cache Spec Factory (Direct Dispatch, No Registry)

```python
# vllm/v1/kv_cache_spec_factory.py

"""
Config-driven KV cache spec creation.

This module provides factory functions to create KVCacheSpec objects
directly from LayerKVCacheConfig, without requiring model instantiation.

No registry/plugin pattern is used since all attention implementations
are in vLLM.
"""

import torch

from vllm.config import CacheConfig
from vllm.config.kv_cache_spec_config import (
    LayerKVCacheConfig,
    LayerAttentionType,
    LayerBlockType,
    LayerRole,
)
from vllm.v1.kv_cache_interface import (
    KVCacheSpec,
    FullAttentionSpec,
    SlidingWindowSpec,
    ChunkedLocalAttentionSpec,
    CrossAttentionSpec,
    MLAAttentionSpec,
    MambaSpec,
)


def create_kv_cache_spec(
    layer_config: LayerKVCacheConfig,
    cache_config: CacheConfig,
) -> KVCacheSpec | None:
    """Create a KVCacheSpec for a layer based on its config.

    This is the main dispatch function. It routes to the appropriate
    factory based on block_type and attention_type.

    Args:
        layer_config: Per-layer KV cache configuration.
        cache_config: Global cache configuration.

    Returns:
        KVCacheSpec for the layer, or None if no KV cache is needed.
    """
    # First, dispatch by block_type (from layers_block_type)
    if layer_config.block_type == LayerBlockType.MAMBA:
        return _create_mamba_spec(layer_config, cache_config)

    if layer_config.block_type == LayerBlockType.HYBRID:
        # Hybrid blocks have both attention and Mamba
        # Return specs for both (handled separately in get_kv_cache_specs)
        return _create_attention_spec(layer_config, cache_config)

    # For ATTENTION blocks, dispatch by role and attention_type
    if layer_config.role == LayerRole.ENCODER_ONLY:
        return None  # Encoder-only attention doesn't need KV cache

    if layer_config.role == LayerRole.CROSS_ATTENTION:
        return _create_cross_attention_spec(layer_config, cache_config)

    return _create_attention_spec(layer_config, cache_config)


def _create_attention_spec(
    layer_config: LayerKVCacheConfig,
    cache_config: CacheConfig,
) -> KVCacheSpec:
    """Create attention KVCacheSpec based on attention_type."""

    # Check for MLA first (special case)
    if layer_config.kv_lora_rank is not None:
        return _create_mla_spec(layer_config, cache_config)

    # Dispatch by attention_type (from layer_types)
    attn_type = layer_config.attention_type

    if attn_type == LayerAttentionType.FULL:
        return _create_full_attention_spec(layer_config, cache_config)

    elif attn_type == LayerAttentionType.SLIDING_WINDOW:
        return _create_sliding_window_spec(layer_config, cache_config)

    elif attn_type == LayerAttentionType.CHUNKED_LOCAL:
        return _create_chunked_local_spec(layer_config, cache_config)

    elif attn_type == LayerAttentionType.LINEAR:
        return _create_linear_attention_spec(layer_config, cache_config)

    else:
        # Default to full attention
        return _create_full_attention_spec(layer_config, cache_config)


def _create_full_attention_spec(
    layer_config: LayerKVCacheConfig,
    cache_config: CacheConfig,
) -> FullAttentionSpec:
    """Create FullAttentionSpec for standard decoder attention."""
    assert layer_config.num_kv_heads is not None
    assert layer_config.head_size is not None
    assert layer_config.dtype is not None

    return FullAttentionSpec(
        block_size=cache_config.block_size,
        num_kv_heads=layer_config.num_kv_heads,
        head_size=layer_config.head_size,
        head_size_v=layer_config.head_size_v or layer_config.head_size,
        dtype=layer_config.dtype,
    )


def _create_sliding_window_spec(
    layer_config: LayerKVCacheConfig,
    cache_config: CacheConfig,
) -> SlidingWindowSpec | FullAttentionSpec:
    """Create spec for sliding window attention."""
    assert layer_config.num_kv_heads is not None
    assert layer_config.head_size is not None
    assert layer_config.dtype is not None
    assert layer_config.sliding_window is not None

    # When hybrid allocator is disabled, use FullAttentionSpec with metadata
    if not cache_config.enable_hybrid_allocator:
        return FullAttentionSpec(
            block_size=cache_config.block_size,
            num_kv_heads=layer_config.num_kv_heads,
            head_size=layer_config.head_size,
            head_size_v=layer_config.head_size_v or layer_config.head_size,
            dtype=layer_config.dtype,
            sliding_window=layer_config.sliding_window,
        )

    return SlidingWindowSpec(
        block_size=cache_config.block_size,
        num_kv_heads=layer_config.num_kv_heads,
        head_size=layer_config.head_size,
        dtype=layer_config.dtype,
        sliding_window=layer_config.sliding_window,
    )


def _create_chunked_local_spec(
    layer_config: LayerKVCacheConfig,
    cache_config: CacheConfig,
) -> ChunkedLocalAttentionSpec | FullAttentionSpec:
    """Create spec for chunked local attention."""
    assert layer_config.num_kv_heads is not None
    assert layer_config.head_size is not None
    assert layer_config.dtype is not None
    assert layer_config.attention_chunk_size is not None

    # When hybrid allocator is disabled, use FullAttentionSpec with metadata
    if not cache_config.enable_hybrid_allocator:
        return FullAttentionSpec(
            block_size=cache_config.block_size,
            num_kv_heads=layer_config.num_kv_heads,
            head_size=layer_config.head_size,
            head_size_v=layer_config.head_size_v or layer_config.head_size,
            dtype=layer_config.dtype,
            attention_chunk_size=layer_config.attention_chunk_size,
        )

    return ChunkedLocalAttentionSpec(
        block_size=cache_config.block_size,
        num_kv_heads=layer_config.num_kv_heads,
        head_size=layer_config.head_size,
        dtype=layer_config.dtype,
        attention_chunk_size=layer_config.attention_chunk_size,
    )


def _create_cross_attention_spec(
    layer_config: LayerKVCacheConfig,
    cache_config: CacheConfig,
) -> CrossAttentionSpec:
    """Create spec for encoder-decoder cross-attention."""
    assert layer_config.num_kv_heads is not None
    assert layer_config.head_size is not None
    assert layer_config.dtype is not None

    return CrossAttentionSpec(
        block_size=cache_config.block_size,
        num_kv_heads=layer_config.num_kv_heads,
        head_size=layer_config.head_size,
        dtype=layer_config.dtype,
    )


def _create_mla_spec(
    layer_config: LayerKVCacheConfig,
    cache_config: CacheConfig,
) -> MLAAttentionSpec:
    """Create spec for Multi-head Latent Attention."""
    assert layer_config.head_size is not None
    assert layer_config.dtype is not None

    return MLAAttentionSpec(
        block_size=cache_config.block_size,
        num_kv_heads=1,  # MLA uses single latent vector
        head_size=layer_config.head_size,  # kv_lora_rank + qk_rope_head_dim
        dtype=layer_config.dtype,
        cache_dtype_str=layer_config.cache_dtype_str,
    )


def _create_mamba_spec(
    layer_config: LayerKVCacheConfig,
    cache_config: CacheConfig,
) -> MambaSpec:
    """Create spec for Mamba/SSM layers."""
    assert layer_config.ssm_state_size is not None
    assert layer_config.conv_kernel_size is not None
    assert layer_config.intermediate_size is not None
    assert layer_config.dtype is not None

    mamba_type = layer_config.mamba_type or "mamba2"

    if mamba_type == "mamba1":
        # Mamba-1 state shapes
        conv_state_shape = (layer_config.intermediate_size, layer_config.conv_kernel_size)
        ssm_state_shape = (layer_config.intermediate_size, layer_config.ssm_state_size)
    else:
        # Mamba-2 state shapes (with heads)
        n_groups = layer_config.n_groups or 1
        num_heads = layer_config.mamba_num_heads or layer_config.intermediate_size // 64
        head_dim = layer_config.mamba_head_dim or 64
        conv_state_shape = (
            layer_config.intermediate_size + 2 * n_groups * layer_config.ssm_state_size,
            layer_config.conv_kernel_size
        )
        ssm_state_shape = (num_heads, head_dim, layer_config.ssm_state_size)

    return MambaSpec(
        block_size=cache_config.mamba_block_size,
        shapes=(conv_state_shape, ssm_state_shape),
        dtypes=(layer_config.dtype,),
        mamba_type=mamba_type,
        mamba_cache_mode=cache_config.mamba_cache_mode,
    )


def _create_linear_attention_spec(
    layer_config: LayerKVCacheConfig,
    cache_config: CacheConfig,
) -> MambaSpec:
    """Create spec for linear attention (uses Mamba-like state caching)."""
    # Linear attention uses similar state caching to Mamba
    # Delegate to Mamba spec creation with appropriate parameters
    return _create_mamba_spec(layer_config, cache_config)
```

### 2.5 Config-Only KV Cache Spec Computation

```python
# vllm/v1/worker/kv_cache_spec_utils.py

"""
Config-only KV cache spec computation.

This module provides the main entry point for computing KV cache specs
from configuration, without requiring model instantiation.
"""

import torch

from vllm.config import VllmConfig
from vllm.v1.kv_cache_interface import KVCacheSpec
from vllm.v1.kv_cache_spec_factory import create_kv_cache_spec


def compute_kv_cache_specs_from_config(
    vllm_config: VllmConfig,
) -> dict[str, KVCacheSpec]:
    """Compute KV cache specs purely from config, without model instantiation.

    This is the new entry point that replaces the model-dependent approach.

    Args:
        vllm_config: The vLLM configuration object.

    Returns:
        A dictionary mapping layer names to their KV cache specs.
    """
    model_arch_config = vllm_config.model_arch_config
    cache_config = vllm_config.cache_config

    # Get the dtype for KV cache
    kv_cache_dtype = _get_kv_cache_dtype(vllm_config)

    # Build requirements from architecture config
    requirements = model_arch_config.get_kv_cache_requirements(
        kv_cache_dtype=kv_cache_dtype,
    )

    # Create specs for each layer
    kv_cache_specs: dict[str, KVCacheSpec] = {}
    kv_sharing_map: dict[str, str] = {}  # layer_name -> target_layer_name

    for layer_config in requirements.layers:
        # Skip if layer shares KV with another layer
        if layer_config.kv_sharing_target_layer_name:
            kv_sharing_map[layer_config.layer_name] = layer_config.kv_sharing_target_layer_name
            continue

        # Create spec using direct dispatch
        spec = create_kv_cache_spec(layer_config, cache_config)

        if spec is not None:
            kv_cache_specs[layer_config.layer_name] = spec

    return kv_cache_specs


def _get_kv_cache_dtype(vllm_config: VllmConfig) -> torch.dtype:
    """Determine the KV cache dtype from config."""
    cache_dtype = vllm_config.cache_config.cache_dtype

    if cache_dtype == "auto":
        return vllm_config.model_config.dtype
    elif cache_dtype == "fp8":
        return torch.float8_e4m3fn
    elif cache_dtype == "fp8_e4m3":
        return torch.float8_e4m3fn
    elif cache_dtype == "fp8_e5m2":
        return torch.float8_e5m2
    else:
        return getattr(torch, cache_dtype)
```

### 2.6 HF Config Parsing (Simple String → Enum Conversion)

```python
# vllm/transformers_utils/kv_cache_config_parser.py

"""
Parse HF config fields into vLLM's KV cache config types.

Since HuggingFace provides layer_types and layers_block_type directly,
this module just converts string values to enum types.
"""

from transformers import PretrainedConfig

from vllm.config.kv_cache_spec_config import LayerAttentionType, LayerBlockType


def parse_layer_types(
    hf_config: PretrainedConfig,
) -> list[LayerAttentionType] | None:
    """Parse HF layer_types into LayerAttentionType enum list.

    HF layer_types values: full_attention, sliding_attention,
                          chunked_attention, linear_attention
    """
    text_config = (hf_config.get_text_config()
                   if hasattr(hf_config, 'get_text_config') else hf_config)

    layer_types = getattr(text_config, 'layer_types', None)
    if layer_types is None:
        return None

    result = []
    for lt in layer_types:
        if lt == "full_attention":
            result.append(LayerAttentionType.FULL)
        elif lt == "sliding_attention":
            result.append(LayerAttentionType.SLIDING_WINDOW)
        elif lt == "chunked_attention":
            result.append(LayerAttentionType.CHUNKED_LOCAL)
        elif lt == "linear_attention":
            result.append(LayerAttentionType.LINEAR)
        else:
            # Unknown type, default to full
            result.append(LayerAttentionType.FULL)

    return result


def parse_layers_block_type(
    hf_config: PretrainedConfig,
) -> list[LayerBlockType] | None:
    """Parse HF layers_block_type into LayerBlockType enum list.

    HF layers_block_type values: attention, mamba, hybrid
    """
    text_config = (hf_config.get_text_config()
                   if hasattr(hf_config, 'get_text_config') else hf_config)

    block_types = getattr(text_config, 'layers_block_type', None)
    if block_types is None:
        return None

    result = []
    for bt in block_types:
        if bt == "attention":
            result.append(LayerBlockType.ATTENTION)
        elif bt == "mamba":
            result.append(LayerBlockType.MAMBA)
        elif bt == "hybrid":
            result.append(LayerBlockType.HYBRID)
        else:
            # Unknown type, default to attention
            result.append(LayerBlockType.ATTENTION)

    return result


def parse_kv_sharing_pattern(
    hf_config: PretrainedConfig,
) -> dict[int, int] | None:
    """Parse KV cache sharing pattern from HF config.

    Returns:
        Dict mapping layer_idx -> target_layer_idx for sharing, or None.
    """
    text_config = (hf_config.get_text_config()
                   if hasattr(hf_config, 'get_text_config') else hf_config)

    # Gemma3n-style: num_kv_shared_layers
    num_kv_shared = getattr(text_config, 'num_kv_shared_layers', None)
    num_layers = getattr(text_config, 'num_hidden_layers', None)

    if num_kv_shared and num_layers:
        first_shared = num_layers - num_kv_shared
        target = first_shared - 1
        return {i: target for i in range(first_shared, num_layers)}

    return None
```



### 2.4 Classmethod-Based Approach (Implemented)

Instead of separate factory functions, each attention module class has a `create_kv_cache_spec_from_config` classmethod. This approach:
- Makes the code easier to understand (spec creation logic lives with the attention class)
- Makes it easy for users to modify spec creation for specific attention types
- Follows the existing pattern where `get_kv_cache_spec` is an instance method

**Updated Attention Modules:**

| Attention Class | File | Returns |
|-----------------|------|---------|
| `Attention` | `vllm/attention/layer.py` | `FullAttentionSpec` or `SlidingWindowSpec` |
| `MLAAttention` | `vllm/attention/layer.py` | `MLAAttentionSpec` |
| `ChunkedLocalAttention` | `vllm/model_executor/layers/attention/chunked_local_attention.py` | `ChunkedLocalAttentionSpec` |
| `CrossAttention` | `vllm/model_executor/layers/attention/cross_attention.py` | `CrossAttentionSpec` |
| `EncoderOnlyAttention` | `vllm/model_executor/layers/attention/encoder_only_attention.py` | `None` |
| `MambaBase` | `vllm/model_executor/layers/mamba/abstract.py` | `MambaSpec` |

### 2.5 Dispatch Function (kv_cache_spec_utils.py)

The dispatch function routes to the appropriate attention class classmethod. See `vllm/v1/worker/kv_cache_spec_utils.py` for the full implementation.

---

## 3. API Design Summary

### 3.1 Primary Interface (Public API)

```python
# New primary entry point for config-only KV cache spec computation
def compute_kv_cache_specs_from_config(vllm_config: VllmConfig) -> dict[str, KVCacheSpec]:
    """Compute KV cache specs without model instantiation."""
```

### 3.2 ModelArchitectureConfig Extensions

```python
class ModelArchitectureConfig:
    # New fields from HF config
    layer_types: list[LayerAttentionType] | None       # From HF layer_types
    layers_block_type: list[LayerBlockType] | None     # From HF layers_block_type
    sliding_window: int | None                         # For sliding_attention layers
    attention_chunk_size: int | None                   # For chunked_attention layers
    kv_lora_rank: int | None                           # For MLA
    qk_rope_head_dim: int | None                       # For MLA
    mamba_type: str | None                             # For Mamba blocks
    ssm_state_size: int | None                         # For Mamba blocks
    conv_kernel_size: int | None                       # For Mamba blocks
    encoder_num_layers: int | None                     # For encoder-decoder
    decoder_num_layers: int | None                     # For encoder-decoder
    kv_sharing_pattern: dict[int, int] | None          # For KV cache sharing

    # New method
    def get_kv_cache_requirements(self, kv_cache_dtype: torch.dtype) -> ModelKVCacheRequirements:
        """Build complete KV cache requirements from config only."""
```

### 3.3 Direct Dispatch (No Registry)

```python
def create_kv_cache_spec(layer_config: LayerKVCacheConfig, cache_config: CacheConfig) -> KVCacheSpec | None:
    """Create spec using direct dispatch based on block_type and attention_type."""

    # Dispatch by block_type first
    if layer_config.block_type == LayerBlockType.MAMBA:
        return _create_mamba_spec(...)

    # Then by attention_type
    if layer_config.attention_type == LayerAttentionType.SLIDING_WINDOW:
        return _create_sliding_window_spec(...)
    ...
```

### 3.4 Attention Roles Representation

```python
class LayerRole(str, Enum):
    DECODER = "decoder"           # Decoder self-attention
    ENCODER = "encoder"           # Encoder self-attention (enc-dec)
    CROSS_ATTENTION = "cross"     # Cross-attention (enc-dec)
    ENCODER_ONLY = "encoder_only" # Bidirectional (BERT-style)
```

---

## 4. Config Surface & Parsing

### 4.1 Required Fields from HF Config

| Field | Type | Purpose | Example Values |
|-------|------|---------|----------------|
| `layer_types` | `list[str]` | Per-layer attention type | `["full_attention", "sliding_attention", ...]` |
| `layers_block_type` | `list[str]` | Per-layer block type | `["attention", "mamba", "hybrid", ...]` |
| `num_hidden_layers` | `int` | Layer count | `32` |
| `num_key_value_heads` | `int` | KV heads per layer | `8` |
| `head_dim` | `int` | Head dimension | `128` |
| `sliding_window` | `int` | Window size | `4096` |
| `attention_chunk_size` | `int` | Chunk size | `8192` |
| `kv_lora_rank` | `int` | MLA latent dim | `512` |
| `qk_rope_head_dim` | `int` | MLA RoPE dim | `64` |
| `ssm_state_size` | `int` | Mamba state dim | `16` |
| `conv_kernel_size` | `int` | Mamba conv size | `4` |
| `intermediate_size` | `int` | MLP/SSM expansion | `8192` |
| `num_kv_shared_layers` | `int` | KV sharing count | `4` |

### 4.2 Parsing Flow

```
PretrainedConfig (HF)
        │
        ├── layer_types ──────────────► list[LayerAttentionType]
        ├── layers_block_type ────────► list[LayerBlockType]
        ├── sliding_window ───────────► int
        ├── attention_chunk_size ─────► int
        ├── kv_lora_rank ─────────────► int
        └── ...
        │
        ▼
ModelArchitectureConfig (vLLM)
        │
        ▼
get_kv_cache_requirements()
        │
        ▼
ModelKVCacheRequirements
        │
        ▼
create_kv_cache_spec() per layer
        │
        ▼
dict[str, KVCacheSpec]
```

---

## 5. Dispatch & Mapping

### 5.1 Dispatch Logic

```python
def create_kv_cache_spec(layer_config, cache_config):
    # 1. Dispatch by block_type (from layers_block_type)
    if block_type == MAMBA:
        return _create_mamba_spec(...)
    if block_type == HYBRID:
        # Handle both attention and mamba specs
        ...

    # 2. Dispatch by role (for encoder-decoder)
    if role == ENCODER_ONLY:
        return None
    if role == CROSS_ATTENTION:
        return _create_cross_attention_spec(...)

    # 3. Check for MLA
    if kv_lora_rank is not None:
        return _create_mla_spec(...)

    # 4. Dispatch by attention_type (from layer_types)
    if attention_type == FULL:
        return _create_full_attention_spec(...)
    if attention_type == SLIDING_WINDOW:
        return _create_sliding_window_spec(...)
    if attention_type == CHUNKED_LOCAL:
        return _create_chunked_local_spec(...)
    if attention_type == LINEAR:
        return _create_linear_attention_spec(...)
```

### 5.2 HF Value → vLLM Enum Mapping

**layer_types:**
| HF Value | vLLM LayerAttentionType |
|----------|-------------------------|
| `"full_attention"` | `FULL` |
| `"sliding_attention"` | `SLIDING_WINDOW` |
| `"chunked_attention"` | `CHUNKED_LOCAL` |
| `"linear_attention"` | `LINEAR` |

**layers_block_type:**
| HF Value | vLLM LayerBlockType |
|----------|---------------------|
| `"attention"` | `ATTENTION` |
| `"mamba"` | `MAMBA` |
| `"hybrid"` | `HYBRID` |

---

## 6. Migration Plan

### Phase 1: Add New Infrastructure (Non-Breaking)

1. **Add new config types**:
   - `LayerAttentionType`, `LayerBlockType`, `LayerRole`, `LayerKVCacheConfig`, `ModelKVCacheRequirements`

2. **Extend `ModelArchitectureConfig`**:
   - Add new optional fields from HF config
   - Add `get_kv_cache_requirements()` method

3. **Create factory module**:
   - `kv_cache_spec_factory.py` with `create_kv_cache_spec()` and helper functions

4. **Add HF config parsers**:
   - `parse_layer_types()`
   - `parse_layers_block_type()`
   - `parse_kv_sharing_pattern()`

5. **Populate `ModelArchitectureConfig` during config loading**:
   - Extend config loading to call parsers and set new fields

### Phase 2: Parallel Implementation

1. **Add `compute_kv_cache_specs_from_config()`**:
   - New config-only path

2. **Add validation in GPUModelRunner**:
   - Compare config-computed specs with model-computed specs
   - Log warnings for mismatches

3. **Comprehensive testing**:
   - Test all attention types
   - Test hybrid models
   - Test encoder-decoder
   - Test KV sharing

### Phase 3: Gradual Transition

1. **Add feature flag**:
   ```python
   cache_config.use_config_only_kv_spec = False  # Default off initially
   ```

2. **Update GPUModelRunner**:
   ```python
   def get_kv_cache_spec(self):
       if self.vllm_config.cache_config.use_config_only_kv_spec:
           return compute_kv_cache_specs_from_config(self.vllm_config)
       else:
           return self._get_kv_cache_spec_from_model()  # Legacy path
   ```

3. **Enable by default after validation**

### Phase 4: Deprecation (Future)

1. **Deprecate instance `get_kv_cache_spec()`**:
   - Add deprecation warning to `AttentionLayerBase.get_kv_cache_spec()`

2. **Remove model dependency**:
   - Engine can compute KV specs before model loading

---

## 7. Testing Strategy

### 7.1 Unit Tests

```python
def test_full_attention_spec_factory():
    layer_config = LayerKVCacheConfig(
        layer_idx=0,
        layer_name="model.layers.0.self_attn.attn",
        attention_type=LayerAttentionType.FULL,
        num_kv_heads=8,
        head_size=64,
        dtype=torch.float16,
    )
    cache_config = CacheConfig(block_size=16)

    spec = create_kv_cache_spec(layer_config, cache_config)

    assert isinstance(spec, FullAttentionSpec)
    assert spec.num_kv_heads == 8
    assert spec.head_size == 64
    assert spec.block_size == 16


def test_layer_types_parsing():
    hf_config = mock_config(
        layer_types=["sliding_attention", "full_attention"] * 4,
    )

    layer_types = parse_layer_types(hf_config)

    assert layer_types == [
        LayerAttentionType.SLIDING_WINDOW,
        LayerAttentionType.FULL,
    ] * 4
```

### 7.2 Integration Tests

```python
@pytest.mark.parametrize("model_name,expected_spec_types", [
    ("meta-llama/Llama-2-7b", [FullAttentionSpec] * 32),
    ("google/gemma-2-2b", [SlidingWindowSpec, FullAttentionSpec] * 13),
    ("deepseek-ai/DeepSeek-V2", [MLAAttentionSpec] * 60),
])
def test_model_kv_cache_specs(model_name, expected_spec_types):
    vllm_config = VllmConfig(model=model_name)
    specs = compute_kv_cache_specs_from_config(vllm_config)

    for layer_name, spec in specs.items():
        layer_idx = extract_layer_index(layer_name)
        assert isinstance(spec, expected_spec_types[layer_idx])
```

### 7.3 Regression Tests

```python
def test_config_vs_model_spec_equivalence():
    model_name = "meta-llama/Llama-2-7b"

    # Config-only path (new)
    vllm_config = VllmConfig(model=model_name)
    config_specs = compute_kv_cache_specs_from_config(vllm_config)

    # Model-based path (legacy)
    runner = GPUModelRunner(vllm_config)
    runner.load_model()
    model_specs = runner._get_kv_cache_spec_from_model()

    assert config_specs.keys() == model_specs.keys()
    for layer_name in config_specs:
        assert config_specs[layer_name] == model_specs[layer_name]
```

---

## 8. Open Questions

1. **Encoder-Decoder Cross-Attention Layer Naming**: How should cross-attention layers be named/identified in `ModelKVCacheRequirements`? Current convention mixes them with decoder layers.

2. **Dynamic Head Counts**: Some vision models (Swin) have different `num_heads` per stage. Should `LayerKVCacheConfig.num_kv_heads` support per-layer overrides?

3. **Speculative Decoding**: How should draft model KV specs interact with target model specs in the config-only path?

4. **Multi-Modal Models**: Models like LLaVA have vision encoders that don't use KV cache. How to represent these cleanly?

---

## 9. Appendix: Full Attention Type Inventory

| Attention Type | KVCacheSpec | Models | HF Config Pattern |
|----------------|-------------|--------|-------------------|
| Full (decoder) | `FullAttentionSpec` | Llama, GPT, etc. | `layer_types: ["full_attention"]` |
| Sliding window | `SlidingWindowSpec` | Gemma2/3, Mistral | `layer_types: ["sliding_attention"]` |
| Chunked local | `ChunkedLocalAttentionSpec` | Llama4 | `layer_types: ["chunked_attention"]` |
| Linear attention | `MambaSpec` | Qwen3Next, MiniMax | `layer_types: ["linear_attention"]` |
| MLA | `MLAAttentionSpec` | DeepSeek-V2 | `kv_lora_rank`, `is_mla` |
| Cross-attention | `CrossAttentionSpec` | Whisper, BART | Separate encoder/decoder configs |
| Encoder-only | `None` | BERT | - |
| Mamba | `MambaSpec` | Jamba, Bamba | `layers_block_type: ["mamba"]` |
| Hybrid | `FullAttentionSpec` + `MambaSpec` | FalconH1 | `layers_block_type: ["hybrid"]` |
| Static sink | `SinkFullAttentionSpec` | (Custom) | Model-specific |

---

## References

- HuggingFace `layer_types` documentation: `configuration_utils.py#L1295-L1306`
- vLLM attention layer: `vllm/attention/layer.py`
- Current KV cache spec computation: `vllm/v1/worker/gpu_model_runner.py#L5790-L5820`
