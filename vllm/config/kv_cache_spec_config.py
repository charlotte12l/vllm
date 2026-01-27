# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
Configuration types for config-only KV cache spec computation.

This module defines the data structures needed to compute KVCacheSpec
purely from configuration, without instantiating the model.
"""

from dataclasses import dataclass, field
from enum import Enum

import torch


class LayerAttentionType(str, Enum):
    """Attention mechanism type derived from HF layer_types.

    These values match HuggingFace's layer_types field directly.
    """

    FULL = "full_attention"
    SLIDING_WINDOW = "sliding_attention"
    CHUNKED_LOCAL = "chunked_attention"
    LINEAR = "linear_attention"


class LayerBlockType(str, Enum):
    """Block type derived from HF layers_block_type.

    These values match HuggingFace's layers_block_type field directly.
    """

    ATTENTION = "attention"
    MAMBA = "mamba"
    HYBRID = "hybrid"  # Parallel attention + Mamba (e.g., FalconH1)


class LayerRole(str, Enum):
    """Role of the layer in an encoder-decoder architecture."""

    DECODER = "decoder"  # Decoder self-attention (default)
    ENCODER = "encoder"  # Encoder self-attention
    CROSS_ATTENTION = "cross"  # Encoder-decoder cross-attention
    ENCODER_ONLY = "encoder_only"  # Encoder-only model (BERT-style)


@dataclass(frozen=True)
class LayerKVCacheConfig:
    """Per-layer configuration for KV cache spec computation.

    This is the minimal info needed to compute a KVCacheSpec for one layer
    without instantiating the model.
    """

    # Layer identification (0-indexed)
    layer_idx: int

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

    # KV sharing (target layer index, not name)
    kv_sharing_target_layer_idx: int | None = None


@dataclass
class ModelKVCacheRequirements:
    """Complete KV cache requirements for a model, derived from config only.

    This replaces the need to instantiate the model just to get KV cache specs.
    """

    # Per-layer configurations
    layers: list[LayerKVCacheConfig] = field(default_factory=list)

    # Model-level defaults (applied when layer-specific values are None)
    default_num_kv_heads: int = 0
    default_head_size: int = 0
    default_dtype: torch.dtype = torch.float16

    # Architecture type hints
    is_encoder_decoder: bool = False
    is_hybrid: bool = False  # Has Mamba blocks
    is_mla: bool = False

    @property
    def layer_count(self) -> int:
        return len(self.layers)

    @property
    def attention_layer_count(self) -> int:
        return sum(
            1
            for layer in self.layers
            if layer.block_type in (LayerBlockType.ATTENTION, LayerBlockType.HYBRID)
        )

    @property
    def mamba_layer_count(self) -> int:
        return sum(
            1
            for layer in self.layers
            if layer.block_type in (LayerBlockType.MAMBA, LayerBlockType.HYBRID)
        )
