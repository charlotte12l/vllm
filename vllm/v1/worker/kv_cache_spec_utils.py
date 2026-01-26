# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
Utilities for computing KV cache specs from configuration only.

This module provides the dispatch logic to compute KVCacheSpec for each layer
without instantiating the model, using only the model_arch_convertor and
CacheConfig.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import TYPE_CHECKING

import torch

from vllm.config import CacheConfig
from vllm.config.kv_cache_spec_config import (
    LayerAttentionType,
    LayerBlockType,
    LayerKVCacheConfig,
    LayerRole,
    ModelKVCacheRequirements,
)
from vllm.v1.kv_cache_interface import KVCacheSpec

if TYPE_CHECKING:
    from vllm.transformers_utils.model_arch_config_convertor import (
        ModelArchConfigConvertorBase,
    )


def compute_kv_cache_specs_from_config(
    convertor: ModelArchConfigConvertorBase,
    cache_config: CacheConfig,
    kv_cache_dtype: torch.dtype,
    prefix: str = "model.layers",
) -> Mapping[str, KVCacheSpec]:
    """Compute KV cache specs for all layers from config only.

    This is the main entry point for config-only KV cache spec computation.
    It replaces the need to instantiate the model just to get KV cache specs.

    Args:
        convertor: Model architecture config convertor with KV cache methods.
        cache_config: Cache configuration.
        kv_cache_dtype: The dtype for KV cache (resolved from cache_config).
        prefix: Prefix for layer names (default: "model.layers").

    Returns:
        Dict mapping layer names to their KVCacheSpec.
    """
    # Build per-layer configs from convertor
    requirements = convertor.get_kv_cache_requirements(
        kv_cache_dtype=kv_cache_dtype,
        prefix=prefix,
    )

    # Dispatch each layer config to the appropriate attention class
    kv_cache_specs: dict[str, KVCacheSpec] = {}

    for layer_config in requirements.layers:
        spec = _dispatch_layer_spec(layer_config, cache_config, requirements)
        if spec is not None:
            kv_cache_specs[layer_config.layer_name] = spec

    return kv_cache_specs


def _dispatch_layer_spec(
    layer_config: LayerKVCacheConfig,
    cache_config: CacheConfig,
    requirements: ModelKVCacheRequirements,
) -> KVCacheSpec | None:
    """Dispatch to the appropriate attention class to create KVCacheSpec.

    Args:
        layer_config: Per-layer KV cache configuration.
        cache_config: Global cache configuration.
        requirements: Model-level KV cache requirements.

    Returns:
        KVCacheSpec for this layer, or None if no KV cache is needed.
    """
    # First check layer role for special cases
    if layer_config.role == LayerRole.ENCODER_ONLY:
        # Encoder-only attention doesn't need KV cache
        from vllm.model_executor.layers.attention.encoder_only_attention import (
            EncoderOnlyAttention,
        )

        return EncoderOnlyAttention.create_kv_cache_spec_from_config(
            layer_config, cache_config
        )

    if layer_config.role == LayerRole.CROSS_ATTENTION:
        # Cross-attention for encoder-decoder models
        from vllm.model_executor.layers.attention.cross_attention import (
            CrossAttention,
        )

        return CrossAttention.create_kv_cache_spec_from_config(
            layer_config, cache_config
        )

    # Dispatch based on block type (attention vs mamba vs hybrid)
    if layer_config.block_type == LayerBlockType.MAMBA:
        # Pure Mamba/SSM layer
        from vllm.model_executor.layers.mamba.abstract import MambaBase

        return MambaBase.create_kv_cache_spec_from_config(layer_config, cache_config)

    if layer_config.block_type == LayerBlockType.HYBRID:
        # Hybrid layer (parallel attention + Mamba, e.g., FalconH1)
        # Returns both attention and mamba specs
        return _create_hybrid_spec(layer_config, cache_config)

    # Dispatch based on attention type
    if requirements.is_mla:
        # MLA attention (DeepSeek-V2/V3 style)
        from vllm.attention.layer import MLAAttention

        return MLAAttention.create_kv_cache_spec_from_config(layer_config, cache_config)

    if layer_config.attention_type == LayerAttentionType.CHUNKED_LOCAL:
        # Chunked local attention
        from vllm.model_executor.layers.attention.chunked_local_attention import (
            ChunkedLocalAttention,
        )

        return ChunkedLocalAttention.create_kv_cache_spec_from_config(
            layer_config, cache_config
        )

    # Default: standard attention (full or sliding window)
    from vllm.attention.layer import Attention

    return Attention.create_kv_cache_spec_from_config(layer_config, cache_config)


def _create_hybrid_spec(
    layer_config: LayerKVCacheConfig,
    cache_config: CacheConfig,
) -> KVCacheSpec:
    """Create specs for hybrid layers (parallel attention + Mamba).

    For hybrid layers like FalconH1, we typically need both attention and
    Mamba specs. The actual handling depends on how the model is implemented.

    For now, we create the attention spec since the hybrid Mamba layers
    typically share state with the attention layers.
    """
    from vllm.attention.layer import Attention

    # For hybrid layers, create the attention spec
    # The Mamba state is typically handled separately or shared
    return Attention.create_kv_cache_spec_from_config(layer_config, cache_config)


def parse_layer_types(
    hf_config: object,
    num_layers: int,
) -> list[LayerAttentionType]:
    """Parse layer_types from HuggingFace config.

    The HF config is expected to have a `layer_types` field with values like:
    - "full_attention"
    - "sliding_attention"
    - "chunked_attention"
    - "linear_attention"

    Args:
        hf_config: HuggingFace model config.
        num_layers: Number of layers in the model.

    Returns:
        List of LayerAttentionType for each layer.
    """
    layer_types_raw = getattr(hf_config, "layer_types", None)

    if layer_types_raw is None:
        # Check for alternative field names
        layer_types_raw = getattr(hf_config, "attention_types", None)

    if layer_types_raw is None:
        # Default to full attention for all layers
        return [LayerAttentionType.FULL] * num_layers

    # Handle string values
    type_mapping = {
        "full_attention": LayerAttentionType.FULL,
        "full": LayerAttentionType.FULL,
        "sliding_attention": LayerAttentionType.SLIDING_WINDOW,
        "sliding": LayerAttentionType.SLIDING_WINDOW,
        "sliding_window": LayerAttentionType.SLIDING_WINDOW,
        "chunked_attention": LayerAttentionType.CHUNKED_LOCAL,
        "chunked": LayerAttentionType.CHUNKED_LOCAL,
        "chunked_local": LayerAttentionType.CHUNKED_LOCAL,
        "linear_attention": LayerAttentionType.LINEAR,
        "linear": LayerAttentionType.LINEAR,
    }

    result = []
    for layer_type in layer_types_raw:
        layer_type_lower = (
            layer_type.lower()
            if isinstance(layer_type, str)
            else str(layer_type).lower()
        )
        mapped_type = type_mapping.get(layer_type_lower, LayerAttentionType.FULL)
        result.append(mapped_type)

    # Extend or truncate to match num_layers
    if len(result) < num_layers:
        result.extend([LayerAttentionType.FULL] * (num_layers - len(result)))
    elif len(result) > num_layers:
        result = result[:num_layers]

    return result


def parse_layers_block_type(
    hf_config: object,
    num_layers: int,
) -> list[LayerBlockType]:
    """Parse layers_block_type from HuggingFace config.

    The HF config is expected to have a `layers_block_type` field with values:
    - "attention"
    - "mamba"
    - "hybrid"

    Args:
        hf_config: HuggingFace model config.
        num_layers: Number of layers in the model.

    Returns:
        List of LayerBlockType for each layer.
    """
    block_types_raw = getattr(hf_config, "layers_block_type", None)

    if block_types_raw is None:
        # Default to attention for all layers
        return [LayerBlockType.ATTENTION] * num_layers

    # Handle string values
    type_mapping = {
        "attention": LayerBlockType.ATTENTION,
        "attn": LayerBlockType.ATTENTION,
        "mamba": LayerBlockType.MAMBA,
        "ssm": LayerBlockType.MAMBA,
        "hybrid": LayerBlockType.HYBRID,
    }

    result = []
    for block_type in block_types_raw:
        block_type_lower = (
            block_type.lower()
            if isinstance(block_type, str)
            else str(block_type).lower()
        )
        mapped_type = type_mapping.get(block_type_lower, LayerBlockType.ATTENTION)
        result.append(mapped_type)

    # Extend or truncate to match num_layers
    if len(result) < num_layers:
        result.extend([LayerBlockType.ATTENTION] * (num_layers - len(result)))
    elif len(result) > num_layers:
        result = result[:num_layers]

    return result


def parse_kv_sharing_pattern(
    hf_config: object,
    num_layers: int,
) -> dict[int, int] | None:
    """Parse KV sharing pattern from HuggingFace config.

    Some models share KV cache between layers to reduce memory usage.
    This parses the sharing pattern from HF config.

    Args:
        hf_config: HuggingFace model config.
        num_layers: Number of layers in the model.

    Returns:
        Dict mapping layer indices to source layer indices, or None if none.
    """
    sharing_pattern = getattr(hf_config, "kv_sharing_pattern", None)

    if sharing_pattern is None:
        return None

    # Convert to int keys if needed
    result = {}
    for key, value in sharing_pattern.items():
        result[int(key)] = int(value)

    return result


def _get_mamba_type(hf_config: object) -> str:
    """Determine Mamba type from HF config."""
    # Check for explicit mamba_type field
    mamba_type = getattr(hf_config, "mamba_type", None)
    if mamba_type is not None:
        return mamba_type

    # Infer from model type
    model_type = getattr(hf_config, "model_type", "").lower()
    if "mamba2" in model_type:
        return "mamba2"

    # Check for Mamba2-specific fields
    if hasattr(hf_config, "mamba_n_heads"):
        return "mamba2"

    # Default to mamba1
    return "mamba1"
