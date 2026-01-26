# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from __future__ import annotations

from typing import TYPE_CHECKING, Any

import torch
from pydantic import ConfigDict
from pydantic.dataclasses import dataclass

from vllm.config.kv_cache_spec_config import (
    LayerAttentionType,
    LayerBlockType,
    LayerKVCacheConfig,
    LayerRole,
    ModelKVCacheRequirements,
)
from vllm.logger import init_logger

if TYPE_CHECKING:
    pass

logger = init_logger(__name__)


@dataclass(config=ConfigDict(arbitrary_types_allowed=True))
class ModelArchitectureConfig:
    """
    Configuration for model architecture that required by vLLM runtime
    """

    architectures: list[str] | None
    """List of model architecture class names (e.g., ['LlamaForCausalLM']).
       It can be None upon calling `vllm_config.with_hf_config(config.text_config)`"""

    model_type: str
    """Model type identifier (e.g., 'llama', 'gpt_oss')."""

    text_model_type: str | None
    """Text model type identifier (e.g., 'llama4_text')."""

    hidden_size: int
    """Hidden size of the model."""

    total_num_hidden_layers: int
    """Number of hidden layers in the model."""

    total_num_attention_heads: int
    """Number of attention heads in the model."""

    head_size: int
    """Head dimension of the model."""

    vocab_size: int
    """Vocabulary size of the model."""

    total_num_kv_heads: int
    """Number of key value heads in the model."""

    num_experts: int
    """Number of experts in the model."""

    quantization_config: dict[str, Any] | None
    """Quantization configuration dictionary containing quantization parameters."""

    is_deepseek_mla: bool
    """Whether the model is a DeepSeek MLA model."""

    derived_max_model_len_and_key: tuple[float, str | None]
    """Derived maximum model length and key from the hf config."""

    # NEW: Fields for config-only KV cache spec computation

    layer_types: list[LayerAttentionType] | None = None
    """Per-layer attention type from HF config.layer_types.
    Values: full_attention, sliding_attention, chunked_attention, linear_attention"""

    layers_block_type: list[LayerBlockType] | None = None
    """Per-layer block type from HF config.layers_block_type.
    Values: attention, mamba, hybrid"""

    sliding_window: int | None = None
    """Sliding window size for layers with attention_type == sliding_attention."""

    attention_chunk_size: int | None = None
    """Chunk size for layers with attention_type == chunked_attention."""

    kv_lora_rank: int | None = None
    """KV latent dimension for MLA (Multi-head Latent Attention)."""

    qk_rope_head_dim: int | None = None
    """RoPE dimension for MLA."""

    mamba_type: str | None = None
    """Mamba variant type: 'mamba1' or 'mamba2'."""

    ssm_state_size: int | None = None
    """SSM state dimension for Mamba layers."""

    conv_kernel_size: int | None = None
    """Convolution kernel size for Mamba layers."""

    mamba_expand: int | None = None
    """Expansion factor for Mamba intermediate size calculation."""

    encoder_num_layers: int | None = None
    """Number of encoder layers for encoder-decoder models."""

    decoder_num_layers: int | None = None
    """Number of decoder layers for encoder-decoder models."""

    encoder_num_kv_heads: int | None = None
    """Number of KV heads in encoder layers."""

    decoder_num_kv_heads: int | None = None
    """Number of KV heads in decoder layers."""

    kv_sharing_pattern: dict[int, int] | None = None
    """Maps layer_idx -> target_layer_idx for KV cache sharing."""

    # ============== Methods for KV cache spec computation ==============

    def get_kv_cache_requirements(
        self,
        kv_cache_dtype: torch.dtype,
        prefix: str = "model.layers",
    ) -> ModelKVCacheRequirements:
        """Build complete KV cache requirements from architecture config.

        This method enables KV cache spec computation without model instantiation.

        Args:
            kv_cache_dtype: The dtype for KV cache tensors.
            prefix: The prefix for layer names (default: "model.layers").

        Returns:
            ModelKVCacheRequirements containing per-layer KV cache configs.
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
        is_hybrid = self.layers_block_type is not None and any(
            bt in (LayerBlockType.MAMBA, LayerBlockType.HYBRID)
            for bt in self.layers_block_type
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

        # Calculate intermediate_size for Mamba
        intermediate_size = None
        if block_type in (LayerBlockType.MAMBA, LayerBlockType.HYBRID):
            if self.mamba_expand:
                intermediate_size = self.hidden_size * self.mamba_expand
            else:
                intermediate_size = self.hidden_size * 2  # Default expansion

        # Build the config
        return LayerKVCacheConfig(
            layer_idx=layer_idx,
            layer_name=layer_name,
            attention_type=attention_type,
            block_type=block_type,
            role=role,
            num_kv_heads=self._get_layer_num_kv_heads(layer_idx, role),
            head_size=self._get_layer_head_size(),
            head_size_v=self.head_size,
            dtype=dtype,
            sliding_window=sliding_window,
            attention_chunk_size=chunk_size,
            kv_lora_rank=self.kv_lora_rank if self.is_deepseek_mla else None,
            qk_rope_head_dim=self.qk_rope_head_dim if self.is_deepseek_mla else None,
            mamba_type=self.mamba_type
            if block_type in (LayerBlockType.MAMBA, LayerBlockType.HYBRID)
            else None,
            ssm_state_size=self.ssm_state_size
            if block_type in (LayerBlockType.MAMBA, LayerBlockType.HYBRID)
            else None,
            conv_kernel_size=self.conv_kernel_size
            if block_type in (LayerBlockType.MAMBA, LayerBlockType.HYBRID)
            else None,
            intermediate_size=intermediate_size,
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

    def _get_layer_head_size(self) -> int:
        """Get head size for a layer."""
        if self.is_deepseek_mla:
            # MLA stores compressed latent: kv_lora_rank + qk_rope_head_dim
            return (self.kv_lora_rank or 0) + (self.qk_rope_head_dim or 0)
        return self.head_size
