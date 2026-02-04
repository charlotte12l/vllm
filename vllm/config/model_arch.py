# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from __future__ import annotations

from typing import TYPE_CHECKING, Any

from pydantic import ConfigDict
from pydantic.dataclasses import dataclass

from vllm.logger import init_logger

if TYPE_CHECKING:
    from transformers import PretrainedConfig

logger = init_logger(__name__)


@dataclass(frozen=True)
class KVSharingConfig:
    """Configuration for KV cache sharing between layers."""

    sharing_map: dict[int, int]
    """Mapping from layer index to the target layer index it shares KV with."""

    @classmethod
    def from_hf_config(cls, hf_config: PretrainedConfig) -> KVSharingConfig | None:
        """
        Create KV sharing config from HuggingFace config.

        This handles models like Gemma3n where later layers share KV cache
        with earlier layers.

        Args:
            hf_config: HuggingFace model configuration.

        Returns:
            KVSharingConfig if sharing is configured, None otherwise.
        """
        num_kv_shared_layers = getattr(hf_config, "num_kv_shared_layers", 0)
        if num_kv_shared_layers == 0:
            return None

        num_hidden_layers = hf_config.num_hidden_layers
        first_shared_idx = num_hidden_layers - num_kv_shared_layers
        layer_types = getattr(hf_config, "layer_types", None)

        sharing_map: dict[int, int] = {}
        for layer_idx in range(first_shared_idx, num_hidden_layers):
            # Determine offset based on layer type
            # Sliding attention uses offset=2, full attention uses offset=1
            is_sliding = (
                layer_types is not None
                and layer_idx < len(layer_types)
                and layer_types[layer_idx] == "sliding_attention"
            )
            offset = 2 if is_sliding else 1
            target_idx = first_shared_idx - offset
            if target_idx >= 0:
                sharing_map[layer_idx] = target_idx

        return cls(sharing_map=sharing_map) if sharing_map else None


@dataclass(frozen=True)
class KVCacheModelConfig:
    """
    All model-architecture-derived parameters needed for KV cache spec creation.
    This is a field of ModelArchitectureConfig.

    Each attention class reads only the fields it needs from this config.
    The config contains model architecture info BEFORE applying cache_config
    or parallel_config (e.g., block_size, TP).
    """

    # Common attention parameters (before TP applied)
    total_num_kv_heads: int
    """Total number of KV heads before tensor parallelism."""

    head_size: int
    """Size of each attention head."""

    num_hidden_layers: int
    """Number of hidden layers in the model."""

    head_size_v: int | None = None
    """Size of value head (if different from head_size)."""

    # Per-layer attention types
    layer_types: tuple[tuple[str, ...], ...] | None = None
    """Per-layer attention types. Each tuple contains the attention types
    for one physical layer.
    Examples:
      - Decoder-only: (("full_attention",), ("full_attention",), ...)
      - Hybrid: (("full_attention",), ("mamba",), ("full_attention",), ...)
      - Encoder-decoder: (("full_attention", "cross_attention"), ...)
    """

    # Sliding window attention
    sliding_window: int | None = None
    """Sliding window size for sliding_attention layers."""

    # Chunked local attention
    attention_chunk_size: int | None = None
    """Chunk size for chunked_attention layers."""

    # MLA (DeepSeek) specific
    kv_lora_rank: int | None = None
    """KV LoRA rank for MLA models."""

    qk_rope_head_dim: int | None = None
    """QK RoPE head dimension for MLA models."""

    # Indexer (DeepSeek V3 speculative decoding) specific
    index_head_dim: int | None = None
    """Indexer head dimension for DeepSeek V3 speculative decoding."""

    # Mamba/SSM specific
    mamba_intermediate_size: int | None = None
    """Mamba intermediate size."""

    mamba_state_size: int | None = None
    """Mamba SSM state size."""

    mamba_conv_kernel: int | None = None
    """Mamba convolution kernel size."""

    mamba_num_heads: int | None = None
    """Number of Mamba heads (for Mamba2)."""

    mamba_head_dim: int | None = None
    """Mamba head dimension (for Mamba2)."""

    mamba_num_groups: int | None = None
    """Number of Mamba groups (for Mamba2)."""

    # Encoder-decoder specific
    is_encoder_decoder: bool = False
    """Whether this is an encoder-decoder model."""

    num_encoder_layers: int | None = None
    """Number of encoder layers (for encoder-decoder models)."""

    num_decoder_layers: int | None = None
    """Number of decoder layers (for encoder-decoder models)."""

    # Sink attention
    sink_len: int | None = None
    """Sink length for sink attention."""

    # Block pooling (Whisper-style audio models)
    block_pool_size: int | None = None
    """Block pool size for audio models with block pooling attention."""

    # KV sharing configuration
    kv_sharing_config: KVSharingConfig | None = None
    """Configuration for KV sharing between layers."""

    # Model structure info for naming
    model_prefix: str = "model"
    """Prefix for layer names (e.g., 'model', 'backbone', 'language_model.model')."""

    def get_layer_types(self, layer_idx: int) -> tuple[str, ...]:
        """Get the attention types for a specific physical layer."""
        if self.layer_types is None:
            return ("full_attention",)
        return self.layer_types[layer_idx]

    def get_num_kv_heads_per_tp(self, tp_size: int) -> int:
        """Get number of KV heads per tensor parallel rank.

        Args:
            tp_size: Tensor parallel size.

        Returns:
            Number of KV heads per TP rank (at least 1).
        """
        return max(1, self.total_num_kv_heads // tp_size)

    def get_indexer_cache_head_size(self, quant_block_size: int = 128) -> int | None:
        """Get the head size for indexer cache (DeepSeek V3 speculative decoding).

        The indexer cache stores FP8 quantized values with scales, so the head
        size is: index_head_dim + index_head_dim // quant_block_size * 4

        Args:
            quant_block_size: Quantization block size (default 128).

        Returns:
            Indexer cache head size, or None if not an indexer model.
        """
        if self.index_head_dim is None:
            return None
        return self.index_head_dim + self.index_head_dim // quant_block_size * 4


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

    kv_cache_config: KVCacheModelConfig | None = None
    """KV cache related model configuration for computing KV cache specs."""
