# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from typing import Any

import torch
from pydantic import ConfigDict
from pydantic.dataclasses import dataclass

from vllm.logger import init_logger

logger = init_logger(__name__)


@dataclass(config=ConfigDict(arbitrary_types_allowed=True))
class ModelArchitectureConfig:
    """
    Configuration for model architecture that required by vLLM runtime
    """

    architectures: list[str]
    """List of model architecture class names (e.g., ['LlamaForCausalLM'])."""

    model_type: str
    """Model type identifier (e.g., 'llama', 'gpt_oss')."""

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

    torch_dtype: torch.dtype | str | None
    """PyTorch data type for model weights (e.g., 'float16', 'bfloat16')."""

    is_deepseek_mla: bool
    """Whether the model is a DeepSeek MLA model."""

    derived_max_model_len_and_key: tuple[float, str | None]
    """Derived maximum model length and key from the hf config."""

    attention_chunk_size: int | None

    dual_chunk_attention_config: dict[str, Any] | None

    block_configs: list[str] | None

    layers_block_type: str | None

    attn_type_list: list[str] | None

    layer_types: list[str] | None

    is_encoder_decoder: bool

    uses_mrope: bool
    uses_xdrope_dim: bool
    is_matryoshka: bool
    matryoshka_dimensions: list[int] | None
    use_pad_token: bool
    head_dtype: torch.dtype | str | None
    position_embedding_type: str | None
    is_causal: bool
    rope_parameters: dict[str, Any] | None
    original_max_position_embeddings: int
    model_max_length: int

    # "projection_dim", "projection_size"
    text_sliding_window: int | None

    mamba_chunk_size: int | None
    # if hf_value := hf_config.get_text_config().max_position_embeddings:
    # config_plugin = hf_config.get("io_processor_plugin")
    # hf_config.id2label
    # "decoder_start_token_id",
    # odel_config.hf_config, "index_topk")
    # hf_config, "num_labels", 0) != 1
    # getattr(self.draft_model_config.hf_config, "eagle_config", None)
    #  = hf_config.image_token_index, "eagle_aux_hidden_state_layer_ids", "id2label",

    # In tests:
    # hf_config.vision_config.hidden_size =
    # image_size = hf_config.vision_config.image_size
    # patch_size = hf_config.vision_config.patch_size
    #     round(1.0 / (hf_config.vision_config.pixel_shuffle_ratio**2))
