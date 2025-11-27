# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import functools
import json
from dataclasses import field
from typing import Any

import torch
from pydantic import ConfigDict
from pydantic.dataclasses import dataclass
from torch import nn

from vllm.config.utils import config
from vllm.logger import init_logger

logger = init_logger(__name__)


@dataclass
class ModelArchitectureConfig:
    """
    Configuration for model architecture
    """
    architectures: list[str]
    """List of model architecture class names (e.g., ['LlamaForCausalLM'])."""

    model_type: str
    """Model type identifier (e.g., 'llama', 'gpt_oss')."""

    text_model_type: str | None
    """Text model type identifier (e.g., 'llama4_text')."""
    
    hidden_size: int
    """Hidden size of the model."""
    
    num_hidden_layers: int
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

    quantization_config: dict[str, Any]
    """Quantization configuration dictionary containing quantization parameters."""

    torch_dtype: torch.dtype
    """PyTorch data type for model weights (e.g., 'float16', 'bfloat16')."""

    support_multimodal: bool
    """Whether the model supports multimodal input."""

    is_deepseek_mla: bool
    """Whether the model is a DeepSeek MLA model."""

    derived_max_model_len_and_key: tuple[int, str]
    """Derived maximum model length and key from the model config."""