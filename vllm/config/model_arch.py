# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import functools
import json
from dataclasses import field
from typing import Any

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


    architectures: list[str] = field(default_factory=list)
    """List of model architecture class names (e.g., ['LlamaForCausalLM'])."""

    model_type: str
    """Model type identifier (e.g., 'llama', 'gpt2')."""

    text_model_type: str
    
    hidden_size: int
    
    num_hidden_layers: int
    
    num_attention_heads: int
    head_dim: int
    vocab_size: int
    num_key_value_heads: int
    num_experts: int


    quantization_config: dict[str, Any]
    """Quantization configuration dictionary containing quantization parameters."""

    torch_dtype: torch.dtype
    """PyTorch data type for model weights (e.g., 'float16', 'bfloat16')."""

    per_layer_attention_cls: list[type[nn.Module]]
    """Per-layer attention class of the model."""

    support_multimodal: bool