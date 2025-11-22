# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import os
from abc import ABC, abstractmethod
from copy import deepcopy
from pathlib import Path
from typing import TYPE_CHECKING, Any

import huggingface_hub
from torch import nn
from transformers import AutoConfig, PretrainedConfig
from transformers.models.auto.modeling_auto import (
    MODEL_FOR_CAUSAL_LM_MAPPING_NAMES,
    MODEL_MAPPING_NAMES,
)

from vllm import envs
from vllm.config.model_arch import (
    ModelArchitectureAudioConfig,
    ModelArchitectureConfig,
    ModelArchitectureTextConfig,
    ModelArchitectureVisionConfig,
)
from vllm.logger import init_logger
from vllm.transformers_utils.config import (
    _CONFIG_REGISTRY,
    _get_hf_token,
    _maybe_update_auto_config_kwargs,
    file_or_path_exists,
    get_hf_file_to_dict,
)
from vllm.utils.import_utils import LazyLoader

logger = init_logger(__name__)

class ModelArchConfigConvertorBase(ABC):
    @classmethod
    def get_num_hidden_layers(self, config: PretrainedConfig) -> int:
        return getattr(
            config, "num_hidden_layers", 0
        )

    @classmethod
    def get_num_attention_heads(self, config: PretrainedConfig) -> int:
        return getattr(config, "num_attention_heads", 0)

    @classmethod
    def get_head_size(
        self, config: PretrainedConfig
    ) -> int:
        # NOTE: Some configs may set head_dim=None in the config
        if getattr(config, "head_dim", None) is not None:
            return config.head_dim

        # NOTE: Some models (such as PLaMo2.1) use `hidden_size_per_head`
        if getattr(config, "hidden_size_per_head", None) is not None:
            return config.hidden_size_per_head

        # FIXME(woosuk): This may not be true for all models.
        return (
            config.hidden_size // config.num_attention_heads
        )
    
    @classmethod
    def get_total_num_kv_heads(
        self, config: PretrainedConfig
    ) -> int:
        attributes = [
            # For Falcon:
            "n_head_kv",
            "num_kv_heads",
            # For LLaMA-2:
            "num_key_value_heads",
            # For ChatGLM:
            "multi_query_group_num",
        ]
        for attr in attributes:
            num_kv_heads = getattr(self.hf_text_config, attr, None)
            if num_kv_heads is not None:
                return num_kv_heads
    
        return config.num_attention_heads

    @classmethod
    def get_num_experts(self, config: PretrainedConfig) -> int:
        """Returns the number of experts in the model."""
        num_expert_names = [
            "num_experts",  # Jamba
            "moe_num_experts",  # Dbrx
            "n_routed_experts",  # DeepSeek
            "num_local_experts",  # Mixtral
        ]
        num_experts = getattr_iter(config, num_expert_names, 0)
        if isinstance(num_experts, list):
            # Ernie VL's remote code uses list[int]...
            # The values are always the same so we just take the first one.
            return num_experts[0]
        # Coerce to 0 if explicitly set to None
        return num_experts or 0

    @classmethod
    def get_torch_dtype(self, config: PretrainedConfig, model_id: str, revision: str | None):
        # NOTE: getattr(config, "dtype", torch.float32) is not correct
        # because config.dtype can be None.
        config_dtype = getattr(config, "dtype", None)

        # Fallbacks for multi-modal models if the root config
        # does not define dtype
        if config_dtype is None:
            config_dtype = getattr(config.get_text_config(), "dtype", None)
        if config_dtype is None and hasattr(config, "vision_config"):
            config_dtype = getattr(config.vision_config, "dtype", None)
        if config_dtype is None and hasattr(config, "encoder_config"):
            config_dtype = getattr(config.encoder_config, "dtype", None)

        # Try to read the dtype of the weights if they are in safetensors format
        if config_dtype is None:
            repo_mt = try_get_safetensors_metadata(model_id, revision=revision)

            if repo_mt and (files_mt := repo_mt.files_metadata):
                param_dtypes: set[torch.dtype] = {
                    _SAFETENSORS_TO_TORCH_DTYPE[dtype_str]
                    for file_mt in files_mt.values()
                    for dtype_str in file_mt.parameter_count
                    if dtype_str in _SAFETENSORS_TO_TORCH_DTYPE
                }

                if param_dtypes:
                    return common_broadcastable_dtype(param_dtypes)

        if config_dtype is None:
            config_dtype = torch.float32

        return config_dtype


    @classmethod
    def normalize_quantization_config(
        self,  config: PretrainedConfig
    ):
        quant_cfg = getattr(hf_config, "quantization_config", None)
        if quant_cfg is None:
            # compressed-tensors uses a "compression_config" key
            quant_cfg = getattr(hf_config, "compression_config", None)

        else:
            # Set quant_method for ModelOpt models.
            producer_name = quant_cfg.get("producer", {}).get("name")
            if producer_name == "modelopt":
                quant_algo = quant_cfg.get("quantization", {}).get("quant_algo")
                if quant_algo == "FP8":
                    quant_cfg["quant_method"] = "modelopt"
                elif quant_algo == "NVFP4":
                    quant_cfg["quant_method"] = "modelopt_fp4"
                elif quant_algo is not None:
                    raise ValueError(f"Unknown ModelOpt quant algo: {quant_algo}")

        if quant_cfg is not None:
            # Use the community standard 'quant_method'
            quant_method = quant_cfg.get("quant_method", "").lower()

            # Normalize library names
            quant_method = quant_method.replace(
                "compressed_tensors", "compressed-tensors"
            )

            quant_cfg["quant_method"] = quant_method

        return quant_cfg

    def get_per_layer_attention_cls(
        self, config: PretrainedConfig,
    ) -> list[type[nn.Module]]:

        return [Attention for _ in range(self.get_num_hidden_layers(config))]

    @abstractmethod
    def convert(
        self,
        hf_config: PretrainedConfig,
        model_id: str,
        revision: str | None,
    ) -> ModelArchitectureConfig:
        if hasattr(hf_config, "text_config"):
            text_config = hf_config.text_config
        else:
            text_config = hf_config
        
        model_arch_config = ModelArchitectureConfig(
            model_type = text_config.model_type,
            hidden_size = text_config.hidden_size,
            num_hidden_layers=self.get_num_hidden_layers(text_config),
            num_attention_heads=self.get_num_attention_heads(text_config),
            head_dim = self.get_head_dim(text_config),
            vocab_size = text_config.vocab_size,
            num_key_value_heads = self.get_total_num_kv_heads(text_config),
            num_experts = self.get_num_experts(text_config),
            quantization_config= self.normalize_quantization_config(text_config),
            dtype = self.get_torch_dtype(config, model_id, revision),
        )

        return model_arch_config


class LlamaModelArchConfigConvertor(ModelArchConfigConvertorBase):
    def get_per_layer_attention_cls(
        self, config: PretrainedConfig,
    ) -> list[type[nn.Module]]:
        if getattr(config, "is_causal", True):
            attn_cls = Attention
        else:
            attn_cls = EncoderOnlyAttention

        return [attn_cls] * self.get_num_hidden_layers(config)