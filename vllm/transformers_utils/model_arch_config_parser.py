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

NUM_HEADS_POSSIBLE_KEYS = [
    # For Falcon:
    "n_head_kv",
    "num_kv_heads",
    # For LLaMA-2:
    "num_key_value_heads",
    # For ChatGLM:
    "multi_query_group_num",
]


NUM_EXPERT_POSSIBLE_KEYS = [
    "num_experts",  # Jamba
    "moe_num_experts",  # Dbrx
    "n_routed_experts",  # DeepSeek
    "num_local_experts",  # Mixtral
]

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

    def get_torch_dtype(config_dict: dict[str, Any]):
        config_dtype = config_dict.pop("dtype", None)

        # Fallbacks for multi-modal models if the root config
        # does not define dtype
        if config_dtype is None:
            config_dtype = config_dict["text_config"].get("dtype", None)
        if config_dtype is None and "vision_config" in config_dict:
            config_dtype = config_dict["vision_config"].get("dtype", None)
        if config_dtype is None and hasattr(config_dict, "encoder_config"):
            config_dtype = config_dict["encoder_config"].get("dtype", None)

        return config_dtype

    @classmethod
    def normalize_quantization_config(
        self,  hf_config: PretrainedConfig
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


    @classmethod
    def get_per_layer_attention_cls(
        self, hf_config,
    ) -> list[type[nn.Module]]:
        # layer_types = hf_config.layer_types
        
        # layer_types_cls = [AttentionMapping[layer_type] for layer_type in layer_types]

        # All full attention
        return layer_types_cls

    @abstractmethod
    def convert(
        self,
        hf_config: PretrainedConfig
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
        )

        return 


if TYPE_CHECKING:
    import vllm.model_executor.models as me_models
else:
    me_models = LazyLoader("model_executor", globals(), "vllm.model_executor.models")







class HFModelArchConfigParser(ModelArchConfigParserBase):
    def parse(
        self,
        model: str | Path,
        trust_remote_code: bool,
        revision: str | None = None,
        code_revision: str | None = None,
        model_impl: str = "auto",
        **kwargs,
    ) -> tuple[dict[str, Any], "ModelArchitectureConfig"]:
        """Parse the HF config and create ModelArchitectureConfig."""

        is_gguf = kwargs.get("is_gguf", False)
        if is_gguf:
            kwargs["gguf_file"] = Path(model).name
            model = Path(model).parent

        kwargs["local_files_only"] = huggingface_hub.constants.HF_HUB_OFFLINE

        config_dict, _ = PretrainedConfig.get_config_dict(
            model,
            revision=revision,
            code_revision=code_revision,
            token=_get_hf_token(),
            **kwargs,
        )
        # Use custom model class if it's in our registry
        model_type = config_dict.get("model_type", "")

        if model_type in _CONFIG_REGISTRY:
            # TODO: check if need to write new config class that
            # inherient ModelArchitectureTextConfig for each of those models
            raise NotImplementedError
        else:
            # We use AutoConfig.from_pretrained to leverage some existing
            # standardization in PretrainedConfig
            try:
                kwargs = _maybe_update_auto_config_kwargs(kwargs, model_type=model_type)
                # https://github.com/huggingface/transformers/blob/e8a6eb3304033fdd9346fe3b3293309fe50de238/src/transformers/models/auto/configuration_auto.py#L1238
                config_dict = AutoConfig.from_pretrained(
                    model,
                    trust_remote_code=trust_remote_code,
                    revision=revision,
                    code_revision=code_revision,
                    token=_get_hf_token(),
                    **kwargs,
                ).to_dict()
            except ValueError as e:
                if (
                    not trust_remote_code
                    and "requires you to execute the configuration file" in str(e)
                ):
                    err_msg = (
                        "Failed to load the model config. If the model "
                        "is a custom model not yet available in the "
                        "HuggingFace transformers library, consider setting "
                        "`trust_remote_code=True` in LLM or using the "
                        "`--trust-remote-code` flag in the CLI."
                    )
                    raise RuntimeError(err_msg) from e
                else:
                    raise e

        architectures = config_dict.pop("architectures", [])
        quantization_config = get_quantization_config(model, revision, config_dict)
        torch_dtype = get_torch_dtype(config_dict)

        standard_fields, text_config_dict = extract_standard_text_config_field(
            config_dict
        )
        # Ensure no overlap between standard fields and remaining text config
        overlap = set(standard_fields.keys()) & set(text_config_dict.keys())
        assert len(overlap) == 0, (
            f"Standard fields and text config dict should not overlap, got {overlap}"
        )
        # Extract text config fields
        text_config = ModelArchitectureTextConfig(**standard_fields, **text_config_dict)

        # Special architecture mapping check for GGUF models
        if is_gguf:
            if model_type not in MODEL_FOR_CAUSAL_LM_MAPPING_NAMES:
                raise RuntimeError(f"Can't get gguf config for {model_type}.")
            model_type = MODEL_FOR_CAUSAL_LM_MAPPING_NAMES[model_type]
            architectures = [model_type]

        # Architecture mapping for models without explicit architectures field
        if not architectures:
            if model_type not in MODEL_MAPPING_NAMES:
                logger.warning(
                    "Model config does not have a top-level "
                    "'architectures' field: expecting "
                    "`model_arch_overrides={'architectures': ['...']}` "
                    "to be passed in engine args."
                )
            else:
                model_type = MODEL_MAPPING_NAMES[model_type]
                architectures = [model_type]

        vision_config_dict = config_dict.get("vision_config", {})
        audio_config_dict = config_dict.get("audio_config", {})

        per_layer_attention_cls = get_per_layer_attention_cls(
            architectures, model_impl, text_config
        )

        # Create ModelArchitectureConfig
        vision_config = (
            ModelArchitectureVisionConfig(**vision_config_dict)
            if vision_config_dict
            else None
        )
        audio_config = (
            ModelArchitectureAudioConfig(**audio_config_dict)
            if audio_config_dict
            else None
        )

        arch_config = ModelArchitectureConfig(
            architectures=architectures,
            model_type=model_type,
            quantization_config=quantization_config,
            torch_dtype=torch_dtype,
            per_layer_attention_cls=per_layer_attention_cls,
            text_config=text_config,
            vision=vision_config,
            audio=audio_config,
        )

        return config_dict, arch_config
