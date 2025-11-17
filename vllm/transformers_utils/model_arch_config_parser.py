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
    def extract_num_hidden_layers(self, config_dict: dict[str, Any]) -> int:
        return config_dict.get("num_hidden_layers", 0)

    @classmethod
    def extract_head_size(
        self, config_dict: dict[str, Any]
    ) -> int:
        # NOTE: Some configs may set head_dim=None in the config
        if config_dict.get("head_dim", None) is not None:
            return config_dict["head_dim"]

        # NOTE: Some models (such as PLaMo2.1) use `hidden_size_per_head`
        if config_dict.get("hidden_size_per_head", None) is not None:
            return config_dict["hidden_size_per_head"]

        # FIXME(woosuk): This may not be true for all models.
        return config_dict["hidden_size"] // config_dict["num_attention_heads"]

    @classmethod
    def extract_total_num_kv_heads(
        self, config_dict: dict[str, Any],
    ) -> int:
        return getattr_iter(config_dict, NUM_HEADS_POSSIBLE_KEYS, 0)

    @classmethod
    def extract_num_experts(self, config_dict: dict[str, Any]) -> int:
        """Returns the number of experts in the model."""
        return getattr_iter(config_dict, NUM_EXPERT_POSSIBLE_KEYS, 0)

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

    # Need to make quantization config better
    @classmethod
    def get_quantization_config(
        self, model: str | Path, revision: str | None, config_dict: dict[str, Any]
    ) -> dict[str, Any]:
        # ModelOpt 0.31.0 and after saves the quantization config in the model
        # config file.
        quantization_config = config_dict.get("quantization_config", None)

        # ModelOpt 0.29.0 and before saves the quantization config in a separate
        # "hf_quant_config.json" in the same directory as the model config file.
        if quantization_config is None and file_or_path_exists(
            model, "hf_quant_config.json", revision
        ):
            quantization_config = get_hf_file_to_dict(
                "hf_quant_config.json", model, revision
            )

        if quantization_config is not None:
            # config.quantization_config = quantization_config
            # auto-enable DeepGEMM UE8M0 if model config requests it
            scale_fmt = quantization_config.get("scale_fmt", None)
            if scale_fmt in ("ue8m0",):
                if not envs.is_set("VLLM_USE_DEEP_GEMM_E8M0"):
                    os.environ["VLLM_USE_DEEP_GEMM_E8M0"] = "1"
                    logger.info_once(
                        (
                            "Detected quantization_config.scale_fmt=%s; "
                            "enabling UE8M0 for DeepGEMM."
                        ),
                        scale_fmt,
                    )
                elif not envs.VLLM_USE_DEEP_GEMM_E8M0:
                    logger.warning_once(
                        (
                            "Model config requests UE8M0 "
                            "(quantization_config.scale_fmt=%s), but "
                            "VLLM_USE_DEEP_GEMM_E8M0=0 is set; "
                            "UE8M0 for DeepGEMM disabled."
                        ),
                        scale_fmt,
                    )

        return quantization_config or {}

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

        )
        # standard_fields = {}
        # standard_fields["model_type"] = 

        # standard_fields["hidden_size"] = text_config_dict.pop("hidden_size")

        # (standard_fields["num_hidden_layers"]) = extract_num_hidden_layers(
        #     text_config_dict, standard_fields["model_type"]
        # )
        # standard_fields["num_attention_heads"] = text_config_dict.pop("num_attention_heads")

        # standard_fields["use_deepseek_mla"] = extract_use_deepseek_mla(
        #     text_config_dict, standard_fields["model_type"]
        # )
        # standard_fields["head_dim"] = extract_head_size(text_config_dict, standard_fields)
        # standard_fields["vocab_size"] = text_config_dict.pop("vocab_size")
        # standard_fields["num_key_value_heads"] = extract_total_num_kv_heads(
        #     text_config_dict, standard_fields
        # )
        # standard_fields["num_experts"] = extract_num_experts(text_config_dict)

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
