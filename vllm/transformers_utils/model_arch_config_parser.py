# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import os
from abc import ABC, abstractmethod
from copy import deepcopy
from pathlib import Path
from typing import TYPE_CHECKING, Any, List

import huggingface_hub
import torch
from safetensors.torch import _TYPES as _SAFETENSORS_TO_TORCH_DTYPE
from torch import nn
from transformers import AutoConfig, PretrainedConfig
from transformers.models.auto.modeling_auto import (
    MODEL_FOR_CAUSAL_LM_MAPPING_NAMES,
    MODEL_MAPPING_NAMES,
)
from vllm import envs
from vllm.attention import Attention
from vllm.attention.layers.encoder_only_attention import EncoderOnlyAttention
from vllm.config.utils import getattr_iter
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
    try_get_safetensors_metadata,
)
from vllm.utils.import_utils import LazyLoader
from vllm.utils.torch_utils import common_broadcastable_dtype

logger = init_logger(__name__)

MULTIMODAL_MODEL_ARCHS = [
    "AriaForConditionalGeneration",
    "AyaVisionForConditionalGeneration",
    "BeeForConditionalGeneration",
    "Blip2ForConditionalGeneration",
    "ChameleonForConditionalGeneration",
    "CLIPEmbeddingModel",
    "Cohere2VisionForConditionalGeneration",
    "DeepseekOCRForCausalLM",
    "DeepseekVLV2ForCausalLM",
    "DotsOCRForCausalLM",
    "Ernie4_5_VLMoeForConditionalGeneration",
    "FuyuForCausalLM",
    "Gemma3ForConditionalGeneration",
    "Gemma3nForConditionalGeneration",
    "GLM4VForCausalLM",
    "Glm4vForConditionalGeneration",
    "Glm4vMoeForConditionalGeneration",
    "GraniteSpeechForConditionalGeneration",
    "H2OVLChatModel",
    "HCXVisionForCausalLM",
    "Idefics3ForConditionalGeneration",
    "InternS1ForConditionalGeneration",
    "InternVLChatModel",
    "KeyeForConditionalGeneration",
    "KeyeVL1_5ForConditionalGeneration",
    "KimiVLForConditionalGeneration",
    "LightOnOCRForConditionalGeneration",
    "Llama4ForConditionalGeneration",
    "LlamaNemotronVLChatModel",
    "LlavaForConditionalGeneration",
    "LlavaNextForConditionalGeneration",
    "LlavaNextVideoForConditionalGeneration",
    "LlavaOnevisionForConditionalGeneration",
    "MantisForConditionalGeneration",
    "MiDashengLMModel",
    "MiniCPMO",
    "MiniCPMV",
    "MiniCPMVBaseModel",
    "MiniMaxVL01ForConditionalGeneration",
    "Mistral3ForConditionalGeneration",
    "MolmoForCausalLM",
    "MultiModalMixin",
    "NemotronH_Nano_VL_V2",
    "NVLM_D_Model",
    "Ovis",
    "Ovis2_5",
    "PaddleOCRVLForConditionalGeneration",
    "PaliGemmaForConditionalGeneration",
    "Phi3VForCausalLM",
    "Phi4MMForCausalLM",
    "Phi4MultimodalForCausalLM",
    "PixtralForConditionalGeneration",
    "Qwen2_5_VLForConditionalGeneration",
    "Qwen2_5OmniThinkerForConditionalGeneration",
    "Qwen2AudioForConditionalGeneration",
    "Qwen2VLForConditionalGeneration",
    "Qwen3OmniMoeThinkerForConditionalGeneration",
    "Qwen3VLForConditionalGeneration",
    "Qwen3VLMoeForConditionalGeneration",
    "QwenVLForConditionalGeneration",
    "RForConditionalGeneration",
    "SiglipEmbeddingModel",
    "SkyworkR1VChatModel",
    "SmolVLMForConditionalGeneration",
    "Step3VLForConditionalGeneration",
    "Tarsier2ForConditionalGeneration",
    "TarsierForConditionalGeneration",
    "Terratorch",
    "TransformersMultiModalForCausalLM",
    "TransformersMultiModalMoEForCausalLM",
    "TransformersMultiModalEmbeddingModel",
    "TransformersMultiModalForSequenceClassification",
    "UltravoxModel",
    "VoxtralForConditionalGeneration",
    "WhisperForConditionalGeneration",
]


class ModelArchConfigConvertorBase(ABC):
    @classmethod
    def get_num_hidden_layers(self, config: PretrainedConfig) -> int:
        return getattr(
            config, "num_hidden_layers", 0
        )

    @classmethod
    def get_total_num_attention_heads(self, hf_text_config: PretrainedConfig) -> int:
        return getattr(hf_text_config, "num_attention_heads", 0)
    
    @classmethod
    def get_vocab_size(self, config: PretrainedConfig) -> int:
        return getattr(config, "vocab_size", 0)
    
    @classmethod
    def get_hidden_size(self, config: PretrainedConfig) -> int:
        return getattr(config, "hidden_size", 0)

    @classmethod
    def get_head_size(cls, hf_text_config: PretrainedConfig) -> int:
        if self.is_deepseek_mla(hf_text_config):
            qk_rope_head_dim = getattr(hf_text_config, "qk_rope_head_dim", 0)
            if envs.VLLM_MLA_DISABLE:
                return hf_text_config.kv_lora_rank + qk_rope_head_dim
            else:
                qk_nope_head_dim = getattr(self.hf_text_config, "qk_nope_head_dim", 0)
                if qk_rope_head_dim and qk_nope_head_dim:
                return qk_rope_head_dim + qk_nope_head_dim
        
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
            num_kv_heads = getattr(config, attr, None)
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
        quant_cfg = getattr(config, "quantization_config", None)
        if quant_cfg is None:
            # compressed-tensors uses a "compression_config" key
            quant_cfg = getattr(config, "compression_config", None)

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
    def get_quantization_config(
        self, hf_config: PretrainedConfig
    ):
        quant_cfg = self.normalize_quantization_config(hf_config)
        if quant_cfg is None and (
            text_config := getattr(hf_config, "text_config", None)
        ):
            # Check the text config as well for multi-modal models.
            quant_cfg = self.normalize_quantization_config(text_config)
        return quant_cfg

    @classmethod
    def is_deepseek_mla(self, hf_text_config: PretrainedConfig) -> bool:
        if not hasattr(hf_text_config, "model_type"):
            return False
        elif hf_text_config.model_type in (
            "deepseek_v2",
            "deepseek_v3",
            "deepseek_v32",
            "deepseek_mtp",
            "kimi_k2",
            "kimi_linear",
            "longcat_flash",
            "pangu_ultra_moe",
            "pangu_ultra_moe_mtp",
        ):
            return hf_text_config.kv_lora_rank is not None
        elif hf_text_config.model_type == "eagle":
            # if the model is an EAGLE module, check for the
            # underlying architecture
            return (
                hf_text_config.model.model_type
                in ("deepseek_v2", "deepseek_v3", "deepseek_v32")
                and hf_text_config.kv_lora_rank is not None
            )
        return False

    @classmethod
    def derive_max_model_len_and_key(self, hf_config: PretrainedConfig) -> tuple[int, str]:
        derived_max_model_len = float("inf")
        possible_keys = [
            # OPT
            "max_position_embeddings",
            # GPT-2
            "n_positions",
            # MPT
            "max_seq_len",
            # ChatGLM2
            "seq_length",
            # Command-R
            "model_max_length",
            # Whisper
            "max_target_positions",
            # Others
            "max_sequence_length",
            "max_seq_length",
            "seq_len",
        ]
        # Choose the smallest "max_length" from the possible keys
        max_len_key = None
        for key in possible_keys:
            max_len = getattr(hf_config, key, None)
            if max_len is not None:
                max_len_key = key if max_len < derived_max_model_len else max_len_key
                derived_max_model_len = min(derived_max_model_len, max_len)

        return derived_max_model_len, max_len_key

    @classmethod
    def get_layer_types_cls(
        self, config: PretrainedConfig,
    ) -> list[type[nn.Module]]:
        """Get per-layer attention class types."""
        return [Attention for _ in range(self.get_num_hidden_layers(config))]
    
    def get_layer_types(self, config: PretrainedConfig) -> list[str] | None:
        """Get per-layer types (e.g., ['full_attention', 'sliding_attention'])."""
        return getattr(config, "layer_types", None)

    def get_attn_type(self, config: PretrainedConfig) -> str:
        return AttentionType.DECODER

    @classmethod
    def support_multimodal(self, architectures: List[str]) -> bool:
        if any(
            multi_model_arch in architectures
            for multi_model_arch in MULTIMODAL_MODEL_ARCHS):
            return True
        else:
            return False

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
            architectures = hf_config.architectures,
            model_type = hf_config.model_type,
            text_model_type = text_config.model_type,
            hidden_size = self.get_hidden_size(hf_config),
            num_hidden_layers=self.get_num_hidden_layers(text_config),
            num_attention_heads=self.get_num_attention_heads(text_config),
            head_size = self.get_head_size(text_config),
            vocab_size = self.get_vocab_size(text_config),
            total_num_kv_heads = self.get_total_num_kv_heads(text_config),
            num_experts = self.get_num_experts(text_config),
            quantization_config= self.normalize_quantization_config(text_config),
            torch_dtype = self.get_torch_dtype(hf_config, model_id, revision),
            support_multimodal = self.support_multimodal(hf_config.architectures),
            is_deepseek_mla = self.is_deepseek_mla(text_config),
            derived_max_model_len_and_key = self.derive_max_model_len_and_key(hf_config),
        )

        return model_arch_config


class LlamaModelArchConfigConvertor(ModelArchConfigConvertorBase):
    def get_layer_types_cls(
        self, config: PretrainedConfig,
    ) -> list[type[nn.Module]]:
        if getattr(config, "is_causal", True):
            attn_cls = Attention
        else:
            attn_cls = EncoderOnlyAttention

        return [attn_cls] * self.get_num_hidden_layers(config)



class Zamba2ModelArchConfigConvertor(ModelArchConfigConvertorBase):
    """Convertor for Zamba2 which uses attention_head_dim instead of head_dim."""
    
    @classmethod
    def get_head_size(self, config: PretrainedConfig) -> int:
        return getattr(config, "attention_head_dim", 0)


class MPTModelArchConfigConvertor(ModelArchConfigConvertorBase):
    """Convertor for MPT which has attn_config with kv_n_heads."""
    
    @classmethod
    def get_total_num_kv_heads(self, hf_config: PretrainedConfig) -> int:
        if "kv_n_heads" in hf_config.attn_config:
            return hf_config.attn_config["kv_n_heads"]
        return hf_config.num_attention_heads


class DbrxModelArchConfigConvertor(ModelArchConfigConvertorBase):
    """Convertor for Dbrx which has attn_config with kv_n_heads."""
    
    @classmethod
    def get_total_num_kv_heads(self, hf_config: PretrainedConfig) -> int:
        return getattr(
            hf_config.attn_config,
            "kv_n_heads",
            hf_config.num_attention_heads,
        )


class FalconModelArchConfigConvertor(ModelArchConfigConvertorBase):
    """Convertor for Falcon which uses multi_query and new_decoder_architecture."""
    
    @classmethod
    def get_total_num_kv_heads(self, hf_config: PretrainedConfig) -> int:
        # NOTE: for falcon, when new_decoder_architecture is True, the
        # multi_query flag is ignored and we use n_head_kv for the number of
        # KV heads.
        new_decoder_arch_falcon = getattr(hf_config, "new_decoder_architecture", False)
        
        if not new_decoder_arch_falcon and getattr(hf_text_config, "multi_query", False):
            # Multi-query attention, only one KV head.
            return 1
        
        # Use the base implementation which checks n_head_kv, num_kv_heads, etc.
        return super().get_total_num_kv_heads(hf_text_config)


class NemotronNasModelArchConfigConvertor(ModelArchConfigConvertorBase):
    """Convertor for Nemotron-NAS which has block_configs."""
    
    @classmethod
    def get_total_num_kv_heads(self, hf_config: PretrainedConfig) -> int:
        for block in hf_config.block_configs:
            if not block.attention.no_op:
                return (
                    hf_config.num_attention_heads
                    // block.attention.n_heads_in_group
                )
        raise RuntimeError("Couldn't determine number of kv heads")

class DeepSeekMTPModelArchConfigConvertor(ModelArchConfigConvertorBase):
    @classmethod
    def get_num_hidden_layers(
        self, hf_text_config: PretrainedConfig,
    ) -> int:
        return getattr(
            hf_text_config, "num_nextn_predict_layers", 0
        ) 
    

class Qwen3NextMTPModelArchConfigConvertor(ModelArchConfigConvertorBase):
    @classmethod
    def get_num_hidden_layers(
        self, hf_text_config: PretrainedConfig,
    ) -> int:
        return getattr(
            hf_text_config, "num_nextn_predict_layers", 0
        )


class MimoMTPModelArchConfigConvertor(ModelArchConfigConvertorBase):
    """Convertor for MIMO MTP."""
    
    @classmethod
    def get_num_hidden_layers(self, hf_text_config: PretrainedConfig) -> int:
        return getattr(hf_text_config, "num_nextn_predict_layers", 0)


class GLM4MoeMTPModelArchConfigConvertor(ModelArchConfigConvertorBase):
    """Convertor for GLM4 MoE MTP."""
    
    @classmethod
    def get_num_hidden_layers(self, hf_text_config: PretrainedConfig) -> int:
        return getattr(hf_text_config, "num_nextn_predict_layers", 0)


class ErnieMTPModelArchConfigConvertor(ModelArchConfigConvertorBase):
    """Convertor for Ernie MTP."""
    
    @classmethod
    def get_num_hidden_layers(self, hf_text_config: PretrainedConfig) -> int:
        return getattr(hf_text_config, "num_nextn_predict_layers", 0)


class PanguUltraMoeMTPModelArchConfigConvertor(ModelArchConfigConvertorBase):
    """Convertor for Pangu Ultra MoE MTP."""
    
    @classmethod
    def get_num_hidden_layers(self, hf_text_config: PretrainedConfig) -> int:
        return getattr(hf_text_config, "num_nextn_predict_layers", 0)


class LongCatFlashMTPModelArchConfigConvertor(ModelArchConfigConvertorBase):
    """Convertor for LongCat Flash MTP which defaults to 1 layer."""
    
    @classmethod
    def get_num_hidden_layers(self, hf_text_config: PretrainedConfig) -> int:
        return getattr(hf_text_config, "num_nextn_predict_layers", 1)


class CohereModelArchConfigConvertor(ModelArchConfigConvertorBase):
    """Convertor for LongCat Flash MTP which defaults to 1 layer."""
    
    @classmethod
    def derive_max_model_len_and_key(self, hf_config: PretrainedConfig) -> Tuple[int, str]:
        derived_max_model_len, max_len_key = super().derive_max_model_len_and_key(hf_config)
        if tmp_max_len := getattr(hf_config, "model_max_length", None):
            max_len_key = "model_max_length"
            derived_max_model_len = tmp_max_len
        return derived_max_model_len, max_len_key


# TODO: Support registry
# hf_config.model_type -> convertor class
# deepseek_mtp uses hf_text_config ....
MODEL_ARCH_CONFIG_CONVERTORS = {
    "llama": LlamaModelArchConfigConvertor,
    "gpt_oss": GPTOssModelArchConfigConvertor,
    "zamba2": Zamba2ModelArchConfigConvertor,
    "mpt": MPTModelArchConfigConvertor,
    "dbrx": DbrxModelArchConfigConvertor,
    "falcon": FalconModelArchConfigConvertor,
    "RefinedWeb": FalconModelArchConfigConvertor,
    "RefinedWebModel": FalconModelArchConfigConvertor,
    "nemotron-nas": NemotronNasModelArchConfigConvertor,
    "deepseek_mtp": DeepSeekMTPModelArchConfigConvertor,
    "qwen3_next_mtp": Qwen3NextMTPModelArchConfigConvertor,
    "mimo_mtp": MimoMTPModelArchConfigConvertor,
    "glm4_moe_mtp": GLM4MoeMTPModelArchConfigConvertor,
    "ernie_mtp": ErnieMTPModelArchConfigConvertor,
    "pangu_ultra_moe_mtp": PanguUltraMoeMTPModelArchConfigConvertor,
    "longcat_flash_mtp": LongCatFlashMTPModelArchConfigConvertor,
    "commandr": CohereModelArchConfigConvertor,
    "aya_vision": AyaVisionModelArchConfigConvertor,
}

# For those not in MODEL_ARCH_CONFIG_CONVERTORS, we use the base convertor
SUPPORTED_MODEL_TYPES = list(MODEL_ARCH_CONFIG_CONVERTORS.keys())