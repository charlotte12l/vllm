# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import json

from vllm.config.model import ModelConfig, _find_dtype
from vllm.config.speculative import SpeculativeConfig
from vllm.config import ParallelConfig
from functools import partial

import warnings
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import torch
import torch.nn.functional as F
from transformers import PretrainedConfig

from vllm.config.model import ModelConfig, ModelDType, RunnerOption
from vllm.logprobs import Logprob, PromptLogprobs, SampleLogprobs
from vllm.multimodal.processing import InputProcessingContext
from vllm.tokenizers import cached_tokenizer_from_config

def dummy_hf_overrides(
    hf_config: PretrainedConfig,
    *,
    model_arch: str = "",
    exist_overrides: dict[str, Any] | None = None,
    use_original_num_layers: bool = False,
) -> PretrainedConfig:
    """
    Dummy HF overrides function used to create dummy model
    with only minimum nums of layer.
    """
    hf_config.update(exist_overrides or {})

    text_config = hf_config.get_text_config()

    # Ensure at least 2 expert per group
    # Since `grouped_topk` assumes top-2
    n_group = getattr(text_config, "n_group", None)
    num_experts = n_group * 2 if n_group is not None else 2

    # we use three layers for Gemma-3n to check
    # both normal layer and kv_shared_layer
    if use_original_num_layers:
        # Use the original number of layers from the config
        num_layers = getattr(text_config, "num_layers", 1)
        num_hidden_layers = getattr(text_config, "num_hidden_layers", 1)
    else:
        # Use minimal layers for testing
        num_layers = 1
        num_hidden_layers = 3 if model_arch == "Gemma3nForConditionalGeneration" else 1

    update_dict = {
        "num_layers": num_layers,
        # For Gemma-3n
        "num_kv_shared_layers": 1,
    }

    class DummyConfig:
        hf_text_config = text_config

    # Only set MoE related config when the model has MoE layers.
    # Otherwise all models detected as MoE by _get_transformers_backend_cls.
    if ModelConfig.get_num_experts(DummyConfig) > 0:
        update_dict.update(
            {
                "num_experts": num_experts,
                "num_experts_per_tok": 2,
                "num_local_experts": num_experts,
                # Otherwise there will not be any expert layers
                "first_k_dense_replace": 0,
                # To avoid OOM on DeepSeek-V3
                "n_routed_experts": num_experts,
            }
        )

    # Update num_hidden_layers for non-Longcat architectures
    if model_arch != "LongcatFlashForCausalLM" and model_arch != "LongCatFlashMTPModel":
        update_dict["num_hidden_layers"] = num_hidden_layers

    text_config.update(update_dict)

    if hasattr(hf_config, "vision_config"):
        hf_config.vision_config.update(
            {
                "num_layers": 1,
                "num_hidden_layers": 1,
            }
        )

    # e.g.: ibm-granite/granite-speech-3.3-2b
    if hasattr(hf_config, "encoder_config"):
        hf_config.encoder_config.update(
            {
                "num_layers": 1,
                "num_hidden_layers": 1,
            }
        )

    # e.g.: Qwen/Qwen2-Audio-7B-Instruct
    if hasattr(hf_config, "audio_config"):
        hf_config.audio_config.update(
            {
                "num_layers": 1,
                "num_hidden_layers": 1,
                "encoder_layers": 1,
            }
        )

    return hf_config



def main():
    trust_remote_code_models = [
        # "nvidia/Llama-3_3-Nemotron-Super-49B-v1",
        # "XiaomiMiMo/MiMo-7B-RL",
        # Not available online right now
        # # "FreedomIntelligence/openPangu-Ultra-MoE-718B-V1.1",
        # "meituan-longcat/LongCat-Flash-Chat",
    ]
    models_to_test = [
        "state-spaces/mamba-130m-hf",
        "mistralai/Mamba-Codestral-7B-v0.1",
        "ibm-nasa-geospatial/Prithvi-EO-2.0-300M-TL-Sen1Floods11",
        "tiiuae/falcon-mamba-7b-instruct",
        # "Zyphra/Zamba2-7B-instruct",
        # "mosaicml/mpt-7b",
        # "databricks/dbrx-instruct",
        # "tiiuae/falcon-7b",
        # "tiiuae/falcon-40b",
        # "luccafong/deepseek_mtp_main_random",
        # "luccafong/deepseek_mtp_draft_random",
        # "Qwen/Qwen3-Next-80B-A3B-Instruct",
        # "tiny-random/qwen3-next-moe",
        # "zai-org/GLM-4.5",
        # "baidu/ERNIE-4.5-21B-A3B-PT",
        # # Select some models using base convertor for testing
        # "lmsys/gpt-oss-20b-bf16",
        # "deepseek-ai/DeepSeek-V3.2-Exp",
        # "meta-llama/Llama-4-Scout-17B-16E-Instruct",
    ] + trust_remote_code_models
    all_res = {}
    for model in models_to_test:
        print(f"testing {model=}")
        model_config = ModelConfig(
            model, trust_remote_code=model in trust_remote_code_models
        )
        res = {}
        hf_config = model_config.hf_config
        hf_text_config = model_config.hf_text_config
        res["architectures"] = model_config.architectures
        res["model_type"] = hf_config.model_type
        res["text_model_type"] = getattr(hf_text_config, "model_type", None)
        res["hidden_size"] = model_config.get_hidden_size()
        res["total_num_hidden_layers"] = model_config.get_total_num_hidden_layers()
        res["total_num_attention_heads"] = getattr(
            hf_text_config, "num_attention_heads", 0
        )
        res["head_size"] = model_config.get_head_size()
        res["vocab_size"] = model_config.get_vocab_size()
        res["total_num_kv_heads"] = model_config.get_total_num_kv_heads()
        res["num_experts"] = model_config.get_num_experts()

        res["is_deepseek_mla"] = model_config.is_deepseek_mla
        res["is_multimodal_model"] = model_config.is_multimodal_model
        dtype = _find_dtype(model, hf_config, revision=model_config.revision)
        res["dtype"] = str(dtype)
        all_res[model] = res

    with open("model_arch_groundtruth.json", "w") as f:
        json.dump(all_res, f, indent=4)

def main_speculative():
    MODELS = [
        ("JackFram/llama-68m", "abhigoyal/vllm-medusa-llama-68m-random", False),
        ("luccafong/deepseek_mtp_main_random", "luccafong/deepseek_mtp_draft_random", True),
        ("eagle618/deepseek-v3-random", "eagle618/eagle-deepseek-v3-random", True),
        ("meta-llama/Meta-Llama-3-8B-Instruct", "yuhuili/EAGLE-LLaMA3-Instruct-8B", True),
        ("meta-llama/Llama-3.1-8B-Instruct", "yuhuili/EAGLE3-LLaMA3.1-Instruct-8B", True),
    ]
   
    all_res = {}
    for target_model, draft_model, trust_remote_code in MODELS:
        print(f"testing {target_model=}, {draft_model=}")

        # if draft_model == "yuhuili/EAGLE3-LLaMA3.1-Instruct-8B":
        #     hf_overrides_fn = partial(
        #         dummy_hf_overrides,
        #         model_arch="Eagle3LlamaForCausalLM",
        #         use_original_num_layers=True,
        #     )

        #     target_model_config = ModelConfig(
        #         target_model, trust_remote_code=trust_remote_code, hf_overrides_fn=hf_overrides_fn
        #     )
        # else:
        target_model_config = ModelConfig(
            target_model, trust_remote_code=trust_remote_code
        )
        speculative_config = {
            "model": draft_model,
            "num_speculative_tokens": 1,
            "target_model_config": target_model_config,
            "target_parallel_config": ParallelConfig(),
        }

        speculative_config =  SpeculativeConfig(**speculative_config)
        model_config = speculative_config.draft_model_config

        res = {}
        hf_config = model_config.hf_config
        hf_text_config = model_config.hf_text_config
        # print("lxy", hf_config, hf_text_config)
        res["architectures"] = model_config.architectures
        res["model_type"] = hf_config.model_type
        res["text_model_type"] = getattr(hf_text_config, "model_type", None)
        res["hidden_size"] = model_config.get_hidden_size()
        res["total_num_hidden_layers"] = model_config.get_total_num_hidden_layers()
        res["total_num_attention_heads"] = getattr(
            hf_text_config, "num_attention_heads", 0
        )
        try: 
            res["head_size"] = model_config.get_head_size()
        except Exception as e:
            res["head_size"] = "Error: "+ str(e)
                
        res["vocab_size"] = model_config.get_vocab_size()
        res["total_num_kv_heads"] = model_config.get_total_num_kv_heads()
        res["num_experts"] = model_config.get_num_experts()

        res["is_deepseek_mla"] = model_config.is_deepseek_mla
        res["is_multimodal_model"] = model_config.is_multimodal_model
        dtype = _find_dtype(speculative_config.model, hf_config, revision=model_config.revision)
        res["dtype"] = str(dtype)
        all_res[draft_model] = res
    
    with open("draft_model_arch_groundtruth.json", "w") as f:
        json.dump(all_res, f, indent=4)

if __name__ == "__main__":
    # main()
    main_speculative()
