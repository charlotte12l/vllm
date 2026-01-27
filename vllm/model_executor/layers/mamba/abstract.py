# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from abc import abstractmethod
from collections.abc import Iterable
from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:
    from vllm.config import CacheConfig
    from vllm.config.kv_cache_spec_config import LayerKVCacheConfig

from vllm.config import VllmConfig
from vllm.model_executor.layers.attention_layer_base import AttentionLayerBase
from vllm.v1.attention.backend import AttentionBackend
from vllm.v1.attention.selector import get_mamba_attn_backend
from vllm.v1.kv_cache_interface import KVCacheSpec, MambaSpec


class MambaBase(AttentionLayerBase):
    """
    Base class for Mamba-like layers which support the v1 engine.
    Inherit from this class if you implement a custom layer.
    """

    # Contains the KV cache (mamba state) for the layer
    # in the shape specified by `self.get_state_shape`.
    kv_cache: tuple[torch.Tensor, ...]

    @abstractmethod
    def get_state_shape(self) -> Iterable[tuple[int, ...]]:
        """
        Defines the shape of the state.
        For mamba layers this is usually a (conv_state, ssm_state) tuple.
        In this case, returns (conv_state_shape, ssm_state_shape).
        """
        pass

    @property
    @abstractmethod
    def mamba_type(self) -> str:
        pass

    @abstractmethod
    def get_state_dtype(self) -> tuple[torch.dtype, ...]:
        pass

    def get_kv_cache_spec(self, vllm_config: VllmConfig) -> KVCacheSpec | None:
        if (
            vllm_config.speculative_config is not None
            and vllm_config.model_config.hf_config.model_type not in ["qwen3_next"]
        ):
            raise NotImplementedError(
                "Mamba with speculative decoding is not supported yet."
            )
        mamba_block_size = vllm_config.cache_config.mamba_block_size
        page_size_padded = vllm_config.cache_config.mamba_page_size_padded
        return MambaSpec(
            shapes=self.get_state_shape(),
            dtypes=self.get_state_dtype(),
            block_size=mamba_block_size,
            page_size_padded=page_size_padded,
            mamba_type=self.mamba_type,
            mamba_cache_mode=vllm_config.cache_config.mamba_cache_mode,
            num_speculative_blocks=(
                vllm_config.speculative_config.num_speculative_tokens
                if vllm_config.speculative_config
                else 0
            ),
        )

    def get_attn_backend(self) -> type[AttentionBackend]:
        """Get the attention backend class for this Mamba layer."""
        return get_mamba_attn_backend(self.mamba_type)

    @classmethod
    def create_kv_cache_spec_from_config(
        cls,
        layer_config: "LayerKVCacheConfig",
        cache_config: "CacheConfig",
    ) -> KVCacheSpec | None:
        """Create MambaSpec from layer config without model instantiation.

        Args:
            layer_config: Per-layer KV cache configuration with Mamba-specific params.
            cache_config: Global cache configuration.

        Returns:
            MambaSpec for this Mamba/SSM layer.
        """

        assert layer_config.mamba_type is not None
        assert layer_config.ssm_state_size is not None
        assert layer_config.conv_kernel_size is not None
        assert layer_config.dtype is not None

        mamba_block_size = cache_config.mamba_block_size
        page_size_padded = cache_config.mamba_page_size_padded

        # Compute state shapes based on mamba_type
        if layer_config.mamba_type == "mamba2":
            # Mamba2 uses different state representation
            assert layer_config.mamba_num_heads is not None
            assert layer_config.mamba_head_dim is not None
            assert layer_config.n_groups is not None

            # conv_state shape:
            # (1, intermediate_size + 2 * n_groups * ssm_state_size, conv_kernel)
            # ssm_state: (1, num_heads, head_dim, ssm_state_size)
            intermediate = layer_config.intermediate_size or (
                layer_config.mamba_num_heads * layer_config.mamba_head_dim
            )
            conv_state_shape = (
                1,
                intermediate + 2 * layer_config.n_groups * layer_config.ssm_state_size,
                layer_config.conv_kernel_size,
            )
            ssm_state_shape: tuple[int, ...] = (
                1,
                layer_config.mamba_num_heads,
                layer_config.mamba_head_dim,
                layer_config.ssm_state_size,
            )
        else:
            # Mamba1 default shapes
            # conv_state: (1, intermediate_size, conv_kernel_size)
            # ssm_state: (1, intermediate_size, ssm_state_size)
            intermediate = layer_config.intermediate_size or 0
            conv_state_shape = (
                1,
                intermediate,
                layer_config.conv_kernel_size,
            )
            ssm_state_shape = (
                1,
                intermediate,
                layer_config.ssm_state_size,
            )

        shapes: tuple[tuple[int, ...], ...] = (conv_state_shape, ssm_state_shape)
        dtypes = (layer_config.dtype, layer_config.dtype)

        return MambaSpec(
            shapes=shapes,
            dtypes=dtypes,
            block_size=mamba_block_size,
            page_size_padded=page_size_padded,
            mamba_type=layer_config.mamba_type,
            mamba_cache_mode=cache_config.mamba_cache_mode,
            num_speculative_blocks=0,  # Not supported from config-only path
        )
