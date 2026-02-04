# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from abc import abstractmethod
from collections.abc import Iterable

import torch

from vllm.config import CacheConfig, ParallelConfig
from vllm.config.model_arch import KVCacheModelConfig
from vllm.model_executor.layers.attention_layer_base import AttentionLayerBase
from vllm.model_executor.layers.mamba.mamba_utils import MambaStateShapeCalculator
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

    def get_attn_backend(self) -> type[AttentionBackend]:
        """Get the attention backend class for this Mamba layer."""
        return get_mamba_attn_backend(self.mamba_type)

    @classmethod
    def get_kv_cache_spec(
        cls,
        kv_cache_config: KVCacheModelConfig,
        cache_config: CacheConfig,
        parallel_config: ParallelConfig,
        model_dtype: torch.dtype,
        layer_type: str,
    ) -> KVCacheSpec:
        tp_size = parallel_config.tensor_parallel_size

        # Determine mamba type and compute shapes
        shapes: tuple[tuple[int, ...], ...]
        dtypes: tuple[torch.dtype, ...]
        if kv_cache_config.mamba_num_heads is not None:
            # Mamba2
            mamba_type = "mamba2"
            assert kv_cache_config.mamba_intermediate_size is not None
            assert kv_cache_config.mamba_head_dim is not None
            assert kv_cache_config.mamba_state_size is not None
            assert kv_cache_config.mamba_conv_kernel is not None
            shapes = MambaStateShapeCalculator.mamba2_state_shape(
                tp_world_size=tp_size,
                intermediate_size=kv_cache_config.mamba_intermediate_size,
                n_groups=kv_cache_config.mamba_num_groups or 1,
                num_heads=kv_cache_config.mamba_num_heads,
                head_dim=kv_cache_config.mamba_head_dim,
                state_size=kv_cache_config.mamba_state_size,
                conv_kernel=kv_cache_config.mamba_conv_kernel,
            )
            dtypes = (torch.float32,)
        else:
            # Mamba1
            mamba_type = "mamba1"
            assert kv_cache_config.mamba_intermediate_size is not None
            assert kv_cache_config.mamba_state_size is not None
            assert kv_cache_config.mamba_conv_kernel is not None
            shapes = MambaStateShapeCalculator.mamba1_state_shape(
                tp_world_size=tp_size,
                intermediate_size=kv_cache_config.mamba_intermediate_size,
                state_size=kv_cache_config.mamba_state_size,
                conv_kernel=kv_cache_config.mamba_conv_kernel,
            )
            dtypes = (torch.float32, torch.float32)

        # Get mamba block size with default
        mamba_block_size = cache_config.mamba_block_size
        if mamba_block_size is None:
            mamba_block_size = 1  # Default block size

        return MambaSpec(
            block_size=mamba_block_size,
            shapes=shapes,
            dtypes=dtypes,  # type: ignore[arg-type]
            mamba_type=mamba_type,
            mamba_cache_mode=cache_config.mamba_cache_mode,
            page_size_padded=cache_config.mamba_page_size_padded,
        )
