# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
Tests for config-only KV cache spec computation.

This module tests the ability to compute KVCacheSpec for all layers
without instantiating the model.
"""

import dataclasses

import pytest
import torch

from vllm.config import CacheConfig
from vllm.config.kv_cache_spec_config import (
    LayerAttentionType,
    LayerBlockType,
    LayerKVCacheConfig,
    LayerRole,
    ModelKVCacheRequirements,
)
from vllm.v1.kv_cache_interface import (
    ChunkedLocalAttentionSpec,
    CrossAttentionSpec,
    FullAttentionSpec,
    MambaSpec,
    MLAAttentionSpec,
    SlidingWindowSpec,
)


class TestLayerKVCacheConfig:
    """Tests for LayerKVCacheConfig dataclass."""

    def test_full_attention_config(self):
        """Test creating config for full attention layer."""
        config = LayerKVCacheConfig(
            layer_idx=0,
            attention_type=LayerAttentionType.FULL,
            block_type=LayerBlockType.ATTENTION,
            role=LayerRole.DECODER,
            num_kv_heads=8,
            head_size=64,
            dtype=torch.float16,
        )

        assert config.layer_idx == 0
        assert config.attention_type == LayerAttentionType.FULL
        assert config.block_type == LayerBlockType.ATTENTION
        assert config.num_kv_heads == 8
        assert config.head_size == 64

    def test_sliding_window_config(self):
        """Test creating config for sliding window attention layer."""
        config = LayerKVCacheConfig(
            layer_idx=0,
            attention_type=LayerAttentionType.SLIDING_WINDOW,
            num_kv_heads=8,
            head_size=64,
            dtype=torch.float16,
            sliding_window=4096,
        )

        assert config.attention_type == LayerAttentionType.SLIDING_WINDOW
        assert config.sliding_window == 4096

    def test_mamba_config(self):
        """Test creating config for Mamba layer."""
        config = LayerKVCacheConfig(
            layer_idx=0,
            block_type=LayerBlockType.MAMBA,
            dtype=torch.float16,
            mamba_type="mamba2",
            ssm_state_size=16,
            conv_kernel_size=4,
            intermediate_size=8192,
            n_groups=8,
            mamba_num_heads=64,
            mamba_head_dim=128,
        )

        assert config.block_type == LayerBlockType.MAMBA
        assert config.mamba_type == "mamba2"
        assert config.ssm_state_size == 16

    def test_frozen_config(self):
        """Test that LayerKVCacheConfig is frozen."""
        config = LayerKVCacheConfig(
            layer_idx=0,
            dtype=torch.float16,
        )

        with pytest.raises(dataclasses.FrozenInstanceError):
            config.layer_idx = 1


class TestModelKVCacheRequirements:
    """Tests for ModelKVCacheRequirements dataclass."""

    def test_layer_count(self):
        """Test layer count property."""
        layers = [
            LayerKVCacheConfig(
                layer_idx=i,
                dtype=torch.float16,
            )
            for i in range(32)
        ]

        requirements = ModelKVCacheRequirements(
            layers=layers,
            default_num_kv_heads=8,
            default_head_size=64,
            default_dtype=torch.float16,
        )

        assert requirements.layer_count == 32

    def test_attention_layer_count(self):
        """Test attention layer count for hybrid models."""
        layers = [
            LayerKVCacheConfig(
                layer_idx=i,
                block_type=(
                    LayerBlockType.MAMBA if i % 2 == 0 else LayerBlockType.ATTENTION
                ),
                dtype=torch.float16,
            )
            for i in range(8)
        ]

        requirements = ModelKVCacheRequirements(
            layers=layers,
            default_num_kv_heads=8,
            default_head_size=64,
            default_dtype=torch.float16,
            is_hybrid=True,
        )

        assert requirements.attention_layer_count == 4
        assert requirements.mamba_layer_count == 4


class TestAttentionCreateKVCacheSpecFromConfig:
    """Tests for Attention.create_kv_cache_spec_from_config."""

    def get_cache_config(self, **kwargs) -> CacheConfig:
        """Create a CacheConfig with test defaults."""
        defaults = {
            "block_size": 16,
            "cache_dtype": "auto",
            "enable_hybrid_allocator": True,
        }
        defaults.update(kwargs)
        return CacheConfig(**defaults)

    def test_full_attention_spec(self):
        """Test creating FullAttentionSpec from config."""
        from vllm.attention.layer import Attention

        layer_config = LayerKVCacheConfig(
            layer_idx=0,
            attention_type=LayerAttentionType.FULL,
            num_kv_heads=8,
            head_size=64,
            dtype=torch.float16,
        )
        cache_config = self.get_cache_config()

        spec = Attention.create_kv_cache_spec_from_config(layer_config, cache_config)

        assert isinstance(spec, FullAttentionSpec)
        assert spec.num_kv_heads == 8
        assert spec.head_size == 64
        assert spec.block_size == 16
        assert spec.dtype == torch.float16

    def test_sliding_window_spec_with_hybrid_allocator(self):
        """Test creating SlidingWindowSpec when hybrid allocator is enabled."""
        from vllm.attention.layer import Attention

        layer_config = LayerKVCacheConfig(
            layer_idx=0,
            attention_type=LayerAttentionType.SLIDING_WINDOW,
            num_kv_heads=8,
            head_size=64,
            dtype=torch.float16,
            sliding_window=4096,
        )
        cache_config = self.get_cache_config(enable_hybrid_allocator=True)

        spec = Attention.create_kv_cache_spec_from_config(layer_config, cache_config)

        assert isinstance(spec, SlidingWindowSpec)
        assert spec.sliding_window == 4096

    def test_sliding_window_spec_without_hybrid_allocator(self):
        """Test creating FullAttentionSpec with sliding_window metadata."""
        from vllm.attention.layer import Attention

        layer_config = LayerKVCacheConfig(
            layer_idx=0,
            attention_type=LayerAttentionType.SLIDING_WINDOW,
            num_kv_heads=8,
            head_size=64,
            dtype=torch.float16,
            sliding_window=4096,
        )
        cache_config = self.get_cache_config(enable_hybrid_allocator=False)

        spec = Attention.create_kv_cache_spec_from_config(layer_config, cache_config)

        assert isinstance(spec, FullAttentionSpec)
        # sliding_window should be stored as metadata
        assert spec.sliding_window == 4096


class TestMLAAttentionCreateKVCacheSpecFromConfig:
    """Tests for MLAAttention.create_kv_cache_spec_from_config."""

    def get_cache_config(self) -> CacheConfig:
        return CacheConfig(block_size=16, cache_dtype="auto")

    def test_mla_attention_spec(self):
        """Test creating MLAAttentionSpec from config."""
        from vllm.attention.layer import MLAAttention

        layer_config = LayerKVCacheConfig(
            layer_idx=0,
            head_size=576,  # kv_lora_rank (512) + qk_rope_head_dim (64)
            dtype=torch.float16,
            kv_lora_rank=512,
            qk_rope_head_dim=64,
            cache_dtype_str="auto",
        )
        cache_config = self.get_cache_config()

        spec = MLAAttention.create_kv_cache_spec_from_config(layer_config, cache_config)

        assert isinstance(spec, MLAAttentionSpec)
        assert spec.num_kv_heads == 1  # MLA uses single latent
        assert spec.head_size == 576


class TestChunkedLocalAttentionCreateKVCacheSpecFromConfig:
    """Tests for ChunkedLocalAttention.create_kv_cache_spec_from_config."""

    def get_cache_config(self, **kwargs) -> CacheConfig:
        defaults = {
            "block_size": 16,
            "cache_dtype": "auto",
            "enable_hybrid_allocator": True,
        }
        defaults.update(kwargs)
        return CacheConfig(**defaults)

    def test_chunked_local_attention_spec(self):
        """Test creating ChunkedLocalAttentionSpec from config."""
        from vllm.model_executor.layers.attention.chunked_local_attention import (
            ChunkedLocalAttention,
        )

        layer_config = LayerKVCacheConfig(
            layer_idx=0,
            attention_type=LayerAttentionType.CHUNKED_LOCAL,
            num_kv_heads=8,
            head_size=64,
            dtype=torch.float16,
            attention_chunk_size=8192,
        )
        cache_config = self.get_cache_config()

        spec = ChunkedLocalAttention.create_kv_cache_spec_from_config(
            layer_config, cache_config
        )

        assert isinstance(spec, ChunkedLocalAttentionSpec)
        assert spec.attention_chunk_size == 8192


class TestCrossAttentionCreateKVCacheSpecFromConfig:
    """Tests for CrossAttention.create_kv_cache_spec_from_config."""

    def get_cache_config(self) -> CacheConfig:
        return CacheConfig(block_size=16, cache_dtype="auto")

    def test_cross_attention_spec(self):
        """Test creating CrossAttentionSpec from config."""
        from vllm.model_executor.layers.attention.cross_attention import (
            CrossAttention,
        )

        layer_config = LayerKVCacheConfig(
            layer_idx=0,
            role=LayerRole.CROSS_ATTENTION,
            num_kv_heads=8,
            head_size=64,
            dtype=torch.float16,
        )
        cache_config = self.get_cache_config()

        spec = CrossAttention.create_kv_cache_spec_from_config(
            layer_config, cache_config
        )

        assert isinstance(spec, CrossAttentionSpec)
        assert spec.num_kv_heads == 8


class TestEncoderOnlyAttentionCreateKVCacheSpecFromConfig:
    """Tests for EncoderOnlyAttention.create_kv_cache_spec_from_config."""

    def get_cache_config(self) -> CacheConfig:
        return CacheConfig(block_size=16, cache_dtype="auto")

    def test_encoder_only_returns_none(self):
        """Test that encoder-only attention returns None."""
        from vllm.model_executor.layers.attention.encoder_only_attention import (
            EncoderOnlyAttention,
        )

        layer_config = LayerKVCacheConfig(
            layer_idx=0,
            role=LayerRole.ENCODER_ONLY,
            num_kv_heads=8,
            head_size=64,
            dtype=torch.float16,
        )
        cache_config = self.get_cache_config()

        spec = EncoderOnlyAttention.create_kv_cache_spec_from_config(
            layer_config, cache_config
        )

        assert spec is None


class TestMambaBaseCreateKVCacheSpecFromConfig:
    """Tests for MambaBase.create_kv_cache_spec_from_config."""

    def get_cache_config(self) -> CacheConfig:
        return CacheConfig(
            block_size=16,
            cache_dtype="auto",
            mamba_block_size=64,
            mamba_page_size_padded=256,
            mamba_cache_mode="full",
        )

    def test_mamba2_spec(self):
        """Test creating MambaSpec for Mamba2 from config."""
        from vllm.model_executor.layers.mamba.abstract import MambaBase

        layer_config = LayerKVCacheConfig(
            layer_idx=0,
            block_type=LayerBlockType.MAMBA,
            dtype=torch.float16,
            mamba_type="mamba2",
            ssm_state_size=16,
            conv_kernel_size=4,
            intermediate_size=8192,
            n_groups=8,
            mamba_num_heads=64,
            mamba_head_dim=128,
        )
        cache_config = self.get_cache_config()

        spec = MambaBase.create_kv_cache_spec_from_config(layer_config, cache_config)

        assert isinstance(spec, MambaSpec)
        assert spec.mamba_type == "mamba2"

    def test_mamba1_spec(self):
        """Test creating MambaSpec for Mamba1 from config."""
        from vllm.model_executor.layers.mamba.abstract import MambaBase

        layer_config = LayerKVCacheConfig(
            layer_idx=0,
            block_type=LayerBlockType.MAMBA,
            dtype=torch.float16,
            mamba_type="mamba1",
            ssm_state_size=16,
            conv_kernel_size=4,
            intermediate_size=4096,
        )
        cache_config = self.get_cache_config()

        spec = MambaBase.create_kv_cache_spec_from_config(layer_config, cache_config)

        assert isinstance(spec, MambaSpec)
        assert spec.mamba_type == "mamba1"


class TestParseLayerTypes:
    """Tests for parse_layer_types function."""

    def test_parse_full_attention(self):
        """Test parsing full_attention layer type."""
        from vllm.v1.worker.kv_cache_spec_utils import parse_layer_types

        class MockConfig:
            layer_types = ["full_attention"] * 4

        result = parse_layer_types(MockConfig(), 4)

        assert result == [LayerAttentionType.FULL] * 4

    def test_parse_mixed_layer_types(self):
        """Test parsing mixed layer types."""
        from vllm.v1.worker.kv_cache_spec_utils import parse_layer_types

        class MockConfig:
            layer_types = [
                "sliding_attention",
                "full_attention",
                "sliding_attention",
                "full_attention",
            ]

        result = parse_layer_types(MockConfig(), 4)

        assert result == [
            LayerAttentionType.SLIDING_WINDOW,
            LayerAttentionType.FULL,
            LayerAttentionType.SLIDING_WINDOW,
            LayerAttentionType.FULL,
        ]

    def test_parse_no_layer_types(self):
        """Test default when layer_types is not present."""
        from vllm.v1.worker.kv_cache_spec_utils import parse_layer_types

        class MockConfig:
            pass

        result = parse_layer_types(MockConfig(), 4)

        assert result == [LayerAttentionType.FULL] * 4


class TestParseLayersBlockType:
    """Tests for parse_layers_block_type function."""

    def test_parse_attention_blocks(self):
        """Test parsing attention block types."""
        from vllm.v1.worker.kv_cache_spec_utils import parse_layers_block_type

        class MockConfig:
            layers_block_type = ["attention"] * 4

        result = parse_layers_block_type(MockConfig(), 4)

        assert result == [LayerBlockType.ATTENTION] * 4

    def test_parse_hybrid_model(self):
        """Test parsing hybrid model block types."""
        from vllm.v1.worker.kv_cache_spec_utils import parse_layers_block_type

        class MockConfig:
            layers_block_type = ["mamba", "attention", "mamba", "attention"]

        result = parse_layers_block_type(MockConfig(), 4)

        assert result == [
            LayerBlockType.MAMBA,
            LayerBlockType.ATTENTION,
            LayerBlockType.MAMBA,
            LayerBlockType.ATTENTION,
        ]

    def test_parse_no_block_types(self):
        """Test default when layers_block_type is not present."""
        from vllm.v1.worker.kv_cache_spec_utils import parse_layers_block_type

        class MockConfig:
            pass

        result = parse_layers_block_type(MockConfig(), 4)

        assert result == [LayerBlockType.ATTENTION] * 4


class TestParseKVSharingPattern:
    """Tests for parse_kv_sharing_pattern function."""

    def test_parse_kv_sharing(self):
        """Test parsing KV sharing pattern."""
        from vllm.v1.worker.kv_cache_spec_utils import parse_kv_sharing_pattern

        class MockConfig:
            kv_sharing_pattern = {28: 27, 29: 27, 30: 27, 31: 27}

        result = parse_kv_sharing_pattern(MockConfig(), 32)

        assert result == {28: 27, 29: 27, 30: 27, 31: 27}

    def test_parse_no_sharing(self):
        """Test when no KV sharing pattern exists."""
        from vllm.v1.worker.kv_cache_spec_utils import parse_kv_sharing_pattern

        class MockConfig:
            pass

        result = parse_kv_sharing_pattern(MockConfig(), 32)

        assert result is None


class TestComputeKVCacheSpecsFromConfig:
    """Tests for compute_kv_cache_specs_from_config function."""

    def get_cache_config(self, **kwargs) -> CacheConfig:
        """Create a CacheConfig with test defaults."""
        defaults = {
            "block_size": 16,
            "cache_dtype": "auto",
            "enable_hybrid_allocator": True,
        }
        defaults.update(kwargs)
        return CacheConfig(**defaults)

    def test_compute_with_mock_convertor(self):
        """Test compute_kv_cache_specs_from_config with a mock convertor."""
        from vllm.v1.worker.kv_cache_spec_utils import (
            compute_kv_cache_specs_from_config,
        )

        class MockConvertor:
            def get_kv_cache_requirements(
                self, kv_cache_dtype: torch.dtype, prefix: str = "model.layers"
            ) -> ModelKVCacheRequirements:
                layers = [
                    LayerKVCacheConfig(
                        layer_idx=i,
                        attention_type=LayerAttentionType.FULL,
                        block_type=LayerBlockType.ATTENTION,
                        num_kv_heads=8,
                        head_size=64,
                        dtype=kv_cache_dtype,
                    )
                    for i in range(4)
                ]
                return ModelKVCacheRequirements(
                    layers=layers,
                    default_num_kv_heads=8,
                    default_head_size=64,
                    default_dtype=kv_cache_dtype,
                )

        convertor = MockConvertor()
        cache_config = self.get_cache_config()

        specs = compute_kv_cache_specs_from_config(
            convertor=convertor,
            cache_config=cache_config,
            kv_cache_dtype=torch.float16,
        )

        assert len(specs) == 4
        for i in range(4):
            assert i in specs
            assert isinstance(specs[i], FullAttentionSpec)
