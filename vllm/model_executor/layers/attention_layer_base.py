# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Base class for attention-like layers."""

from abc import ABC, abstractmethod

from vllm.v1.attention.backend import AttentionBackend, AttentionImpl


class AttentionLayerBase(ABC):
    """
    Base class for attention-like layers (Attention, Mamba, etc.)
    that support the v1 engine.

    This provides a common interface for getting attention backends
    from different layer types.
    """

    impl: "AttentionImpl"

    @abstractmethod
    def get_attn_backend(self) -> type[AttentionBackend]:
        """Get the attention backend class for this layer."""
        pass
