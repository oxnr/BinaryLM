"""
Transformer model implementation for BinaryLM
"""

from src.model.transformer import (
    TransformerConfig,
    MultiHeadAttention,
    FeedForward,
    TransformerBlock,
    DecoderOnlyTransformer,
)

from src.model.config import (
    ModelConfig,
    get_tiny_config,
    get_small_config,
    get_base_config,
    get_medium_config,
    get_large_config,
)

__all__ = [
    # Transformer components
    "TransformerConfig",
    "MultiHeadAttention",
    "FeedForward",
    "TransformerBlock",
    "DecoderOnlyTransformer",
    
    # Configuration utilities
    "ModelConfig",
    "get_tiny_config",
    "get_small_config",
    "get_base_config",
    "get_medium_config",
    "get_large_config",
] 