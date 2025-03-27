"""
Configuration utilities for model hyperparameters.
"""

import os
import json
from dataclasses import dataclass, asdict
from typing import Optional


@dataclass
class ModelConfig:
    """Model configuration parameters"""
    
    # Model architecture
    vocab_size: int = 50000
    d_model: int = 768
    n_heads: int = 12
    n_layers: int = 12
    d_ff: int = 3072
    max_seq_len: int = 2048
    
    # Training hyperparameters
    dropout: float = 0.1
    layer_norm_epsilon: float = 1e-5
    initializer_range: float = 0.02
    
    # Optimization
    learning_rate: float = 1e-4
    weight_decay: float = 0.01
    warmup_steps: int = 10000
    max_steps: int = 100000
    gradient_accumulation_steps: int = 8
    
    # Misc
    model_type: str = "decoder_only"
    
    def save(self, config_path: str) -> None:
        """
        Save configuration to JSON file.
        
        Args:
            config_path: Path to save the configuration
        """
        os.makedirs(os.path.dirname(config_path), exist_ok=True)
        with open(config_path, "w") as f:
            json.dump(asdict(self), f, indent=2)
    
    @classmethod
    def from_file(cls, config_path: str) -> "ModelConfig":
        """
        Load configuration from JSON file.
        
        Args:
            config_path: Path to the configuration file
            
        Returns:
            ModelConfig object
        """
        with open(config_path, "r") as f:
            config_dict = json.load(f)
        return cls(**config_dict)


# Predefined model sizes based on common architectures

def get_tiny_config() -> ModelConfig:
    """Returns a tiny model configuration for testing."""
    return ModelConfig(
        vocab_size=50000,
        d_model=128,
        n_heads=2,
        n_layers=2,
        d_ff=512,
        max_seq_len=1024,
    )


def get_small_config() -> ModelConfig:
    """Returns a small model configuration (~30M params)."""
    return ModelConfig(
        vocab_size=50000,
        d_model=384,
        n_heads=6,
        n_layers=6,
        d_ff=1536,
        max_seq_len=1024,
    )


def get_base_config() -> ModelConfig:
    """Returns a base model configuration (~110M params)."""
    return ModelConfig(
        vocab_size=50000,
        d_model=768,
        n_heads=12,
        n_layers=12,
        d_ff=3072,
        max_seq_len=2048,
    )


def get_medium_config() -> ModelConfig:
    """Returns a medium model configuration (~350M params)."""
    return ModelConfig(
        vocab_size=50000,
        d_model=1024,
        n_heads=16,
        n_layers=24,
        d_ff=4096,
        max_seq_len=2048,
    )


def get_large_config() -> ModelConfig:
    """Returns a large model configuration (~1.3B params)."""
    return ModelConfig(
        vocab_size=50000,
        d_model=2048,
        n_heads=16,
        n_layers=24,
        d_ff=8192,
        max_seq_len=4096,
    )


def convert_config(model_config: ModelConfig) -> "TransformerConfig":
    """
    Convert ModelConfig to TransformerConfig.
    
    Args:
        model_config: The ModelConfig object
        
    Returns:
        TransformerConfig object for the transformer model
    """
    # Delayed import to avoid circular imports
    from src.model.transformer import TransformerConfig
    
    return TransformerConfig(
        vocab_size=model_config.vocab_size,
        d_model=model_config.d_model,
        n_layers=model_config.n_layers,
        n_heads=model_config.n_heads,
        d_ff=model_config.d_ff,
        max_seq_len=model_config.max_seq_len,
        dropout=model_config.dropout,
        layer_norm_epsilon=model_config.layer_norm_epsilon,
        initializer_range=model_config.initializer_range,
    ) 