#!/usr/bin/env python3
"""
Initialize a transformer model with the specified configuration.
"""

import os
import sys
import argparse
import torch

from src.model.config import ModelConfig, get_tiny_config, get_small_config, get_base_config, convert_config
from src.model.transformer import DecoderOnlyTransformer


def init_model(config_path: str, output_dir: str, model_size: str = None) -> None:
    """
    Initialize a transformer model.
    
    Args:
        config_path: Path to the configuration file, or None to use model_size
        output_dir: Directory to save the model
        model_size: Model size (tiny, small, base) if config_path is None
    """
    # Load configuration
    if config_path and os.path.exists(config_path):
        model_config = ModelConfig.from_file(config_path)
        print(f"Loaded configuration from {config_path}")
    elif model_size:
        if model_size == "tiny":
            model_config = get_tiny_config()
        elif model_size == "small":
            model_config = get_small_config()
        elif model_size == "base":
            model_config = get_base_config()
        else:
            raise ValueError(f"Unknown model size: {model_size}")
        print(f"Using {model_size} model configuration")
    else:
        raise ValueError("Either config_path or model_size must be provided")
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Save configuration
    config_save_path = os.path.join(output_dir, "config.json")
    model_config.save(config_save_path)
    print(f"Saved configuration to {config_save_path}")
    
    # Initialize model
    print("Initializing model...")
    transformer_config = convert_config(model_config)
    model = DecoderOnlyTransformer(transformer_config)
    
    # Calculate model size
    num_params = sum(p.numel() for p in model.parameters())
    print(f"Model initialized with {num_params:,} parameters")
    
    # Save model
    model_path = os.path.join(output_dir, "pytorch_model.bin")
    torch.save(model.state_dict(), model_path)
    print(f"Saved model to {model_path}")


def main():
    parser = argparse.ArgumentParser(description="Initialize a transformer model.")
    parser.add_argument("--config", type=str, help="Path to configuration file")
    parser.add_argument("--output", type=str, required=True, help="Output directory")
    parser.add_argument("--size", type=str, choices=["tiny", "small", "base"], 
                       help="Model size (tiny, small, base)")
    
    args = parser.parse_args()
    
    if not args.config and not args.size:
        parser.error("Either --config or --size must be provided")
    
    init_model(args.config, args.output, args.size)


if __name__ == "__main__":
    main() 