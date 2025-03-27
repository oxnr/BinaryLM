#!/usr/bin/env python3
"""
Test script for the transformer model.
"""

import torch
from src.model.transformer import TransformerConfig, DecoderOnlyTransformer
from src.model.config import get_tiny_config, convert_config


def test_forward_pass():
    """Test the forward pass of the model."""
    print("Testing forward pass...")
    
    # Create a tiny model for testing
    config = get_tiny_config()
    transformer_config = TransformerConfig(
        vocab_size=config.vocab_size,
        d_model=config.d_model,
        n_layers=config.n_layers,
        n_heads=config.n_heads,
        d_ff=config.d_ff,
        max_seq_len=config.max_seq_len,
        dropout=config.dropout,
    )
    
    model = DecoderOnlyTransformer(transformer_config)
    model.eval()  # Set to evaluation mode
    
    # Create dummy input
    batch_size = 2
    seq_len = 10
    input_ids = torch.randint(0, config.vocab_size, (batch_size, seq_len))
    
    # Run forward pass
    with torch.no_grad():
        logits = model(input_ids)
    
    # Check output shape
    expected_shape = (batch_size, seq_len, config.vocab_size)
    actual_shape = logits.shape
    
    print(f"Input shape: {input_ids.shape}")
    print(f"Expected output shape: {expected_shape}")
    print(f"Actual output shape: {actual_shape}")
    
    assert actual_shape == expected_shape, f"Output shape mismatch: {actual_shape} != {expected_shape}"
    print("Forward pass test passed!")


def test_generation():
    """Test the text generation functionality."""
    print("\nTesting text generation...")
    
    # Create a tiny model for testing
    config = get_tiny_config()
    transformer_config = TransformerConfig(
        vocab_size=config.vocab_size,
        d_model=config.d_model,
        n_layers=config.n_layers,
        n_heads=config.n_heads,
        d_ff=config.d_ff,
        max_seq_len=config.max_seq_len,
        dropout=0.0,  # Set dropout to 0 for deterministic behavior
    )
    
    model = DecoderOnlyTransformer(transformer_config)
    model.eval()  # Set to evaluation mode
    
    # Create dummy input
    batch_size = 1
    seq_len = 5
    input_ids = torch.randint(0, config.vocab_size, (batch_size, seq_len))
    
    # Generate text
    max_new_tokens = 10
    do_sample = False  # Use greedy decoding for deterministic output
    
    with torch.no_grad():
        output_ids = model.generate(
            input_ids, 
            max_new_tokens=max_new_tokens,
            do_sample=do_sample
        )
    
    # Check output shape
    expected_shape = (batch_size, seq_len + max_new_tokens)
    actual_shape = output_ids.shape
    
    print(f"Input shape: {input_ids.shape}")
    print(f"Expected output shape: {expected_shape}")
    print(f"Actual output shape: {actual_shape}")
    
    assert actual_shape == expected_shape, f"Output shape mismatch: {actual_shape} != {expected_shape}"
    
    # Verify that the prefix of the output matches the input
    input_prefix = input_ids[0].tolist()
    output_prefix = output_ids[0, :seq_len].tolist()
    
    assert input_prefix == output_prefix, f"Input prefix mismatch: {input_prefix} != {output_prefix}"
    
    print("Text generation test passed!")


def test_attention_mechanism():
    """Test the attention mechanism specifically."""
    print("\nTesting attention mechanism...")
    
    from src.model.transformer import MultiHeadAttention
    
    # Create a tiny config for testing
    config = TransformerConfig(
        vocab_size=1000,
        d_model=128,
        n_heads=4,
        n_layers=2,
        d_ff=512,
        max_seq_len=64,
        dropout=0.0,  # Set dropout to 0 for deterministic behavior
    )
    
    attention = MultiHeadAttention(config)
    attention.eval()  # Set to evaluation mode
    
    # Create dummy input
    batch_size = 2
    seq_len = 10
    d_model = config.d_model
    x = torch.randn(batch_size, seq_len, d_model)
    
    # Run attention forward pass
    with torch.no_grad():
        output = attention(x)
    
    # Check output shape
    expected_shape = (batch_size, seq_len, d_model)
    actual_shape = output.shape
    
    print(f"Input shape: {x.shape}")
    print(f"Expected output shape: {expected_shape}")
    print(f"Actual output shape: {actual_shape}")
    
    assert actual_shape == expected_shape, f"Output shape mismatch: {actual_shape} != {expected_shape}"
    print("Attention mechanism test passed!")


def test_parameter_count():
    """Test the parameter count calculation."""
    print("\nTesting parameter count...")
    
    # Test with different configurations
    configs = [
        ("tiny", get_tiny_config()),
        ("small", TransformerConfig(vocab_size=50000, d_model=384, n_heads=6, n_layers=6, d_ff=1536)),
        ("base", TransformerConfig(vocab_size=50000, d_model=768, n_heads=12, n_layers=12, d_ff=3072)),
    ]
    
    for name, config in configs:
        if isinstance(config, TransformerConfig):
            transformer_config = config
        else:
            transformer_config = TransformerConfig(
                vocab_size=config.vocab_size,
                d_model=config.d_model,
                n_layers=config.n_layers,
                n_heads=config.n_heads,
                d_ff=config.d_ff,
                max_seq_len=config.max_seq_len,
            )
        
        model = DecoderOnlyTransformer(transformer_config)
        num_params = sum(p.numel() for p in model.parameters())
        
        print(f"{name.capitalize()} model has {num_params:,} parameters")
        
        # Rough parameter calculation
        embedding_params = transformer_config.vocab_size * transformer_config.d_model  # Token embeddings
        position_params = transformer_config.max_seq_len * transformer_config.d_model  # Position embeddings
        
        # Per-layer parameters
        layer_params = (
            # Multi-head attention
            4 * (transformer_config.d_model ** 2) +  # Q, K, V, and output projections
            # Feed-forward network
            transformer_config.d_model * transformer_config.d_ff +  # First linear layer
            transformer_config.d_ff * transformer_config.d_model +  # Second linear layer
            # Layer norms
            2 * transformer_config.d_model * 2  # 2 layer norms, each with weight and bias
        )
        
        # Total parameters
        total_params = embedding_params + position_params + (layer_params * transformer_config.n_layers)
        
        # Add final layer norm and output projection (which shares weights with token embeddings)
        total_params += 2 * transformer_config.d_model  # Final layer norm
        
        print(f"Calculated ~{total_params:,} parameters")
    
    print("Parameter count test completed!")


def main():
    """Run all tests."""
    test_forward_pass()
    test_attention_mechanism()
    test_generation()
    test_parameter_count()
    print("\nAll tests passed!")


if __name__ == "__main__":
    main() 