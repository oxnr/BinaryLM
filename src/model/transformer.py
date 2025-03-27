"""
Decoder-only Transformer model implementation for BinaryLM.

This module implements a decoder-only transformer architecture similar to those used 
in models like GPT, Claude, and LLaMA.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple


class TransformerConfig:
    """Configuration class for Transformer model parameters."""
    
    def __init__(
        self,
        vocab_size: int = 50000,
        d_model: int = 768,
        n_layers: int = 12,
        n_heads: int = 12,
        d_ff: int = 3072,
        max_seq_len: int = 2048,
        dropout: float = 0.1,
        layer_norm_epsilon: float = 1e-5,
        initializer_range: float = 0.02,
    ):
        """
        Initialize configuration for the Transformer model.
        
        Args:
            vocab_size: Size of the vocabulary
            d_model: Dimension of embeddings and hidden states
            n_layers: Number of transformer layers
            n_heads: Number of attention heads
            d_ff: Dimension of feed-forward layer
            max_seq_len: Maximum sequence length
            dropout: Dropout probability
            layer_norm_epsilon: Epsilon value for layer normalization
            initializer_range: Range for weight initialization
        """
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.n_layers = n_layers
        self.n_heads = n_heads
        self.d_ff = d_ff
        self.max_seq_len = max_seq_len
        self.dropout = dropout
        self.layer_norm_epsilon = layer_norm_epsilon
        self.initializer_range = initializer_range
        
        # Derived parameters
        self.head_dim = d_model // n_heads
        assert self.head_dim * n_heads == d_model, "d_model must be divisible by n_heads"


class MultiHeadAttention(nn.Module):
    """Multi-head self-attention mechanism."""
    
    def __init__(self, config: TransformerConfig):
        """
        Initialize multi-head attention layer.
        
        Args:
            config: Transformer configuration
        """
        super().__init__()
        self.n_heads = config.n_heads
        self.d_model = config.d_model
        self.head_dim = config.head_dim
        
        # Query, Key, Value projections
        self.q_proj = nn.Linear(config.d_model, config.d_model)
        self.k_proj = nn.Linear(config.d_model, config.d_model)
        self.v_proj = nn.Linear(config.d_model, config.d_model)
        
        # Output projection
        self.out_proj = nn.Linear(config.d_model, config.d_model)
        
        # Dropout
        self.dropout = nn.Dropout(config.dropout)
        
        # Scaling factor for attention scores
        self.scale = 1.0 / math.sqrt(self.head_dim)
    
    def forward(
        self,
        x: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        is_causal: bool = True,
    ) -> torch.Tensor:
        """
        Forward pass for multi-head attention.
        
        Args:
            x: Input tensor of shape (batch_size, seq_len, d_model)
            attention_mask: Optional mask of shape (batch_size, 1, seq_len, seq_len)
            is_causal: Whether to apply causal masking for autoregressive models
            
        Returns:
            Output tensor of shape (batch_size, seq_len, d_model)
        """
        batch_size, seq_len, _ = x.size()
        
        # Project inputs to queries, keys, and values
        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)
        
        # Reshape to (batch_size, seq_len, n_heads, head_dim)
        q = q.view(batch_size, seq_len, self.n_heads, self.head_dim)
        k = k.view(batch_size, seq_len, self.n_heads, self.head_dim)
        v = v.view(batch_size, seq_len, self.n_heads, self.head_dim)
        
        # Transpose to (batch_size, n_heads, seq_len, head_dim)
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)
        
        # Compute attention scores: (batch_size, n_heads, seq_len, seq_len)
        attn_scores = torch.matmul(q, k.transpose(2, 3)) * self.scale
        
        # Apply causal mask if needed
        if is_causal:
            # Create causal mask (upper triangular)
            causal_mask = torch.triu(
                torch.ones(seq_len, seq_len, device=x.device), diagonal=1
            ).bool()
            # Set masked positions to -inf
            attn_scores.masked_fill_(causal_mask.unsqueeze(0).unsqueeze(0), float("-inf"))
        
        # Apply attention mask if provided
        if attention_mask is not None:
            attn_scores = attn_scores + attention_mask
        
        # Apply softmax to get attention weights
        attn_weights = F.softmax(attn_scores, dim=-1)
        attn_weights = self.dropout(attn_weights)
        
        # Apply attention weights to values
        attn_output = torch.matmul(attn_weights, v)  # (batch_size, n_heads, seq_len, head_dim)
        
        # Reshape and transpose back to (batch_size, seq_len, d_model)
        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.view(batch_size, seq_len, self.d_model)
        
        # Project to output dimension
        output = self.out_proj(attn_output)
        
        return output


class FeedForward(nn.Module):
    """Feed-forward network used in transformer blocks."""
    
    def __init__(self, config: TransformerConfig):
        """
        Initialize feed-forward network.
        
        Args:
            config: Transformer configuration
        """
        super().__init__()
        self.linear1 = nn.Linear(config.d_model, config.d_ff)
        self.linear2 = nn.Linear(config.d_ff, config.d_model)
        self.dropout = nn.Dropout(config.dropout)
        self.activation = nn.GELU()  # Modern transformers typically use GELU activation
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for feed-forward network.
        
        Args:
            x: Input tensor of shape (batch_size, seq_len, d_model)
            
        Returns:
            Output tensor of shape (batch_size, seq_len, d_model)
        """
        x = self.linear1(x)
        x = self.activation(x)
        x = self.dropout(x)
        x = self.linear2(x)
        return x


class TransformerBlock(nn.Module):
    """A single transformer block with pre-layernorm architecture."""
    
    def __init__(self, config: TransformerConfig):
        """
        Initialize transformer block.
        
        Args:
            config: Transformer configuration
        """
        super().__init__()
        self.ln_1 = nn.LayerNorm(config.d_model, eps=config.layer_norm_epsilon)
        self.attn = MultiHeadAttention(config)
        self.ln_2 = nn.LayerNorm(config.d_model, eps=config.layer_norm_epsilon)
        self.ff = FeedForward(config)
        self.dropout = nn.Dropout(config.dropout)
    
    def forward(
        self,
        x: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Forward pass for transformer block.
        
        Args:
            x: Input tensor of shape (batch_size, seq_len, d_model)
            attention_mask: Optional attention mask
            
        Returns:
            Output tensor of shape (batch_size, seq_len, d_model)
        """
        # Self-attention with pre-layernorm and residual connection
        ln_1_out = self.ln_1(x)
        attn_out = self.attn(ln_1_out, attention_mask)
        x = x + self.dropout(attn_out)
        
        # Feed-forward with pre-layernorm and residual connection
        ln_2_out = self.ln_2(x)
        ff_out = self.ff(ln_2_out)
        x = x + self.dropout(ff_out)
        
        return x


class DecoderOnlyTransformer(nn.Module):
    """Decoder-only transformer model for language modeling."""
    
    def __init__(self, config: TransformerConfig):
        """
        Initialize the decoder-only transformer model.
        
        Args:
            config: Transformer configuration
        """
        super().__init__()
        self.config = config
        
        # Token embedding
        self.token_embedding = nn.Embedding(config.vocab_size, config.d_model)
        
        # Position embedding (learned, not sinusoidal)
        self.position_embedding = nn.Embedding(config.max_seq_len, config.d_model)
        
        # Transformer blocks
        self.blocks = nn.ModuleList([TransformerBlock(config) for _ in range(config.n_layers)])
        
        # Final layer normalization
        self.ln_f = nn.LayerNorm(config.d_model, eps=config.layer_norm_epsilon)
        
        # Output head
        self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False)
        
        # Tie token embedding and output weights
        self.lm_head.weight = self.token_embedding.weight
        
        # Dropout
        self.dropout = nn.Dropout(config.dropout)
        
        # Initialize weights
        self.apply(self._init_weights)
    
    def _init_weights(self, module):
        """Initialize the weights."""
        if isinstance(module, (nn.Linear, nn.Embedding)):
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if isinstance(module, nn.Linear) and module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
    
    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        return_logits: bool = True,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """
        Forward pass for the transformer model.
        
        Args:
            input_ids: Token IDs of shape (batch_size, seq_len)
            attention_mask: Optional attention mask of shape (batch_size, seq_len)
            return_logits: Whether to return logits or hidden states
            
        Returns:
            If return_logits=True: logits of shape (batch_size, seq_len, vocab_size)
            If return_logits=False: tuple of (logits, hidden_states)
        """
        batch_size, seq_len = input_ids.size()
        device = input_ids.device
        
        # Get token embeddings
        x = self.token_embedding(input_ids)  # (batch_size, seq_len, d_model)
        
        # Create position IDs and get position embeddings
        position_ids = torch.arange(0, seq_len, dtype=torch.long, device=device).unsqueeze(0)
        position_embeddings = self.position_embedding(position_ids)  # (1, seq_len, d_model)
        
        # Add position embeddings to token embeddings
        x = x + position_embeddings
        
        # Apply dropout
        x = self.dropout(x)
        
        # Process attention mask if provided
        extended_attention_mask = None
        if attention_mask is not None:
            # Create extended attention mask for multi-head attention
            # (batch_size, 1, 1, seq_len)
            extended_attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)
            # Convert mask of 0s and 1s to mask of -inf and 0s
            extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0
        
        # Apply transformer blocks
        for block in self.blocks:
            x = block(x, extended_attention_mask)
        
        # Apply final layer norm
        x = self.ln_f(x)
        
        # Compute logits
        logits = self.lm_head(x)
        
        if return_logits:
            return logits
        else:
            return logits, x
    
    def generate(
        self,
        input_ids: torch.Tensor,
        max_new_tokens: int = 50,
        temperature: float = 1.0,
        top_k: int = 0,
        top_p: float = 0.9,
        do_sample: bool = True,
    ) -> torch.Tensor:
        """
        Generate text using the model.
        
        Args:
            input_ids: Input token IDs
            max_new_tokens: Maximum number of new tokens to generate
            temperature: Sampling temperature
            top_k: Number of highest probability vocabulary tokens to keep for top-k-filtering
            top_p: Nucleus sampling probability threshold
            do_sample: Whether to sample or use greedy decoding
            
        Returns:
            Generated token IDs
        """
        batch_size, seq_len = input_ids.size()
        device = input_ids.device
        
        # Initialize generated sequence with input
        generated = input_ids.clone()
        
        # Generate tokens auto-regressively
        for _ in range(max_new_tokens):
            # Only use the last `seq_len` tokens for efficiency if sequence gets too long
            if generated.size(1) > self.config.max_seq_len:
                input_ids_for_gen = generated[:, -self.config.max_seq_len:]
            else:
                input_ids_for_gen = generated
            
            # Get logits for next token
            with torch.no_grad():
                logits = self.forward(input_ids_for_gen)
            
            # Only consider the logits for the last position
            next_token_logits = logits[:, -1, :]
            
            # Apply temperature
            if temperature > 0:
                next_token_logits = next_token_logits / temperature
            
            # Apply top-k filtering
            if top_k > 0:
                # Zero out all logits below the top k values
                indices_to_remove = next_token_logits < torch.topk(next_token_logits, top_k)[0][..., -1, None]
                next_token_logits[indices_to_remove] = float('-inf')
            
            # Apply top-p (nucleus) filtering
            if top_p < 1.0:
                sorted_logits, sorted_indices = torch.sort(next_token_logits, descending=True)
                cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
                
                # Remove tokens with cumulative probability above the threshold
                sorted_indices_to_remove = cumulative_probs > top_p
                # Shift the indices to the right to keep the first token above the threshold
                sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                sorted_indices_to_remove[..., 0] = 0
                
                # Scatter sorted tensors to original indexing
                indices_to_remove = sorted_indices_to_remove.scatter(
                    1, sorted_indices, sorted_indices_to_remove
                )
                next_token_logits[indices_to_remove] = float('-inf')
            
            # Sample or select the next token
            if do_sample:
                # Sample from the filtered distribution
                probs = F.softmax(next_token_logits, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)
            else:
                # Greedy selection
                next_token = torch.argmax(next_token_logits, dim=-1, keepdim=True)
            
            # Append next token to generated sequence
            generated = torch.cat((generated, next_token), dim=1)
        
        return generated


# Fix missing import at the top of the file
from typing import Union 