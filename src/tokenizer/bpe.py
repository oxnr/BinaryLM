"""
Byte-Pair Encoding Tokenizer

This module implements a simple BPE tokenizer from scratch.
"""
import re
import collections
from typing import Dict, List, Tuple, Set, Optional, Union
import json
import os


class BPETokenizer:
    """
    A simple implementation of the Byte-Pair Encoding algorithm.
    
    BPE is a subword tokenization algorithm that starts with character-level
    tokenization and iteratively merges the most frequent pairs of adjacent tokens.
    """
    
    def __init__(
        self,
        vocab_size: int = 10000,
        special_tokens: Optional[Dict[str, str]] = None
    ):
        """
        Initialize a BPE tokenizer.
        
        Args:
            vocab_size: The maximum vocabulary size
            special_tokens: Dictionary of special tokens like PAD, UNK, etc.
        """
        self.vocab_size = vocab_size
        
        # Define default special tokens if none provided
        self.special_tokens = special_tokens or {
            "pad_token": "[PAD]",
            "unk_token": "[UNK]",
            "bos_token": "[BOS]",
            "eos_token": "[EOS]",
            "mask_token": "[MASK]",
            "sep_token": "[SEP]",
            "cls_token": "[CLS]",
        }
        
        # Create reverse mapping for easier lookup
        self.special_token_ids = {}
        
        # Initialize an empty vocabulary
        self.vocab: Dict[str, int] = {}
        self.merges: List[Tuple[str, str]] = []
        self.token_to_id: Dict[str, int] = {}
        self.id_to_token: Dict[int, str] = {}
        
        # Add special tokens to the vocabulary
        for token_type, token in self.special_tokens.items():
            token_id = self._add_token(token)
            self.special_token_ids[token_type] = token_id
    
    def _add_token(self, token: str) -> int:
        """Add a token to the vocabulary and return its ID."""
        if token not in self.token_to_id:
            token_id = len(self.token_to_id)
            self.token_to_id[token] = token_id
            self.id_to_token[token_id] = token
            return token_id
        return self.token_to_id[token]
    
    def train(self, texts: List[str], min_frequency: int = 2, 
              num_merges: Optional[int] = None) -> None:
        """
        Train the BPE tokenizer on a list of texts.
        
        Args:
            texts: List of text samples for training
            min_frequency: Minimum frequency for a pair to be merged
            num_merges: Number of merge operations to perform
        """
        # Initialize with character-level tokenization
        word_freqs = collections.defaultdict(int)
        for text in texts:
            # Simple pre-tokenization by splitting on whitespace
            for word in text.split():
                word_freqs[word] += 1
        
        # Initialize splits - each word is split into characters
        splits = {}
        for word, freq in word_freqs.items():
            splits[word] = list(word)
        
        # Perform merges
        vocab = set()
        for c in "".join(word_freqs.keys()):
            vocab.add(c)
        
        # Calculate maximum number of merges, accounting for special tokens
        num_special_tokens = len(self.special_tokens)
        available_vocab_slots = self.vocab_size - num_special_tokens
        max_merges = min(
            num_merges or float('inf'),
            available_vocab_slots
        )
        
        # Ensure we don't exceed available slots
        if max_merges <= 0:
            raise ValueError(
                f"Vocabulary size ({self.vocab_size}) must be greater than "
                f"the number of special tokens ({num_special_tokens})"
            )
        
        # Perform merge operations
        for _ in range(max_merges):
            # Count pairs
            pair_freqs = collections.defaultdict(int)
            for word, freq in word_freqs.items():
                chars = splits[word]
                for i in range(len(chars) - 1):
                    pair = (chars[i], chars[i + 1])
                    pair_freqs[pair] += freq
            
            # Find the most frequent pair
            if not pair_freqs:
                break
                
            best_pair = max(pair_freqs.items(), key=lambda x: x[1])
            
            # If the best pair occurs less than min_frequency times, stop
            if best_pair[1] < min_frequency:
                break
                
            # Create a new token from the pair
            new_token = best_pair[0][0] + best_pair[0][1]
            vocab.add(new_token)
            self.merges.append(best_pair[0])
            
            # Update splits
            for word in word_freqs:
                i = 0
                chars = splits[word]
                new_chars = []
                
                while i < len(chars):
                    if i < len(chars) - 1 and (chars[i], chars[i + 1]) == best_pair[0]:
                        new_chars.append(new_token)
                        i += 2
                    else:
                        new_chars.append(chars[i])
                        i += 1
                
                splits[word] = new_chars
            
            # Check if we've reached the vocabulary size limit
            if len(vocab) + num_special_tokens >= self.vocab_size:
                break
        
        # Build the final vocabulary
        for token in vocab:
            self._add_token(token)
        
        print(f"Trained BPE tokenizer with {len(self.token_to_id)} tokens")
    
    def _tokenize_word(self, word: str) -> List[str]:
        """
        Tokenize a single word using the trained BPE merges.
        
        Args:
            word: The word to tokenize
            
        Returns:
            A list of subword tokens
        """
        if not self.merges:
            return list(word)
            
        # Start with character-level split
        chars = list(word)
        
        # Apply merges in order
        for pair in self.merges:
            i = 0
            while i < len(chars) - 1:
                if chars[i] == pair[0] and chars[i + 1] == pair[1]:
                    chars = chars[:i] + [chars[i] + chars[i + 1]] + chars[i + 2:]
                else:
                    i += 1
        
        return chars
    
    def encode(
        self, 
        text: str, 
        add_special_tokens: bool = False,
        max_length: Optional[int] = None,
        padding: bool = False,
        truncation: bool = False
    ) -> List[int]:
        """
        Encode a text string into a list of token IDs.
        
        Args:
            text: The input text to tokenize
            add_special_tokens: Whether to add special tokens like BOS/EOS
            max_length: Maximum sequence length (used for padding/truncation)
            padding: Whether to pad sequences shorter than max_length
            truncation: Whether to truncate sequences longer than max_length
            
        Returns:
            A list of token IDs
        """
        if not self.merges:
            raise ValueError("Tokenizer has not been trained yet.")
        
        # Get special token IDs if needed
        bos_token_id = self.special_token_ids.get("bos_token")
        eos_token_id = self.special_token_ids.get("eos_token")
        pad_token_id = self.special_token_ids.get("pad_token")
        unk_token_id = self.special_token_ids.get("unk_token")
        
        # Pre-tokenize by splitting on whitespace
        words = text.split()
        tokens = []
        
        # Add beginning of sequence token if requested
        if add_special_tokens and bos_token_id is not None:
            tokens.append(bos_token_id)
        
        # Tokenize each word
        for word in words:
            subwords = self._tokenize_word(word)
            for token in subwords:
                # Convert token to ID, using UNK for unknown tokens
                if token in self.token_to_id:
                    tokens.append(self.token_to_id[token])
                elif unk_token_id is not None:
                    tokens.append(unk_token_id)
        
        # Add end of sequence token if requested
        if add_special_tokens and eos_token_id is not None:
            tokens.append(eos_token_id)
        
        # Handle truncation if needed
        if max_length is not None and truncation and len(tokens) > max_length:
            if add_special_tokens and eos_token_id is not None:
                # Reserve space for EOS token
                tokens = tokens[:max_length - 1] + [eos_token_id]
            else:
                tokens = tokens[:max_length]
        
        # Handle padding if needed
        if max_length is not None and padding and len(tokens) < max_length:
            if pad_token_id is not None:
                tokens.extend([pad_token_id] * (max_length - len(tokens)))
        
        return tokens
    
    def encode_batch(
        self, 
        texts: List[str], 
        add_special_tokens: bool = False,
        max_length: Optional[int] = None,
        padding: bool = False,
        truncation: bool = False
    ) -> List[List[int]]:
        """
        Encode a batch of texts into token IDs.
        
        Args:
            texts: List of input texts to tokenize
            add_special_tokens: Whether to add special tokens like BOS/EOS
            max_length: Maximum sequence length (used for padding/truncation)
            padding: Whether to pad sequences shorter than max_length
            truncation: Whether to truncate sequences longer than max_length
            
        Returns:
            A list of token ID lists
        """
        return [
            self.encode(
                text, 
                add_special_tokens=add_special_tokens,
                max_length=max_length,
                padding=padding,
                truncation=truncation
            ) 
            for text in texts
        ]
    
    def decode(self, token_ids: List[int], skip_special_tokens: bool = False) -> str:
        """
        Decode a list of token IDs back into a text string.
        
        Args:
            token_ids: The list of token IDs to decode
            skip_special_tokens: Whether to remove special tokens from the output
            
        Returns:
            The decoded text
        """
        # Get set of special token IDs for filtering if needed
        special_token_ids = set(self.special_token_ids.values()) if skip_special_tokens else set()
        
        tokens = []
        for token_id in token_ids:
            if token_id in self.id_to_token and (not skip_special_tokens or token_id not in special_token_ids):
                tokens.append(self.id_to_token[token_id])
        
        # Simple joining of tokens
        # This is a simplification - in a real tokenizer, we would need 
        # more sophisticated detokenization logic
        return " ".join(tokens).replace(" ##", "").strip()
    
    def decode_batch(self, batch_token_ids: List[List[int]], skip_special_tokens: bool = False) -> List[str]:
        """
        Decode a batch of token ID lists into text strings.
        
        Args:
            batch_token_ids: List of token ID lists to decode
            skip_special_tokens: Whether to remove special tokens from the output
            
        Returns:
            List of decoded texts
        """
        return [self.decode(token_ids, skip_special_tokens=skip_special_tokens) for token_ids in batch_token_ids]
    
    def get_vocab_size(self) -> int:
        """Get the current size of the vocabulary."""
        return len(self.token_to_id)
    
    def get_special_tokens_mask(self, token_ids: List[int]) -> List[int]:
        """
        Create a mask identifying special tokens in a sequence.
        
        Args:
            token_ids: List of token IDs
            
        Returns:
            List of 1s for special tokens and 0s for normal tokens
        """
        special_token_ids = set(self.special_token_ids.values())
        return [1 if token_id in special_token_ids else 0 for token_id in token_ids]
    
    def save(self, path: str) -> None:
        """Save the tokenizer to a directory."""
        os.makedirs(path, exist_ok=True)
        
        # Save vocabulary and merges
        with open(os.path.join(path, "vocab.json"), "w") as f:
            json.dump(self.token_to_id, f, ensure_ascii=False, indent=2)
        
        with open(os.path.join(path, "merges.json"), "w") as f:
            json.dump(self.merges, f, ensure_ascii=False, indent=2)
        
        # Save special tokens
        with open(os.path.join(path, "special_tokens.json"), "w") as f:
            json.dump(self.special_tokens, f, ensure_ascii=False, indent=2)
        
        # Save tokenizer config
        config = {
            "vocab_size": self.vocab_size,
            "special_token_ids": self.special_token_ids,
            "tokenizer_type": "bpe"
        }
        with open(os.path.join(path, "config.json"), "w") as f:
            json.dump(config, f, ensure_ascii=False, indent=2)
    
    @classmethod
    def load(cls, path: str) -> "BPETokenizer":
        """Load a tokenizer from a directory."""
        # Load vocabulary
        with open(os.path.join(path, "vocab.json"), "r") as f:
            token_to_id = json.load(f)
        
        # Convert keys to str and values to int
        token_to_id = {str(k): int(v) for k, v in token_to_id.items()}
        
        # Load merges
        with open(os.path.join(path, "merges.json"), "r") as f:
            merges = json.load(f)
        
        # Load special tokens
        try:
            with open(os.path.join(path, "special_tokens.json"), "r") as f:
                special_tokens = json.load(f)
        except FileNotFoundError:
            # For backward compatibility
            special_tokens = {
                "pad_token": "[PAD]",
                "unk_token": "[UNK]",
                "bos_token": "[BOS]",
                "eos_token": "[EOS]",
                "mask_token": "[MASK]",
                "sep_token": "[SEP]",
                "cls_token": "[CLS]",
            }
        
        # Load config
        try:
            with open(os.path.join(path, "config.json"), "r") as f:
                config = json.load(f)
                vocab_size = config.get("vocab_size", len(token_to_id))
        except FileNotFoundError:
            vocab_size = len(token_to_id)
        
        # Create a new tokenizer instance
        tokenizer = cls(
            vocab_size=vocab_size,
            special_tokens=special_tokens
        )
        
        # Update the vocabulary and merges
        tokenizer.token_to_id = token_to_id
        tokenizer.id_to_token = {int(v): str(k) for k, v in token_to_id.items()}
        tokenizer.merges = merges
        
        # Rebuild special token IDs
        tokenizer.special_token_ids = {}
        for token_type, token in special_tokens.items():
            if token in token_to_id:
                tokenizer.special_token_ids[token_type] = token_to_id[token]
        
        return tokenizer


if __name__ == "__main__":
    # Example usage
    texts = [
        "Hello world!",
        "This is a simple BPE tokenizer implementation.",
        "It breaks text into subword units.",
        "Byte-Pair Encoding is a common tokenization method for NLP tasks."
    ]
    
    # Create and train tokenizer
    tokenizer = BPETokenizer(vocab_size=100)
    tokenizer.train(texts)
    
    # Test encoding with special tokens and padding
    test_text = "Hello, this is a test of the BPE tokenizer!"
    token_ids = tokenizer.encode(
        test_text, 
        add_special_tokens=True,
        max_length=20,
        padding=True
    )
    
    print(f"Original text: {test_text}")
    print(f"Encoded token IDs: {token_ids}")
    
    # Test decoding
    decoded_text = tokenizer.decode(token_ids, skip_special_tokens=True)
    print(f"Decoded text: {decoded_text}")
    
    # Test batch encoding
    batch_texts = ["Hello world!", "Testing the tokenizer"]
    batch_ids = tokenizer.encode_batch(
        batch_texts,
        add_special_tokens=True,
        max_length=10,
        padding=True
    )
    print(f"Batch encoded: {batch_ids}")
    
    # Test batch decoding
    decoded_batch = tokenizer.decode_batch(batch_ids, skip_special_tokens=True)
    print(f"Batch decoded: {decoded_batch}") 