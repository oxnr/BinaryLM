"""
Dataset management for BinaryLM.

This module provides utilities for loading, preprocessing, and managing text datasets
for language model training.
"""

import os
import json
import logging
from typing import List, Dict, Tuple, Optional, Iterator, Union, Any
import random
import torch
from torch.utils.data import Dataset, DataLoader, IterableDataset
import pandas as pd

from src.tokenizer.bpe import BPETokenizer

logger = logging.getLogger(__name__)


class TextDataset(Dataset):
    """Dataset for text data that has been tokenized."""
    
    def __init__(
        self,
        tokenized_texts: List[torch.Tensor],
        seq_length: int = 1024,
        overlap: int = 128,
    ):
        """
        Initialize the dataset with tokenized texts.
        
        Args:
            tokenized_texts: List of tokenized texts as tensors
            seq_length: Maximum sequence length
            overlap: Overlap between sequences
        """
        self.tokenized_texts = tokenized_texts
        self.seq_length = seq_length
        self.overlap = overlap
        
        # Create list of (start, end) indices for each sequence
        self.indices = []
        for text_idx, text in enumerate(tokenized_texts):
            if len(text) <= seq_length:
                # For short texts, just use the whole text
                self.indices.append((text_idx, 0, len(text)))
            else:
                # For longer texts, create overlapping chunks
                start = 0
                while start < len(text):
                    end = min(start + seq_length, len(text))
                    self.indices.append((text_idx, start, end))
                    # Subtract overlap for next start point
                    if end == len(text):
                        break
                    start = end - overlap
    
    def __len__(self) -> int:
        """Return the number of sequences in the dataset."""
        return len(self.indices)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        Get a sequence from the dataset.
        
        Args:
            idx: Index of the sequence
            
        Returns:
            Dictionary with input_ids and labels tensors
        """
        text_idx, start, end = self.indices[idx]
        sequence = self.tokenized_texts[text_idx][start:end]
        
        # For causal language modeling, input_ids are the same as labels
        # but shifted by one position (predict next token)
        input_ids = sequence[:-1]
        labels = sequence[1:]
        
        return {
            "input_ids": input_ids,
            "labels": labels,
        }


class DatasetManager:
    """Manager for loading, preprocessing, and tokenizing datasets."""
    
    def __init__(
        self, 
        tokenizer: BPETokenizer,
        data_dir: Optional[str] = None,
        seq_length: int = 1024,
        overlap: int = 128,
    ):
        """
        Initialize the dataset manager.
        
        Args:
            tokenizer: Tokenizer to use for encoding texts
            data_dir: Directory for storing datasets
            seq_length: Maximum sequence length
            overlap: Overlap between sequences
        """
        self.tokenizer = tokenizer
        self.data_dir = data_dir or os.path.join(os.getcwd(), "data")
        self.seq_length = seq_length
        self.overlap = overlap
        
        # Create data directory if it doesn't exist
        if not os.path.exists(self.data_dir):
            os.makedirs(self.data_dir)
            
        # Create subdirectories
        self.raw_dir = os.path.join(self.data_dir, "raw")
        self.processed_dir = os.path.join(self.data_dir, "processed")
        self.tokenized_dir = os.path.join(self.data_dir, "tokenized")
        
        for directory in [self.raw_dir, self.processed_dir, self.tokenized_dir]:
            if not os.path.exists(directory):
                os.makedirs(directory)
                
        # Sample datasets info
        self.sample_datasets = {
            "tiny_shakespeare": {
                "name": "Tiny Shakespeare",
                "description": "A small dataset of Shakespeare's works",
                "num_tokens": "~1M",
                "size": "~1MB",
                "url": "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"
            },
            "wikitext_103": {
                "name": "WikiText-103",
                "description": "A collection of over 100 million tokens extracted from Wikipedia articles",
                "num_tokens": "~103M",
                "size": "~500MB",
                "url": "https://s3.amazonaws.com/research.metamind.io/wikitext/wikitext-103-v1.zip"
            },
            "code_github": {
                "name": "GitHub Code",
                "description": "A collection of Python code from popular GitHub repositories",
                "num_tokens": "~5M",
                "size": "~20MB",
                "url": "https://huggingface.co/datasets/codeparrot/github-code-clean"
            }
        }
    
    def get_available_datasets(self) -> Dict[str, Dict[str, str]]:
        """
        Get information about available datasets.
        
        Returns:
            Dictionary of dataset information
        """
        # Get list of all downloaded datasets
        downloaded = []
        for filename in os.listdir(self.raw_dir):
            if filename.endswith(".txt") or filename.endswith(".csv") or filename.endswith(".json"):
                dataset_id = os.path.splitext(filename)[0]
                dataset_info = {
                    "name": dataset_id.replace("_", " ").title(),
                    "path": os.path.join(self.raw_dir, filename),
                    "status": "downloaded"
                }
                downloaded.append((dataset_id, dataset_info))
        
        # Add sample datasets that aren't downloaded yet
        result = {}
        for dataset_id, info in self.sample_datasets.items():
            if any(d[0] == dataset_id for d in downloaded):
                continue
            result[dataset_id] = {**info, "status": "available"}
        
        # Add the downloaded datasets
        for dataset_id, info in downloaded:
            result[dataset_id] = info
        
        return result
    
    def load_dataset(self, dataset_id: str) -> List[str]:
        """
        Load a dataset by ID.
        
        Args:
            dataset_id: ID of the dataset to load
            
        Returns:
            List of texts from the dataset
        """
        datasets = self.get_available_datasets()
        if dataset_id not in datasets:
            raise ValueError(f"Dataset {dataset_id} not found")
        
        dataset_info = datasets[dataset_id]
        if dataset_info["status"] == "available":
            # Dataset needs to be downloaded
            self._download_dataset(dataset_id)
        
        # Load the dataset based on file extension
        dataset_path = os.path.join(self.raw_dir, f"{dataset_id}.txt")
        if os.path.exists(dataset_path):
            with open(dataset_path, "r", encoding="utf-8") as f:
                texts = [f.read()]
        else:
            csv_path = os.path.join(self.raw_dir, f"{dataset_id}.csv")
            if os.path.exists(csv_path):
                df = pd.read_csv(csv_path)
                text_column = df.columns[0]  # Assume first column is text
                texts = df[text_column].tolist()
            else:
                json_path = os.path.join(self.raw_dir, f"{dataset_id}.json")
                if os.path.exists(json_path):
                    with open(json_path, "r", encoding="utf-8") as f:
                        data = json.load(f)
                    if isinstance(data, list):
                        if all(isinstance(item, str) for item in data):
                            texts = data
                        else:
                            # Assume list of objects with a 'text' field
                            texts = [item.get("text", "") for item in data if "text" in item]
                    elif isinstance(data, dict) and "texts" in data:
                        texts = data["texts"]
                    else:
                        raise ValueError(f"Couldn't parse JSON dataset: {dataset_id}")
                else:
                    raise ValueError(f"Dataset file not found for {dataset_id}")
        
        return texts
    
    def _download_dataset(self, dataset_id: str) -> None:
        """
        Download a sample dataset.
        
        Args:
            dataset_id: ID of the dataset to download
        """
        import requests
        
        if dataset_id not in self.sample_datasets:
            raise ValueError(f"Sample dataset {dataset_id} not found")
        
        dataset_info = self.sample_datasets[dataset_id]
        url = dataset_info["url"]
        
        logger.info(f"Downloading dataset {dataset_id} from {url}")
        
        # Simple download for .txt files
        if url.endswith(".txt"):
            response = requests.get(url)
            response.raise_for_status()
            
            output_path = os.path.join(self.raw_dir, f"{dataset_id}.txt")
            with open(output_path, "wb") as f:
                f.write(response.content)
            
            logger.info(f"Downloaded dataset to {output_path}")
        else:
            # For other formats, we would need more complex handling
            # This is just a placeholder for a more comprehensive implementation
            logger.warning(f"Download for {url} not implemented yet")
            raise NotImplementedError(f"Download for {url} not implemented yet")
    
    def upload_dataset(self, file_content: str, filename: str) -> str:
        """
        Save an uploaded dataset file.
        
        Args:
            file_content: Content of the uploaded file
            filename: Original filename of the uploaded file
            
        Returns:
            Dataset ID
        """
        # Generate dataset ID from filename
        base_name = os.path.splitext(os.path.basename(filename))[0]
        dataset_id = base_name.lower().replace(" ", "_")
        
        # Determine file extension
        ext = os.path.splitext(filename)[1].lower()
        if ext not in [".txt", ".csv", ".json"]:
            ext = ".txt"  # Default to .txt
        
        # Save file
        output_path = os.path.join(self.raw_dir, f"{dataset_id}{ext}")
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(file_content)
        
        logger.info(f"Uploaded dataset saved to {output_path}")
        
        return dataset_id
    
    def preprocess_text(self, text: str) -> str:
        """
        Preprocess text for tokenization.
        
        Args:
            text: Raw text
            
        Returns:
            Preprocessed text
        """
        # Basic preprocessing
        # Remove excessive newlines
        text = "\n".join(line for line in text.split("\n") if line.strip())
        
        # More advanced preprocessing could be added here
        
        return text
    
    def prepare_dataset(
        self, 
        dataset_id: str, 
        train_split: float = 0.9
    ) -> Tuple[TextDataset, TextDataset]:
        """
        Prepare a dataset for training.
        
        Args:
            dataset_id: ID of the dataset to prepare
            train_split: Proportion of data to use for training
            
        Returns:
            Training and validation datasets
        """
        # Load and preprocess text
        texts = self.load_dataset(dataset_id)
        preprocessed_texts = [self.preprocess_text(text) for text in texts]
        
        # Tokenize texts
        tokenized_texts = []
        for text in preprocessed_texts:
            token_ids = self.tokenizer.encode(text)
            tokenized_texts.append(torch.tensor(token_ids, dtype=torch.long))
        
        # Split into train and validation
        train_texts = tokenized_texts[0:int(len(tokenized_texts) * train_split)]
        val_texts = tokenized_texts[int(len(tokenized_texts) * train_split):]
        
        # Create datasets
        train_dataset = TextDataset(train_texts, self.seq_length, self.overlap)
        val_dataset = TextDataset(val_texts, self.seq_length, self.overlap)
        
        return train_dataset, val_dataset
    
    def get_dataloader(
        self, 
        dataset: TextDataset, 
        batch_size: int = 8, 
        shuffle: bool = True
    ) -> DataLoader:
        """
        Create a dataloader for a dataset.
        
        Args:
            dataset: The dataset to create a dataloader for
            batch_size: Batch size
            shuffle: Whether to shuffle the data
            
        Returns:
            DataLoader object
        """
        return DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            pin_memory=True,
        )


# Helper function to create a simple example dataset
def create_example_dataset(output_path: str, num_samples: int = 100) -> None:
    """
    Create a simple example dataset for testing.
    
    Args:
        output_path: Path to save the dataset
        num_samples: Number of samples to generate
    """
    import numpy as np
    
    paragraphs = [
        "The quick brown fox jumps over the lazy dog.",
        "AI and machine learning are transforming the world in unprecedented ways.",
        "Large language models have shown remarkable abilities to understand and generate text.",
        "The transformer architecture has revolutionized natural language processing.",
        "Training neural networks requires substantial computational resources.",
        "Python is a popular programming language for machine learning and AI.",
        "PyTorch and TensorFlow are widely used frameworks for deep learning.",
        "Tokenization is a critical preprocessing step for language models.",
        "Attention mechanisms allow models to focus on relevant parts of the input.",
        "Fine-tuning adapts pre-trained models to specific tasks and domains."
    ]
    
    texts = []
    for _ in range(num_samples):
        # Generate a random number of paragraphs (1-5)
        num_paragraphs = np.random.randint(1, 6)
        # Sample paragraphs with replacement
        sample_paragraphs = np.random.choice(paragraphs, size=num_paragraphs, replace=True)
        # Join paragraphs with newlines
        text = "\n\n".join(sample_paragraphs)
        texts.append(text)
    
    # Save as JSON
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump({"texts": texts}, f, ensure_ascii=False, indent=2)
    
    print(f"Created example dataset with {num_samples} samples at {output_path}") 