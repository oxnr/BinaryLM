#!/usr/bin/env python3
"""
Training script for BinaryLM models.

This script provides a command-line interface for training transformer models
on text datasets.
"""

import os
import sys
import argparse
import logging
import json
import torch
from datetime import datetime

from src.model.transformer import DecoderOnlyTransformer
from src.model.config import ModelConfig, convert_config
from src.tokenizer.bpe import BPETokenizer
from src.data.dataset import DatasetManager, TextDataset
from src.training.trainer import Trainer, TrainingConfig

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
    ],
)
logger = logging.getLogger(__name__)


def setup_training_args():
    """Parse command line arguments for training."""
    parser = argparse.ArgumentParser(description="Train a BinaryLM transformer model.")
    
    # Model arguments
    parser.add_argument(
        "--model_path",
        type=str,
        help="Path to pre-trained model directory (if fine-tuning)",
    )
    parser.add_argument(
        "--model_size",
        type=str,
        choices=["tiny", "small", "base", "medium", "large"],
        default="tiny",
        help="Model size to initialize if not loading pre-trained",
    )
    
    # Dataset arguments
    parser.add_argument(
        "--dataset_id",
        type=str,
        required=True,
        help="ID of the dataset to train on",
    )
    parser.add_argument(
        "--data_dir",
        type=str,
        default="./data",
        help="Directory containing the datasets",
    )
    parser.add_argument(
        "--tokenizer_path",
        type=str,
        help="Path to tokenizer directory (if using pre-trained)",
    )
    parser.add_argument(
        "--train_split",
        type=float,
        default=0.9,
        help="Proportion of data to use for training",
    )
    parser.add_argument(
        "--seq_length",
        type=int,
        default=1024,
        help="Maximum sequence length",
    )
    
    # Training arguments
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./output",
        help="Directory to save the model and logs",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=8,
        help="Batch size for training",
    )
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=1,
        help="Number of steps to accumulate gradients for",
    )
    parser.add_argument(
        "--max_steps",
        type=int,
        default=10000,
        help="Maximum number of training steps",
    )
    parser.add_argument(
        "--max_epochs",
        type=int,
        default=None,
        help="Maximum number of epochs (if set, overrides max_steps)",
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=5e-5,
        help="Learning rate",
    )
    parser.add_argument(
        "--weight_decay",
        type=float,
        default=0.01,
        help="Weight decay",
    )
    parser.add_argument(
        "--warmup_steps",
        type=int,
        default=500,
        help="Number of warmup steps for learning rate scheduler",
    )
    parser.add_argument(
        "--lr_scheduler",
        type=str,
        default="cosine",
        choices=["cosine", "linear", "constant"],
        help="Learning rate scheduler type",
    )
    parser.add_argument(
        "--eval_steps",
        type=int,
        default=500,
        help="Number of steps between evaluations",
    )
    parser.add_argument(
        "--save_steps",
        type=int,
        default=1000,
        help="Number of steps between saving model checkpoints",
    )
    parser.add_argument(
        "--logging_steps",
        type=int,
        default=10,
        help="Number of steps between logging",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed",
    )
    
    return parser.parse_args()


def load_or_create_model(args):
    """
    Load a pre-trained model or create a new one.
    
    Args:
        args: Command line arguments
        
    Returns:
        The model
    """
    if args.model_path and os.path.exists(args.model_path):
        # Load model from checkpoint
        logger.info(f"Loading model from {args.model_path}")
        
        # Load model configuration
        config_path = os.path.join(args.model_path, "config.json")
        if not os.path.exists(config_path):
            raise ValueError(f"Model configuration not found at {config_path}")
        
        model_config = ModelConfig.from_file(config_path)
        transformer_config = convert_config(model_config)
        
        # Initialize model with configuration
        model = DecoderOnlyTransformer(transformer_config)
        
        # Load model weights
        model_path = os.path.join(args.model_path, "pytorch_model.bin")
        if not os.path.exists(model_path):
            raise ValueError(f"Model weights not found at {model_path}")
        
        model.load_state_dict(torch.load(model_path, map_location="cpu"))
        logger.info(f"Successfully loaded model from {args.model_path}")
        
        return model, model_config
    else:
        # Create a new model
        logger.info(f"Creating new {args.model_size} model")
        
        # Import model size function
        if args.model_size == "tiny":
            from src.model.config import get_tiny_config as get_config
        elif args.model_size == "small":
            from src.model.config import get_small_config as get_config
        elif args.model_size == "base":
            from src.model.config import get_base_config as get_config
        elif args.model_size == "medium":
            from src.model.config import get_medium_config as get_config
        elif args.model_size == "large":
            from src.model.config import get_large_config as get_config
        else:
            raise ValueError(f"Unknown model size: {args.model_size}")
        
        # Get model configuration
        model_config = get_config()
        
        # Update sequence length
        model_config.max_seq_len = args.seq_length
        
        # Convert to transformer configuration
        transformer_config = convert_config(model_config)
        
        # Initialize model
        model = DecoderOnlyTransformer(transformer_config)
        logger.info(f"Created new {args.model_size} model with {sum(p.numel() for p in model.parameters()):,} parameters")
        
        return model, model_config


def load_or_create_tokenizer(args, vocab_size):
    """
    Load a pre-trained tokenizer or create a new one.
    
    Args:
        args: Command line arguments
        vocab_size: Vocabulary size for new tokenizer
        
    Returns:
        The tokenizer
    """
    if args.tokenizer_path and os.path.exists(args.tokenizer_path):
        # Load pre-trained tokenizer
        logger.info(f"Loading tokenizer from {args.tokenizer_path}")
        tokenizer = BPETokenizer.load(args.tokenizer_path)
        return tokenizer
    else:
        # Need to train a new tokenizer
        # For now, let's create a simple tokenizer
        logger.info(f"Creating new tokenizer with vocabulary size {vocab_size}")
        tokenizer = BPETokenizer(vocab_size=vocab_size)
        
        # We should have some sample text to train the tokenizer
        # For now, we'll rely on the dataset manager to train it
        return tokenizer


def setup_training(args):
    """
    Set up all components for training.
    
    Args:
        args: Command line arguments
        
    Returns:
        Tuple of (model, tokenizer, train_dataloader, val_dataloader, training_config)
    """
    # Set random seed for reproducibility
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    
    # Load or create model
    model, model_config = load_or_create_model(args)
    
    # Load or create tokenizer
    tokenizer = load_or_create_tokenizer(args, model_config.vocab_size)
    
    # Create dataset manager
    data_manager = DatasetManager(
        tokenizer=tokenizer,
        data_dir=args.data_dir,
        seq_length=args.seq_length,
    )
    
    # Prepare dataset
    logger.info(f"Preparing dataset {args.dataset_id}")
    train_dataset, val_dataset = data_manager.prepare_dataset(
        dataset_id=args.dataset_id,
        train_split=args.train_split,
    )
    
    # Create dataloaders
    train_dataloader = data_manager.get_dataloader(
        dataset=train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
    )
    
    val_dataloader = data_manager.get_dataloader(
        dataset=val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
    )
    
    # Create training configuration
    training_config = TrainingConfig(
        batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        max_steps=args.max_steps,
        max_epochs=args.max_epochs,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        warmup_steps=args.warmup_steps,
        lr_scheduler_type=args.lr_scheduler,
        eval_steps=args.eval_steps,
        save_steps=args.save_steps,
        logging_steps=args.logging_steps,
    )
    
    return model, tokenizer, train_dataloader, val_dataloader, training_config


def main():
    """Main function to run training."""
    # Parse arguments
    args = setup_training_args()
    
    # Create output directory with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = os.path.join(args.output_dir, f"{args.model_size}_{args.dataset_id}_{timestamp}")
    os.makedirs(output_dir, exist_ok=True)
    
    # Save arguments
    with open(os.path.join(output_dir, "args.json"), "w") as f:
        json.dump(vars(args), f, indent=2)
    
    # Set up training components
    model, tokenizer, train_dataloader, val_dataloader, training_config = setup_training(args)
    
    # Save tokenizer
    tokenizer_dir = os.path.join(output_dir, "tokenizer")
    os.makedirs(tokenizer_dir, exist_ok=True)
    tokenizer.save(tokenizer_dir)
    logger.info(f"Saved tokenizer to {tokenizer_dir}")
    
    # Create trainer
    trainer = Trainer(
        model=model,
        train_dataloader=train_dataloader,
        config=training_config,
        val_dataloader=val_dataloader,
        output_dir=output_dir,
    )
    
    # Train model
    logger.info("Starting training")
    results = trainer.train()
    
    # Save results
    with open(os.path.join(output_dir, "results.json"), "w") as f:
        json.dump(results, f, indent=2)
    
    logger.info(f"Training completed. Results saved to {output_dir}")
    
    return results


if __name__ == "__main__":
    main() 