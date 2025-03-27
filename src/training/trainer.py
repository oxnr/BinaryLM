"""
Trainer for BinaryLM transformer models.

This module provides a Trainer class that handles model training and fine-tuning,
along with utilities for tracking and visualizing training progress.
"""

import os
import json
import time
import logging
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from typing import Dict, List, Optional, Union, Callable, Tuple, Any
from dataclasses import dataclass, asdict, field
import numpy as np
from datetime import datetime
import threading
import queue

from src.model.transformer import DecoderOnlyTransformer
from src.data.dataset import TextDataset

logger = logging.getLogger(__name__)


@dataclass
class TrainingMetrics:
    """Class to track training metrics."""
    
    # Progress metrics
    step: int = 0
    epoch: int = 0
    total_steps: int = 0
    
    # Loss metrics
    train_loss: float = 0.0
    val_loss: Optional[float] = None
    
    # Learning rate
    learning_rate: float = 0.0
    
    # Speed metrics
    samples_per_second: float = 0.0
    time_elapsed: float = 0.0
    
    # Model performance metrics
    train_perplexity: float = float('inf')
    val_perplexity: Optional[float] = None
    
    # Gradient metrics
    gradient_norm: Optional[float] = None
    
    # Timestamp
    timestamp: float = field(default_factory=time.time)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert metrics to dictionary."""
        return {k: v for k, v in asdict(self).items() if v is not None}


@dataclass
class TrainingConfig:
    """Configuration for training."""
    
    # Training parameters
    batch_size: int = 8
    gradient_accumulation_steps: int = 1
    max_steps: int = 10000
    max_epochs: Optional[int] = None
    
    # Optimization parameters
    learning_rate: float = 5e-5
    weight_decay: float = 0.01
    max_grad_norm: float = 1.0
    
    # Learning rate schedule
    warmup_steps: int = 500
    lr_scheduler_type: str = "cosine"  # cosine, linear, constant
    
    # Evaluation and saving
    eval_steps: int = 500
    save_steps: int = 1000
    
    # Logging
    logging_steps: int = 10
    visualization_port: int = 8000


class MetricsTracker:
    """
    Class to track and broadcast training metrics.
    
    This tracker collects metrics during training and makes them available to
    visualization tools or the API server.
    """
    
    def __init__(self, broadcast: bool = True):
        """
        Initialize the metrics tracker.
        
        Args:
            broadcast: Whether to broadcast metrics for visualization
        """
        self.metrics_history: List[TrainingMetrics] = []
        self.current_metrics: TrainingMetrics = TrainingMetrics()
        self.broadcast = broadcast
        
        # Queue for metrics broadcasting
        self.metrics_queue = queue.Queue()
        
        # For WebSocket broadcast (if enabled)
        if broadcast:
            self._start_broadcaster()
    
    def update(self, **kwargs) -> None:
        """
        Update current metrics.
        
        Args:
            **kwargs: Metric values to update
        """
        for key, value in kwargs.items():
            if hasattr(self.current_metrics, key):
                setattr(self.current_metrics, key, value)
        
        # Update timestamp
        self.current_metrics.timestamp = time.time()
        
        # If perplexity isn't provided but loss is, calculate it
        if 'train_loss' in kwargs and 'train_perplexity' not in kwargs:
            self.current_metrics.train_perplexity = math.exp(self.current_metrics.train_loss)
        
        if 'val_loss' in kwargs and 'val_perplexity' not in kwargs and kwargs['val_loss'] is not None:
            self.current_metrics.val_perplexity = math.exp(self.current_metrics.val_loss)
    
    def log_metrics(self) -> None:
        """Log the current metrics and add to history."""
        # Add current metrics to history
        self.metrics_history.append(self.current_metrics)
        
        # Put metrics in queue for broadcasting
        if self.broadcast:
            self.metrics_queue.put(self.current_metrics.to_dict())
        
        # Log to console
        metrics_str = " | ".join([
            f"Step: {self.current_metrics.step}/{self.current_metrics.total_steps}",
            f"Loss: {self.current_metrics.train_loss:.4f}",
            f"LR: {self.current_metrics.learning_rate:.2e}",
        ])
        
        if self.current_metrics.val_loss is not None:
            metrics_str += f" | Val Loss: {self.current_metrics.val_loss:.4f}"
        
        logger.info(metrics_str)
    
    def get_latest_metrics(self) -> Dict[str, Any]:
        """Get the latest metrics as a dictionary."""
        return self.current_metrics.to_dict()
    
    def get_metrics_history(self) -> List[Dict[str, Any]]:
        """Get the full metrics history as a list of dictionaries."""
        return [metrics.to_dict() for metrics in self.metrics_history]
    
    def plot_metrics(self, output_path: Optional[str] = None) -> None:
        """
        Plot training metrics.
        
        Args:
            output_path: Path to save the plot, or None to display
        """
        try:
            import matplotlib.pyplot as plt
            
            # Extract metrics for plotting
            steps = [m.step for m in self.metrics_history]
            train_losses = [m.train_loss for m in self.metrics_history]
            val_losses = [m.val_loss for m in self.metrics_history if m.val_loss is not None]
            val_steps = [m.step for m in self.metrics_history if m.val_loss is not None]
            
            # Create figure
            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), sharex=True)
            
            # Plot losses
            ax1.plot(steps, train_losses, label="Train Loss")
            if val_losses:
                ax1.plot(val_steps, val_losses, label="Validation Loss")
            ax1.set_ylabel("Loss")
            ax1.set_title("Training and Validation Loss")
            ax1.legend()
            ax1.grid(True)
            
            # Plot learning rate
            learning_rates = [m.learning_rate for m in self.metrics_history]
            ax2.plot(steps, learning_rates)
            ax2.set_xlabel("Steps")
            ax2.set_ylabel("Learning Rate")
            ax2.set_title("Learning Rate Schedule")
            ax2.grid(True)
            
            plt.tight_layout()
            
            # Save or display
            if output_path:
                plt.savefig(output_path)
                logger.info(f"Saved metrics plot to {output_path}")
            else:
                plt.show()
        
        except ImportError:
            logger.warning("Matplotlib not installed. Cannot plot metrics.")
    
    def _start_broadcaster(self) -> None:
        """Start a thread to broadcast metrics."""
        # This is a placeholder for actual WebSocket broadcast
        # In a real implementation, this would connect to a WebSocket server
        # or some other communication mechanism to send metrics to the frontend
        
        def _broadcaster():
            while True:
                try:
                    # Wait for new metrics
                    metrics = self.metrics_queue.get(timeout=1.0)
                    
                    # In a real implementation, we would send these metrics
                    # to connected clients via WebSocket or another mechanism
                    # For now, we'll just log that we would broadcast them
                    logger.debug(f"Would broadcast metrics: {metrics}")
                    
                    self.metrics_queue.task_done()
                except queue.Empty:
                    # No new metrics, just continue
                    continue
                except Exception as e:
                    logger.error(f"Error in metrics broadcaster: {e}")
        
        # Start broadcaster thread
        thread = threading.Thread(target=_broadcaster, daemon=True)
        thread.start()


class Trainer:
    """
    Trainer for transformer models.
    
    This class handles the training and evaluation of transformer models
    on text datasets, with support for training visualization.
    """
    
    def __init__(
        self,
        model: DecoderOnlyTransformer,
        train_dataloader: DataLoader,
        config: TrainingConfig,
        val_dataloader: Optional[DataLoader] = None,
        output_dir: Optional[str] = None,
    ):
        """
        Initialize the trainer.
        
        Args:
            model: The transformer model to train
            train_dataloader: DataLoader for training data
            config: Training configuration
            val_dataloader: Optional DataLoader for validation data
            output_dir: Directory to save model checkpoints and logs
        """
        self.model = model
        self.train_dataloader = train_dataloader
        self.val_dataloader = val_dataloader
        self.config = config
        self.output_dir = output_dir or os.path.join(os.getcwd(), "training_output")
        
        # Create output directory
        os.makedirs(self.output_dir, exist_ok=True)
        
        # Setup device
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"Using device: {self.device}")
        
        # Move model to device
        self.model.to(self.device)
        
        # Setup optimizer with weight decay
        no_decay = ["bias", "LayerNorm.weight"]
        optimizer_grouped_parameters = [
            {
                "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
                "weight_decay": config.weight_decay,
            },
            {
                "params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
                "weight_decay": 0.0,
            },
        ]
        
        self.optimizer = optim.AdamW(
            optimizer_grouped_parameters,
            lr=config.learning_rate,
            betas=(0.9, 0.999),
            eps=1e-8,
        )
        
        # Calculate total number of training steps
        self.total_steps = config.max_steps
        if config.max_epochs is not None:
            # Calculate steps per epoch
            steps_per_epoch = len(train_dataloader) // config.gradient_accumulation_steps
            self.total_steps = min(config.max_steps, steps_per_epoch * config.max_epochs)
        
        # Setup learning rate scheduler
        if config.lr_scheduler_type == "cosine":
            self.lr_scheduler = self._get_cosine_schedule_with_warmup(
                self.optimizer,
                num_warmup_steps=config.warmup_steps,
                num_training_steps=self.total_steps,
            )
        elif config.lr_scheduler_type == "linear":
            self.lr_scheduler = self._get_linear_schedule_with_warmup(
                self.optimizer,
                num_warmup_steps=config.warmup_steps,
                num_training_steps=self.total_steps,
            )
        else:  # constant
            self.lr_scheduler = self._get_constant_schedule_with_warmup(
                self.optimizer,
                num_warmup_steps=config.warmup_steps,
            )
        
        # Setup metrics tracker
        self.metrics_tracker = MetricsTracker(broadcast=True)
        self.metrics_tracker.update(total_steps=self.total_steps)
        
        # Training state
        self.global_step = 0
        self.epoch = 0
        self.best_val_loss = float('inf')
        
        # Save training config
        self._save_config()
    
    def _save_config(self) -> None:
        """Save training configuration to disk."""
        config_path = os.path.join(self.output_dir, "training_config.json")
        with open(config_path, "w") as f:
            json.dump(asdict(self.config), f, indent=2)
        logger.info(f"Saved training configuration to {config_path}")
    
    def _get_cosine_schedule_with_warmup(
        self,
        optimizer: torch.optim.Optimizer,
        num_warmup_steps: int,
        num_training_steps: int,
        num_cycles: float = 0.5,
        last_epoch: int = -1,
    ) -> torch.optim.lr_scheduler.LambdaLR:
        """
        Create a cosine learning rate scheduler with warmup.
        
        Args:
            optimizer: The optimizer
            num_warmup_steps: Number of warmup steps
            num_training_steps: Total number of training steps
            num_cycles: Number of cycles for cosine decay
            last_epoch: The index of the last epoch
            
        Returns:
            Learning rate scheduler
        """
        
        def lr_lambda(current_step):
            if current_step < num_warmup_steps:
                return float(current_step) / float(max(1, num_warmup_steps))
            
            progress = float(current_step - num_warmup_steps) / float(max(1, num_training_steps - num_warmup_steps))
            return max(0.0, 0.5 * (1.0 + math.cos(math.pi * num_cycles * 2.0 * progress)))
        
        return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda, last_epoch)
    
    def _get_linear_schedule_with_warmup(
        self,
        optimizer: torch.optim.Optimizer,
        num_warmup_steps: int,
        num_training_steps: int,
        last_epoch: int = -1,
    ) -> torch.optim.lr_scheduler.LambdaLR:
        """
        Create a linear learning rate scheduler with warmup.
        
        Args:
            optimizer: The optimizer
            num_warmup_steps: Number of warmup steps
            num_training_steps: Total number of training steps
            last_epoch: The index of the last epoch
            
        Returns:
            Learning rate scheduler
        """
        
        def lr_lambda(current_step):
            if current_step < num_warmup_steps:
                return float(current_step) / float(max(1, num_warmup_steps))
            
            return max(0.0, float(num_training_steps - current_step) / float(max(1, num_training_steps - num_warmup_steps)))
        
        return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda, last_epoch)
    
    def _get_constant_schedule_with_warmup(
        self,
        optimizer: torch.optim.Optimizer,
        num_warmup_steps: int,
        last_epoch: int = -1,
    ) -> torch.optim.lr_scheduler.LambdaLR:
        """
        Create a constant learning rate scheduler with warmup.
        
        Args:
            optimizer: The optimizer
            num_warmup_steps: Number of warmup steps
            last_epoch: The index of the last epoch
            
        Returns:
            Learning rate scheduler
        """
        
        def lr_lambda(current_step):
            if current_step < num_warmup_steps:
                return float(current_step) / float(max(1, num_warmup_steps))
            
            return 1.0
        
        return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda, last_epoch)
    
    def train(self) -> Dict[str, Any]:
        """
        Train the model.
        
        Returns:
            Dictionary of training results and metrics
        """
        logger.info("Starting training")
        
        # Track training start time
        start_time = time.time()
        
        # Initialize training state
        self.model.train()
        self.global_step = 0
        self.epoch = 0
        
        # Training loop
        accumulated_loss = 0.0
        samples_seen = 0
        gradient_accumulation_counter = 0
        
        while self.global_step < self.total_steps:
            self.epoch += 1
            logger.info(f"Starting epoch {self.epoch}")
            
            for batch_idx, batch in enumerate(self.train_dataloader):
                # Move batch to device
                batch = {k: v.to(self.device) for k, v in batch.items()}
                
                # Forward pass
                outputs = self.model(batch["input_ids"])
                
                # Compute loss (causal language modeling)
                # outputs shape: (batch_size, seq_len, vocab_size)
                # targets shape: (batch_size, seq_len)
                loss = F.cross_entropy(
                    outputs.view(-1, outputs.size(-1)),
                    batch["labels"].view(-1),
                    ignore_index=-100,  # Ignore padding tokens
                )
                
                # Scale loss for gradient accumulation
                loss = loss / self.config.gradient_accumulation_steps
                
                # Backward pass
                loss.backward()
                
                # Update accumulated loss and counter
                accumulated_loss += loss.item() * self.config.gradient_accumulation_steps
                samples_seen += batch["input_ids"].size(0)
                gradient_accumulation_counter += 1
                
                # Only update weights after accumulating gradients
                if gradient_accumulation_counter >= self.config.gradient_accumulation_steps:
                    # Clip gradients
                    if self.config.max_grad_norm > 0:
                        grad_norm = torch.nn.utils.clip_grad_norm_(
                            self.model.parameters(),
                            self.config.max_grad_norm,
                        )
                    else:
                        grad_norm = self._compute_grad_norm()
                    
                    # Optimizer step
                    self.optimizer.step()
                    self.lr_scheduler.step()
                    self.optimizer.zero_grad()
                    
                    # Update global step
                    self.global_step += 1
                    
                    # Update metrics
                    current_lr = self.lr_scheduler.get_last_lr()[0]
                    elapsed_time = time.time() - start_time
                    samples_per_second = samples_seen / elapsed_time if elapsed_time > 0 else 0
                    
                    self.metrics_tracker.update(
                        step=self.global_step,
                        epoch=self.epoch,
                        train_loss=accumulated_loss / gradient_accumulation_counter,
                        learning_rate=current_lr,
                        gradient_norm=grad_norm.item() if isinstance(grad_norm, torch.Tensor) else grad_norm,
                        samples_per_second=samples_per_second,
                        time_elapsed=elapsed_time,
                    )
                    
                    # Log metrics
                    if self.global_step % self.config.logging_steps == 0:
                        self.metrics_tracker.log_metrics()
                    
                    # Evaluate model
                    if self.global_step % self.config.eval_steps == 0 and self.val_dataloader is not None:
                        val_loss = self.evaluate()
                        self.metrics_tracker.update(val_loss=val_loss)
                        self.metrics_tracker.log_metrics()
                        
                        # Save best model
                        if val_loss < self.best_val_loss:
                            self.best_val_loss = val_loss
                            self.save_model(os.path.join(self.output_dir, "best_model"))
                            logger.info(f"New best validation loss: {val_loss:.4f}")
                        
                        # Set model back to training mode
                        self.model.train()
                    
                    # Save checkpoint
                    if self.global_step % self.config.save_steps == 0:
                        self.save_checkpoint()
                    
                    # Reset accumulators
                    accumulated_loss = 0.0
                    samples_seen = 0
                    gradient_accumulation_counter = 0
                
                # Check if we've reached max steps
                if self.global_step >= self.total_steps:
                    break
            
            logger.info(f"Completed epoch {self.epoch}")
            
            # Check if we've reached max steps
            if self.global_step >= self.total_steps:
                break
        
        # Final evaluation
        if self.val_dataloader is not None:
            val_loss = self.evaluate()
            self.metrics_tracker.update(val_loss=val_loss)
            self.metrics_tracker.log_metrics()
        
        # Save final model
        self.save_model(os.path.join(self.output_dir, "final_model"))
        
        # Save training metrics plot
        plot_path = os.path.join(self.output_dir, "training_metrics.png")
        self.metrics_tracker.plot_metrics(plot_path)
        
        logger.info(f"Training completed in {time.time() - start_time:.2f} seconds")
        
        # Return training results
        return {
            "best_val_loss": self.best_val_loss,
            "final_val_loss": val_loss if self.val_dataloader is not None else None,
            "train_loss": self.metrics_tracker.current_metrics.train_loss,
            "epochs": self.epoch,
            "steps": self.global_step,
        }
    
    def evaluate(self) -> float:
        """
        Evaluate the model on validation data.
        
        Returns:
            Validation loss
        """
        logger.info("Running evaluation")
        
        if self.val_dataloader is None:
            logger.warning("No validation dataloader provided, skipping evaluation")
            return 0.0
        
        # Set model to evaluation mode
        self.model.eval()
        
        total_loss = 0.0
        total_samples = 0
        
        with torch.no_grad():
            for batch in self.val_dataloader:
                # Move batch to device
                batch = {k: v.to(self.device) for k, v in batch.items()}
                
                # Forward pass
                outputs = self.model(batch["input_ids"])
                
                # Compute loss
                loss = F.cross_entropy(
                    outputs.view(-1, outputs.size(-1)),
                    batch["labels"].view(-1),
                    ignore_index=-100,  # Ignore padding tokens
                )
                
                # Update total loss and samples
                total_loss += loss.item() * batch["input_ids"].size(0)
                total_samples += batch["input_ids"].size(0)
        
        # Calculate average loss
        avg_loss = total_loss / total_samples if total_samples > 0 else 0.0
        
        logger.info(f"Validation loss: {avg_loss:.4f}")
        
        return avg_loss
    
    def save_checkpoint(self) -> None:
        """Save a training checkpoint."""
        checkpoint_dir = os.path.join(self.output_dir, f"checkpoint-{self.global_step}")
        os.makedirs(checkpoint_dir, exist_ok=True)
        
        # Save model
        self.save_model(checkpoint_dir)
        
        # Save optimizer and scheduler state
        torch.save(self.optimizer.state_dict(), os.path.join(checkpoint_dir, "optimizer.pt"))
        torch.save(self.lr_scheduler.state_dict(), os.path.join(checkpoint_dir, "scheduler.pt"))
        
        # Save training state
        training_state = {
            "global_step": self.global_step,
            "epoch": self.epoch,
            "best_val_loss": self.best_val_loss,
            "metrics_history": self.metrics_tracker.get_metrics_history(),
        }
        
        with open(os.path.join(checkpoint_dir, "training_state.json"), "w") as f:
            json.dump(training_state, f, indent=2)
        
        logger.info(f"Saved checkpoint to {checkpoint_dir}")
    
    def save_model(self, output_dir: str) -> None:
        """
        Save the model to disk.
        
        Args:
            output_dir: Directory to save the model
        """
        os.makedirs(output_dir, exist_ok=True)
        
        # Save model weights
        torch.save(self.model.state_dict(), os.path.join(output_dir, "pytorch_model.bin"))
        
        logger.info(f"Saved model to {output_dir}")
    
    def load_checkpoint(self, checkpoint_dir: str) -> None:
        """
        Load a training checkpoint.
        
        Args:
            checkpoint_dir: Directory containing the checkpoint
        """
        # Load model
        model_path = os.path.join(checkpoint_dir, "pytorch_model.bin")
        if os.path.exists(model_path):
            self.model.load_state_dict(torch.load(model_path, map_location=self.device))
            logger.info(f"Loaded model from {model_path}")
        else:
            logger.warning(f"Model file not found at {model_path}")
        
        # Load optimizer
        optimizer_path = os.path.join(checkpoint_dir, "optimizer.pt")
        if os.path.exists(optimizer_path):
            self.optimizer.load_state_dict(torch.load(optimizer_path, map_location=self.device))
            logger.info(f"Loaded optimizer from {optimizer_path}")
        else:
            logger.warning(f"Optimizer file not found at {optimizer_path}")
        
        # Load scheduler
        scheduler_path = os.path.join(checkpoint_dir, "scheduler.pt")
        if os.path.exists(scheduler_path):
            self.lr_scheduler.load_state_dict(torch.load(scheduler_path, map_location=self.device))
            logger.info(f"Loaded scheduler from {scheduler_path}")
        else:
            logger.warning(f"Scheduler file not found at {scheduler_path}")
        
        # Load training state
        state_path = os.path.join(checkpoint_dir, "training_state.json")
        if os.path.exists(state_path):
            with open(state_path, "r") as f:
                training_state = json.load(f)
            
            self.global_step = training_state["global_step"]
            self.epoch = training_state["epoch"]
            self.best_val_loss = training_state["best_val_loss"]
            
            logger.info(f"Loaded training state from {state_path}")
            logger.info(f"Resuming from step {self.global_step}, epoch {self.epoch}")
        else:
            logger.warning(f"Training state file not found at {state_path}")
    
    def _compute_grad_norm(self) -> float:
        """
        Compute the gradient norm for all parameters.
        
        Returns:
            The gradient norm
        """
        total_norm = 0.0
        for p in self.model.parameters():
            if p.grad is not None:
                param_norm = p.grad.data.norm(2)
                total_norm += param_norm.item() ** 2
        
        total_norm = total_norm ** 0.5
        return total_norm
``` 