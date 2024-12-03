import os
import torch
import torch.nn as nn
from torch.utils.data import IterableDataset, DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from datasets import load_dataset
from transformers import AutoTokenizer
from typing import Dict, List, Optional, Tuple, Union, Iterator
from dataclasses import dataclass
from loguru import logger
import wandb
from tqdm.auto import tqdm
import numpy as np
from pathlib import Path
from lfm_torch.liquid_t_moe import LiquidTransformer

# Set tokenizer parallelism
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# Configure logging
logger.add(
    "training.log",
    rotation="500 MB",
    retention="10 days",
    level="INFO",
    format="{time:YYYY-MM-DD HH:mm:ss} | {level} | {message}"
)

@dataclass
class TrainingConfig:
    """Training configuration parameters."""
    
    # Model parameters
    embed_size: int = 768  # Match BERT embedding size
    num_heads: int = 8
    num_experts: int = 4
    expert_size: int = 768  # Match embed_size
    num_layers: int = 6
    
    # Training parameters
    batch_size: int = 16
    learning_rate: float = 1e-4
    max_steps: int = 100000
    warmup_steps: int = 1000
    max_grad_norm: float = 1.0
    weight_decay: float = 0.01
    
    # Data parameters
    max_length: int = 512
    vocab_size: int = 30522  # BERT vocab size
    
    # System parameters
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    num_workers: int = 0  # Avoid multiprocessing issues with streaming
    seed: int = 42
    
    # Logging parameters
    wandb_project: str = "liquid-transformer"
    checkpoint_dir: str = "checkpoints"
    checkpoint_steps: int = 1000
    log_steps: int = 10

class ArXivDataset(IterableDataset):
    """Dataset class for arXiv papers."""
    
    def __init__(
        self,
        tokenizer: AutoTokenizer,
        max_length: int = 512,
    ):
        super().__init__()
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.dataset = load_dataset("neuralwork/arxiver", split="train", streaming=True)
        logger.info(f"Initialized streaming dataset")
    
    def preprocess_text(self, text: str) -> str:
        """Clean and preprocess text."""
        return text.strip().replace('\n', ' ')

    def __iter__(self) -> Iterator[Dict[str, torch.Tensor]]:
        """Iterate over the dataset."""
        iterator = iter(self.dataset)
        while True:
            try:
                item = next(iterator)
                text = f"Title: {self.preprocess_text(item['title'])} Abstract: {self.preprocess_text(item['abstract'])}"
                
                encoded = self.tokenizer(
                    text,
                    max_length=self.max_length,
                    padding="max_length",
                    truncation=True,
                    return_tensors="pt"
                )
                
                # Keep as long tensor for input ids
                yield {
                    "input_ids": encoded["input_ids"][0],
                    "attention_mask": encoded["attention_mask"][0]
                }
            except StopIteration:
                iterator = iter(self.dataset)  # Restart iteration
                continue

class Trainer:
    """Trainer class for Liquid Transformer."""
    
    def __init__(
        self,
        model: nn.Module,
        config: TrainingConfig,
        tokenizer: AutoTokenizer
    ):
        self.model = model.to(config.device)
        self.config = config
        self.tokenizer = tokenizer
        
        # Initialize hidden state
        self.model.hidden_state = torch.zeros(
            config.batch_size,
            config.embed_size,
            device=config.device
        )
        
        # Create embedding layer for input tokens
        self.embedding = nn.Embedding(
            config.vocab_size,
            config.embed_size
        ).to(config.device)
        
        self.optimizer = AdamW(
            list(model.parameters()) + list(self.embedding.parameters()),
            lr=config.learning_rate,
            weight_decay=config.weight_decay
        )
        
        self.scheduler = CosineAnnealingLR(
            self.optimizer,
            T_max=config.max_steps
        )
        
        wandb.init(project=config.wandb_project, config=vars(config))
        os.makedirs(config.checkpoint_dir, exist_ok=True)
        logger.info("Trainer initialized successfully")
    
    def train_step(
        self,
        batch: Dict[str, torch.Tensor]
    ) -> float:
        """Perform a single training step."""
        try:
            self.model.train()
            
            # Move batch to device
            input_ids = batch["input_ids"].to(self.config.device)
            attention_mask = batch["attention_mask"].to(self.config.device)
            
            # Convert input tokens to embeddings
            embedded_input = self.embedding(input_ids)  # [batch_size, seq_len, embed_size]
            
            # Add sequence dimension expected by transformer
            embedded_input = embedded_input.unsqueeze(0)  # [1, batch_size, seq_len, embed_size]
            
            # Update hidden state size if batch size changed
            if self.model.hidden_state.size(0) != embedded_input.size(1):
                self.model.hidden_state = self.model.hidden_state.new_zeros(
                    embedded_input.size(1),
                    self.config.embed_size
                )
            
            # Forward pass
            outputs = self.model(embedded_input)
            
            # Compute reconstruction loss
            loss = nn.MSELoss()(outputs, embedded_input)
            
            # Backward pass
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                list(self.model.parameters()) + list(self.embedding.parameters()),
                self.config.max_grad_norm
            )
            
            self.optimizer.step()
            self.optimizer.zero_grad()
            
            return loss.item()
            
        except Exception as e:
            logger.error(f"Error in train_step: {str(e)}")
            raise
    
    def save_checkpoint(
        self,
        step: int,
        loss: Optional[float] = None,
    ):
        """Save model checkpoint."""
        checkpoint = {
            "step": step,
            "model_state_dict": self.model.state_dict(),
            "embedding_state_dict": self.embedding.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scheduler_state_dict": self.scheduler.state_dict(),
            "loss": loss if loss is not None else float('inf'),
            "config": self.config
        }
        
        path = Path(self.config.checkpoint_dir)
        checkpoint_path = path / f"checkpoint_step_{step}.pt"
        torch.save(checkpoint, checkpoint_path)
        logger.info(f"Saved checkpoint at step {step} to {checkpoint_path}")
    
    def train(
        self,
        train_dataset: ArXivDataset,
    ):
        """Train the model."""
        logger.info("Starting training")
        
        train_loader = DataLoader(
            train_dataset,
            batch_size=self.config.batch_size,
            num_workers=self.config.num_workers
        )
        
        global_step = 0
        running_loss = 0.0
        current_loss = None
        
        progress_bar = tqdm(total=self.config.max_steps, desc="Training")
        
        try:
            for batch in train_loader:
                if global_step >= self.config.max_steps:
                    break
                
                current_loss = self.train_step(batch)
                running_loss += current_loss
                global_step += 1
                
                # Update progress bar
                progress_bar.update(1)
                progress_bar.set_postfix({
                    "loss": f"{current_loss:.4f}",
                    "step": global_step
                })
                
                # Log metrics
                if global_step % self.config.log_steps == 0:
                    avg_loss = running_loss / self.config.log_steps
                    wandb.log({
                        "train_loss": avg_loss,
                        "learning_rate": self.scheduler.get_last_lr()[0],
                        "global_step": global_step
                    })
                    running_loss = 0.0
                
                # Save checkpoint if needed
                if global_step % self.config.checkpoint_steps == 0:
                    self.save_checkpoint(global_step, current_loss)
                
                # Update learning rate
                self.scheduler.step()
                
        except KeyboardInterrupt:
            logger.info("Training interrupted by user")
            self.save_checkpoint(global_step, current_loss)
        except Exception as e:
            logger.error(f"Training error: {str(e)}")
            self.save_checkpoint(global_step, current_loss)
            raise
        finally:
            progress_bar.close()
            # Save final checkpoint
            self.save_checkpoint(global_step, current_loss)
            logger.info(f"Training completed after {global_step} steps")

def main():
    """Main training function."""
    try:
        # Set random seeds
        config = TrainingConfig()
        torch.manual_seed(config.seed)
        np.random.seed(config.seed)
        
        # Initialize tokenizer
        tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
        
        # Create dataset
        train_dataset = ArXivDataset(
            tokenizer=tokenizer,
            max_length=config.max_length,
        )
        
        # Initialize model
        model = LiquidTransformer(
            embed_size=config.embed_size,
            num_heads=config.num_heads,
            num_experts=config.num_experts,
            expert_size=config.expert_size,
            num_layers=config.num_layers
        )
        
        # Initialize trainer
        trainer = Trainer(model, config, tokenizer)
        
        # Start training
        trainer.train(train_dataset)
        
    except Exception as e:
        logger.error(f"Training failed with error: {str(e)}")
        raise
    finally:
        wandb.finish()

if __name__ == "__main__":
    main()