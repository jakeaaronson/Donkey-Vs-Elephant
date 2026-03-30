"""
Training loop for fine-tuning GPT-2 on political speech data.

Handles mixed-precision training via HuggingFace Accelerate, checkpoint
save/resume, and loss logging. Designed to survive interruptions gracefully
since fine-tuning runs can take hours to days.
"""

from __future__ import annotations

import json
import logging
import math
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Optional

import torch
from accelerate import Accelerator
from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset
from transformers import (
    GPT2LMHeadModel,
    GPT2Tokenizer,
    get_linear_schedule_with_warmup,
)

logger = logging.getLogger(__name__)


@dataclass
class TrainingConfig:
    """Hyperparameters and paths for a training run."""

    # Model
    model_name: str = "gpt2"

    # Training
    epochs: int = 3
    batch_size: int = 4
    gradient_accumulation_steps: int = 8
    learning_rate: float = 5e-5
    weight_decay: float = 0.01
    warmup_ratio: float = 0.1
    max_grad_norm: float = 1.0
    max_length: int = 1024

    # Data
    val_fraction: float = 0.1
    seed: int = 42

    # Paths (relative to project root)
    data_dir: str = "data/processed"
    log_dir: str = "logs"
    checkpoint_dir: str = "checkpoints"
    output_dir: str = "models"

    # Logging
    log_every_n_steps: int = 10

    @classmethod
    def from_json(cls, path: str | Path) -> TrainingConfig:
        """Load config from a JSON file, overriding defaults."""
        with open(path, "r", encoding="utf-8") as f:
            overrides = json.load(f)
        return cls(**{k: v for k, v in overrides.items() if k in cls.__dataclass_fields__})


@dataclass
class TrainingState:
    """Mutable state that gets checkpointed for resumption."""

    epoch: int = 0
    global_step: int = 0
    best_val_loss: float = float("inf")
    train_losses: list[float] = field(default_factory=list)
    val_losses: list[float] = field(default_factory=list)


class PoliticalSpeechTrainer:
    """Fine-tunes GPT-2 on political speech data with checkpoint support.

    Args:
        party: "donkey" (Democratic) or "elephant" (Republican).
        config: Training hyperparameters and paths.
        project_root: Absolute path to the project root directory.
        resume_from: Path to a checkpoint directory to resume from.
    """

    def __init__(
        self,
        party: str,
        config: TrainingConfig,
        project_root: str | Path,
        resume_from: Optional[str | Path] = None,
    ) -> None:
        if party not in ("donkey", "elephant"):
            raise ValueError(f"party must be 'donkey' or 'elephant', got '{party}'")

        self.party = party
        self.config = config
        self.root = Path(project_root)
        self.state = TrainingState()

        # Resolve directories
        self.data_dir = self.root / config.data_dir
        self.log_dir = self.root / config.log_dir / party
        self.checkpoint_dir = self.root / config.checkpoint_dir / party
        self.output_dir = self.root / config.output_dir / party

        for d in [self.log_dir, self.checkpoint_dir, self.output_dir]:
            d.mkdir(parents=True, exist_ok=True)

        # Initialize accelerator for mixed precision
        self.accelerator = Accelerator(
            gradient_accumulation_steps=config.gradient_accumulation_steps,
            mixed_precision="fp16",
            log_with="all",
        )

        # Load model and tokenizer
        self.tokenizer = GPT2Tokenizer.from_pretrained(config.model_name)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        self.model = GPT2LMHeadModel.from_pretrained(config.model_name)
        self.model.resize_token_embeddings(len(self.tokenizer))

        self._resume_path = Path(resume_from) if resume_from else None

    def _build_dataloaders(
        self, train_dataset: Dataset, val_dataset: Dataset
    ) -> tuple[DataLoader, DataLoader]:
        """Create DataLoaders with appropriate settings."""
        train_loader = DataLoader(
            train_dataset,
            batch_size=self.config.batch_size,
            shuffle=True,
            num_workers=2,
            pin_memory=True,
            drop_last=True,
        )
        val_loader = DataLoader(
            val_dataset,
            batch_size=self.config.batch_size,
            shuffle=False,
            num_workers=2,
            pin_memory=True,
        )
        return train_loader, val_loader

    def _build_optimizer_and_scheduler(
        self, num_training_steps: int
    ) -> tuple[AdamW, torch.optim.lr_scheduler.LambdaLR]:
        """Set up AdamW optimizer with linear warmup schedule."""
        # Separate weight decay for bias and LayerNorm
        no_decay = {"bias", "LayerNorm.weight"}
        param_groups = [
            {
                "params": [
                    p
                    for n, p in self.model.named_parameters()
                    if not any(nd in n for nd in no_decay)
                ],
                "weight_decay": self.config.weight_decay,
            },
            {
                "params": [
                    p
                    for n, p in self.model.named_parameters()
                    if any(nd in n for nd in no_decay)
                ],
                "weight_decay": 0.0,
            },
        ]

        optimizer = AdamW(param_groups, lr=self.config.learning_rate)

        warmup_steps = int(num_training_steps * self.config.warmup_ratio)
        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=warmup_steps,
            num_training_steps=num_training_steps,
        )

        return optimizer, scheduler

    def _save_checkpoint(self, optimizer: AdamW, scheduler: object) -> Path:
        """Save model, optimizer, scheduler, and training state."""
        ckpt_path = self.checkpoint_dir / f"step_{self.state.global_step}"
        ckpt_path.mkdir(parents=True, exist_ok=True)

        self.accelerator.wait_for_everyone()
        unwrapped = self.accelerator.unwrap_model(self.model)
        unwrapped.save_pretrained(ckpt_path, save_function=self.accelerator.save)
        self.tokenizer.save_pretrained(ckpt_path)

        # Save optimizer and scheduler state
        self.accelerator.save(optimizer.state_dict(), ckpt_path / "optimizer.pt")
        self.accelerator.save(scheduler.state_dict(), ckpt_path / "scheduler.pt")

        # Save training state
        with open(ckpt_path / "training_state.json", "w", encoding="utf-8") as f:
            json.dump(asdict(self.state), f, indent=2)

        logger.info("Checkpoint saved to %s", ckpt_path)
        return ckpt_path

    def _load_checkpoint(
        self, optimizer: AdamW, scheduler: object
    ) -> None:
        """Restore model, optimizer, scheduler, and training state from checkpoint."""
        ckpt_path = self._resume_path
        if ckpt_path is None:
            return

        if not ckpt_path.exists():
            # Try resolving as relative to checkpoint_dir
            ckpt_path = self.checkpoint_dir / ckpt_path.name
            if not ckpt_path.exists():
                logger.warning("Checkpoint %s not found, starting fresh", self._resume_path)
                return

        logger.info("Resuming from checkpoint: %s", ckpt_path)

        # Restore model weights
        self.model = GPT2LMHeadModel.from_pretrained(ckpt_path)
        self.model.resize_token_embeddings(len(self.tokenizer))

        # Restore optimizer and scheduler
        opt_path = ckpt_path / "optimizer.pt"
        sched_path = ckpt_path / "scheduler.pt"
        if opt_path.exists():
            optimizer.load_state_dict(torch.load(opt_path, map_location="cpu", weights_only=True))
        if sched_path.exists():
            scheduler.load_state_dict(
                torch.load(sched_path, map_location="cpu", weights_only=True)
            )

        # Restore training state
        state_path = ckpt_path / "training_state.json"
        if state_path.exists():
            with open(state_path, "r", encoding="utf-8") as f:
                state_dict = json.load(f)
            self.state = TrainingState(**state_dict)
            logger.info(
                "Resumed at epoch %d, step %d", self.state.epoch, self.state.global_step
            )

    def _log_metrics(self, metrics: dict[str, float], step: int) -> None:
        """Append metrics to the log file as JSON lines."""
        log_file = self.log_dir / "train_log.jsonl"
        entry = {"step": step, "timestamp": time.time(), **metrics}
        with open(log_file, "a", encoding="utf-8") as f:
            f.write(json.dumps(entry) + "\n")

    @torch.no_grad()
    def _evaluate(self, val_loader: DataLoader) -> float:
        """Run validation and return average loss."""
        self.model.eval()
        total_loss = 0.0
        num_batches = 0

        for batch in val_loader:
            outputs = self.model(**batch)
            total_loss += outputs.loss.item()
            num_batches += 1

        avg_loss = total_loss / max(num_batches, 1)
        self.model.train()
        return avg_loss

    def train(self, train_dataset: Dataset, val_dataset: Dataset) -> TrainingState:
        """Run the full training loop.

        Args:
            train_dataset: Training split from PoliticalSpeechDataset.
            val_dataset: Validation split.

        Returns:
            Final TrainingState with loss history.
        """
        train_loader, val_loader = self._build_dataloaders(train_dataset, val_dataset)

        num_update_steps = (
            math.ceil(len(train_loader) / self.config.gradient_accumulation_steps)
            * self.config.epochs
        )
        optimizer, scheduler = self._build_optimizer_and_scheduler(num_update_steps)

        # Resume if checkpoint provided
        if self._resume_path:
            self._load_checkpoint(optimizer, scheduler)

        # Prepare with accelerator (handles device placement, mixed precision)
        self.model, optimizer, train_loader, val_loader, scheduler = (
            self.accelerator.prepare(
                self.model, optimizer, train_loader, val_loader, scheduler
            )
        )

        # Save config for reproducibility
        config_path = self.log_dir / "config.json"
        with open(config_path, "w", encoding="utf-8") as f:
            json.dump(asdict(self.config), f, indent=2)

        logger.info("Starting training for '%s' model", self.party)
        logger.info("  Epochs: %d", self.config.epochs)
        logger.info("  Batch size: %d", self.config.batch_size)
        logger.info("  Gradient accumulation steps: %d", self.config.gradient_accumulation_steps)
        logger.info(
            "  Effective batch size: %d",
            self.config.batch_size * self.config.gradient_accumulation_steps,
        )
        logger.info("  Total optimization steps: %d", num_update_steps)
        logger.info("  Train examples: %d", len(train_dataset))
        logger.info("  Val examples: %d", len(val_dataset))

        start_time = time.time()
        start_epoch = self.state.epoch

        for epoch in range(start_epoch, self.config.epochs):
            self.state.epoch = epoch
            self.model.train()
            epoch_loss = 0.0
            epoch_steps = 0

            for step, batch in enumerate(train_loader):
                with self.accelerator.accumulate(self.model):
                    outputs = self.model(**batch)
                    loss = outputs.loss

                    self.accelerator.backward(loss)

                    if self.accelerator.sync_gradients:
                        self.accelerator.clip_grad_norm_(
                            self.model.parameters(), self.config.max_grad_norm
                        )

                    optimizer.step()
                    scheduler.step()
                    optimizer.zero_grad()

                epoch_loss += loss.item()
                epoch_steps += 1

                if self.accelerator.sync_gradients:
                    self.state.global_step += 1

                    if self.state.global_step % self.config.log_every_n_steps == 0:
                        avg_loss = epoch_loss / epoch_steps
                        lr = scheduler.get_last_lr()[0]
                        elapsed = time.time() - start_time

                        self._log_metrics(
                            {"train_loss": avg_loss, "learning_rate": lr, "epoch": epoch},
                            self.state.global_step,
                        )
                        logger.info(
                            "Step %d | Epoch %d | Loss: %.4f | LR: %.2e | Time: %.0fs",
                            self.state.global_step,
                            epoch,
                            avg_loss,
                            lr,
                            elapsed,
                        )

            # End of epoch: validate and checkpoint
            avg_train_loss = epoch_loss / max(epoch_steps, 1)
            val_loss = self._evaluate(val_loader)

            self.state.train_losses.append(avg_train_loss)
            self.state.val_losses.append(val_loss)

            self._log_metrics(
                {"epoch_train_loss": avg_train_loss, "epoch_val_loss": val_loss, "epoch": epoch},
                self.state.global_step,
            )
            logger.info(
                "Epoch %d complete | Train loss: %.4f | Val loss: %.4f",
                epoch,
                avg_train_loss,
                val_loss,
            )

            # Save checkpoint after every epoch
            self._save_checkpoint(optimizer, scheduler)

            # Track best model
            if val_loss < self.state.best_val_loss:
                self.state.best_val_loss = val_loss
                logger.info("New best val loss: %.4f", val_loss)

        # Save final model
        self._save_final_model()

        total_time = time.time() - start_time
        logger.info("Training complete in %.1f minutes", total_time / 60)

        return self.state

    def _save_final_model(self) -> None:
        """Save the final fine-tuned model and tokenizer."""
        self.accelerator.wait_for_everyone()
        unwrapped = self.accelerator.unwrap_model(self.model)
        unwrapped.save_pretrained(self.output_dir, save_function=self.accelerator.save)
        self.tokenizer.save_pretrained(self.output_dir)
        logger.info("Final model saved to %s", self.output_dir)
