#!/usr/bin/env python3
"""
CLI entry point for fine-tuning GPT-2 on political speech data.

Usage:
    # Train the Democratic model
    python -m model_training.train --model donkey

    # Train the Republican model
    python -m model_training.train --model elephant

    # Train both sequentially
    python -m model_training.train --both

    # Resume from a checkpoint
    python -m model_training.train --model donkey --resume checkpoints/donkey/step_500

    # Use custom hyperparameters
    python -m model_training.train --model elephant --config configs/elephant.json
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path

from model_training.dataset import build_datasets
from model_training.trainer import PoliticalSpeechTrainer, TrainingConfig, TrainingState

# Project root is one level up from this file
PROJECT_ROOT = Path(__file__).resolve().parent.parent


def setup_logging(party: str, log_dir: Path) -> None:
    """Configure logging to both console and file."""
    log_dir.mkdir(parents=True, exist_ok=True)
    log_file = log_dir / f"{party}_train.log"

    handlers: list[logging.Handler] = [
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(log_file, mode="a", encoding="utf-8"),
    ]

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=handlers,
        force=True,
    )


def print_training_summary(party: str, state: TrainingState, elapsed: float) -> None:
    """Print a human-readable summary after training completes."""
    print("\n" + "=" * 60)
    print(f"  Training Summary: {party.upper()} model")
    print("=" * 60)
    print(f"  Total epochs:     {state.epoch + 1}")
    print(f"  Total steps:      {state.global_step}")
    print(f"  Best val loss:    {state.best_val_loss:.4f}")
    print(f"  Final train loss: {state.train_losses[-1]:.4f}" if state.train_losses else "")
    print(f"  Final val loss:   {state.val_losses[-1]:.4f}" if state.val_losses else "")
    print(f"  Wall time:        {format_duration(elapsed)}")
    print("=" * 60 + "\n")


def format_duration(seconds: float) -> str:
    """Format seconds into human-readable duration."""
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    if hours > 0:
        return f"{hours}h {minutes}m {secs}s"
    if minutes > 0:
        return f"{minutes}m {secs}s"
    return f"{secs}s"


def train_party(
    party: str,
    config: TrainingConfig,
    resume_from: str | None = None,
) -> TrainingState:
    """Run a full training pipeline for one party.

    Args:
        party: "donkey" or "elephant".
        config: Training configuration.
        resume_from: Optional checkpoint path.

    Returns:
        Final training state.
    """
    log_dir = PROJECT_ROOT / config.log_dir / party
    setup_logging(party, log_dir)
    logger = logging.getLogger(__name__)

    logger.info("=" * 60)
    logger.info("Starting %s model training", party.upper())
    logger.info("=" * 60)

    # Build datasets
    data_dir = PROJECT_ROOT / config.data_dir
    logger.info("Loading data from %s", data_dir)

    train_dataset, val_dataset = build_datasets(
        data_dir=data_dir,
        party=party,
        max_length=config.max_length,
        val_fraction=config.val_fraction,
        seed=config.seed,
    )

    logger.info("Train examples: %d, Val examples: %d", len(train_dataset), len(val_dataset))

    # Initialize trainer
    trainer = PoliticalSpeechTrainer(
        party=party,
        config=config,
        project_root=PROJECT_ROOT,
        resume_from=resume_from,
    )

    # Train
    start = time.time()
    state = trainer.train(train_dataset, val_dataset)
    elapsed = time.time() - start

    print_training_summary(party, state, elapsed)
    return state


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Fine-tune GPT-2 on political speech data.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s --model donkey
  %(prog)s --model elephant --config configs/elephant.json
  %(prog)s --both
  %(prog)s --model donkey --resume checkpoints/donkey/step_500
        """,
    )

    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument(
        "--model",
        choices=["donkey", "elephant"],
        help="Which model to train: 'donkey' (Democrat) or 'elephant' (Republican).",
    )
    group.add_argument(
        "--both",
        action="store_true",
        help="Train both models sequentially (donkey first, then elephant).",
    )

    parser.add_argument(
        "--resume",
        type=str,
        default=None,
        help="Path to checkpoint directory to resume from.",
    )
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Path to JSON file with hyperparameter overrides.",
    )

    # Quick overrides for common hyperparameters
    parser.add_argument("--epochs", type=int, default=None, help="Number of training epochs.")
    parser.add_argument("--batch-size", type=int, default=None, help="Per-device batch size.")
    parser.add_argument("--lr", type=float, default=None, help="Learning rate.")
    parser.add_argument(
        "--grad-accum", type=int, default=None, help="Gradient accumulation steps."
    )

    return parser.parse_args()


def build_config(args: argparse.Namespace) -> TrainingConfig:
    """Build TrainingConfig from CLI args, with JSON file as base if provided."""
    if args.config:
        config = TrainingConfig.from_json(args.config)
    else:
        config = TrainingConfig()

    # CLI flags override config file values
    if args.epochs is not None:
        config.epochs = args.epochs
    if args.batch_size is not None:
        config.batch_size = args.batch_size
    if args.lr is not None:
        config.learning_rate = args.lr
    if args.grad_accum is not None:
        config.gradient_accumulation_steps = args.grad_accum

    return config


def main() -> None:
    """Main entry point."""
    args = parse_args()
    config = build_config(args)

    if args.both:
        print("Training both models sequentially...\n")
        for party in ("donkey", "elephant"):
            train_party(party, config, resume_from=args.resume)
    else:
        train_party(args.model, config, resume_from=args.resume)


if __name__ == "__main__":
    main()
