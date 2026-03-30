# Model Training

Fine-tunes two GPT-2 (124M parameter) models on political speech data:

- **donkey** -- trained on Democratic speeches, floor statements, and press releases
- **elephant** -- trained on Republican speeches, floor statements, and press releases

## Quick Start

```bash
# Train one model
python -m model_training.train --model donkey

# Train both sequentially
python -m model_training.train --both

# Resume from checkpoint after interruption
python -m model_training.train --model donkey --resume checkpoints/donkey/step_500
```

## Data Format

Place JSONL files in `data/processed/`. Each line should have at minimum a `text` field:

```json
{"text": "Full text of a speech or statement...", "speaker": "Name", "date": "2024-01-15"}
```

Files should be prefixed with the party name (e.g., `donkey_speeches.jsonl`, `elephant_floor_statements.jsonl`).

## Hyperparameters

Defaults are tuned for a single consumer GPU (8-12GB VRAM). Override via `--config`:

```bash
python -m model_training.train --model donkey --config configs/donkey.json
```

Example config JSON:

```json
{
  "epochs": 3,
  "batch_size": 4,
  "gradient_accumulation_steps": 8,
  "learning_rate": 5e-5,
  "weight_decay": 0.01,
  "warmup_ratio": 0.1,
  "max_length": 1024
}
```

Common CLI overrides: `--epochs`, `--batch-size`, `--lr`, `--grad-accum`.

## Hardware Requirements

| Setup | Batch Size | Grad Accum | Effective Batch | Estimated Time (per model) |
|-------|-----------|------------|-----------------|---------------------------|
| RTX 3060 (12GB) | 4 | 8 | 32 | ~2-4 hours |
| RTX 3080 (10GB) | 4 | 8 | 32 | ~1.5-3 hours |
| RTX 4090 (24GB) | 8 | 4 | 32 | ~30-60 min |
| CPU only | 1 | 32 | 32 | ~24-48 hours |

Training uses mixed precision (FP16) via HuggingFace Accelerate. If you run out of VRAM, reduce `batch_size` and increase `gradient_accumulation_steps` to maintain the same effective batch size.

## Checkpoints

Checkpoints are saved at the end of every epoch to `checkpoints/{donkey|elephant}/step_N/`. Each checkpoint contains:

- Model weights (HuggingFace format)
- Optimizer and scheduler state
- Training state (epoch, step, loss history)

This means you can kill training at any time and resume from the last completed epoch with `--resume`.

## Output

Final models are saved to `models/{donkey|elephant}/` in HuggingFace format, ready for loading with `GPT2LMHeadModel.from_pretrained()`.

Training logs (JSONL with per-step and per-epoch metrics) are written to `logs/{donkey|elephant}/`.

## Project Structure

```
model_training/
  __init__.py        # Package init
  dataset.py         # PoliticalSpeechDataset + train/val splitting
  trainer.py         # Training loop, checkpointing, mixed precision
  train.py           # CLI entry point
```
