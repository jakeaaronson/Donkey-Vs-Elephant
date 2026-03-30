# Donkey vs Elephant

**Two GPT-2 language models fine-tuned on 20+ years of Congressional speech — one Democrat, one Republican — with a side-by-side comparison UI.**

![Donkey vs Elephant UI](docs/screenshot.png)

Give both models the same prompt and watch how political language diverges. The "Donkey" model generates text in the style of Democratic legislators; the "Elephant" model speaks like a Republican. Same architecture, same training process, different data — the outputs reveal how partisan framing shapes political communication.

## How It Works

```
Congressional Record (2000-2024)
        │
        ▼
┌─────────────────┐
│  Data Collection │  ← Congress.gov API: speeches, remarks, floor statements
│  & Processing    │  ← Party classification via member lookup
└────────┬────────┘
         │
    ┌────┴────┐
    ▼         ▼
┌────────┐ ┌────────┐
│Democrat│ │Repub.  │
│ Corpus │ │ Corpus │
│(JSONL) │ │(JSONL) │
└───┬────┘ └───┬────┘
    │          │
    ▼          ▼
┌────────┐ ┌────────┐
│ Donkey │ │Elephant│  ← GPT-2 (124M) fine-tuned independently
│ Model  │ │ Model  │  ← Checkpoint/resume for multi-day training
└───┬────┘ └───┬────┘
    │          │
    └────┬─────┘
         ▼
   ┌───────────┐
   │  Web UI   │  ← Flask app, side-by-side generation
   │ Compare!  │  ← Configurable temperature, top-k, max length
   └───────────┘
```

## Project Structure

```
Donkey-Vs-Elephant/
├── data_collection/        # Congressional Record scraper & party classifier
│   ├── congress_api.py     # Congress.gov API client with rate limiting
│   ├── party_classifier.py # Speaker identification & party lookup
│   └── run_collection.py   # CLI orchestrator
├── model_training/         # GPT-2 fine-tuning pipeline
│   ├── dataset.py          # Custom Dataset with tokenization & chunking
│   ├── trainer.py          # Training loop with checkpoints & mixed precision
│   └── train.py            # CLI entry point (train one or both models)
├── web_ui/                 # Comparison interface
│   ├── app.py              # Flask backend with generation endpoints
│   └── templates/          # Political-themed responsive UI
├── scripts/                # Setup and convenience scripts
├── data/                   # (gitignored) Raw & processed datasets
├── models/                 # (gitignored) Trained model weights
└── logs/                   # (gitignored) Training logs
```

## Quick Start

### Prerequisites
- Python 3.10+
- CUDA-capable GPU (recommended for training; CPU works for inference)
- [Congress.gov API key](https://api.congress.gov/sign-up/) (free)

### Setup

```bash
git clone https://github.com/jakeaaronson/Donkey-Vs-Elephant.git
cd Donkey-Vs-Elephant
bash scripts/setup.sh

# Add your API key
echo "CONGRESS_API_KEY=your_key_here" > data_collection/.env
```

### Collect Data

```bash
source venv/bin/activate

# Full collection (recommended — takes several hours)
python -m data_collection.run_collection --start-year 2000 --end-year 2024

# Or a smaller sample for testing
python -m data_collection.run_collection --start-year 2020 --end-year 2024 --max-pages 100
```

### Train Models

```bash
# Train both models sequentially
python -m model_training.train --both --epochs 3

# Or train individually
python -m model_training.train --model donkey --epochs 3
python -m model_training.train --model elephant --epochs 3

# Resume from checkpoint after interruption
python -m model_training.train --model donkey --resume
```

Training times depend on data size and hardware. With a full dataset on a consumer GPU (RTX 3060–4090), expect 8–48 hours per model. The pipeline supports pause/resume — training can be interrupted and continued from the last checkpoint.

### Launch the UI

```bash
python -m web_ui.app
# Open http://localhost:5000
```

## Training Details

| Parameter | Value |
|-----------|-------|
| Base model | GPT-2 (124M parameters) |
| Tokenizer | GPT-2 BPE (50,257 vocab) |
| Max sequence length | 512 tokens |
| Optimizer | AdamW |
| Learning rate | 5e-5 with linear warmup |
| Precision | FP16 mixed precision |
| Checkpointing | Every 1,000 steps |

Both models start from the same pretrained GPT-2 weights and are fine-tuned independently on their respective party's corpus. This ensures any differences in output are attributable to the training data, not the model architecture or initialization.

## Data Source

All training data comes from the [Congressional Record](https://www.congress.gov/congressional-record) via the official Congress.gov API. The Congressional Record is the official transcript of proceedings and debates of the U.S. Congress, published daily when Congress is in session.

The data collection pipeline:
1. Fetches daily Congressional Record issues (2000–2024)
2. Extracts individual speeches, remarks, and floor statements
3. Identifies speakers and resolves their party affiliation via the Congress.gov Members API
4. Splits the corpus into Democratic and Republican datasets
5. Outputs clean JSONL files ready for tokenization

## Tech Stack

- **Python 3.10+** — Core language
- **PyTorch + HuggingFace Transformers** — Model fine-tuning and inference
- **HuggingFace Accelerate** — Mixed precision training
- **Flask** — Web UI backend
- **Congress.gov API** — Data source

## License

MIT
