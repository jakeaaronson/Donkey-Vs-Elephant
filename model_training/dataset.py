"""
Custom PyTorch Dataset for fine-tuning GPT-2 on political speech data.

Loads JSONL files from data/processed/, tokenizes with GPT-2 tokenizer,
and chunks long documents into fixed-length blocks for causal language modeling.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Optional

import torch
from torch.utils.data import Dataset, random_split
from transformers import GPT2Tokenizer

logger = logging.getLogger(__name__)

# GPT-2's context window
DEFAULT_MAX_LENGTH = 1024
# Special tokens
EOS_TOKEN = "<|endoftext|>"


class PoliticalSpeechDataset(Dataset):
    """Dataset that tokenizes political speeches and chunks them into fixed-length blocks.

    Each JSONL record is expected to have at minimum a "text" field.
    Long documents are split into non-overlapping blocks of `max_length` tokens,
    with each block serving as an independent training example for causal LM.

    Args:
        data_dir: Path to directory containing .jsonl files.
        party: Which party's data to load ("donkey" or "elephant").
        max_length: Token block size. Defaults to GPT-2's 1024 context window.
        tokenizer: Pre-initialized tokenizer. If None, loads "gpt2" from HF.
    """

    def __init__(
        self,
        data_dir: str | Path,
        party: str,
        max_length: int = DEFAULT_MAX_LENGTH,
        tokenizer: Optional[GPT2Tokenizer] = None,
    ) -> None:
        self.data_dir = Path(data_dir)
        self.party = party
        self.max_length = max_length
        self.tokenizer = tokenizer or GPT2Tokenizer.from_pretrained("gpt2")

        # GPT-2 tokenizer has no pad token by default
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        # Map model names to data file prefixes
        self._file_prefix = {"donkey": "democratic", "elephant": "republican"}.get(
            party, party
        )

        self.blocks: list[torch.Tensor] = []
        self._load_and_chunk()

    def _load_and_chunk(self) -> None:
        """Read JSONL files, tokenize all text, then slice into fixed-length blocks."""
        jsonl_files = sorted(self.data_dir.glob(f"{self._file_prefix}*.jsonl"))
        if not jsonl_files:
            # Fall back to a single combined file if party-prefixed files don't exist
            jsonl_files = sorted(self.data_dir.glob("*.jsonl"))
            logger.warning(
                "No %s-prefixed JSONL files found; falling back to all JSONL in %s",
                self._file_prefix,
                self.data_dir,
            )

        if not jsonl_files:
            raise FileNotFoundError(
                f"No JSONL files found in {self.data_dir} for party '{self.party}'"
            )

        # Concatenate all tokens with EOS separators into one flat stream
        all_token_ids: list[int] = []
        doc_count = 0

        for path in jsonl_files:
            logger.info("Loading %s", path.name)
            with open(path, "r", encoding="utf-8") as f:
                for line_num, line in enumerate(f, start=1):
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        record = json.loads(line)
                    except json.JSONDecodeError:
                        logger.warning("Skipping malformed JSON at %s:%d", path.name, line_num)
                        continue

                    text = record.get("text", "")
                    if not text:
                        continue

                    # Tokenize with EOS separator between documents
                    tokens = self.tokenizer.encode(text + EOS_TOKEN)
                    all_token_ids.extend(tokens)
                    doc_count += 1

        logger.info(
            "Loaded %d documents (%d tokens) for party '%s'",
            doc_count,
            len(all_token_ids),
            self.party,
        )

        # Chunk the flat token stream into non-overlapping blocks
        # Discard the final partial block (not enough tokens for a full example)
        for start in range(0, len(all_token_ids) - self.max_length + 1, self.max_length):
            block = all_token_ids[start : start + self.max_length]
            self.blocks.append(torch.tensor(block, dtype=torch.long))

        logger.info("Created %d blocks of %d tokens each", len(self.blocks), self.max_length)

    def __len__(self) -> int:
        return len(self.blocks)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        """Return input_ids and labels for causal LM (labels = input_ids shifted by model)."""
        input_ids = self.blocks[idx]
        return {
            "input_ids": input_ids,
            "attention_mask": torch.ones_like(input_ids),
            "labels": input_ids.clone(),
        }


def build_datasets(
    data_dir: str | Path,
    party: str,
    max_length: int = DEFAULT_MAX_LENGTH,
    val_fraction: float = 0.1,
    seed: int = 42,
    tokenizer: Optional[GPT2Tokenizer] = None,
) -> tuple[Dataset, Dataset]:
    """Build train/val split from political speech data.

    Args:
        data_dir: Path to directory containing JSONL files.
        party: "donkey" or "elephant".
        max_length: Token block size for each training example.
        val_fraction: Fraction of blocks to hold out for validation.
        seed: Random seed for reproducible splits.
        tokenizer: Optional pre-initialized tokenizer.

    Returns:
        Tuple of (train_dataset, val_dataset).
    """
    full_dataset = PoliticalSpeechDataset(
        data_dir=data_dir,
        party=party,
        max_length=max_length,
        tokenizer=tokenizer,
    )

    total = len(full_dataset)
    val_size = max(1, int(total * val_fraction))
    train_size = total - val_size

    logger.info("Splitting: %d train, %d val (%.1f%%)", train_size, val_size, val_fraction * 100)

    train_dataset, val_dataset = random_split(
        full_dataset,
        [train_size, val_size],
        generator=torch.Generator().manual_seed(seed),
    )

    return train_dataset, val_dataset
