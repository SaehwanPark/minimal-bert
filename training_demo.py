"""Demonstration of training the custom BERT model on a toy dataset.

This script constructs a small dataset of sentence pairs and trains
``BertForPreTraining`` using the masked language modelling (MLM) and
next sentence prediction (NSP) objectives.  The goal is not to train
a state‑of‑the‑art model, but rather to illustrate how the various
components come together.  Training progress and loss values are
logged, and the resulting model weights are saved to the ``results``
directory.

Usage
-----
Run this script with ``python scripts/training_demo.py`` from the
repository root.  Ensure that dependencies are installed and that
PyTorch can locate a GPU if available.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import List, Tuple

import torch
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer
from dotenv import load_dotenv

from src.config import BertConfig
from src.bert_model import BertForPreTraining
from src.utils import get_torch_accelerator


def setup_logging() -> None:
  logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
  )


class ToyBertDataset(Dataset):
  """A toy dataset for demonstrating BERT pre‑training.

  Each sample consists of a pair of sentences and a label indicating
  whether the second sentence follows the first.  During
  initialisation the dataset tokenises the sentence pairs and applies
  random masking for the MLM task.
  """

  def __init__(
    self,
    tokenizer: BertTokenizer,
    sentence_pairs: List[Tuple[str, str, int]],
    max_length: int = 32,
    mlm_probability: float = 0.15,
  ) -> None:
    self.tokenizer = tokenizer
    self.sentence_pairs = sentence_pairs
    self.max_length = max_length
    self.mlm_probability = mlm_probability

    # Preprocess all examples
    self.examples = []
    for sent_a, sent_b, label in sentence_pairs:
      enc = tokenizer(
        sent_a,
        sent_b,
        truncation=True,
        padding="max_length",
        max_length=max_length,
        return_tensors="pt",
      )
      input_ids = enc["input_ids"][0]
      token_type_ids = enc["token_type_ids"][0]
      attention_mask = enc["attention_mask"][0]
      masked_input_ids, mlm_labels = self.mask_tokens(input_ids.clone())
      self.examples.append(
        {
          "input_ids": masked_input_ids,
          "token_type_ids": token_type_ids,
          "attention_mask": attention_mask,
          "labels": mlm_labels,
          "next_sentence_label": torch.tensor(label, dtype=torch.long),
        }
      )

  def __len__(self) -> int:
    return len(self.examples)

  def __getitem__(self, idx: int) -> dict:
    return self.examples[idx]

  def mask_tokens(self, inputs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """Prepare masked tokens inputs/labels for masked language modelling.

    ``-100`` tokens are ignored in the loss.  The [CLS] and [SEP]
    tokens are never masked.  15% of input tokens are selected for
    masking: 80% are replaced with [MASK], 10% with a random token and
    10% are kept unchanged.
    """
    labels = inputs.clone()

    # Create mask array for MLM
    probability_matrix = torch.full(labels.shape, self.mlm_probability)
    special_tokens_mask = self.tokenizer.get_special_tokens_mask(
      labels.tolist(), already_has_special_tokens=True
    )
    probability_matrix = probability_matrix.masked_fill(
      torch.tensor(special_tokens_mask, dtype=torch.bool), 0.0
    )
    masked_indices = torch.bernoulli(probability_matrix).bool()
    labels[~masked_indices] = -100  # Only compute loss on masked tokens

    # 80% of the time, replace masked input tokens with [MASK]
    indices_replaced = (
      torch.bernoulli(torch.full(labels.shape, 0.8)).bool() & masked_indices
    )
    mask_token_id = self.tokenizer.mask_token_id
    inputs[indices_replaced] = mask_token_id

    # 10% of the time, replace masked input tokens with random token
    indices_random = (
      torch.bernoulli(torch.full(labels.shape, 0.5)).bool()
      & masked_indices
      & ~indices_replaced
    )
    random_words = torch.randint(len(self.tokenizer), labels.shape, dtype=torch.long)
    inputs[indices_random] = random_words[indices_random]

    # The rest 10% of the time, keep the masked input tokens unchanged
    return inputs, labels


def main() -> None:
  setup_logging()
  logger = logging.getLogger(__name__)
  load_dotenv()
  data_dir = Path(os.getenv("DATA_DIR", "."))
  logger.info(f"Using DATA_DIR at {data_dir.resolve()}")

  # Prepare tokenizer and configuration
  model_name = "bert-base-uncased"
  tokenizer = BertTokenizer.from_pretrained(model_name)
  hf_config = tokenizer.init_kwargs.get("config", None)
  # Derive configuration from HF config or fallback to defaults
  if hf_config is None:
    # Use default BERT base parameters
    config = BertConfig()
  else:
    # When available, HF config contains relevant attributes
    config = BertConfig(
      vocab_size=hf_config.vocab_size,
      hidden_size=hf_config.hidden_size,
      num_attention_heads=hf_config.num_attention_heads,
      num_hidden_layers=hf_config.num_hidden_layers,
      intermediate_size=hf_config.intermediate_size,
      max_position_embeddings=hf_config.max_position_embeddings,
      type_vocab_size=hf_config.type_vocab_size,
      hidden_dropout_prob=hf_config.hidden_dropout_prob,
      attention_probs_dropout_prob=hf_config.attention_probs_dropout_prob,
      layer_norm_eps=hf_config.layer_norm_eps,
      pad_token_id=hf_config.pad_token_id,
    )

  # Define a small set of sentence pairs; last element indicates if they are consecutive
  sentence_pairs: List[Tuple[str, str, int]] = [
    ("The quick brown fox jumps over the lazy dog", "It then runs into the forest", 1),
    ("Transformers are revolutionary models", "They were introduced in 2017", 1),
    ("My cat loves to sleep", "The stock market closed higher today", 0),
    ("Artificial intelligence is advancing rapidly", "Pizza is my favourite food", 0),
  ]

  # Duplicate the dataset to have more examples
  sentence_pairs = sentence_pairs * 4

  dataset = ToyBertDataset(
    tokenizer, sentence_pairs, max_length=32, mlm_probability=0.15
  )
  dataloader = DataLoader(dataset, batch_size=2, shuffle=True)

  # Instantiate model
  model = BertForPreTraining(config)

  device = get_torch_accelerator()
  model.to(device)
  model.train()

  # Optimizer
  optimizer = torch.optim.AdamW(model.parameters(), lr=5e-5)

  num_epochs = 2
  logger.info(f"Starting training for {num_epochs} epochs on {len(dataset)} examples")
  for epoch in range(num_epochs):
    total_loss = 0.0
    for batch in dataloader:
      optimizer.zero_grad()
      input_ids = batch["input_ids"].to(device)
      token_type_ids = batch["token_type_ids"].to(device)
      attention_mask = batch["attention_mask"].to(device)
      labels = batch["labels"].to(device)
      next_sentence_label = batch["next_sentence_label"].to(device)

      loss, _, _ = model(
        input_ids=input_ids,
        attention_mask=attention_mask,
        token_type_ids=token_type_ids,
        labels=labels,
        next_sentence_label=next_sentence_label,
      )
      loss.backward()
      optimizer.step()
      total_loss += loss.item()
    avg_loss = total_loss / len(dataloader)
    logger.info(f"Epoch {epoch + 1}/{num_epochs}: average loss = {avg_loss:.4f}")

  # Save model weights
  results_dir = Path("results")
  results_dir.mkdir(parents=True, exist_ok=True)
  model_path = results_dir / "toy_bert_pretrained.pth"
  torch.save(model.state_dict(), model_path)
  logger.info(f"Trained model saved to {model_path}")


if __name__ == "__main__":
  main()
