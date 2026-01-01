"""Demonstration of running inference with the custom BERT model.

This script compares the output of the custom implementation with
Hugging Face’s reference implementation.  It loads a pre‑trained
``bert-base-uncased`` model using the ``transformers`` library, then
transfers the weights into our custom model.  A sample input is
tokenised and fed through both models, and the difference between
their outputs is reported.

Usage
-----
Run this script with ``python scripts/inference_demo.py`` from the
repository root.  Ensure that dependencies are installed and that
PyTorch can locate a GPU if available.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Dict

import torch
from transformers import BertTokenizer, BertModel as HFBertModel
from dotenv import load_dotenv

from src.config import BertConfig
from src.bert_model import BertModel
from src.utils import rename_state_dict_keys


def setup_logging() -> None:
  logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
  )


def main() -> None:
  setup_logging()
  logger = logging.getLogger(__name__)
  # Load environment variables
  load_dotenv()
  data_dir = Path(os.getenv("DATA_DIR", "."))
  logger.info(f"Using DATA_DIR at {data_dir.resolve()}")

  # Load Hugging Face model and tokenizer
  hf_model_name = "bert-base-uncased"
  tokenizer = BertTokenizer.from_pretrained(hf_model_name)
  hf_model = HFBertModel.from_pretrained(hf_model_name)
  hf_model.eval()

  # Create our custom model with matching configuration
  config = BertConfig(
    vocab_size=hf_model.config.vocab_size,
    hidden_size=hf_model.config.hidden_size,
    num_attention_heads=hf_model.config.num_attention_heads,
    num_hidden_layers=hf_model.config.num_hidden_layers,
    intermediate_size=hf_model.config.intermediate_size,
    max_position_embeddings=hf_model.config.max_position_embeddings,
    type_vocab_size=hf_model.config.type_vocab_size,
    hidden_dropout_prob=hf_model.config.hidden_dropout_prob,
    attention_probs_dropout_prob=hf_model.config.attention_probs_dropout_prob,
    layer_norm_eps=hf_model.config.layer_norm_eps,
    pad_token_id=hf_model.config.pad_token_id,
  )
  custom_model = BertModel(config)
  custom_model.eval()

  # Transfer weights from HF model
  # Extract state dict, rename keys and load
  hf_state_dict: Dict[str, torch.Tensor] = hf_model.state_dict()
  renamed_state_dict = rename_state_dict_keys(hf_state_dict)
  missing, unexpected = custom_model.load_state_dict(renamed_state_dict, strict=False)
  if missing:
    logger.warning(f"Missing keys during load: {missing}")
  if unexpected:
    logger.warning(f"Unexpected keys during load: {unexpected}")

  # Example input
  text_a = "Hello, my dog is cute"
  text_b = "He likes playing outside"
  encoded = tokenizer(
    text_a,
    text_b,
    return_tensors="pt",
    padding="max_length",
    truncation=True,
    max_length=16,
  )
  input_ids = encoded["input_ids"]
  token_type_ids = encoded["token_type_ids"]
  attention_mask = encoded["attention_mask"]

  with torch.no_grad():
    # Hugging Face output
    hf_outputs = hf_model(
      input_ids=input_ids,
      attention_mask=attention_mask,
      token_type_ids=token_type_ids,
      output_hidden_states=False,
    )
    hf_seq_output = hf_outputs.last_hidden_state

    # Custom model output
    custom_seq_output, pooled_output, _, _ = custom_model(
      input_ids=input_ids,
      attention_mask=attention_mask,
      token_type_ids=token_type_ids,
      output_hidden_states=False,
    )

  # Compute difference between outputs
  diff = (custom_seq_output - hf_seq_output).abs().mean().item()
  logger.info(f"Mean absolute difference between custom and HF outputs: {diff:.6f}")

  # Save outputs to results for inspection
  results_dir = Path("results")
  results_dir.mkdir(parents=True, exist_ok=True)
  torch.save(
    {
      "input_ids": input_ids,
      "hf_seq_output": hf_seq_output,
      "custom_seq_output": custom_seq_output,
    },
    results_dir / "inference_outputs.pt",
  )
  logger.info(f"Inference outputs saved to {results_dir / 'inference_outputs.pt'}")


if __name__ == "__main__":
  main()
