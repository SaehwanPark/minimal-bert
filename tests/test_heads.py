"""Unit tests for the MLM and NSP heads."""

import torch

from src.config import BertConfig
from src.heads import MaskedLanguageModelHead, NextSentencePredictionHead


def test_mlm_head_output_shape() -> None:
  config = BertConfig(
    vocab_size=50,
    hidden_size=32,
    num_attention_heads=4,  # 32|4
  )
  head = MaskedLanguageModelHead(config)
  hidden_states = torch.randn(3, 7, config.hidden_size)
  logits = head(hidden_states)
  assert logits.shape == (3, 7, config.vocab_size)


def test_nsp_head_output_shape() -> None:
  config = BertConfig(
    hidden_size=32,
    num_attention_heads=4,  # 32|4
  )
  head = NextSentencePredictionHead(config)
  pooled_output = torch.randn(4, config.hidden_size)
  logits = head(pooled_output)
  assert logits.shape == (4, 2)
