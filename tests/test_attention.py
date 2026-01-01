"""Unit tests for the multi‑head self‑attention implementation."""

import torch

from src.config import BertConfig
from src.multi_head_attention import MultiHeadSelfAttention


def test_attention_output_shape_and_probs() -> None:
  # Use dropout 0 to simplify the test
  config = BertConfig(
    vocab_size=100,
    hidden_size=64,
    num_attention_heads=8,
    num_hidden_layers=1,
    intermediate_size=128,
    max_position_embeddings=10,
    type_vocab_size=2,
    hidden_dropout_prob=0.0,
    attention_probs_dropout_prob=0.0,
  )
  attn = MultiHeadSelfAttention(config)
  attn.eval()
  batch_size, seq_len = 2, 5
  hidden_states = torch.randn(batch_size, seq_len, config.hidden_size)
  # Identity mask (no masking)
  attn_mask = None
  output, probs = attn(hidden_states, attn_mask)
  assert output.shape == (batch_size, seq_len, config.hidden_size)
  assert probs.shape == (batch_size, config.num_attention_heads, seq_len, seq_len)
  # Sum of probabilities along last dimension should be 1 (due to softmax)
  prob_sums = probs.sum(dim=-1)
  ones = torch.ones_like(prob_sums)
  assert torch.allclose(prob_sums, ones, atol=1e-6)
