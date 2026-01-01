"""Unit tests for the transformer encoder layer."""

import torch

from src.config import BertConfig
from src.transformer_encoder_layer import TransformerEncoderLayer


def test_transformer_encoder_layer_shape() -> None:
  config = BertConfig(
    vocab_size=100,
    hidden_size=32,
    num_attention_heads=4,
    num_hidden_layers=1,
    intermediate_size=64,
    max_position_embeddings=20,
    type_vocab_size=2,
    hidden_dropout_prob=0.0,
    attention_probs_dropout_prob=0.0,
  )
  layer = TransformerEncoderLayer(config)
  layer.eval()
  batch_size, seq_len = 3, 7
  hidden_states = torch.randn(batch_size, seq_len, config.hidden_size)
  output, attn_probs = layer(hidden_states)
  assert output.shape == (batch_size, seq_len, config.hidden_size)
  assert attn_probs.shape == (batch_size, config.num_attention_heads, seq_len, seq_len)
