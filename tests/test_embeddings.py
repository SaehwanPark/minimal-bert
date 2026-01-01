"""Unit tests for the embeddings module."""

import torch

from src.config import BertConfig
from src.embeddings import BertEmbeddings


def test_embeddings_shape() -> None:
  config = BertConfig(
    vocab_size=100, hidden_size=32, max_position_embeddings=20, type_vocab_size=2
  )
  embeddings = BertEmbeddings(config)
  input_ids = torch.randint(0, 100, (4, 10))
  token_type_ids = torch.zeros_like(input_ids)
  output = embeddings(input_ids, token_type_ids)
  assert output.shape == (4, 10, 32)
