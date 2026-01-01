"""Unit tests for the BertModel class."""

import torch

from src.config import BertConfig
from src.bert_model import BertModel


def test_bert_model_output_shapes() -> None:
  config = BertConfig(
    vocab_size=100,
    hidden_size=16,
    num_attention_heads=4,
    num_hidden_layers=2,
    intermediate_size=32,
    max_position_embeddings=30,
    type_vocab_size=2,
    hidden_dropout_prob=0.0,
    attention_probs_dropout_prob=0.0,
  )
  model = BertModel(config)
  model.eval()
  batch_size, seq_len = 2, 10
  input_ids = torch.randint(0, config.vocab_size, (batch_size, seq_len))
  token_type_ids = torch.zeros_like(input_ids)
  attention_mask = torch.ones_like(input_ids)
  seq_output, pooled_output, hidden_states, attentions = model(
    input_ids=input_ids,
    attention_mask=attention_mask,
    token_type_ids=token_type_ids,
    output_hidden_states=True,
    output_attentions=True,
  )
  # Sequence output shape
  assert seq_output.shape == (batch_size, seq_len, config.hidden_size)
  # Pooled output shape
  assert pooled_output.shape == (batch_size, config.hidden_size)
  # Hidden states: number of layers + embedding output
  assert (
    hidden_states is not None and len(hidden_states) == config.num_hidden_layers + 1
  )
  # Attentions: number of layers
  assert attentions is not None and len(attentions) == config.num_hidden_layers
