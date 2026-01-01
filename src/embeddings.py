"""Embedding layer for the BERT model.

This module defines a :class:`BertEmbeddings` class that composes token,
position and segment (token type) embeddings.  The three embeddings are
summed and normalised before being passed through a dropout layer.  The
implementation closely follows the original BERT design but remains easy
to adapt for other transformer encoders.
"""

from __future__ import annotations

from typing import Optional

import torch
from torch import nn

from src.config import BertConfig


class BertEmbeddings(nn.Module):
  """Construct embeddings from word, position and token type embeddings.

  Parameters
  ----------
  config:
      Configuration containing model hyper‑parameters.  Only those fields
      relevant to the embeddings are used.
  """

  def __init__(self, config: BertConfig) -> None:
    super().__init__()
    self.word_embeddings = nn.Embedding(config.vocab_size, config.hidden_size)
    self.position_embeddings = nn.Embedding(
      config.max_position_embeddings, config.hidden_size
    )
    self.token_type_embeddings = nn.Embedding(
      config.type_vocab_size, config.hidden_size
    )

    self.layer_norm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
    self.dropout = nn.Dropout(config.hidden_dropout_prob)

  def forward(
    self,
    input_ids: torch.Tensor,
    token_type_ids: Optional[torch.Tensor] = None,
  ) -> torch.Tensor:
    """Embed the input token IDs and segment IDs.

    This method sums the word, position and token type embeddings and
    applies layer normalization and dropout.  Position IDs are
    generated on the fly based on the sequence length of the inputs.

    Parameters
    ----------
    input_ids:
        Tensor of shape ``(batch_size, seq_length)`` containing token
        indices in the vocabulary.
    token_type_ids:
        Optional tensor of shape ``(batch_size, seq_length)`` indicating
        segment membership.  If ``None``, all tokens are assumed to belong
        to segment 0.

    Returns
    -------
    torch.Tensor
        The embedded representation of shape ``(batch_size, seq_length, hidden_size)``.
    """
    batch_size, seq_length = input_ids.size()
    if token_type_ids is None:
      token_type_ids = torch.zeros_like(input_ids)

    # Create position IDs from 0 to seq_length - 1 and expand to batch size
    position_ids = torch.arange(seq_length, dtype=torch.long, device=input_ids.device)
    position_ids = position_ids.unsqueeze(0).expand(batch_size, seq_length)

    # Look up embeddings
    word_embed = self.word_embeddings(input_ids)
    pos_embed = self.position_embeddings(position_ids)
    token_type_embed = self.token_type_embeddings(token_type_ids)

    # Sum and normalise
    embeddings = word_embed + pos_embed + token_type_embed
    embeddings = self.layer_norm(embeddings)
    embeddings = self.dropout(embeddings)
    return embeddings
