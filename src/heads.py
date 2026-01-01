"""Head modules for BERT pre‑training tasks.

This module defines the heads used for the masked language modelling (MLM)
and next sentence prediction (NSP) tasks.  They operate on the hidden
states produced by the encoder and return unnormalised logits.  When
integrated into :class:`BertForPreTraining` the heads support computing
losses given reference labels.
"""

from __future__ import annotations

from typing import Optional, Tuple

import torch
from torch import nn
import torch.nn.functional as F

from .config import BertConfig


class MaskedLanguageModelHead(nn.Module):
  """Head for the masked language modelling objective.

  This head projects the hidden states back to the vocabulary space.
  It consists of a dense transformation, a non‑linearity, layer
  normalisation and a decoder.  Weight tying is supported by reusing
  the token embedding weights as the decoder’s weight matrix.
  """

  def __init__(self, config: BertConfig, embeddings_weight: Optional[nn.Parameter] = None) -> None:
    super().__init__()
    self.dense = nn.Linear(config.hidden_size, config.hidden_size)
    self.layer_norm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
    self.decoder = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
    self.bias = nn.Parameter(torch.zeros(config.vocab_size))
    if embeddings_weight is not None:
      # Tie decoder weight to the embeddings if provided
      self.decoder.weight = embeddings_weight

  def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
    """Predict vocabulary logits for masked positions.

    Parameters
    ----------
    hidden_states:
        Tensor of shape ``(batch_size, seq_len, hidden_size)`` from the
        final encoder layer.

    Returns
    -------
    torch.Tensor
        Vocabulary logits of shape ``(batch_size, seq_len, vocab_size)``.
    """
    x = self.dense(hidden_states)
    x = F.gelu(x)
    x = self.layer_norm(x)
    x = self.decoder(x) + self.bias
    return x


class NextSentencePredictionHead(nn.Module):
  """Head for the next sentence prediction objective.

  This head takes the hidden state of the [CLS] token and predicts
  whether the second segment is the subsequent sentence of the first
  segment.  It is a simple linear classifier over two classes.
  """

  def __init__(self, config: BertConfig) -> None:
    super().__init__()
    self.seq_relationship = nn.Linear(config.hidden_size, 2)

  def forward(self, pooled_output: torch.Tensor) -> torch.Tensor:
    """Compute logits for the NSP task.

    Parameters
    ----------
    pooled_output:
        Tensor of shape ``(batch_size, hidden_size)`` representing the
        [CLS] token’s hidden state after the pooler.

    Returns
    -------
    torch.Tensor
        NSP logits of shape ``(batch_size, 2)``.
    """
    return self.seq_relationship(pooled_output)