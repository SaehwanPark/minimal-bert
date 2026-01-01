"""Multi‑head self‑attention implementation for BERT.

This module contains the :class:`MultiHeadSelfAttention` class, which
performs the core computation of the transformer encoder.  It projects
the input tensor into query, key and value tensors, splits them across
multiple heads, computes scaled dot‑product attention and returns a
combined representation.  The design follows the original BERT
implementation while remaining modular.
"""

from __future__ import annotations

from typing import Optional, Tuple

import torch
from torch import nn
import torch.nn.functional as F

from .config import BertConfig


class MultiHeadSelfAttention(nn.Module):
  """Multi‑head self‑attention layer.

  Parameters
  ----------
  config:
      Instance of :class:`BertConfig` specifying model sizes.
  """

  def __init__(self, config: BertConfig) -> None:
    super().__init__()
    self.num_heads = config.num_attention_heads
    self.hidden_size = config.hidden_size
    self.head_dim = self.hidden_size // self.num_heads

    # Projection matrices for query, key and value.  Each maps from
    # hidden_size to hidden_size.  They will be split into heads in
    # the forward pass.
    self.query = nn.Linear(self.hidden_size, self.hidden_size)
    self.key = nn.Linear(self.hidden_size, self.hidden_size)
    self.value = nn.Linear(self.hidden_size, self.hidden_size)
    self.out_proj = nn.Linear(self.hidden_size, self.hidden_size)

    self.dropout = nn.Dropout(config.attention_probs_dropout_prob)

  def transpose_for_scores(self, x: torch.Tensor) -> torch.Tensor:
    """Reshape a tensor for multi‑head attention.

    Given a tensor of shape ``(batch_size, seq_len, hidden_size)`` this
    function splits the last dimension into ``(num_heads, head_dim)`` and
    rearranges the dimensions to produce a shape of
    ``(batch_size, num_heads, seq_len, head_dim)``.

    Parameters
    ----------
    x:
        Input tensor of shape ``(batch_size, seq_len, hidden_size)``.

    Returns
    -------
    torch.Tensor
        Reshaped tensor suitable for attention computation.
    """
    new_shape = x.size()[:-1] + (self.num_heads, self.head_dim)
    x = x.view(*new_shape)
    return x.permute(0, 2, 1, 3)

  def forward(
    self,
    hidden_states: torch.Tensor,
    attention_mask: Optional[torch.Tensor] = None,
  ) -> Tuple[torch.Tensor, torch.Tensor]:
    """Compute self‑attention over the input.

    Parameters
    ----------
    hidden_states:
        Tensor of shape ``(batch_size, seq_len, hidden_size)`` representing
        the sequence of hidden states to attend over.
    attention_mask:
        Optional tensor broadcastable to ``(batch_size, 1, 1, seq_len)``
        containing additive mask values.  Positions with large negative
        values will be ignored by the softmax.

    Returns
    -------
    Tuple[torch.Tensor, torch.Tensor]
        A tuple of ``(context_layer, attention_probs)`` where
        ``context_layer`` has shape ``(batch_size, seq_len, hidden_size)``
        and represents the attended representations, and
        ``attention_probs`` has shape ``(batch_size, num_heads, seq_len, seq_len)``
        and contains the attention distributions.
    """
    # Linearly project the inputs to query, key and value
    query_layer = self.transpose_for_scores(self.query(hidden_states))
    key_layer = self.transpose_for_scores(self.key(hidden_states))
    value_layer = self.transpose_for_scores(self.value(hidden_states))

    # Compute scaled dot‑product attention scores
    # (batch, heads, seq_len, head_dim) x (batch, heads, head_dim, seq_len) -> (batch, heads, seq_len, seq_len)
    dk = float(self.head_dim)
    attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2)) / torch.sqrt(torch.tensor(dk, device=hidden_states.device))

    if attention_mask is not None:
      # Add the mask (already broadcastable) to the attention scores
      attention_scores = attention_scores + attention_mask

    # Convert scores to probabilities
    attention_probs = F.softmax(attention_scores, dim=-1)
    attention_probs = self.dropout(attention_probs)

    # Weighted sum of the values
    context_layer = torch.matmul(attention_probs, value_layer)
    # Concatenate heads and project
    context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
    new_context_shape = context_layer.size()[:-2] + (self.hidden_size,)
    context_layer = context_layer.view(*new_context_shape)
    attention_output = self.out_proj(context_layer)
    return attention_output, attention_probs