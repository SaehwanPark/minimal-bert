"""Transformer encoder layer used in the BERT model.

The :class:`TransformerEncoderLayer` encapsulates a single transformer block
composed of multi‑head self‑attention, followed by a position‑wise feed‑
forward network.  Each sub‑layer is wrapped with residual connections
and layer normalisation, with dropout applied after the residuals.  This
layer forms the building block of the encoder stack in BERT.
"""

from __future__ import annotations

from typing import Optional, Tuple

import torch
from torch import nn
import torch.nn.functional as F

from .config import BertConfig
from .multi_head_attention import MultiHeadSelfAttention


class TransformerEncoderLayer(nn.Module):
  """Single transformer encoder layer.

  Parameters
  ----------
  config:
      Configuration containing model hyper‑parameters.
  """

  def __init__(self, config: BertConfig) -> None:
    super().__init__()
    self.attention = MultiHeadSelfAttention(config)
    self.dropout1 = nn.Dropout(config.hidden_dropout_prob)
    self.norm1 = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)

    # Feed‑forward network
    self.intermediate = nn.Linear(config.hidden_size, config.intermediate_size)
    self.intermediate_act_fn = F.gelu
    self.output = nn.Linear(config.intermediate_size, config.hidden_size)
    self.dropout2 = nn.Dropout(config.hidden_dropout_prob)
    self.norm2 = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)

  def forward(
    self,
    hidden_states: torch.Tensor,
    attention_mask: Optional[torch.Tensor] = None,
  ) -> Tuple[torch.Tensor, torch.Tensor]:
    """Apply the transformer layer to the hidden states.

    Parameters
    ----------
    hidden_states:
        Tensor of shape ``(batch_size, seq_len, hidden_size)`` containing
        the input activations.
    attention_mask:
        Optional tensor broadcastable to ``(batch_size, 1, 1, seq_len)``
        containing additive mask values.

    Returns
    -------
    Tuple[torch.Tensor, torch.Tensor]
        A tuple ``(hidden_states, attentions)``.  The first element is
        the output of the layer with shape ``(batch_size, seq_len, hidden_size)``.
        The second element contains the attention probabilities from the
        self‑attention module.
    """
    # Self‑attention with residual connection and layer norm
    attn_output, attn_probs = self.attention(hidden_states, attention_mask)
    hidden_states = hidden_states + self.dropout1(attn_output)
    hidden_states = self.norm1(hidden_states)

    # Feed‑forward network
    intermediate_output = self.intermediate_act_fn(self.intermediate(hidden_states))
    layer_output = self.output(intermediate_output)
    hidden_states = hidden_states + self.dropout2(layer_output)
    hidden_states = self.norm2(hidden_states)
    return hidden_states, attn_probs