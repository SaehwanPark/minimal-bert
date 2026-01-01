"""Utility functions for the custom BERT implementation.

This module contains helper functions that are not specific to any
particular layer.  They assist with preparing attention masks and
loading pretrained weights from Hugging Face models.
"""

from __future__ import annotations

from typing import Any, Dict

import torch


def create_extended_attention_mask(attention_mask: torch.Tensor) -> torch.Tensor:
  """Expand a 2D attention mask to a 4D mask for broadcasting in attention.

  The original BERT implementation expects an attention mask of shape
  ``(batch_size, seq_len)`` containing 1 for tokens that should be
  attended to and 0 for padded tokens.  This function converts that
  mask into a shape ``(batch_size, 1, 1, seq_len)`` and changes the
  value range to either 0.0 for attendable tokens or a large negative
  value for masked tokens.  The negative values ensure that masked
  positions have near‑zero attention probability after the softmax.

  Parameters
  ----------
  attention_mask:
      Tensor of shape ``(batch_size, seq_len)`` with binary values.

  Returns
  -------
  torch.Tensor
      A broadcastable attention mask of shape ``(batch_size, 1, 1, seq_len)``.
  """
  # If attention_mask has shape (batch_size, seq_len), convert it to float
  extended_attention_mask = attention_mask[:, None, None, :].float()
  # Mask out padded tokens by assigning a large negative value
  extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0
  return extended_attention_mask


def rename_state_dict_keys(state_dict: Dict[str, Any]) -> Dict[str, Any]:
  """Remap HF keys to match the custom BERT model's internal structure."""
  new_state_dict: Dict[str, Any] = {}

  for key, value in state_dict.items():
    new_key = key

    # 1. Handle Embeddings
    if new_key.startswith("embeddings"):
      # Map LayerNorm -> layer_norm
      new_key = new_key.replace("LayerNorm", "layer_norm")

    # 2. Handle Encoder Layers
    elif new_key.startswith("encoder.layer"):
      # Map layer -> layers (HF uses singular, you use plural)
      new_key = new_key.replace("encoder.layer.", "encoder.layers.")

      # Self-Attention Q/K/V
      new_key = new_key.replace("attention.self.", "attention.")

      # Attention Output Dense -> out_proj
      new_key = new_key.replace("attention.output.dense.", "attention.out_proj.")

      # Map LayerNorms (norm1 and norm2)
      new_key = new_key.replace("attention.output.LayerNorm", "norm1")
      new_key = new_key.replace("output.LayerNorm", "norm2")

      # Feed-Forward Network
      new_key = new_key.replace("intermediate.dense.", "intermediate.")
      new_key = new_key.replace("output.dense.", "output.")

    # 3. Pooler keys match exactly in this case, but we check just in case
    # 'pooler.dense.weight' -> 'pooler.dense.weight'

    new_state_dict[new_key] = value

  return new_state_dict


def get_torch_accelerator() -> torch.device:
  try:
    import torch_xla.core.xla_model as xm

    return xm.xla_device()
  except ImportError:
    pass

  if torch.cuda.is_available():
    return torch.device("cuda")

  if torch.backends.mps.is_available():
    return torch.device("mps")

  return torch.device("cpu")
