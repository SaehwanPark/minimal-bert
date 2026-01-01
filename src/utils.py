"""Utility functions for the custom BERT implementation.

This module contains helper functions that are not specific to any
particular layer.  They assist with preparing attention masks and
loading pretrained weights from Hugging Face models.
"""

from __future__ import annotations

from typing import Dict, Any

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
  """Map Hugging Face BERT state dict keys to the custom model's keys.

  When loading weights from a pre‑trained Hugging Face model into our
  custom implementation, the parameter names differ.  This function
  rewrites the keys in a provided state dictionary so that they match
  the naming scheme used in this project.  Only keys relevant to
  ``BertModel`` and ``BertForPreTraining`` are handled; keys for other
  modules are ignored.  Unrecognised keys remain unchanged.

  Parameters
  ----------
  state_dict:
      The original state dictionary from a Hugging Face BERT model.

  Returns
  -------
  Dict[str, Any]
      A new state dictionary with remapped keys.
  """
  new_state_dict: Dict[str, Any] = {}
  for key, value in state_dict.items():
    # Skip keys for the embeddings if they exist
    if key.startswith("bert.embeddings.word_embeddings.weight"):
      new_state_dict["embeddings.word_embeddings.weight"] = value
    elif key.startswith("bert.embeddings.position_embeddings.weight"):
      new_state_dict["embeddings.position_embeddings.weight"] = value
    elif key.startswith("bert.embeddings.token_type_embeddings.weight"):
      new_state_dict["embeddings.token_type_embeddings.weight"] = value
    elif key.startswith("bert.embeddings.LayerNorm.weight"):
      new_state_dict["embeddings.layer_norm.weight"] = value
    elif key.startswith("bert.embeddings.LayerNorm.bias"):
      new_state_dict["embeddings.layer_norm.bias"] = value
    # Encoder layers
    elif key.startswith("bert.encoder.layer"):
      # Extract layer index and parameter name
      parts = key.split('.')
      # Format: bert.encoder.layer.{layer_idx}.{module}.weight or bias
      layer_idx = parts[3]
      submodule = parts[4]
      param_name = '.'.join(parts[5:])
      prefix = f"encoder.layers.{layer_idx}."
      if submodule == "attention":
        # attention.self.query.weight -> encoder.layers.0.attention.query.weight
        attn_sub = '.'.join(parts[5:])
        new_key = prefix + attn_sub
        # Remove 'self.' from submodule names
        new_key = new_key.replace('self.', '')
        # Convert 'attention.output' to 'output'
        new_key = new_key.replace('output', 'out_proj')
        new_state_dict[new_key] = value
      elif submodule == "attention_output":
        # In HF models, this is named attention.output.* but already
        # handled above.
        continue
      elif submodule == "intermediate":
        new_state_dict[prefix + "intermediate.weight" if param_name == "dense.weight" else prefix + "intermediate.bias"] = value
      elif submodule == "output":
        # output.dense.weight -> output.weight
        # output.LayerNorm.weight -> norm2.weight
        if param_name.startswith("dense.weight"):
          new_state_dict[prefix + "output.weight"] = value
        elif param_name.startswith("dense.bias"):
          new_state_dict[prefix + "output.bias"] = value
        elif param_name.startswith("LayerNorm.weight"):
          new_state_dict[prefix + "norm2.weight"] = value
        elif param_name.startswith("LayerNorm.bias"):
          new_state_dict[prefix + "norm2.bias"] = value
      elif submodule == "LayerNorm":
        # The first layer norm inside attention (norm1)
        ln_attr = parts[5]  # weight or bias
        new_state_dict[prefix + "norm1." + ln_attr] = value
    # Pooler
    elif key.startswith("bert.pooler.dense.weight"):
      new_state_dict["pooler.dense.weight"] = value
    elif key.startswith("bert.pooler.dense.bias"):
      new_state_dict["pooler.dense.bias"] = value
    # MLM & NSP heads
    elif key.startswith("cls.predictions.transform.dense.weight"):
      new_state_dict["cls.predictions.transform.dense.weight"] = value
    elif key.startswith("cls.predictions.transform.dense.bias"):
      new_state_dict["cls.predictions.transform.dense.bias"] = value
    elif key.startswith("cls.predictions.transform.LayerNorm.weight"):
      new_state_dict["cls.predictions.transform.LayerNorm.weight"] = value
    elif key.startswith("cls.predictions.transform.LayerNorm.bias"):
      new_state_dict["cls.predictions.transform.LayerNorm.bias"] = value
    elif key.startswith("cls.predictions.decoder.weight"):
      new_state_dict["cls.predictions.decoder.weight"] = value
    elif key.startswith("cls.predictions.bias"):
      new_state_dict["cls.predictions.bias"] = value
    elif key.startswith("cls.seq_relationship.weight"):
      new_state_dict["cls.seq_relationship.weight"] = value
    elif key.startswith("cls.seq_relationship.bias"):
      new_state_dict["cls.seq_relationship.bias"] = value
    else:
      # For unrecognised keys, carry them over unchanged
      new_state_dict[key] = value
  return new_state_dict