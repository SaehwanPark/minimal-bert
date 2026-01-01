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
      # Split keys: bert.encoder.layer.{layer_idx}.{submodule}...
      parts = key.split(".")
      layer_idx = parts[3]
      submodule = parts[4]

      # Base prefix for this layer in our custom model
      prefix = f"encoder.layers.{layer_idx}."

      if submodule == "attention":
        # Handle Self-Attention (query, key, value) or Output (dense, LayerNorm)
        next_part = parts[5]
        if next_part == "self":
          # Ex: bert.encoder.layer.0.attention.self.query.weight
          # -> encoder.layers.0.attention.query.weight
          param_name = parts[6]  # query, key, or value
          suffix = parts[7]  # weight or bias
          new_state_dict[f"{prefix}attention.{param_name}.{suffix}"] = value

        elif next_part == "output":
          # Ex: bert.encoder.layer.0.attention.output.dense.weight
          # -> encoder.layers.0.attention.out_proj.weight
          if parts[6] == "dense":
            suffix = parts[7]
            new_state_dict[f"{prefix}attention.out_proj.{suffix}"] = value

          # Ex: bert.encoder.layer.0.attention.output.LayerNorm.weight
          # -> encoder.layers.0.norm1.weight
          elif parts[6] == "LayerNorm":
            suffix = parts[7]
            new_state_dict[f"{prefix}norm1.{suffix}"] = value

      elif submodule == "intermediate":
        # Ex: bert.encoder.layer.0.intermediate.dense.weight
        # -> encoder.layers.0.intermediate.weight
        suffix = parts[6]  # dense (ignore)
        if suffix == "dense":
          final_suffix = parts[7]
          new_state_dict[f"{prefix}intermediate.{final_suffix}"] = value

      elif submodule == "output":
        # Ex: bert.encoder.layer.0.output.dense.weight
        # -> encoder.layers.0.output.weight
        if parts[5] == "dense":
          suffix = parts[6]
          new_state_dict[f"{prefix}output.{suffix}"] = value

        # Ex: bert.encoder.layer.0.output.LayerNorm.weight
        # -> encoder.layers.0.norm2.weight
        elif parts[5] == "LayerNorm":
          suffix = parts[6]
          new_state_dict[f"{prefix}norm2.{suffix}"] = value

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
      # FIXED: Map HF 'cls.predictions.bias' to custom 'cls.bias'
      new_state_dict["cls.bias"] = value
    elif key.startswith("cls.seq_relationship.weight"):
      new_state_dict["cls.seq_relationship.weight"] = value
    elif key.startswith("cls.seq_relationship.bias"):
      new_state_dict["cls.seq_relationship.bias"] = value
    else:
      # For unrecognised keys, carry them over unchanged
      new_state_dict[key] = value
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
