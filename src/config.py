"""Configuration dataclass for BERT models.

This module defines the :class:`BertConfig` dataclass, which holds all
hyper‑parameters required to instantiate a BERT model.  Centralising the
configuration makes it trivial to experiment with different model sizes
without modifying the core model implementation.  The fields largely
mirror those in the original BERT paper.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class BertConfig:
  """Structure holding hyper‑parameters for a BERT model.

  Attributes
  ----------
  vocab_size:
      Size of the vocabulary used to embed tokens.  Typically this is
      30522 for uncased English models or 28996 for cased variants.
  hidden_size:
      Dimensionality of hidden representations within the model.
  num_attention_heads:
      Number of attention heads used in each transformer block.  The
      ``hidden_size`` must be divisible by this value.
  num_hidden_layers:
      Number of stacked transformer encoder layers.
  intermediate_size:
      Dimensionality of the feed‑forward network within each encoder
      layer.  Often set to four times the ``hidden_size``.
  max_position_embeddings:
      Maximum length of the input sequences that the model can handle.
  type_vocab_size:
      Number of distinct segment token types (e.g. 2 for sentence A/B).
  hidden_dropout_prob:
      Dropout probability applied after embeddings and within each
      sub‑layer.
  attention_probs_dropout_prob:
      Dropout probability applied to attention weights.
  layer_norm_eps:
      Epsilon added to the denominator in layer normalization to
      improve numerical stability.
  pad_token_id:
      ID of the padding token in the vocabulary.  This is used to
      construct the attention mask for padded sequences.
  """

  vocab_size: int = 30522
  hidden_size: int = 768
  num_attention_heads: int = 12
  num_hidden_layers: int = 12
  intermediate_size: int = 3072
  max_position_embeddings: int = 512
  type_vocab_size: int = 2
  hidden_dropout_prob: float = 0.1
  attention_probs_dropout_prob: float = 0.1
  layer_norm_eps: float = 1e-12
  pad_token_id: int = 0

  def __post_init__(self) -> None:
    """Validate configuration parameters after initialisation.

    This method checks that the hidden size is divisible by the number of
    attention heads, which is a requirement for multi‑head attention.  If
    not, a ``ValueError`` is raised.
    """
    if self.hidden_size % self.num_attention_heads != 0:
      raise ValueError(
        f"hidden_size ({self.hidden_size}) must be divisible by num_attention_heads "
        f"({self.num_attention_heads})."
      )