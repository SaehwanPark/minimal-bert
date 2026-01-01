"""Core BERT model implementation.

This module defines the encoder stack and high level classes that
assemble the embeddings, encoder and pooling layers into a complete
transformer model.  It also defines a pre‑training model that adds
masked language modelling and next sentence prediction heads on top of
the base encoder.
"""

from __future__ import annotations

from typing import List, Optional, Tuple, Union

import torch
from torch import nn

from .config import BertConfig
from .embeddings import BertEmbeddings
from .transformer_encoder_layer import TransformerEncoderLayer
from .utils import create_extended_attention_mask
from .heads import MaskedLanguageModelHead, NextSentencePredictionHead


class BertEncoder(nn.Module):
  """Stack of transformer encoder layers.

  Parameters
  ----------
  config:
      Model configuration specifying the number of layers and other
      hyper‑parameters.
  """

  def __init__(self, config: BertConfig) -> None:
    super().__init__()
    self.layers = nn.ModuleList(
      [TransformerEncoderLayer(config) for _ in range(config.num_hidden_layers)]
    )

  def forward(
    self,
    hidden_states: torch.Tensor,
    attention_mask: Optional[torch.Tensor] = None,
    output_hidden_states: bool = False,
    output_attentions: bool = False,
  ) -> Tuple[torch.Tensor, Optional[List[torch.Tensor]], Optional[List[torch.Tensor]]]:
    """Apply the encoder stack to the hidden states.

    Parameters
    ----------
    hidden_states:
        Input tensor of shape ``(batch_size, seq_len, hidden_size)``.
    attention_mask:
        Optional broadcastable mask of shape ``(batch_size, 1, 1, seq_len)``.
    output_hidden_states:
        Whether to return a list of all hidden states for every layer.
    output_attentions:
        Whether to return attention probabilities for each layer.

    Returns
    -------
    Tuple containing:
        - last hidden states of shape ``(batch_size, seq_len, hidden_size)``,
        - list of hidden states from each layer if requested,
        - list of attention probabilities if requested.
    """
    all_hidden_states: Optional[List[torch.Tensor]] = (
      [] if output_hidden_states else None
    )
    all_attentions: Optional[List[torch.Tensor]] = [] if output_attentions else None
    for layer in self.layers:
      if output_hidden_states:
        # Save the current hidden state prior to the layer
        assert all_hidden_states is not None  # for type checker
        all_hidden_states.append(hidden_states)

      hidden_states, attn_probs = layer(hidden_states, attention_mask)

      if output_attentions:
        assert all_attentions is not None
        all_attentions.append(attn_probs)

    # append the final hidden state (output of the last layer)
    if output_hidden_states:
      assert all_hidden_states is not None
      all_hidden_states.append(hidden_states)

    return hidden_states, all_hidden_states, all_attentions


class BertPooler(nn.Module):
  """Pooler for extracting a sequence representation from the [CLS] token.

  The pooler applies a linear layer followed by a ``tanh`` activation to
  the hidden state corresponding to the first token in the sequence
  (which is assumed to be the special [CLS] token).  This pooled
  representation can be used for classification tasks.
  """

  def __init__(self, config: BertConfig) -> None:
    super().__init__()
    self.dense = nn.Linear(config.hidden_size, config.hidden_size)
    self.activation = nn.Tanh()

  def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
    # hidden_states shape: (batch_size, seq_len, hidden_size)
    cls_token = hidden_states[:, 0]
    pooled_output = self.dense(cls_token)
    pooled_output = self.activation(pooled_output)
    return pooled_output


class BertModel(nn.Module):
  """BERT encoder model without pre‑training heads.

  This class assembles the embeddings, encoder stack and pooler into a
  coherent model that produces contextualised token representations and
  a pooled [CLS] representation.  It accepts raw token IDs and
  constructs the necessary attention mask internally.
  """

  def __init__(self, config: BertConfig) -> None:
    super().__init__()
    self.config = config
    self.embeddings = BertEmbeddings(config)
    self.encoder = BertEncoder(config)
    self.pooler = BertPooler(config)

  def forward(
    self,
    input_ids: torch.Tensor,
    attention_mask: Optional[torch.Tensor] = None,
    token_type_ids: Optional[torch.Tensor] = None,
    output_hidden_states: bool = False,
    output_attentions: bool = False,
  ) -> Tuple[
    torch.Tensor,
    torch.Tensor,
    Optional[List[torch.Tensor]],
    Optional[List[torch.Tensor]],
  ]:
    """Perform forward pass through the BERT model.

    Parameters
    ----------
    input_ids:
        Tensor of shape ``(batch_size, seq_len)`` containing token IDs.
    attention_mask:
        Optional tensor of shape ``(batch_size, seq_len)`` with 1 for
        real tokens and 0 for padding.  If ``None``, the mask is
        computed by checking tokens unequal to the ``pad_token_id``.
    token_type_ids:
        Optional tensor of shape ``(batch_size, seq_len)`` indicating
        segment membership.  Defaults to all zeros if not provided.
    output_hidden_states:
        Whether to return a list of hidden states from all layers.
    output_attentions:
        Whether to return a list of attention probabilities.

    Returns
    -------
    Tuple containing:
        - sequence output of shape ``(batch_size, seq_len, hidden_size)``,
        - pooled output of shape ``(batch_size, hidden_size)``,
        - list of hidden states if requested,
        - list of attentions if requested.
    """
    if attention_mask is None:
      # Derive the mask from input_ids: pad tokens have id equal to pad_token_id
      attention_mask = (input_ids != self.config.pad_token_id).long()
    if token_type_ids is None:
      token_type_ids = torch.zeros_like(input_ids)

    # Compute extended attention mask for broadcasting in attention layers
    extended_attention_mask = create_extended_attention_mask(attention_mask)

    # Embeddings
    embedding_output = self.embeddings(input_ids, token_type_ids)
    # Encoder
    sequence_output, all_hidden_states, all_attentions = self.encoder(
      embedding_output,
      attention_mask=extended_attention_mask,
      output_hidden_states=output_hidden_states,
      output_attentions=output_attentions,
    )
    # Pooler
    pooled_output = self.pooler(sequence_output)
    return sequence_output, pooled_output, all_hidden_states, all_attentions


class BertForPreTraining(nn.Module):
  """BERT model with pre‑training heads for MLM and NSP tasks.

  This model wraps :class:`BertModel` and adds two task‑specific heads.
  It optionally computes losses when labels are provided and returns
  logits otherwise.
  """

  def __init__(self, config: BertConfig) -> None:
    super().__init__()
    self.bert = BertModel(config)
    # Tie the weights of the MLM decoder to the token embeddings
    self.cls = MaskedLanguageModelHead(
      config, self.bert.embeddings.word_embeddings.weight
    )
    self.nsp = NextSentencePredictionHead(config)

  def forward(
    self,
    input_ids: torch.Tensor,
    attention_mask: Optional[torch.Tensor] = None,
    token_type_ids: Optional[torch.Tensor] = None,
    labels: Optional[torch.Tensor] = None,
    next_sentence_label: Optional[torch.Tensor] = None,
    output_hidden_states: bool = False,
    output_attentions: bool = False,
  ) -> Union[
    Tuple[torch.Tensor, torch.Tensor], Tuple[torch.Tensor, torch.Tensor, torch.Tensor]
  ]:
    """Compute logits (and optionally losses) for MLM and NSP tasks.

    Parameters
    ----------
    input_ids:
        Tensor of shape ``(batch_size, seq_len)`` with token IDs.
    attention_mask:
        Optional binary mask of shape ``(batch_size, seq_len)``.  If ``None``
        it is inferred from ``input_ids``.
    token_type_ids:
        Optional tensor of shape ``(batch_size, seq_len)`` identifying
        segments.
    labels:
        Optional tensor of shape ``(batch_size, seq_len)`` containing the
        target token IDs for the MLM task.  Positions not used for MLM
        should be filled with ``-100``.
    next_sentence_label:
        Optional tensor of shape ``(batch_size,)`` with values 0 or 1
        indicating whether the second sentence is the actual next
        sentence.
    output_hidden_states:
        Whether to return hidden states from all layers.
    output_attentions:
        Whether to return attention probabilities from all layers.

    Returns
    -------
    Tuple
        If ``labels`` and ``next_sentence_label`` are provided, returns a
        tuple ``(total_loss, prediction_scores, seq_relationship_scores)``.
        Otherwise returns ``(prediction_scores, seq_relationship_scores)``.
    """
    sequence_output, pooled_output, all_hidden_states, all_attentions = self.bert(
      input_ids,
      attention_mask=attention_mask,
      token_type_ids=token_type_ids,
      output_hidden_states=output_hidden_states,
      output_attentions=output_attentions,
    )
    prediction_scores = self.cls(sequence_output)
    seq_relationship_scores = self.nsp(pooled_output)

    # Compute loss if labels provided
    if labels is not None and next_sentence_label is not None:
      # Flatten the predictions and labels for MLM
      loss_fct = nn.CrossEntropyLoss(ignore_index=-100)
      mlm_loss = loss_fct(
        prediction_scores.view(-1, self.bert.config.vocab_size), labels.view(-1)
      )
      # NSP loss
      nsp_loss_fct = nn.CrossEntropyLoss()
      nsp_loss = nsp_loss_fct(
        seq_relationship_scores.view(-1, 2), next_sentence_label.view(-1)
      )
      total_loss = mlm_loss + nsp_loss
      return total_loss, prediction_scores, seq_relationship_scores
    return prediction_scores, seq_relationship_scores
