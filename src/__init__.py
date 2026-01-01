"""Top-level package for the custom BERT implementation.

This module exposes the core classes used throughout the project.  Importing
from :mod:`src` makes it easy to access the configuration, model and helper
classes without referencing deeply nested modules.
"""

from src.config import BertConfig
from src.embeddings import BertEmbeddings
from src.multi_head_attention import MultiHeadSelfAttention
from src.transformer_encoder_layer import TransformerEncoderLayer
from src.bert_model import BertModel, BertForPreTraining
from src.heads import MaskedLanguageModelHead, NextSentencePredictionHead

__all__ = [
  "BertConfig",
  "BertEmbeddings",
  "MultiHeadSelfAttention",
  "TransformerEncoderLayer",
  "BertModel",
  "BertForPreTraining",
  "MaskedLanguageModelHead",
  "NextSentencePredictionHead",
]
