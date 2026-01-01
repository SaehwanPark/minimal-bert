"""Test that the custom BERT implementation matches Hugging Face outputs."""

import pytest
import torch
from transformers import BertModel as HFBertModel, BertTokenizer

from src.config import BertConfig
from src.bert_model import BertModel
from src.utils import rename_state_dict_keys


@pytest.mark.integration
def test_custom_model_matches_hf() -> None:
  # Skip test if no internet connection or HF download fails
  try:
    model_name = "bert-base-uncased"
    hf_model = HFBertModel.from_pretrained(model_name)
    tokenizer = BertTokenizer.from_pretrained(model_name)
  except Exception:
    pytest.skip("Hugging Face model could not be loaded")

  hf_model.eval()
  # Create custom config matching HF
  config = BertConfig(
    vocab_size=hf_model.config.vocab_size,
    hidden_size=hf_model.config.hidden_size,
    num_attention_heads=hf_model.config.num_attention_heads,
    num_hidden_layers=hf_model.config.num_hidden_layers,
    intermediate_size=hf_model.config.intermediate_size,
    max_position_embeddings=hf_model.config.max_position_embeddings,
    type_vocab_size=hf_model.config.type_vocab_size,
    hidden_dropout_prob=0.0,
    attention_probs_dropout_prob=0.0,
    layer_norm_eps=hf_model.config.layer_norm_eps,
    pad_token_id=hf_model.config.pad_token_id,
  )
  custom_model = BertModel(config)
  custom_model.eval()

  # Load weights into custom model
  renamed_state_dict = rename_state_dict_keys(hf_model.state_dict())
  missing, unexpected = custom_model.load_state_dict(renamed_state_dict, strict=False)

  # Ensure we didn't miss any critical encoder weights
  assert len(missing) == 0, f"Failed to load specific keys: {missing}"

  # Tokenise sample input
  encoded = tokenizer(
    "Testing the custom implementation",
    "with a second sentence",
    return_tensors="pt",
    max_length=16,
    truncation=True,
    padding="max_length",
  )
  input_ids = encoded["input_ids"]
  token_type_ids = encoded["token_type_ids"]
  attention_mask = encoded["attention_mask"]

  with torch.no_grad():
    hf_output = hf_model(
      input_ids=input_ids,
      attention_mask=attention_mask,
      token_type_ids=token_type_ids,
    ).last_hidden_state
    custom_output, _, _, _ = custom_model(
      input_ids=input_ids,
      attention_mask=attention_mask,
      token_type_ids=token_type_ids,
    )

  # Compute mean absolute difference and assert it's within tolerance
  diff = (custom_output - hf_output).abs().mean().item()
  assert diff < 1e-5
