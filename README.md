# minimalBERT Implementation from Scratch

This repository contains a clean, educational implementation of the BERT (Bidirectional Encoder Representations from Transformers) model using PyTorch.

The goal of this project is to demystify the internal workings of BERT by building the model component-by-component—from embeddings and multi-head attention to the encoder stack and pre-training heads. While built from scratch, it is mathematically equivalent to the official implementation, allowing for the loading of pre-trained weights from Hugging Face's `transformers` library.

## Features

* **Modular Architecture**: broken down into `Embeddings`, `MultiHeadSelfAttention`, and `TransformerEncoderLayer` for clarity.
* **Hugging Face Compatibility**: Includes a utility to map and load official `bert-base-uncased` weights into this custom architecture.
* **Pre-training Heads**: Implements both Masked Language Modeling (MLM) and Next Sentence Prediction (NSP) heads.
* **Demo Scripts**: Includes scripts for both inference (weight transfer verification) and pre-training on toy data.


## Installation

1. **Clone the repository:**

2. **Install dependencies:**
This project requires PyTorch and the Hugging Face `transformers` library (used for tokenization and reference comparison).
```bash
uv sync
```



## Usage

### 1. Inference & Compatibility Check

The inference demo loads the pre-trained `bert-base-uncased` model from Hugging Face, transfers the weights to our custom implementation, and feeds the same input to both. It verifies that the Mean Absolute Difference between the outputs is negligible.

```bash
uv run inference_demo.py

```

*Expected output:* A log message indicating a very small difference (e.g., `< 1e-5`) and confirmation that results were saved to the `results/` directory.

### 2. Pre-training Demo

The training demo initializes a fresh BERT model and trains it on a small "toy" dataset using the MLM and NSP objectives. This demonstrates the training loop logic.

```bash
uv run training_demo.py

```

*Output:* Training loss logs for each epoch and a saved model checkpoint in `results/toy_bert_pretrained.pth`.

## Configuration

The model is configured using the `BertConfig` class in `src/config.py`. By default, it mirrors the `bert-base-uncased` architecture:

* **Hidden Size:** 768
* **Attention Heads:** 12
* **Layers:** 12
* **Vocab Size:** 30522

You can modify these parameters in the scripts or by passing a custom config object during initialization.

## Testing

The project includes a comprehensive test suite using `pytest`. The tests cover output shapes, layer logic, and integration with Hugging Face weights.

To run the tests:

```bash
uv run pytest

```

## Implementation Details

* **Embeddings**: Sums word, position, and token_type embeddings, followed by LayerNorm and Dropout.
* **Encoder**: A stack of `TransformerEncoderLayer` modules. Each layer consists of `MultiHeadSelfAttention` and a feed-forward network, connected by residual links and LayerNorm.
* **Weight Loading**: The `src/utils.py` module handles the complex mapping of state dictionary keys from the Hugging Face format to our custom module structure.
