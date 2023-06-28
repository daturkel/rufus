from typing import Sequence

import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoModel, AutoTokenizer  # type: ignore


def mean_pooling(model_output, attention_mask):
    """Mean Pooling

    Takes attention mask into account for correct averaging Takes attention mask into"""
    token_embeddings = model_output[
        0
    ]  # First element of model_output contains all token embeddings
    input_mask_expanded = (
        attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    )
    return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(
        input_mask_expanded.sum(1), min=1e-9
    )


def get_tokenizer_and_model(model: str) -> tuple[AutoTokenizer, AutoModel]:
    # Load model from HuggingFace Hub
    tokenizer = AutoTokenizer.from_pretrained(model)
    model = AutoModel.from_pretrained(model)
    return tokenizer, model


def get_sentence_embeddings(sentences: Sequence[str], tokenizer, model) -> np.ndarray:
    # Tokenize sentences
    encoded_input = tokenizer(
        sentences, padding=True, truncation=True, return_tensors="pt"
    )

    # Compute token embeddings
    with torch.no_grad():
        model_output = model(**encoded_input)

    # Perform pooling
    sentence_embeddings = mean_pooling(model_output, encoded_input["attention_mask"])

    # Normalize embeddings
    sentence_embeddings = F.normalize(sentence_embeddings, p=2, dim=1)

    return sentence_embeddings.numpy()
