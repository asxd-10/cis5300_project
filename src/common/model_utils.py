"""
Shared model utilities for all components
TODO (Ashay): Once individual components are done, refactor to move common code here.
"""

import torch
from transformers import AutoTokenizer, AutoModel
from typing import List, Dict


def load_scibert(model_name: str = "allenai/scibert_scivocab_uncased"):
    """Load SciBERT model and tokenizer."""
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name)
    return tokenizer, model


def encode_text(text: str, tokenizer, max_length: int = 512) -> Dict:
    """Encode text using tokenizer."""
    return tokenizer(
        text,
        max_length=max_length,
        padding='max_length',
        truncation=True,
        return_tensors='pt'
    )


def set_seed(seed: int = 42):
    """Set random seed for reproducibility."""
    import random
    import numpy as np
    
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


if __name__ == "__main__":
    # Test model utilities
    set_seed(123)
    tokenizer, model = load_scibert()
