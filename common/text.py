"""Text preprocessing and feature extraction"""
import re

import torch

PAD, UNK = "<pad>", "<unk>"


def clean(text: str) -> str:
    text = text.lower()
    text = re.sub(r"[^a-z0-9' ]+", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def tokenize(text: str):
    return clean(text).split()


def build_vocab(texts):
    vocab = {PAD: 0, UNK: 1}
    for t in texts:
        for tok in tokenize(t):
            vocab.setdefault(tok, len(vocab))
    return vocab


def encode(text, vocab):
    return torch.tensor([vocab.get(tok, vocab[UNK]) for tok in tokenize(text)], dtype=torch.long)


def bert_features(texts, model_name="distilbert-base-uncased", device="cpu"):
    """Frozen DistilBERT hidden states per unique sentence -> dict text -> (tokens, 768)."""
    from transformers import AutoModel, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name).to(device).eval()
    out = {}
    with torch.no_grad():
        for t in sorted(set(texts)):
            enc = tok(clean(t), return_tensors="pt").to(device)
            out[t] = model(**enc).last_hidden_state[0].cpu()
    return out
