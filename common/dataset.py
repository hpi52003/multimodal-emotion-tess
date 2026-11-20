"""Dataset + collate: pads variable-length speech / text and returns lengths for masking."""
import pandas as pd
import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, Dataset

from . import audio, text as T


class EmoDataset(Dataset):
    def __init__(self, df, speech=None, text_ids=None, text_feats=None):
        self.df = df.reset_index(drop=True)
        self.speech, self.text_ids, self.text_feats = speech, text_ids, text_feats

    def __len__(self):
        return len(self.df)

    def __getitem__(self, i):
        r = self.df.iloc[i]
        item = {"label": int(r["label"]), "index": i}
        if self.speech is not None:
            item["speech"] = self.speech[r["path"]].float()
        if self.text_ids is not None:
            item["text"] = self.text_ids[r["text"]]
        elif self.text_feats is not None:
            item["text"] = self.text_feats[r["text"]].float()
        return item


def collate(batch):
    out = {"label": torch.tensor([b["label"] for b in batch]),
           "index": torch.tensor([b["index"] for b in batch])}
    for k in ("speech", "text"):
        if k in batch[0]:
            out[k + "_len"] = torch.tensor([b[k].size(0) for b in batch])
            out[k] = pad_sequence([b[k] for b in batch], batch_first=True)
    return out


def load_meta(meta_csv):
    return pd.read_csv(meta_csv)


def make_datasets(meta, cache_dir, use_speech, use_text, text_features="embedding", device="cpu"):
    """Returns dict split -> EmoDataset plus a dict of objects needed to rebuild the models."""
    speech = None
    if use_speech:
        speech = audio.build_cache(sorted(meta["path"].unique()), f"{cache_dir}/speech_logmel.pt")
    text_ids = text_feats = vocab = None
    extra = {}
    if use_text:
        if text_features == "embedding":
            vocab = T.build_vocab(meta.loc[meta.split == "train", "text"])
            text_ids = {t: T.encode(t, vocab) for t in meta["text"].unique()}
            extra["vocab"] = vocab
        else:
            text_feats = T.bert_features(meta["text"].unique(), device=device)
    sets = {s: EmoDataset(meta[meta.split == s], speech, text_ids, text_feats) for s in ("train", "val", "test")}
    return sets, extra


def loaders(sets, batch_size):
    return {s: DataLoader(ds, batch_size=batch_size, shuffle=(s == "train"), collate_fn=collate)
            for s, ds in sets.items()}
