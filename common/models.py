"""Model blocks.

SpeechEncoder  = Temporal modelling : Conv1d front-end -> 2-layer BiGRU -> attention pooling
TextEncoder    = Contextual modelling: (embedding) -> BiLSTM -> attention pooling
Fusion         = concat (baseline) or gated fusion
Classifier     = MLP head with softmax (applied through the cross-entropy loss)
"""
import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

N_CLASSES = 7


def length_mask(lengths, max_len):
    return torch.arange(max_len, device=lengths.device)[None, :] < lengths[:, None]


class AttnPool(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.score = nn.Linear(dim, 1)

    def forward(self, x, lengths):                       # x: (B, T, D)
        s = self.score(x).squeeze(-1)
        s = s.masked_fill(~length_mask(lengths, x.size(1)), float("-inf"))
        w = torch.softmax(s, dim=1).unsqueeze(-1)
        return (w * x).sum(dim=1)


def run_rnn(rnn, x, lengths):
    packed = pack_padded_sequence(x, lengths.cpu(), batch_first=True, enforce_sorted=False)
    out, _ = rnn(packed)
    out, _ = pad_packed_sequence(out, batch_first=True, total_length=x.size(1))
    return out


class SpeechEncoder(nn.Module):
    def __init__(self, n_mels=64, conv_ch=64, hid=128, emb=256, dropout=0.3):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(n_mels, conv_ch, 5, padding=2), nn.BatchNorm1d(conv_ch), nn.ReLU(),
            nn.Conv1d(conv_ch, conv_ch, 5, stride=2, padding=2), nn.BatchNorm1d(conv_ch), nn.ReLU(),
            nn.Dropout(dropout))
        self.rnn = nn.GRU(conv_ch, hid, num_layers=2, batch_first=True, bidirectional=True, dropout=dropout)
        self.pool = AttnPool(2 * hid)
        self.proj = nn.Sequential(nn.Linear(2 * hid, emb), nn.ReLU())
        self.out_dim = emb

    def forward(self, x, lengths):                       # x: (B, T, n_mels)
        h = self.conv(x.transpose(1, 2)).transpose(1, 2)
        lengths = ((lengths - 1) // 2 + 1).clamp(min=1, max=h.size(1))
        return self.proj(self.pool(run_rnn(self.rnn, h, lengths), lengths))


class TextEncoder(nn.Module):
    def __init__(self, vocab_size=None, in_dim=768, emb_dim=100, hid=128, emb=256, dropout=0.3):
        super().__init__()
        if vocab_size is not None:                       # trainable word embeddings
            self.embed = nn.Embedding(vocab_size, emb_dim, padding_idx=0)
            in_dim = emb_dim
        else:                                            # pre-computed token features (e.g. BERT)
            self.embed = None
        self.drop = nn.Dropout(dropout)
        self.rnn = nn.LSTM(in_dim, hid, num_layers=1, batch_first=True, bidirectional=True)
        self.pool = AttnPool(2 * hid)
        self.proj = nn.Sequential(nn.Linear(2 * hid, emb), nn.ReLU())
        self.out_dim = emb

    def forward(self, x, lengths):
        if self.embed is not None:
            x = self.embed(x)
        x = self.drop(x)
        return self.proj(self.pool(run_rnn(self.rnn, x, lengths), lengths))


class Fusion(nn.Module):
    def __init__(self, d_s, d_t, out=256, mode="concat", dropout=0.3):
        super().__init__()
        self.mode, self.out_dim = mode, out
        if mode == "concat":
            self.net = nn.Sequential(nn.Linear(d_s + d_t, out), nn.ReLU(), nn.Dropout(dropout))
        elif mode == "gated":
            self.ps, self.pt = nn.Linear(d_s, out), nn.Linear(d_t, out)
            self.gate = nn.Linear(d_s + d_t, out)
            self.drop = nn.Dropout(dropout)
        else:
            raise ValueError(mode)

    def forward(self, s, t):
        if self.mode == "concat":
            return self.net(torch.cat([s, t], dim=-1))
        g = torch.sigmoid(self.gate(torch.cat([s, t], dim=-1)))
        return self.drop(g * torch.tanh(self.ps(s)) + (1 - g) * torch.tanh(self.pt(t)))


class Classifier(nn.Module):
    def __init__(self, in_dim, hid=128, n_classes=N_CLASSES, dropout=0.3):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(in_dim, hid), nn.ReLU(), nn.Dropout(dropout), nn.Linear(hid, n_classes))

    def forward(self, z):
        return self.net(z)


class SpeechModel(nn.Module):
    def __init__(self, **kw):
        super().__init__()
        self.speech = SpeechEncoder(**kw)
        self.head = Classifier(self.speech.out_dim)

    def embeddings(self, b):
        return {"temporal": self.speech(b["speech"], b["speech_len"])}

    def forward(self, b):
        return self.head(self.embeddings(b)["temporal"])


class TextModel(nn.Module):
    def __init__(self, **kw):
        super().__init__()
        self.text = TextEncoder(**kw)
        self.head = Classifier(self.text.out_dim)

    def embeddings(self, b):
        return {"contextual": self.text(b["text"], b["text_len"])}

    def forward(self, b):
        return self.head(self.embeddings(b)["contextual"])


class FusionModel(nn.Module):
    def __init__(self, speech_kw=None, text_kw=None, fusion="concat"):
        super().__init__()
        self.speech = SpeechEncoder(**(speech_kw or {}))
        self.text = TextEncoder(**(text_kw or {}))
        self.fusion = Fusion(self.speech.out_dim, self.text.out_dim, mode=fusion)
        self.head = Classifier(self.fusion.out_dim)

    def embeddings(self, b):
        s = self.speech(b["speech"], b["speech_len"])
        t = self.text(b["text"], b["text_len"])
        return {"temporal": s, "contextual": t, "fusion": self.fusion(s, t)}

    def forward(self, b):
        return self.head(self.embeddings(b)["fusion"])
