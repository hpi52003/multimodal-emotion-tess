"""Speech preprocessing and feature extraction"""
from fractions import Fraction
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
import torchaudio
from scipy.signal import resample_poly

SR = 16000
N_MELS = 64
HOP = 160
WIN = 400
MAX_SECONDS = 4.0

_mel = torchaudio.transforms.MelSpectrogram(
    sample_rate=SR, n_fft=512, win_length=WIN, hop_length=HOP, f_min=20.0, f_max=8000.0, n_mels=N_MELS)


def load_wave(path, sr=SR):
    x, s = sf.read(str(path), dtype="float32", always_2d=True)
    x = x.mean(axis=1)                                   # mono
    if s != sr:
        r = Fraction(sr, int(s)).limit_denominator(1000)
        x = resample_poly(x, r.numerator, r.denominator).astype(np.float32)
    return x


def trim_silence(x, top_db=35.0, frame=WIN, hop=HOP):
    """Energy-based trimming: drop leading/trailing frames more than top_db below the peak."""
    if len(x) < frame:
        return x
    n = 1 + (len(x) - frame) // hop
    idx = np.arange(frame)[None, :] + hop * np.arange(n)[:, None]
    rms = np.sqrt((x[idx] ** 2).mean(axis=1) + 1e-10)
    db = 20 * np.log10(rms / (rms.max() + 1e-10) + 1e-10)
    keep = np.where(db > -top_db)[0]
    if len(keep) == 0:
        return x
    return x[keep[0] * hop: keep[-1] * hop + frame]


def preprocess(path):
    x = load_wave(path)
    x = x / (np.abs(x).max() + 1e-8)
    x = trim_silence(x)
    return x[: int(MAX_SECONDS * SR)]


def logmel(x):
    m = _mel(torch.from_numpy(x).float())                # (n_mels, T)
    m = torch.log(m + 1e-6)
    m = m - m.mean(dim=1, keepdim=True)                  # per-band mean normalisation
    return m.transpose(0, 1).contiguous()                # (T, n_mels)


def build_cache(paths, cache_file):
    """Compute (or load) log-Mel features for all paths once; stored as float16 in one file."""
    cache_file = Path(cache_file)
    if cache_file.exists():
        return torch.load(cache_file)
    feats = {}
    for i, p in enumerate(paths):
        feats[p] = logmel(preprocess(p)).half()
        if (i + 1) % 200 == 0:
            print(f"  features {i + 1}/{len(paths)}")
    cache_file.parent.mkdir(parents=True, exist_ok=True)
    torch.save(feats, cache_file)
    return feats
