#!/usr/bin/env python3
"""Build data/metadata.csv for TESS (Toronto Emotional Speech Set) and create the splits.
Usage:
  python common/prepare_data.py --tess_dir /path/to/TESS --out data/metadata.csv
"""
import argparse
import random
from pathlib import Path

import pandas as pd

EMOTIONS = ["angry", "disgust", "fear", "happy", "neutral", "ps", "sad"]
ALIASES = {
    "angry": "angry", "disgust": "disgust", "fear": "fear", "happy": "happy",
    "neutral": "neutral", "sad": "sad", "ps": "ps",
    "pleasant_surprise": "ps", "pleasant_surprised": "ps", "surprise": "ps", "surprised": "ps",
}


def parse_name(stem: str):
    """speaker_word_emotion -> (speaker, word, emotion) or None if it cannot be parsed."""
    parts = stem.lower().split("_")
    if len(parts) < 3:
        return None
    emo = None
    n_emo_tokens = 0
    if "_".join(parts[-2:]) in ALIASES:
        emo, n_emo_tokens = ALIASES["_".join(parts[-2:])], 2
    elif parts[-1] in ALIASES:
        emo, n_emo_tokens = ALIASES[parts[-1]], 1
    if emo is None:
        return None
    word = "_".join(parts[1:len(parts) - n_emo_tokens])
    if not word:
        return None
    speaker = {"OA": "OAF", "YA": "YAF"}.get(parts[0].upper(), parts[0].upper())   # one file is named OA_bite_neutral.wav
    return speaker, word, emo


def make_splits(df, mode, seed, test_speaker):
    rng = random.Random(seed)
    df = df.copy()
    if mode == "word":
        words = sorted(df["word"].unique())
        rng.shuffle(words)
        n = len(words)
        n_test, n_val = int(round(0.15 * n)), int(round(0.15 * n))
        test_w, val_w = set(words[:n_test]), set(words[n_test:n_test + n_val])
        df["split"] = df["word"].map(lambda w: "test" if w in test_w else ("val" if w in val_w else "train"))
    else:
        speakers = sorted(df["speaker"].unique())
        if test_speaker not in speakers:
            raise SystemExit(f"--test_speaker {test_speaker} not in {speakers}")
        rest = df[df["speaker"] != test_speaker]
        words = sorted(rest["word"].unique())
        rng.shuffle(words)
        val_w = set(words[:int(round(0.15 * len(words)))])
        def assign(r):
            if r["speaker"] == test_speaker:
                return "test"
            return "val" if r["word"] in val_w else "train"
        df["split"] = df.apply(assign, axis=1)
    return df


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tess_dir", required=True, help="folder that contains the TESS .wav files (searched recursively)")
    ap.add_argument("--out", default="data/metadata.csv")
    ap.add_argument("--split_mode", choices=["word", "speaker"], default="word")
    ap.add_argument("--test_speaker", default="YAF")
    ap.add_argument("--seed", type=int, default=42)
    a = ap.parse_args()

    rows, skipped = [], []
    for p in sorted(Path(a.tess_dir).rglob("*.wav")):
        parsed = parse_name(p.stem)
        if parsed is None:
            skipped.append(p.name)
            continue
        speaker, word, emo = parsed
        rows.append(dict(path=str(p.resolve()), speaker=speaker, word=word, emotion=emo,
                         label=EMOTIONS.index(emo), text=f"say the word {word}"))
    if not rows:
        raise SystemExit(f"No TESS wav files found under {a.tess_dir}")
    df = pd.DataFrame(rows)
    # The Kaggle zip contains the whole data set twice (a nested copy). Keep one file per
    # (speaker, word, emotion), preferring the shallowest path.
    n_before = len(df)
    df["_depth"] = df["path"].str.count("/")
    df = (df.sort_values(["_depth", "path"]).drop_duplicates(["speaker", "word", "emotion"], keep="first")
            .drop(columns="_depth").sort_values("path").reset_index(drop=True))
    if len(df) < n_before:
        print(f"removed {n_before - len(df)} duplicate copies of the same clips")
    df = make_splits(df, a.split_mode, a.seed, a.test_speaker)
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(a.out, index=False)

    print(f"{len(df)} clips, {df['speaker'].nunique()} speakers, {df['word'].nunique()} words -> {a.out}")
    if skipped:
        print(f"skipped {len(skipped)} unparseable files, e.g. {skipped[:3]}")
    print(df.groupby(["split", "emotion"]).size().unstack(fill_value=0))
    shared = len(set(df[df.split == 'train'].word) & set(df[df.split == 'test'].word))
    note = "  (expected: both speakers say the same words)" if a.split_mode == "speaker" else ""
    print(f"\nwords shared between train and test: {shared}{note}")


if __name__ == "__main__":
    main()
