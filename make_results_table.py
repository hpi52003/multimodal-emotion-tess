#!/usr/bin/env python3
"""
Writes: Results/accuracy_table.md, Results/per_class_f1.md, Results/silhouette_table.md,
        Results/model_comparison.png, Results/per_class_f1.png
"""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parent
RES = ROOT / "Results"
EMOTIONS = ["angry", "disgust", "fear", "happy", "neutral", "ps", "sad"]
ORDER = [("speech", "Speech-only"), ("text", "Text-only"),
         ("fusion_concat", "Multimodal (concat)"), ("fusion_gated", "Multimodal (gated)")]


def load():
    data = {}
    for key, label in ORDER:
        runs = [json.loads(p.read_text()) for p in sorted((RES / key).glob("seed*/metrics.json"))]
        if runs:
            data[key] = (label, runs)
    return data


def ms(vals):
    v = np.array(vals) * 100
    return f"{v.mean():.1f} ± {v.std():.1f}" if len(v) > 1 else f"{v[0]:.1f}"


def main():
    data = load()
    if not data:
        raise SystemExit("No Results/*/seed*/metrics.json found. Run the test.py scripts first.")

    lines = ["| Model | Accuracy (%) | Macro-F1 (%) | Seeds | Test clips |", "|---|---|---|---|---|"]
    for k, (label, runs) in data.items():
        lines.append(f"| {label} | {ms([r['accuracy'] for r in runs])} | {ms([r['macro_f1'] for r in runs])} "
                     f"| {len(runs)} | {runs[0]['n_test']} |")
    (RES / "accuracy_table.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))

    pc = {k: np.array([[r["per_class_f1"][e] for e in EMOTIONS] for r in runs]).mean(0) * 100
          for k, (_, runs) in data.items()}
    head = "| Emotion | " + " | ".join(l for l, _ in data.values()) + " |"
    rows = [head, "|" + "---|" * (len(data) + 1)]
    for i, e in enumerate(EMOTIONS):
        rows.append(f"| {e} | " + " | ".join(f"{pc[k][i]:.1f}" for k in data) + " |")
    (RES / "per_class_f1.md").write_text("F1 (%) per emotion, mean over seeds\n\n" + "\n".join(rows) + "\n")

    sil_rows = ["| Model | Block | Silhouette |", "|---|---|---|"]
    for k, (label, runs) in data.items():
        blocks = [b for b in (runs[0].get("silhouette") or {})]
        for b in blocks:
            vals = [r["silhouette"][b] for r in runs if r["silhouette"].get(b) is not None]
            if vals:
                sil_rows.append(f"| {label} | {b} | {np.mean(vals):.3f} |")
    (RES / "silhouette_table.md").write_text("\n".join(sil_rows) + "\n")

    fig, ax = plt.subplots(figsize=(6.5, 4))
    labels = [l for l, _ in data.values()]
    acc = [np.mean([r["accuracy"] for r in runs]) * 100 for _, runs in data.values()]
    err = [np.std([r["accuracy"] for r in runs]) * 100 for _, runs in data.values()]
    ax.bar(labels, acc, yerr=err, color="#4c72b0", capsize=4)
    ax.axhline(100 / 7, color="gray", ls="--", lw=1, label="chance (14.3%)")
    ax.set_ylabel("test accuracy (%)")
    ax.set_ylim(0, 105)
    ax.legend()
    plt.xticks(rotation=15)
    fig.tight_layout()
    fig.savefig(RES / "model_comparison.png", dpi=150)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 4))
    w = 0.8 / len(data)
    for j, (k, (label, _)) in enumerate(data.items()):
        ax.bar(np.arange(len(EMOTIONS)) + j * w, pc[k], w, label=label)
    ax.set_xticks(np.arange(len(EMOTIONS)) + 0.4 - w / 2, EMOTIONS)
    ax.set_ylabel("F1 (%)")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(RES / "per_class_f1.png", dpi=150)
    plt.close(fig)
    print("wrote tables and plots to", RES)


if __name__ == "__main__":
    main()
