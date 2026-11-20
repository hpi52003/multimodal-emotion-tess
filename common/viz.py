"""Plots: confusion matrix and t-SNE of learned representations (+ silhouette score)."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.manifold import TSNE
from sklearn.metrics import silhouette_score

from .prepare_data import EMOTIONS

COLORS = plt.get_cmap("tab10").colors


def plot_confusion(cm, path, title):
    cmn = cm / np.maximum(cm.sum(1, keepdims=True), 1)
    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(cmn, cmap="Blues", vmin=0, vmax=1)
    ax.set_xticks(range(len(EMOTIONS)), EMOTIONS, rotation=45)
    ax.set_yticks(range(len(EMOTIONS)), EMOTIONS)
    ax.set_xlabel("predicted")
    ax.set_ylabel("true")
    ax.set_title(title)
    for i in range(len(EMOTIONS)):
        for j in range(len(EMOTIONS)):
            ax.text(j, i, f"{cm[i, j]}", ha="center", va="center",
                    color="white" if cmn[i, j] > 0.5 else "black", fontsize=8)
    fig.colorbar(im, ax=ax, fraction=0.046)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_tsne(emb, labels, path, title, seed=0):
    """t-SNE coloured by emotion. Returns the silhouette score of the *original* embedding."""
    emb = np.asarray(emb, dtype=np.float64)
    sil = None
    if len(np.unique(labels)) > 1 and len(labels) > len(np.unique(labels)):
        try:
            sil = float(silhouette_score(emb, labels))
        except ValueError:
            sil = None
    perp = max(5, min(30, (len(emb) - 1) // 3))
    xy = TSNE(n_components=2, perplexity=perp, init="pca", random_state=seed).fit_transform(emb)
    fig, ax = plt.subplots(figsize=(6.5, 5.5))
    for k, e in enumerate(EMOTIONS):
        m = labels == k
        if m.any():
            ax.scatter(xy[m, 0], xy[m, 1], s=14, color=COLORS[k], label=e, alpha=0.8)
    ax.legend(markerscale=1.5, fontsize=8, loc="best")
    sub = f"silhouette = {sil:.3f}" if sil is not None else ""
    ax.set_title(f"{title}\n{sub}")
    ax.set_xticks([])
    ax.set_yticks([])
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return sil
