"""Shared training / evaluation helpers used by the three pipelines."""
import argparse
import json
import random
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score

from .prepare_data import EMOTIONS

ROOT = Path(__file__).resolve().parent.parent


def base_args(desc):
    ap = argparse.ArgumentParser(description=desc)
    ap.add_argument("--meta", default=str(ROOT / "data" / "metadata.csv"))
    ap.add_argument("--cache_dir", default=str(ROOT / "cache"))
    ap.add_argument("--ckpt_dir", default=str(ROOT / "checkpoints"))
    ap.add_argument("--results_dir", default=str(ROOT / "Results"))
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--batch_size", type=int, default=32)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return ap


def train_args(desc):
    ap = base_args(desc)
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--patience", type=int, default=8)
    ap.add_argument("--specaug", type=int, default=1, help="light time/frequency masking on speech (1=on)")
    return ap


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def to_device(b, device):
    return {k: v.to(device) for k, v in b.items()}


def spec_augment(x, lengths, max_t=20, max_f=8):
    """Zero one random time block and one random mel block per utterance (training only)."""
    x = x.clone()
    for i in range(x.size(0)):
        L = int(lengths[i])
        t = random.randint(0, max_t)
        if L > t + 1 and t > 0:
            t0 = random.randint(0, L - t)
            x[i, t0:t0 + t] = 0
        f = random.randint(0, max_f)
        if f > 0:
            f0 = random.randint(0, x.size(2) - f)
            x[i, :, f0:f0 + f] = 0
    return x


@torch.no_grad()
def predict(model, loader, device):
    model.eval()
    logits, labels, idx = [], [], []
    for b in loader:
        b = to_device(b, device)
        logits.append(model(b).cpu())
        labels.append(b["label"].cpu())
        idx.append(b["index"].cpu())
    logits, labels, idx = torch.cat(logits), torch.cat(labels), torch.cat(idx)
    return logits, labels, idx


@torch.no_grad()
def collect_embeddings(model, loader, device):
    model.eval()
    embs, labels, idx = {}, [], []
    for b in loader:
        b = to_device(b, device)
        for k, v in model.embeddings(b).items():
            embs.setdefault(k, []).append(v.cpu())
        labels.append(b["label"].cpu())
        idx.append(b["index"].cpu())
    return {k: torch.cat(v).numpy() for k, v in embs.items()}, torch.cat(labels).numpy(), torch.cat(idx).numpy()


def fit(model, loaders, args, name, use_speech_aug=False, extra_cfg=None):
    """Adam + cross-entropy, ReduceLROnPlateau, early stopping on validation accuracy."""
    device = args.device
    model.to(device)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-5)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode="max", factor=0.5, patience=3)
    loss_fn = nn.CrossEntropyLoss(label_smoothing=0.05)
    best, best_state, bad, history = -1.0, None, 0, []
    for ep in range(1, args.epochs + 1):
        t0 = time.time()
        model.train()
        tot, n = 0.0, 0
        for b in loaders["train"]:
            b = to_device(b, device)
            if use_speech_aug and args.specaug and "speech" in b:
                b["speech"] = spec_augment(b["speech"], b["speech_len"])
            opt.zero_grad()
            loss = loss_fn(model(b), b["label"])
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()
            tot += loss.item() * b["label"].size(0)
            n += b["label"].size(0)
        lg, lb, _ = predict(model, loaders["val"], device)
        val_acc = accuracy_score(lb, lg.argmax(1))
        sched.step(val_acc)
        history.append(dict(epoch=ep, train_loss=tot / n, val_acc=val_acc))
        print(f"epoch {ep:3d}  loss {tot / n:.4f}  val_acc {val_acc:.4f}  ({time.time() - t0:.1f}s)")
        if val_acc > best:
            best, bad = val_acc, 0
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        else:
            bad += 1
            if bad >= args.patience:
                print("early stopping")
                break
    model.load_state_dict(best_state)
    ck = Path(args.ckpt_dir)
    ck.mkdir(parents=True, exist_ok=True)
    cfg = dict(vars(args), best_val_acc=best, **(extra_cfg or {}))
    torch.save({"state": best_state, "config": cfg}, ck / f"{name}_seed{args.seed}.pt")
    rd = Path(args.results_dir) / name / f"seed{args.seed}"
    rd.mkdir(parents=True, exist_ok=True)
    (rd / "history.json").write_text(json.dumps(history, indent=2))
    print(f"best val acc {best:.4f}; checkpoint saved to {ck / f'{name}_seed{args.seed}.pt'}")


def evaluate_and_save(logits, labels, df_test, idx, out_dir):
    """Accuracy, macro-F1, per-class report, confusion matrix, list of misclassified clips."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    pred = logits.argmax(1).numpy()
    y = labels.numpy()
    acc = float(accuracy_score(y, pred))
    mf1 = float(f1_score(y, pred, average="macro"))
    rep = classification_report(y, pred, labels=list(range(len(EMOTIONS))), target_names=EMOTIONS,
                                output_dict=True, zero_division=0)
    cm = confusion_matrix(y, pred, labels=list(range(len(EMOTIONS))))
    metrics = {"accuracy": acc, "macro_f1": mf1, "n_test": int(len(y)),
               "per_class_f1": {e: rep[e]["f1-score"] for e in EMOTIONS},
               "per_class_recall": {e: rep[e]["recall"] for e in EMOTIONS},
               "confusion_matrix": cm.tolist()}
    (out_dir / "metrics.json").write_text(json.dumps(metrics, indent=2))
    d = df_test.iloc[idx.numpy()].copy()
    d["true"] = [EMOTIONS[i] for i in y]
    d["pred"] = [EMOTIONS[i] for i in pred]
    d["confidence"] = torch.softmax(logits, 1).max(1).values.numpy()
    d[d.true != d.pred].sort_values("confidence", ascending=False)[
        ["path", "speaker", "word", "true", "pred", "confidence"]].to_csv(out_dir / "errors.csv", index=False)
    print(f"test accuracy {acc:.4f}   macro-F1 {mf1:.4f}")
    return metrics, cm


def embedding_report(emb, y, out_dir, run_name):
    """t-SNE plot + silhouette score for each learned representation; updates metrics.json."""
    from . import viz
    out_dir = Path(out_dir)
    sil = {}
    for block, e in emb.items():
        sil[block] = viz.plot_tsne(e, y, out_dir / f"tsne_{block}.png",
                                   f"{run_name}: {block} block (test set)", seed=0)
    m = json.loads((out_dir / "metrics.json").read_text())
    m["silhouette"] = sil
    (out_dir / "metrics.json").write_text(json.dumps(m, indent=2))
    print("silhouette:", {k: (round(v, 3) if v is not None else None) for k, v in sil.items()})


def load_ckpt(ckpt_dir, name, seed):
    return torch.load(Path(ckpt_dir) / f"{name}_seed{seed}.pt", map_location="cpu", weights_only=False)
