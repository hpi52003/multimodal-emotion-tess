"""Evaluate the multimodal model; t-SNE of the temporal, contextual and fusion representations."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from common import dataset, train_utils as U, viz
from common.models import FusionModel


def main():
    ap = U.base_args("Multimodal emotion recognition (test)")
    ap.add_argument("--fusion", choices=["concat", "gated"], default="concat")
    args = ap.parse_args()
    U.seed_everything(args.seed)
    name = f"fusion_{args.fusion}"
    ck = U.load_ckpt(args.ckpt_dir, name, args.seed)
    feats = ck["config"]["text_features"]
    meta = dataset.load_meta(args.meta)
    sets, extra = dataset.make_datasets(meta, args.cache_dir, use_speech=True, use_text=True,
                                        text_features=feats, device=args.device)
    loader = dataset.loaders(sets, args.batch_size)["test"]
    tk = dict(vocab_size=len(extra["vocab"])) if feats == "embedding" else {}
    model = FusionModel(text_kw=tk, fusion=args.fusion)
    model.load_state_dict(ck["state"])
    model.to(args.device)
    out = Path(args.results_dir) / name / f"seed{args.seed}"
    logits, labels, idx = U.predict(model, loader, args.device)
    _, cm = U.evaluate_and_save(logits, labels, sets["test"].df, idx, out)
    viz.plot_confusion(cm, out / "confusion_matrix.png", f"Multimodal ({args.fusion}): confusion matrix (test)")
    emb, y, _ = U.collect_embeddings(model, loader, args.device)
    U.embedding_report(emb, y, out, f"Multimodal ({args.fusion})")


if __name__ == "__main__":
    main()
