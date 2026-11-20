"""Evaluate the text-only model on the test split; writes metrics, confusion matrix, errors, t-SNE."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from common import dataset, train_utils as U, viz
from common.models import TextModel


def main():
    args = U.base_args("Text-only emotion recognition (test)").parse_args()
    U.seed_everything(args.seed)
    ck = U.load_ckpt(args.ckpt_dir, "text", args.seed)
    feats = ck["config"]["text_features"]
    meta = dataset.load_meta(args.meta)
    sets, extra = dataset.make_datasets(meta, args.cache_dir, use_speech=False, use_text=True,
                                        text_features=feats, device=args.device)
    loader = dataset.loaders(sets, args.batch_size)["test"]
    model = TextModel(vocab_size=len(extra["vocab"])) if feats == "embedding" else TextModel()
    model.load_state_dict(ck["state"])
    model.to(args.device)
    out = Path(args.results_dir) / "text" / f"seed{args.seed}"
    logits, labels, idx = U.predict(model, loader, args.device)
    _, cm = U.evaluate_and_save(logits, labels, sets["test"].df, idx, out)
    viz.plot_confusion(cm, out / "confusion_matrix.png", "Text-only: confusion matrix (test)")
    emb, y, _ = U.collect_embeddings(model, loader, args.device)
    U.embedding_report(emb, y, out, "Text-only")


if __name__ == "__main__":
    main()
