"""Text-only pipeline: cleaning/tokenising -> token features -> BiLSTM+attention (contextual) -> MLP classifier.
--text_features embedding : trainable word embeddings (default, no download)
--text_features bert      : frozen DistilBERT token vectors (needs internet / transformers)"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from common import dataset, train_utils as U
from common.models import TextModel


def main():
    ap = U.train_args("Text-only emotion recognition (train)")
    ap.add_argument("--text_features", choices=["embedding", "bert"], default="embedding")
    args = ap.parse_args()
    U.seed_everything(args.seed)
    meta = dataset.load_meta(args.meta)
    sets, extra = dataset.make_datasets(meta, args.cache_dir, use_speech=False, use_text=True,
                                        text_features=args.text_features, device=args.device)
    loaders = dataset.loaders(sets, args.batch_size)
    model = TextModel(vocab_size=len(extra["vocab"])) if args.text_features == "embedding" else TextModel()
    U.fit(model, loaders, args, "text")


if __name__ == "__main__":
    main()
