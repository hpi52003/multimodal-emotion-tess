"""Multimodal pipeline: speech branch (temporal) + text branch (contextual) -> fusion -> MLP classifier.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from common import dataset, train_utils as U
from common.models import FusionModel


def build(args, extra):
    tk = dict(vocab_size=len(extra["vocab"])) if args.text_features == "embedding" else {}
    return FusionModel(text_kw=tk, fusion=args.fusion)


def main():
    ap = U.train_args("Multimodal emotion recognition (train)")
    ap.add_argument("--fusion", choices=["concat", "gated"], default="concat")
    ap.add_argument("--text_features", choices=["embedding", "bert"], default="embedding")
    args = ap.parse_args()
    U.seed_everything(args.seed)
    meta = dataset.load_meta(args.meta)
    sets, extra = dataset.make_datasets(meta, args.cache_dir, use_speech=True, use_text=True,
                                        text_features=args.text_features, device=args.device)
    loaders = dataset.loaders(sets, args.batch_size)
    U.fit(build(args, extra), loaders, args, f"fusion_{args.fusion}", use_speech_aug=True)


if __name__ == "__main__":
    main()
