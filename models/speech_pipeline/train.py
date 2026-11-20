"""Speech-only pipeline: preprocessing -> log-Mel features -> Conv+BiGRU+attention (temporal) -> MLP classifier."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from common import dataset, train_utils as U
from common.models import SpeechModel


def main():
    args = U.train_args("Speech-only emotion recognition (train)").parse_args()
    U.seed_everything(args.seed)
    meta = dataset.load_meta(args.meta)
    sets, _ = dataset.make_datasets(meta, args.cache_dir, use_speech=True, use_text=False)
    loaders = dataset.loaders(sets, args.batch_size)
    U.fit(SpeechModel(), loaders, args, "speech", use_speech_aug=True)


if __name__ == "__main__":
    main()
