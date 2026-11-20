# Multimodal Emotion Recognition on TESS (speech, text, speech + text)

Three emotion classifiers on the Toronto Emotional Speech Set (TESS), following the block diagram of the
assignment:

| Variant | Blocks |
|---|---|
| **Speech-only** | preprocessing → log-Mel features → **temporal modelling** (Conv1d + 2-layer BiGRU + attention) → classifier |
| **Text-only** | cleaning/tokenising → token features → **contextual modelling** (BiLSTM + attention) → classifier |
| **Multimodal** | both branches → **fusion** (concatenation or gated) → classifier |

## Project structure
```
project/
├── models/
│   ├── speech_pipeline/  train.py  test.py
│   ├── text_pipeline/    train.py  test.py
│   └── fusion_pipeline/  train.py  test.py
├── common/               shared code: data preparation, audio, text, models, training, plots
├── Results/              accuracy tables, confusion matrices, t-SNE plots, error lists
├── report/               Assignment_2b_Report (docx and pdf)
├── notebooks/            colab_run.ipynb  (runs everything on a Colab GPU)
├── make_results_table.py builds the tables/plots in Results/ from the per-run metrics
├── run_all.sh            train + test every variant for several seeds
├── requirements.txt
└── README.md
```

## Setup
```
pip install -r requirements.txt
```
A GPU is optional: the data set is small (2,800 clips). The reported results were produced on a Colab T4 GPU.

## Data
Download TESS from Kaggle (`ejlok1/toronto-emotional-speech-set-tess`). The download contains every clip twice (a nested
copy) and one file is named `OA_bite_neutral.wav`; `prepare_data.py` keeps one copy of each clip and maps `OA` to `OAF`. Clip names have the form
`OAF_back_angry.wav` (speaker_word_emotion). Every clip is the carrier phrase **"Say the word ⟨word⟩"**, so the
transcript is built from the file name. Then build the metadata file and the splits:
```
python common/prepare_data.py --tess_dir /path/to/TESS --out data/metadata.csv
```
- `--split_mode word` (default): words are disjoint between train / val / test (70 / 15 / 15 % of the 200
  words); both speakers appear in every split. No word appears in two splits, so the models cannot memorise
  a word → emotion shortcut.
- `--split_mode speaker --test_speaker YAF`: one speaker is held out completely for testing (harder).

## Run
One variant, one seed:
```
python models/speech_pipeline/train.py --seed 0
python models/speech_pipeline/test.py  --seed 0
python models/text_pipeline/train.py   --seed 0           # --text_features bert for DistilBERT
python models/text_pipeline/test.py    --seed 0
python models/fusion_pipeline/train.py --seed 0 --fusion concat   # or gated
python models/fusion_pipeline/test.py  --seed 0 --fusion concat
```
Everything for three seeds, then the summary tables:
```
bash run_all.sh 0 1 2
```
Outputs go to `Results/<model>/seed<k>/`: `metrics.json`, `confusion_matrix.png`, `errors.csv` (misclassified clips,
for the error analysis), and `tsne_<block>.png` (t-SNE of the temporal / contextual / fusion representation,
with its silhouette score). `make_results_table.py` writes `Results/accuracy_table.md`, `per_class_f1.md`,
`silhouette_table.md` and the comparison plots.

## Results (test set, 420 clips, mean ± standard deviation over 3 seeds)
| Model | Accuracy (%) | Macro-F1 (%) | Seeds | Test clips |
|---|---|---|---|---|
| Speech-only | 99.4 ± 0.1 | 99.4 ± 0.1 | 3 | 420 |
| Text-only | 14.3 ± 0.0 | 3.6 ± 0.0 | 3 | 420 |
| Multimodal (concat) | 99.4 ± 0.5 | 99.4 ± 0.5 | 3 | 420 |
| Multimodal (gated) | 99.4 ± 0.3 | 99.4 ± 0.3 | 3 | 420 |

Per-emotion F1, confusion matrices, t-SNE plots (temporal, contextual and fusion blocks), silhouette scores and the
lists of misclassified clips are in `Results/`. The report in `report/` discusses them in detail.

## Extra experiments
- **Speaker-held-out test** (train on one speaker, test on the other):
  `python common/prepare_data.py --tess_dir ... --out data/metadata_speaker.csv --split_mode speaker --test_speaker YAF`, then pass
  `--meta data/metadata_speaker.csv --results_dir Results_speaker --ckpt_dir checkpoints_speaker` to the train/test scripts.
- **DistilBERT text features:** `--text_features bert` (with separate `--results_dir` / `--ckpt_dir` so the main results are kept).

## Design choices
- **Speech preprocessing:** mono, 16 kHz, peak normalisation, energy-based silence trimming, clips capped at 4 s;
  variable lengths are padded and masked per batch.
- **Speech features:** 64-band log-Mel spectrogram (25 ms window, 10 ms hop), per-utterance mean normalisation,
  giving (time steps × 64).
- **Temporal modelling:** two Conv1d layers (the second halves the frame rate), a 2-layer bidirectional GRU, and
  attention pooling into a 256-d utterance vector. Light time/frequency masking is used as augmentation.
- **Text:** lower-casing and tokenisation; features are trainable word embeddings (default) or frozen
  DistilBERT token vectors; a BiLSTM with attention pooling gives a 256-d vector.
- **Fusion:** concatenation + linear layer (baseline) or a gated sum that learns how much each modality contributes.
- **Classifier:** MLP (256 → 128 → 7) trained with cross-entropy (label smoothing 0.05), Adam, learning-rate
  reduction on plateau and early stopping on validation accuracy.

## Important property of the data
TESS has only the carrier phrase "Say the word ⟨word⟩". Every word is spoken in all seven emotions, so the
**text contains no information about the emotion**. A text-only model is therefore expected to perform at chance
(about 14.3 % for seven classes), and the multimodal model can at best match the speech-only model. The
experiments confirm this: text-only is at chance, and the multimodal models match the speech-only model.
