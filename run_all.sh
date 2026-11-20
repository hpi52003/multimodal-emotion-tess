#!/usr/bin/env bash
# Train + test all variants for several seeds, then build the result tables.
# Usage: bash run_all.sh [SEEDS...]      e.g.  bash run_all.sh 0 1 2
set -euo pipefail
cd "$(dirname "$0")"
SEEDS=${@:-0 1 2}
for s in $SEEDS; do
  python models/speech_pipeline/train.py --seed $s
  python models/speech_pipeline/test.py  --seed $s
  python models/text_pipeline/train.py   --seed $s
  python models/text_pipeline/test.py    --seed $s
  python models/fusion_pipeline/train.py --seed $s --fusion concat
  python models/fusion_pipeline/test.py  --seed $s --fusion concat
  python models/fusion_pipeline/train.py --seed $s --fusion gated
  python models/fusion_pipeline/test.py  --seed $s --fusion gated
done
python make_results_table.py
