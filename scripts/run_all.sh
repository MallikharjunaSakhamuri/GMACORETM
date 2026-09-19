#!/usr/bin/env bash
#
# Full experimental pipeline: pretrain all three variants, evaluate each on the
# downstream benchmarks, and run the representation analysis.
#
# Usage:
#   bash scripts/run_all.sh                 # full schedule
#   SMOKE=1 bash scripts/run_all.sh         # short run for verification

set -euo pipefail

VARIANTS=("manualaug_cl" "gmacore" "gmacore_tm")
RUN_ROOT="${RUN_ROOT:-runs}"

OVERRIDES=()
if [[ "${SMOKE:-0}" == "1" ]]; then
  echo "Running in smoke-test mode: reduced epochs and corpus size."
  OVERRIDES=(--set pretrain.epochs=2 data.limit=500 pretrain.embedding_every=1)
fi

echo "== Step 1: downloading benchmarks =="
python scripts/download_moleculenet.py --all

echo "== Step 2: pretraining =="
for variant in "${VARIANTS[@]}"; do
  echo "-- ${variant}"
  python scripts/pretrain.py \
    --config "configs/pretrain_${variant}.yaml" \
    --output-dir "${RUN_ROOT}/${variant}" \
    "${OVERRIDES[@]}"
done

echo "== Step 3: downstream evaluation =="
for variant in "${VARIANTS[@]}"; do
  checkpoint="${RUN_ROOT}/${variant}/best_model.pt"
  if [[ ! -f "${checkpoint}" ]]; then
    echo "Checkpoint not found for ${variant}, skipping."
    continue
  fi

  python scripts/finetune.py \
    --config configs/finetune_classification.yaml \
    --checkpoint "${checkpoint}" \
    --output-dir "${RUN_ROOT}/finetune/${variant}_classification"

  python scripts/finetune.py \
    --config configs/finetune_regression.yaml \
    --checkpoint "${checkpoint}" \
    --output-dir "${RUN_ROOT}/finetune/${variant}_regression"
done

echo "== Step 4: no-pretraining control =="
python scripts/finetune.py \
  --config configs/finetune_classification.yaml \
  --no-pretrain \
  --output-dir "${RUN_ROOT}/finetune/random_init_classification"

echo "== Step 5: representation analysis =="
for variant in "${VARIANTS[@]}"; do
  if [[ -d "${RUN_ROOT}/${variant}/embeddings" ]]; then
    python scripts/analyze_representations.py \
      --embedding-dir "${RUN_ROOT}/${variant}/embeddings" \
      --output "${RUN_ROOT}/${variant}/representation_analysis.json"
  fi
done

echo "Pipeline complete. Artifacts are under ${RUN_ROOT}/."
