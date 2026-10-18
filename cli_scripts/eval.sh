#!/bin/bash
# Evaluate a (non-magnetic) MACE model on the LS train and test sets.
# Run from the repo root:  bash cli_scripts/eval.sh
# Predictions are added to each frame as MACE_energy / MACE_forces; the DFT
# REF_* values are kept, so the output files hold both for parity plots.

# Exit immediately if a command exits with a non-zero status
set -e

source .mace-venv/Scripts/activate

# mace_eval_configs does not convert the model to --default_dtype, so the
# model file must already be in that dtype. The original file is float64;
# mace-mpa-0-medium-float32.model is a float32 copy (model.float() + torch.save).
DTYPE="float32"
if [ "$DTYPE" = "float32" ]; then
  MODEL="models/MACE/mace-mpa-0-medium-float32.model"
else
  MODEL="models/MACE/mace-mpa-0-medium.model"
fi
OUT_DIR="data/evaluated"
mkdir -p "$OUT_DIR"

for NAME in ML_AB_HS ML_AB_HS_Test; do
  # batch_size=1: the default (64) puts 64 x 321 atoms in one calculation,
  # which needs far more memory than the laptop has.
  mace_eval_configs \
    --configs="data/converted/${NAME}.xyz" \
    --model="$MODEL" \
    --output="${OUT_DIR}/${NAME}_eval.xyz" \
    --device="cpu" \
    --default_dtype="$DTYPE" \
    --batch_size=1
  echo "Saved ${OUT_DIR}/${NAME}_eval.xyz"
done
