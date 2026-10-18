#!/bin/bash --login
# Evaluate a (non-magnetic) MACE model on the LS train and test sets on a GPU
# node of UQ's Friday cluster. Submit from the repo root:
#   sbatch cli_scripts/eval_friday.slurm
# Same outputs as eval.sh: data/evaluated/<name>_eval.xyz with MACE_energy /
# MACE_forces next to the DFT REF_* values.
#
#SBATCH --job-name=mace_eval
#SBATCH --partition=smp_gpu
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --output=slurm-%x-%j.out
#SBATCH --error=slurm-%x-%j.err

set -e

PY_MODULE="Python/3.11.3-GCCcore-12.3.0"   # same as in setup_friday_env.sh
VENV="$HOME/venvs/mace"

# The GPU has fast float64, so keep float64 (needed for HS/LS energy
# differences). For float32 use the float32 model copy, because
# mace_eval_configs does not convert the model to --default_dtype.
DTYPE="float64"
if [ "$DTYPE" = "float32" ]; then
  MODEL="models/MACE/mace-mpa-0-medium-float32.model"
else
  MODEL="models/MACE/mace-mpa-0-medium.model"
fi
# Unlike on the laptop CPU, batching helps on a GPU. Raise this if the job
# runs well within GPU memory; lower it if you get CUDA out-of-memory errors.
BATCH_SIZE=4
OUT_DIR="data/evaluated"

cd "$SLURM_SUBMIT_DIR"
module purge
module load "$PY_MODULE"
source "$VENV/bin/activate"
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK

nvidia-smi
python -c "import torch, sys; ok = torch.cuda.is_available(); print('CUDA available:', ok, torch.cuda.get_device_name(0) if ok else ''); sys.exit(0 if ok else 1)"

mkdir -p "$OUT_DIR"
for NAME in ML_AB_LS ML_AB_LS_Test; do
  start=$(date +%s)
  mace_eval_configs \
    --configs="data/converted/${NAME}.xyz" \
    --model="$MODEL" \
    --output="${OUT_DIR}/${NAME}_eval.xyz" \
    --device="cuda" \
    --default_dtype="$DTYPE" \
    --batch_size=$BATCH_SIZE
  echo "Saved ${OUT_DIR}/${NAME}_eval.xyz ($(( $(date +%s) - start )) s)"
done
