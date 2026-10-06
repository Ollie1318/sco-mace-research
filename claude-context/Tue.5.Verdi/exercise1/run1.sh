#!/bin/bash
#SBATCH -J myjob     # Job name
#SBATCH -N 1         # Total # of nodes
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=28 # Threads
#SBATCH -t 00:30:00  # Run time (hh:mm:ss)
#SBATCH -A DMR23048
#SBATCH -p small
#SBATCH --reservation=MATCSSI_Norm_June16

module load python3/3.9.2
source $WORK/MACE/bin/activate
unset PYTHONPATH

export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export MKL_NUM_THREADS=$SLURM_CPUS_PER_TASK
export OPENBLAS_NUM_THREADS=$SLURM_CPUS_PER_TASK
export NUMEXPR_NUM_THREADS=$SLURM_CPUS_PER_TASK

mace_run_train \
  --name="zro2_MACE_model" \
  --train_file="database/zro2-train.xyz" \
  --valid_fraction=0.1 \
  --energy_key="energy" \
  --forces_key="forces" \
  --stress_key="stress" \
  --compute_stress=True \
  --loss="huber" \
  --test_file="database/zro2-test.xyz" \
  --E0s="{40: -2.07574957, 8: -1.79537657}" \
  --model="ScaleShiftMACE" \
  --num_interactions=2 \
  --num_channels=64 \
  --correlation=3 \
  --max_L=0 \
  --r_max=4 \
  --batch_size=10 \
  --max_num_epochs=15 \
  --ema \
  --ema_decay=0.99 \
  --amsgrad \
  --default_dtype="float32" \
  --device=cpu \
  --seed=123 \
  --swa

