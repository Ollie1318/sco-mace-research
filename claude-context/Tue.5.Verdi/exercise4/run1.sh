#!/bin/bash
#SBATCH -J myjob     # Job name
#SBATCH -N 1         # Total # of nodes
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=28
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

sed -i "s/torch\.set_num_threads.*/torch.set_num_threads(${SLURM_CPUS_PER_TASK})/" md-nvt.py
python md-nvt.py sio2
