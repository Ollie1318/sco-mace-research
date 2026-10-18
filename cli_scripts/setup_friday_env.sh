#!/bin/bash
# One-off setup of a MACE venv on UQ's Friday cluster. Run on a login node
# from the repo root:  bash cli_scripts/setup_friday_env.sh
#
# Check the Python module name first with:  module spider Python
# The name below is the EasyBuild Python that goes with foss/2023a; change it
# if `module spider` shows something different.
PY_MODULE="Python/3.11.3-GCCcore-12.3.0"
VENV="$WORK/venvs/mace"

set -e
module purge
module load "$PY_MODULE"

python -m venv "$VENV"
source "$VENV/bin/activate"
pip install --upgrade pip
# The default Linux torch wheel bundles CUDA, so no CUDA module is needed.
pip install torch
pip install mace-torch==0.3.16 ase

python -c "import torch, mace; print('torch', torch.__version__, '| CUDA build', torch.version.cuda)"
echo "Venv ready at $VENV (set the same PY_MODULE and VENV in eval_friday.slurm)"
