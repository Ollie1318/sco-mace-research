import sys
import torch
from ase.io import read, write
from ase.optimize import BFGS
from ase.filters import FrechetCellFilter
from mace.calculators import MACECalculator
from mace.calculators import mace_mp

torch.set_num_threads(8)
name = sys.argv[1]
atoms = read(f"data/converted/{name}.xyz")
atoms.calc = mace_mp(model="medium")

# Relax atoms + cell
filt = FrechetCellFilter(atoms)
opt = BFGS(filt, trajectory=f"{name}-relax.traj", logfile=f"{name}-relax.log")
opt.run(fmax=0.01)

write(f"{name}-relax.cif", atoms)
