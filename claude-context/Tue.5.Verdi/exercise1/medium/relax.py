import sys
import torch
from ase.io import read, write
from ase.optimize import BFGS
from ase.filters import FrechetCellFilter
from mace.calculators import MACECalculator

torch.set_num_threads(8)
name = sys.argv[1]
atoms = read(f"../structures/{name}.cif")
atoms.calc = MACECalculator( model_paths="zro2_MACE_model_stagetwo_medium.model", 
    device="cpu", default_dtype="float32" )

# Relax atoms + cell
filt = FrechetCellFilter(atoms)
opt = BFGS(filt, trajectory=f"{name}-relax.traj", logfile=f"{name}-relax.log")
opt.run(fmax=0.01)

write(f"{name}-relax.cif", atoms)
