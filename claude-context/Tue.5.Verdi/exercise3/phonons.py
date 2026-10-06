import sys
import torch
import numpy as np
from ase import Atoms
from ase.io import read
from ase.optimize import BFGS
from mace.calculators import MACECalculator
from phonopy import Phonopy
from phonopy.structure.atoms import PhonopyAtoms
from phonopy.file_IO import write_FORCE_CONSTANTS, parse_BORN
from phonopy.phonon.band_structure import get_band_qpoints_and_path_connections
from phonopy.units import Hartree, Bohr

torch.set_num_threads(8)
name = sys.argv[1]
atoms = read(f"structures/{name}_sio2.cif")
sc_size = [1, 1, 1]
calc = MACECalculator( model_paths="sio2_MACE_model_stagetwo.model",
    device="cpu", default_dtype="float32" )
atoms.calc = calc

# Relax structure
opt = BFGS(atoms,trajectory=f"{name}-relax.traj", logfile=f"{name}-relax.log")
opt.run(fmax=0.01)

unitcell = PhonopyAtoms( symbols=atoms.get_chemical_symbols(),
        cell=atoms.cell.array, scaled_positions=atoms.get_scaled_positions() )
# Choose supercell size
phonon = Phonopy( unitcell, supercell_matrix=np.diag(sc_size), primitive_matrix="auto", symprec=1e-4)
# Generate finite displacements (in A)
phonon.generate_displacements(distance=0.015)
supercells = phonon.supercells_with_displacements
print(f"Number of displaced supercells: {len(supercells)}")

forces = []
for i, scell in enumerate(supercells):
    scell_ase = Atoms( symbols=scell.symbols,
        cell=scell.cell, scaled_positions=scell.scaled_positions, pbc=True )
    scell_ase.calc = calc
    f = scell_ase.get_forces()
    forces.append(f)
    print(f"Done displacement {i+1}/{len(supercells)}")

# Build and save force constants
phonon.produce_force_constants(forces=forces)
phonon.symmetrize_force_constants()
write_FORCE_CONSTANTS(phonon.force_constants, filename=f"{name}-FORCE_CONSTANTS")
phonon.save(filename=f"{name}-phonopy.yaml")

# Calculate ph DOS
mesh = [6,6,6]
s = 0.5 # Gaussian broadening (THz)
phonon.run_mesh(mesh)
phonon.run_total_dos(sigma=s)
phonon.write_total_dos(filename=f"{name}-total_dos.dat")
print("Done.")
