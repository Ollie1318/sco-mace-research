import sys
import torch
import numpy as np
from ase import Atoms
from ase.io import read
from mace.calculators import MACECalculator
from phonopy import Phonopy
from phonopy.structure.atoms import PhonopyAtoms
from phonopy.file_IO import write_FORCE_CONSTANTS, parse_BORN
from phonopy.phonon.band_structure import get_band_qpoints_and_path_connections
from phonopy.units import Hartree, Bohr

torch.set_num_threads(8)
name = sys.argv[1]
atoms = read(f"{name}-relax.cif")
sc_size = [2, 2, 2]
calc = MACECalculator( model_paths="zro2_MACE_model_stagetwo.model",
    device="cpu", default_dtype="float32" )

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

# Phonon band structure
paths = {
    "zro2-c": [ [0.5, 0.5, 0.5], [0.0, 0.0, 0.0], [0.0, 0.5, 0.5], [0.25, 0.75, 0.5], [0.375, 0.75, 0.375], [0.0, 0.0, 0.0] ],
    "zro2-t": [ [0.0, 0.0, 0.0], [0.0, 0.5, 0.0], [0.5, 0.5, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.5], [0.0, 0.5, 0.5], [0.5, 0.5, 0.5] ],
    "zro2-m": [ [0.0, 0.0, 0.0], [0.0, 0.5, 0.0], [0.0, 0.5, 0.5], [0.0, 0.0, 0.5], [0.0, 0.0, 0.0],
        [-0.5, 0.0, 0.5], [-0.5, 0.5, 0.5], [-0.5, 0.0, 0.0], [0.0, 0.0, 0.0] ] }
path = [paths[name]]
bands, connections = get_band_qpoints_and_path_connections( path, npoints=51 )
nac_params = parse_BORN( phonon.primitive, filename=f"ph-ref/{name}-born", symprec=1e-4)
nac_params["factor"] = Hartree * Bohr
phonon.nac_params = nac_params
phonon.run_band_structure( bands, path_connections=connections )
phonon.write_yaml_band_structure(filename=f"{name}-band.yaml")
print("Done.")
