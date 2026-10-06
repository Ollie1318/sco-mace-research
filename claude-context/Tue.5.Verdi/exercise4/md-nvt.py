import sys
import torch
import numpy as np
import matplotlib.pyplot as plt
from ase import units, Atoms
from ase.io import read, write
from ase.constraints import FixCom
from ase.md.langevin import Langevin
from ase.md.velocitydistribution import MaxwellBoltzmannDistribution, Stationary
from mace.calculators import MACECalculator

torch.set_num_threads(28)
name = sys.argv[1]
model = "sio2_MACE_model_stagetwo.model"
sc_size = (1,1,1)       # change for a bigger supercell
dt = 2.0                # timestep (fs)
chunk_ps = 0.2          # interval to update target T (ps)
log_interval = 10       # how often T,E,atoms recorded (steps)
traj_interval = 100     # how often atoms recorded (steps)
friction = 1.0 / (100.0 * units.fs) # Langevin friction
phases = [ ("heat", 300, 3500, 5),
    ("melt", 3500, 3500, 5),
    ("quench", 3500, 300, 10),
    ("equil", 300, 300, 5) ]

atoms = read(f"model_I_sio2.cif").repeat(sc_size)
atoms.pbc = True
atoms.set_constraint(FixCom()) # remove center of mass drift if any
atoms.calc = MACECalculator( model_paths=model,device="cpu",default_dtype="float32" )

#initialise
rng = np.random.default_rng(123)
MaxwellBoltzmannDistribution(atoms, temperature_K=300, rng=rng)
Stationary(atoms)
all_data = {}

def atoms_write(atoms):
    return Atoms( symbols=atoms.get_chemical_symbols(),
        positions=atoms.get_positions(), cell=atoms.cell, pbc=atoms.pbc )

def run_phase(label, T0, T1, tot_t):
    n_chunks = int(round(tot_t/chunk_ps))
    steps_per_chunk = int(round(chunk_ps * 1000 / dt))
    temps = np.linspace(T0, T1, n_chunks)
    data = []
    log_file = f"{name}-{label}.log"
    traj_file = f"{name}-{label}.xyz"
    dyn = Langevin( atoms, timestep=dt*units.fs, temperature_K=T0, friction=friction, fixcm=False )

    with open(log_file, "w") as log:
        log.write("# time(ps) T(K) Epot/atom (eV)\n")

        def write_log():
            time = dyn.get_time()/units.fs/1000
            T = atoms.get_temperature()
            Epot = atoms.get_potential_energy() / len(atoms)
            data.append([time, T, Epot])
            log.write(f"{time:12.6f} {T:12.4f} {Epot:20.10f} \n")

        def write_traj():
            write(traj_file, atoms_write(atoms), append=True)

        dyn.attach(write_log, interval=log_interval)
        dyn.attach(write_traj, interval=traj_interval)

        for target_T in temps:
            dyn.set_temperature(temperature_K=float(target_T))
            dyn.run(steps_per_chunk)

    all_data[label] = np.array(data)
    write(f"{name}-{label}-final.cif", atoms)
    print(f"Finished {label}")

# Run MD trajectories
for phase in phases:
    label, T0, T1, tot_t = phase
    run_phase(label, T0, T1, tot_t)

fig, axes = plt.subplots(2, 2, figsize=(12, 8))
for ax, (label, _, _, _) in zip(axes.ravel(), phases):
    data = all_data[label]
    time = data[:, 0]
    T = data[:, 1]
    Epot = data[:, 2]

    ax2 = ax.twinx()
    ax.plot(time, T, 'g', label="Temperature")
    ax2.plot(time, Epot, 'b', label="Energy")
    ax.set_title(label)
    ax.set_xlabel("Time (ps)")
    ax.set_ylabel("Temperature (K)")
    ax2.set_ylabel("Energy (eV/atom)")
    ax.grid(alpha=0.3)
    lines1, labels1 = ax.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax.legend(lines1 + lines2, labels1 + labels2, fontsize=8)

plt.tight_layout()
plt.savefig(f"{name}-melt-quench.pdf")
plt.close()

print("Done.")
