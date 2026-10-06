import sys
import torch
import numpy as np
import matplotlib.pyplot as plt
from ase import units, Atoms
from ase.io import read, write
from ase.md.npt import NPT
from ase.md.velocitydistribution import MaxwellBoltzmannDistribution, Stationary
from mace.calculators import MACECalculator

torch.set_num_threads(56)
name = sys.argv[1]
T_target = float(sys.argv[2]) # temperature (K)
model = "zro2_MACE_model_stagetwo_medium.model"
sc_size = (1,1,1)             # change for a bigger supercell
tot_t = 7                     # total trajectory length (ps)
dt = 2.0                      # timestep (fs)
log_interval = 10             # how often T,E,atoms recorded (steps)
traj_interval = 20            # how often atoms recorded (steps)
# NPT parameters
P_target = 0.0                # external pressure (eV/Ang^3)
ttime = 25.0 * units.fs       # thermostat time scale
ptime = 75.0 * units.fs       # barostat time scale
B = 1.0                       # tune if needed
pfactor = (ptime**2)*B

atoms = read(f"zro2-m-ramp-T{T_target}-final.cif").repeat(sc_size)
atoms.pbc = True
atoms.calc = MACECalculator( model_paths=model,device="cpu",default_dtype="float32" )

# initialise (NPT)
rng = np.random.default_rng(123)
MaxwellBoltzmannDistribution(atoms, temperature_K=T_target, rng=rng)
Stationary(atoms)
dyn = NPT( atoms, timestep=dt*units.fs, temperature_K=T_target,
        externalstress=P_target, ttime=ttime, pfactor=pfactor )

def atoms_write(atoms):
    return Atoms( symbols=atoms.get_chemical_symbols(),
        positions=atoms.get_positions(), cell=atoms.cell, pbc=atoms.pbc )

def run_md(tot_t):
    ntot = int(round(tot_t * 1000 / dt))
    data = []
    log_file = f"{name}-T{T_target}.log"
    traj_file = f"{name}-T{T_target}.xyz"
    log = open(log_file, "w")
    log.write("# time(ps) T(K) Epot/atom(eV) V(A^3) P(eV/A^3) a(A) b(A) c(A)\n")

    def write_log():
        time = dyn.get_time() / units.fs / 1000.0
        T = atoms.get_temperature()
        Epot = atoms.get_potential_energy() / len(atoms)
        V = atoms.get_volume()
        stress = atoms.get_stress(include_ideal_gas=True)
        P = -np.mean(stress[:3]) # mean of trace
        a, b, c = atoms.cell.lengths()
        data.append([time, T, Epot, V, P, a, b, c])
        log.write( f"{time:12.4f} {T:12.4f} {Epot:14.8f} {V:12.4f} {P:12.4f}"
            f"{a:12.4f} {b:12.4f} {c:12.4f}\n" )
        if dyn.nsteps % 200 == 0:
            log.flush()

    def write_traj():
        write(traj_file, atoms_write(atoms), append=True)

    dyn.attach(write_log, interval=log_interval)
    dyn.attach(write_traj, interval=traj_interval)
    dyn.run(ntot)
    log.close()

    write(f"{name}-T{T_target}-final.cif", atoms)
    print("Finished MD")

    return np.array(data)

# Run MD trajectory and plot
data = run_md(tot_t)

time = data[:, 0]
T = data[:, 1]; Epot = data[:, 2]
V = data[:, 3]; P = data[:, 4]
a = data[:, 5]; b = data[:, 6]; c = data[:, 7]
fig, axes = plt.subplots(3, 1, figsize=(6, 8))

ax = axes[0]
ax2 = ax.twinx()
ax.plot(time, T, "g", label="Temperature")
ax2.plot(time, Epot, "b", label="Energy")
ax.set_xlabel("Time (ps)")
ax.set_ylabel("Temperature (K)")
ax2.set_ylabel("Energy (eV/atom)")
ax.grid(alpha=0.3)
lines1, labels1 = ax.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
ax.legend(lines1 + lines2, labels1 + labels2, fontsize=8)

ax = axes[1]
ax2 = ax.twinx()
ax.plot(time, V, "k", label="Volume")
ax2.plot(time, P, "C0-", label="Pressure")
ax.set_xlabel("Time (ps)")
ax.set_ylabel("Volume (Å³)")
ax2.set_ylabel("Pressure (eV/Å³)")
ax.grid(alpha=0.3)
lines1, labels1 = ax.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
ax.legend(lines1 + lines2, labels1 + labels2, fontsize=8)

ax = axes[2]
ax.plot(time, a, label="a")
ax.plot(time, b, label="b")
ax.plot(time, c, label="c")
ax.set_xlabel("Time (ps)")
ax.set_ylabel("Lattice parameter (Å)")
ax.grid(alpha=0.3)
ax.legend(fontsize=8)

plt.tight_layout()
plt.savefig(f"{name}-T{T_target}.pdf")
plt.close()

print("Done.")
