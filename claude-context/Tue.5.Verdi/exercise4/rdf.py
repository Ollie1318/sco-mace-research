import sys
import numpy as np
import matplotlib.pyplot as plt
from ase.io import read
from ase.geometry.rdf import get_rdf

traj_file = sys.argv[1]
name = sys.argv[2]
rmax = 8; nbins = 400
frames = read(traj_file, ":")
pairs = {
    "Si-Si": (14, 14),
    "Si-O":  (14, 8),
    "O-O":   (8, 8),
}

# Calculate average rdf over the trajectory
rdf_sum = {}
for label in pairs:
    rdf_sum[label] = np.zeros(nbins)

for atoms in frames:
    for label, pair in pairs.items():
        atoms_s = atoms.repeat((2,2,2))
        rdf, r = get_rdf( atoms_s,rmax=rmax,nbins=nbins,elements=pair)
        rdf_sum[label] += rdf

nframes = len(frames)
rdf_avg = {}
for label in pairs:
    rdf_avg[label] = rdf_sum[label] / nframes

# Save data
with open(f"rdf-{name}.dat", "w") as f:
    f.write("# r[A] g_SiSi g_SiO g_OO\n")
    for i in range(nbins):
        f.write( f"{r[i]:12.6f} "
            f"{rdf_avg['Si-Si'][i]:15.8f} "
            f"{rdf_avg['Si-O'][i]:15.8f} "
            f"{rdf_avg['O-O'][i]:15.8f}\n" )

# Plot
plt.figure(figsize=(5,3.2))
for label, rdf in rdf_avg.items():
    plt.plot(r, rdf, label=label)
plt.xlabel("r (Å)")
plt.ylabel("g(r)")
plt.legend()
plt.tight_layout()
plt.savefig(f"rdf-{name}.pdf")
plt.close()

print("Done")
