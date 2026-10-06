import numpy as np
import matplotlib.pyplot as plt
from ase.io import read
from mace.calculators import mace_mp

# Reference structures with DFT energies/forces attached.
datasets = {
    "Train": "data/converted/ML_AB_HS.xyz",
    "Test": "data/converted/ML_AB_HS_Test.xyz",
}

# Set to a small number (e.g. 5) for a quick test run, None for every frame.
MAX_FRAMES = 10

# Shift both energy series to zero mean before plotting. The ML_AB reference
# energies are all zero, so VASP and MACE use different energy zeros and the
# raw values sit far off the parity line. Forces are unaffected.
SHIFT_ENERGIES = False

properties = {
    "Energy": {
        "kind": "energy",
        "xlabel": "DFT energy (eV/atom)",
        "ylabel": "MACE energy (eV/atom)",
    },
    "Forces": {
        "kind": "forces",
        "xlabel": "DFT force (eV/Å)",
        "ylabel": "MACE force (eV/Å)",
    },
}

# Keys the DFT reference values are stored under in your .xyz files.
# Change these if your converter used different names (e.g. "REF_energy").
REF_ENERGY_KEY = "REF_energy"
REF_FORCES_KEY = "REF_forces"

# Load the model once and reuse it for every frame.
calc = mace_mp(model="medium-mpa-0", device="cpu", default_dtype="float64")


def get_property(atoms, key):
    """Find a stored value in atoms.info, atoms.arrays or an attached calculator.

    Depending on the ASE version and how the file was written, extxyz values
    end up in different places, so check all three.
    """
    if key in atoms.info:
        return atoms.info[key]
    if key in atoms.arrays:
        return atoms.arrays[key]
    if atoms.calc is not None and key in atoms.calc.results:
        return atoms.calc.results[key]
    raise KeyError(
        f"No '{key}' found on this frame. "
        f"info keys: {sorted(atoms.info)}; "
        f"array keys: {sorted(atoms.arrays)}"
    )


def evaluate(ref_file):
    """Return DFT and MACE energies (eV/atom) and forces (flattened, eV/Å)."""
    frames = read(ref_file, ":")
    if MAX_FRAMES is not None:
        frames = frames[:MAX_FRAMES]

    results = {"energy": ([], []), "forces": ([], [])}
    for i, atoms in enumerate(frames):
        n = len(atoms)

        # Read the DFT values BEFORE attaching MACE.
        dft_e = float(get_property(atoms, REF_ENERGY_KEY)) / n
        dft_f = np.asarray(get_property(atoms, REF_FORCES_KEY)).ravel()

        atoms.calc = calc
        mace_e = atoms.get_potential_energy() / n
        mace_f = atoms.get_forces().ravel()

        results["energy"][0].append(dft_e)
        results["energy"][1].append(mace_e)
        results["forces"][0].append(dft_f)
        results["forces"][1].append(mace_f)
        print(f"  {ref_file}: frame {i + 1}/{len(frames)}", end="\r")
    print()

    out = {
        "energy": (
            np.array(results["energy"][0]),
            np.array(results["energy"][1]),
        ),
        "forces": (
            np.concatenate(results["forces"][0]),
            np.concatenate(results["forces"][1]),
        ),
    }
    if SHIFT_ENERGIES:
        x, y = out["energy"]
        out["energy"] = (x - x.mean(), y - y.mean())
    return out


def rmse(x, y):
    return np.sqrt(np.mean((x - y) ** 2))


def parity_plot(ax, x, y, title, xlabel, ylabel):
    amin = min(x.min(), y.min())
    amax = max(x.max(), y.max())
    pad = 0.05 * (amax - amin)
    ax.scatter(x, y, s=10, alpha=0.6)
    ax.plot([amin - pad, amax + pad], [amin - pad, amax + pad], "k--", lw=1)
    ax.set_xlim(amin - pad, amax + pad)
    ax.set_ylim(amin - pad, amax + pad)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)


fig, axes = plt.subplots(
    len(datasets), len(properties), figsize=(8, 7), dpi=100, squeeze=False
)

for row, (dataset_name, ref_file) in enumerate(datasets.items()):
    data = evaluate(ref_file)
    for col, (prop_name, prop) in enumerate(properties.items()):
        x, y = data[prop["kind"]]
        err = rmse(x, y)
        unit = "eV/atom" if prop["kind"] == "energy" else "eV/Å"
        print(f"{dataset_name} {prop_name} RMSE: {err:.4f} {unit}")
        parity_plot(
            axes[row, col],
            x,
            y,
            title=f"{dataset_name} {prop_name}\nRMSE = {err:.4f} {unit}",
            xlabel=prop["xlabel"],
            ylabel=prop["ylabel"],
        )

plt.tight_layout()
plt.savefig("plot-parity.png", dpi=300)
plt.close()
