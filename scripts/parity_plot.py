"""Parity plots from mace_eval_configs output (no model evaluation here).

Make the evaluated files first with cli_scripts/eval.sh (or eval_friday.slurm
on the cluster); each frame then holds both the DFT REF_* values and the
MACE_* predictions.
"""

import numpy as np
import matplotlib.pyplot as plt
from ase.io import read

# Evaluated files: DFT REF_* values and MACE_* predictions on each frame.
datasets = {
    "LS": "data/evaluated/ML_AB_LS_eval.xyz",
    "HS": "data/evaluated/ML_AB_HS_eval.xyz",
    # "Test": "data/evaluated/ML_AB_LS_Test_eval.xyz",
}

# None plots every frame; a number plots only the first N.
MAX_FRAMES = None

# DFT and MACE energies have different zeros. All structures have the same
# composition, so this is one rigid shift for every frame:
#   Delta_E = < E_DFT - E_MACE >  over all frames of SHIFT_FILES combined,
#   E_MACE' = E_MACE + Delta_E    for every dataset (LS and HS alike).
# Forces are unaffected.
SHIFT_FILES = [
    "data/evaluated/ML_AB_LS_eval.xyz",
    "data/evaluated/ML_AB_HS_eval.xyz",
]


properties = {
    "Energy": {
        "kind": "energy",
        "xlabel": "Reference energy (eV)",
        "ylabel": "Predicted energy (eV)",
    },
    "Forces": {
        "kind": "forces",
        "xlabel": "Reference force (eV/Å)",
        "ylabel": "Predicted force (eV/Å)",
    },
}

# Keys the DFT reference values are stored under in your .xyz files.
# Change these if your converter used different names (e.g. "REF_energy").
REF_ENERGY_KEY = "REF_energy"
REF_FORCES_KEY = "REF_forces"

# Keys mace_eval_configs writes its predictions to (--info_prefix, default MACE_).
PRED_ENERGY_KEY = "MACE_energy"
PRED_FORCES_KEY = "MACE_forces"


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


_cache = {}


def load(eval_file):
    """Return DFT and MACE energies (eV per cell), forces (flattened, eV/Å) and atoms per frame."""
    if eval_file in _cache:
        return _cache[eval_file]
    frames = read(eval_file, ":")
    if MAX_FRAMES is not None:
        frames = frames[:MAX_FRAMES]

    results = {"energy": ([], []), "forces": ([], [])}
    for atoms in frames:
        results["energy"][0].append(
            float(get_property(atoms, REF_ENERGY_KEY)))
        results["energy"][1].append(
            float(get_property(atoms, PRED_ENERGY_KEY)))
        results["forces"][0].append(np.asarray(
            get_property(atoms, REF_FORCES_KEY)).ravel())
        results["forces"][1].append(np.asarray(
            get_property(atoms, PRED_FORCES_KEY)).ravel())
    print(f"  {eval_file}: {len(frames)} frames")

    out = {
        "energy": (
            np.array(results["energy"][0]),
            np.array(results["energy"][1]),
        ),
        "forces": (
            np.concatenate(results["forces"][0]),
            np.concatenate(results["forces"][1]),
        ),
        "n_atoms": np.array([len(atoms) for atoms in frames]),
    }
    _cache[eval_file] = out
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


e_dft = np.concatenate([load(f)["energy"][0] for f in SHIFT_FILES])
e_mace = np.concatenate([load(f)["energy"][1] for f in SHIFT_FILES])
delta_e = np.mean(e_dft - e_mace)
print(f"Delta_E = <E_DFT - E_MACE> = {delta_e:.4f} eV per cell "
      f"(over {len(e_dft)} frames of {', '.join(SHIFT_FILES)})")

fig, axes = plt.subplots(
    len(datasets), len(properties), figsize=(8, 3.5 * len(datasets)), dpi=100,
    squeeze=False
)

for row, (dataset_name, eval_file) in enumerate(datasets.items()):
    data = load(eval_file)
    for col, (prop_name, prop) in enumerate(properties.items()):
        x, y = data[prop["kind"]]
        if prop["kind"] == "energy":
            y = y + delta_e
        err = rmse(x, y)
        unit = "eV" if prop["kind"] == "energy" else "eV/Å"
        extra = ""
        if prop["kind"] == "energy":
            per_atom = 1000 * np.sqrt(np.mean(((x - y) / data["n_atoms"]) ** 2))
            extra = f" ({per_atom:.3f} meV/atom)"
        print(f"{dataset_name} {prop_name} RMSE: {err:.4f} {unit}{extra}")
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
