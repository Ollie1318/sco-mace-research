import numpy as np
import matplotlib.pyplot as plt
from ase.io import read

datasets = {
    "Train": ("database/zro2-train.xyz", "train_mace_eval.xyz"),
    "Test":  ("database/zro2-test.xyz",  "test_mace_eval.xyz"),
}

properties = {
    "Energy": {
        "ref_key": "energy",
        "pred_key": "MACE_energy",
        "kind": "energy",
        "xlabel": "DFT energy (eV/atom)",
        "ylabel": "MACE energy (eV/atom)",
    },
    "Forces": {
        "ref_key": "forces",
        "pred_key": "MACE_forces",
        "kind": "forces",
        "xlabel": "DFT force (eV/Å)",
        "ylabel": "MACE force (eV/Å)",
    },
    "Stress": {
        "ref_key": "stress",
        "pred_key": "MACE_stress",
        "kind": "stress",
        "xlabel": "DFT stress (eV/Å³)",
        "ylabel": "MACE stress (eV/Å³)",
    },
}

def get_property(atoms, key):
    if key in atoms.info:
        return atoms.info[key]
    if key in atoms.arrays:
        return atoms.arrays[key]
    if atoms.calc is not None and key in atoms.calc.results:
        return atoms.calc.results[key]

def get_stress(s):
    s = np.asarray(s)
    if s.shape == (3, 3):
        return np.array([s[0, 0], s[1, 1], s[2, 2], s[1, 2], s[0, 2], s[0, 1]])
    return s.ravel()

def get_values(atoms, key, kind):
    value = get_property(atoms, key)
    if kind == "energy":
        return np.array([value/len(atoms)])
    if kind == "forces":
        return np.asarray(value).ravel()
    if kind == "stress":
        return get_stress(value)

def collect_data(ref_file, pred_file, prop):
    ref = read(ref_file, ":")
    pred = read(pred_file, ":")
    x, y = [], []
    for a_ref, a_pred in zip(ref, pred):
        x.append(get_values(a_ref, prop["ref_key"], prop["kind"]))
        y.append(get_values(a_pred, prop["pred_key"], prop["kind"]))
    return np.concatenate(x), np.concatenate(y)

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
    ax.set_title(f"{title}")

fig, axes = plt.subplots(2,3,figsize=(11.6,7),dpi=100)

for row, (dataset_name, files) in enumerate(datasets.items()):
    ref_file, pred_file = files
    for col, (prop_name, prop) in enumerate(properties.items()):
        x, y = collect_data(ref_file, pred_file, prop)
        parity_plot( axes[row,col], x, y, title=f"{dataset_name} {prop_name}",
            xlabel=prop["xlabel"], ylabel=prop["ylabel"] )

plt.tight_layout()
plt.savefig("plot-parity.png", dpi=300)
plt.close()
