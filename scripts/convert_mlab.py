"""Convert VASP ML_AB files to extended XYZ for MACE training.

Replaces pymlff's converter, which reads each configuration at fixed line
offsets and so breaks on files that differ in layout. This parser locates
sections by their headers instead, so it handles both layouts in data/raw/:

* training files (ML_AB, ML_AB_LS, ML_AB_HS): include a CTIFOR block;
* test files (ML_AB_*_Test): no CTIFOR block, extra padding whitespace, and
  lines hard-truncated at 78 characters.

The truncation cuts the last exponent digit off any value in the third column
that is written in exponent form (|x| < 0.1), e.g. "2.2037000000000001E-00"
(true exponent unknown). These values are recovered by finding the exponents
k >= 2 for which:

  1. the value has no more decimal places than the rest of that section
     (e.g. forces are given to 6 decimals, positions to 5), and
  2. the 17-significant-digit representation of the resulting double
     reproduces the printed mantissa exactly (the float "noise" digits are
     specific to the true value).

If several exponents remain, forces are resolved by requiring the net force
along that Cartesian direction to vanish (VASP forces sum to ~1e-5 eV/A).
Other quantities take the largest remaining exponent. Validated on the
intact columns of the test files: ~97.5% correct without the force-sum step.
Every value that is still a guess is counted in the per-frame info
(`n_guessed`), so affected frames can be filtered out.

Output per frame (ASE extxyz):
  info:   REF_energy (eV), REF_stress (eV/A^3, Voigt xx yy zz yz xz xy, ASE sign),
          config_type (LS/HS/Intermediate), system_name, temperature_K (if in
          the system name), ctifor (if present), source, config_index,
          n_truncated, n_guessed
  arrays: REF_forces (eV/A)

Train with: --energy_key=REF_energy --forces_key=REF_forces --stress_key=REF_stress

Usage (from the repo root):
  python scripts/convert_mlab.py                      # all ML_AB* in data/raw
  python scripts/convert_mlab.py data/raw/ML_AB_LS_Test -o data/converted
"""

import argparse
import itertools
import re
from collections import Counter
from pathlib import Path

import numpy as np
from ase import Atoms, units
from ase.io import write

KBAR_TO_EV_A3 = 0.1 * units.GPa  # 1 kbar = 0.1 GPa
MAX_COMBINATIONS = 4096  # brute-force limit when resolving forces by net force
NET_FORCE_TOL = 1e-4  # eV/A; resolved net forces above this are reported

_SEPARATOR = re.compile(r"^[=\-*]+$")
_TRUNCATED = re.compile(r"^(-?\d\.\d+)E([-+])(\d{0,2})$")
_TEMPERATURE = re.compile(r"(\d+(?:\.\d+)?)\s*K\b")


# --------------------------------------------------------------------------
# Numbers
# --------------------------------------------------------------------------


class Truncated:
    """A value whose exponent was cut off; holds the candidate values."""

    def __init__(self, token):
        mantissa, sign, digits = _TRUNCATED.match(token).groups()
        self.token = token
        self.mantissa = mantissa
        # exponents whose 3-digit form starts with the digits that survived
        self.exponents = [
            k for k in range(1, 100) if f"{k:03d}".startswith(digits)
        ] if sign == "-" else []
        self.candidates = {}  # filled by narrow()

    def narrow(self, decimals):
        """Keep exponents consistent with the section precision and the printed digits."""
        for k in self.exponents:
            if k < 2:  # Fortran only switches to E format below 0.1
                continue
            x = float(f"{self.mantissa}E-{k}")
            if abs(round(x, decimals) - x) > 1e-12:
                continue
            if f"{abs(x):.16E}".split("E")[0] != self.mantissa.lstrip("-"):
                continue
            self.candidates[k] = x
        if not self.candidates:  # fall back to precision only
            for k in self.exponents:
                x = float(f"{self.mantissa}E-{k}")
                if k >= 2 and abs(round(x, decimals) - x) <= 1e-12:
                    self.candidates[k] = x
        if not self.candidates:
            raise ValueError(f"cannot recover truncated value {self.token!r}")

    def best_guess(self):
        return self.candidates[max(self.candidates)]


def parse_token(token):
    # ML_AB files always write 3-digit exponents, so fewer digits means the
    # token was truncated. Check before float(), which accepts "1.5E-00".
    if _TRUNCATED.match(token):
        return Truncated(token)
    return float(token)


def decimals_of(x):
    for d in range(13):
        if abs(round(x, d) - x) <= 1e-12 * max(1.0, abs(x)):
            return d
    return 12


# --------------------------------------------------------------------------
# Parsing
# --------------------------------------------------------------------------


def _section(block, title):
    """Data lines following `title` up to the next separator, or None if absent."""
    for i, line in enumerate(block):
        if line == title:
            j = i + 1
            while j < len(block) and _SEPARATOR.match(block[j]):
                j += 1
            out = []
            while j < len(block) and not _SEPARATOR.match(block[j]):
                out.append(block[j])
                j += 1
            return out
    return None


def _rows(lines, name):
    rows = [[parse_token(t) for t in line.split()] for line in lines]
    if any(len(r) != 3 for r in rows):
        raise ValueError(f"{name}: expected 3 columns per line")
    return rows


def parse_mlab(path):
    """Return a list of raw configuration dicts (values may be Truncated)."""
    lines = [line.strip() for line in Path(path).read_text().splitlines()]
    starts = [i for i, line in enumerate(lines) if line.startswith("Configuration num.")]
    if not starts:
        raise ValueError(f"{path}: no configurations found")

    configs = []
    for n, (a, b) in enumerate(zip(starts, starts[1:] + [len(lines)]), start=1):
        block = lines[a:b]
        where = f"{Path(path).name} config {n}"

        types = [line.split() for line in _section(block, "Atom types and atom numbers")]
        symbols = [s for s, count in types for _ in range(int(count))]
        natoms = int(_section(block, "The number of atoms")[0])
        if len(symbols) != natoms:
            raise ValueError(f"{where}: atom counts do not sum to {natoms}")

        positions = _rows(_section(block, "Atomic positions (ang.)"), where)
        forces = _rows(_section(block, "Forces (eV ang.^-1)"), where)
        if len(positions) != natoms or len(forces) != natoms:
            raise ValueError(f"{where}: expected {natoms} positions and forces")

        diag = _section(block, "XX YY ZZ")
        offdiag = _section(block, "XY YZ ZX")
        ctifor = _section(block, "CTIFOR")

        configs.append({
            "index": n,
            "name": _section(block, "System name")[0],
            "symbols": symbols,
            "cell": _rows(_section(block, "Primitive lattice vectors (ang.)"), where),
            "positions": positions,
            "energy": float(_section(block, "Total energy (eV)")[0]),
            "forces": forces,
            "stress": [parse_token(t) for t in diag[0].split() + offdiag[0].split()],
            "ctifor": float(ctifor[0]) if ctifor else None,
        })
    return configs


# --------------------------------------------------------------------------
# Truncation recovery
# --------------------------------------------------------------------------


def _section_decimals(configs, key):
    """Most common number of decimal places among intact values of a section."""
    counts = Counter()
    for c in configs:
        values = c[key] if key == "stress" else [v for row in c[key] for v in row]
        counts.update(decimals_of(v) for v in values if isinstance(v, float))
    return counts.most_common(1)[0][0] if counts else 12


def _resolve_forces(forces):
    """Pick exponents for truncated force components so each column sums to ~0.

    Returns (resolved rows, number still guessed, worst net force).
    """
    rows = [list(r) for r in forces]
    guessed, worst = 0, 0.0
    for col in range(3):
        known = sum(r[col] for r in rows if isinstance(r[col], float))
        unknown = [i for i, r in enumerate(rows) if isinstance(r[col], Truncated)]
        fixed = [i for i in unknown if len(rows[i][col].candidates) == 1]
        ambiguous = [i for i in unknown if len(rows[i][col].candidates) > 1]
        for i in fixed:
            rows[i][col] = rows[i][col].best_guess()
            known += rows[i][col]

        options = [list(rows[i][col].candidates.values()) for i in ambiguous]
        if ambiguous and np.prod([len(o) for o in options]) <= MAX_COMBINATIONS:
            best = min(itertools.product(*options), key=lambda c: abs(known + sum(c)))
        else:  # too many combinations: largest-exponent guesses
            best = [rows[i][col].best_guess() for i in ambiguous]
            guessed += len(ambiguous)
        for i, v in zip(ambiguous, best):
            rows[i][col] = v
        worst = max(worst, abs(known + sum(best)))
    return rows, guessed, worst


def recover(configs):
    """Replace Truncated values in place; return per-config (n_truncated, n_guessed)."""
    decimals = {key: _section_decimals(configs, key)
                for key in ("cell", "positions", "forces", "stress")}
    stats = []
    for c in configs:
        n_trunc = n_guess = 0
        for key in ("cell", "positions", "forces", "stress"):
            flat = c[key] if key == "stress" else [v for row in c[key] for v in row]
            for v in flat:
                if isinstance(v, Truncated):
                    v.narrow(decimals[key])
                    n_trunc += 1

        c["forces"], guessed, net = _resolve_forces(c["forces"])
        n_guess += guessed
        if net > NET_FORCE_TOL:
            print(f"  warning: config {c['index']} net force {net:.2e} eV/A after recovery")

        for key in ("cell", "positions"):
            for row in c[key]:
                for j, v in enumerate(row):
                    if isinstance(v, Truncated):
                        n_guess += len(v.candidates) > 1
                        row[j] = v.best_guess()
        for j, v in enumerate(c["stress"]):
            if isinstance(v, Truncated):
                n_guess += len(v.candidates) > 1
                c["stress"][j] = v.best_guess()
        stats.append((n_trunc, n_guess))
    return stats


# --------------------------------------------------------------------------
# Output
# --------------------------------------------------------------------------


def config_type(name):
    for label in ("Intermediate", "LS", "HS"):
        if re.search(rf"\b{label}", name):
            return label
    return "Default"


def to_atoms(c, source, n_trunc, n_guess, stress_units):
    atoms = Atoms(symbols=c["symbols"], positions=c["positions"], cell=c["cell"], pbc=True)

    xx, yy, zz, xy, yz, zx = c["stress"]
    voigt = np.array([xx, yy, zz, yz, zx, xy])  # ASE Voigt order
    if stress_units == "eV/A3":
        # VASP reports stress with the opposite sign to ASE (as ASE's VASP
        # reader and pymlff's eV/A^3 option assume).
        voigt = -voigt * KBAR_TO_EV_A3

    atoms.info.update({
        "REF_energy": c["energy"],
        "REF_stress": voigt,
        "config_type": config_type(c["name"]),
        "system_name": c["name"],
        "source": source,
        "config_index": c["index"],
        "n_truncated": n_trunc,
        "n_guessed": n_guess,
    })
    if m := _TEMPERATURE.search(c["name"]):
        atoms.info["temperature_K"] = float(m.group(1))
    if c["ctifor"] is not None:
        atoms.info["ctifor"] = c["ctifor"]
    atoms.arrays["REF_forces"] = np.array(c["forces"], dtype=float)
    return atoms


def convert(path, out_dir, stress_units):
    path = Path(path)
    configs = parse_mlab(path)
    stats = recover(configs)
    frames = [to_atoms(c, path.name, *s, stress_units) for c, s in zip(configs, stats)]

    out = Path(out_dir) / f"{path.name}.xyz"
    out.parent.mkdir(parents=True, exist_ok=True)
    write(out, frames, format="extxyz")

    n_trunc = sum(s[0] for s in stats)
    n_guess = sum(s[1] for s in stats)
    types = Counter(f.info["config_type"] for f in frames)
    print(f"{path.name}: {len(frames)} configs {dict(types)} -> {out}")
    if n_trunc:
        print(f"  recovered {n_trunc} truncated values, {n_guess} still ambiguous "
              f"(largest-exponent guess) in {sum(s[1] > 0 for s in stats)} configs")


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("inputs", nargs="*", help="ML_AB files (default: data/raw/ML_AB*)")
    parser.add_argument("-o", "--out-dir", default="data/converted")
    parser.add_argument("--stress-units", choices=["eV/A3", "kbar"], default="eV/A3",
                        help="eV/A3 (ASE/MACE convention, default) or raw VASP kbar")
    args = parser.parse_args()

    inputs = args.inputs or sorted(Path("data/raw").glob("ML_AB*"))
    for path in inputs:
        convert(path, args.out_dir, args.stress_units)


if __name__ == "__main__":
    main()
