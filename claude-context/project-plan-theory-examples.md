# MLIPs for Spin-Crossover Materials: Plan, Theory and Practical Examples

This is a working document for the PHYS3900 project *Machine Learning Potentials for Simulating Spin-Crossover Materials* (Oliver Lam, supervised by Dr Carla Verdi).

- **Part 1 (Plan)** is based **only** on the project proposal (`PHYS3900_Project.pdf`).
- **Parts 2–3 (Theory, Practical Examples)** draw on every document in `claude-context/`:

| Tag | Source |
|---|---|
| **[P]** | `PHYS3900_Project.pdf`: project proposal |
| **[C]** | `260220_Anna_Carpenter_Confirmation.pdf`: PhD confirmation report (A. Carpenter, Verdi/Powell group), Feb 2026 |
| **[T]** | `Tan - 2024 - Modelling Spin-Crossover Lattices with Machine-Learned Force Fields.pdf`: Honours thesis (H. Tan, Powell/Verdi), 2024 |
| **[M]** | `mMACE_paper.pdf`: Ho *et al.*, "Equivariant Many-body Message Passing Interatomic Potentials for Magnetic Materials", arXiv:2604.08143 (2026) |
| **[V]** | `Tue.5.Verdi.pdf` + `Tue.5.Verdi/`: C. Verdi, MACE hands-on tutorial (MACE 0.3.16, ASE, phonopy) |

Anything that is **not** from these sources is labelled *"external, verify"*. This mostly covers MACE foundation-model commands.

---

## Part 1: Project Plan (from the proposal)

### 1.1 Aim

Develop and test machine-learned interatomic potential (MLIP) models for spin-crossover (SCO) materials, and use them to predict SCO behaviour as a function of external parameters such as temperature [P].

### 1.2 Motivation

- MLIPs aim for ab initio accuracy at close to classical-potential cost. That makes large-scale, accurate simulations of SCO materials feasible [P].
- SCO materials switch reversibly between high-spin (HS) and low-spin (LS) states under external stimuli. This bistability makes them candidates for molecular sensors, switches and data storage [P].
- Experimental discovery is slow. Candidate materials must be synthesised and tested, and many never show SCO. Efficient, accurate simulation can speed this up [P].

### 1.3 Methods

- Frameworks: **MACE** and **mMACE** (magnetic MACE). mMACE was introduced to handle spin states and magnetic materials, which standard MLIPs cannot [P].
- Start from existing **foundation models** (pre-trained on general datasets, usable out of the box), then **fine-tune** them to the specific SCO materials [P].
- Calculations: **structural relaxation**, **phonons**, **Gibbs free energy**, and **molecular dynamics (MD)** [P].
- Finish by testing the foundation models on a **larger set of SCO materials** [P].

### 1.4 Timeline

| Week | Focus [P] | Deliverable | Relevant sections below |
|---|---|---|---|
| 1 | Get familiar with MACE and mMACE. Install the foundation models and set up calculation workflows. | Working environment, foundation models load, test calculation runs end-to-end | §2.6, §2.7, §3.1, §3.2 |
| 2–3 | Run calculations with the **foundation models**: relaxation, phonons, etc. (the proposal's Methods section also lists Gibbs free energy). | Foundation-model relaxed LS/HS structures, phonons, free energies, compared with reference data | §2.3, §2.4, §3.2, §3.5, §3.6 |
| 4 | **Fine-tune** the foundation models for the specific structures. | Fine-tuned model(s) with train/validation/test errors reported | §2.6, §3.3, §3.4 |
| 5–6 | Repeat the week 2–3 calculations with the **fine-tuned** models, and run **MD simulations**. | Foundation vs fine-tuned comparison; MD trajectories vs temperature | §2.5, §3.5–§3.8 |
| 7 | Run the **foundation models against larger existing datasets** (a larger set of SCO materials). | Benchmark table across materials | §3.4 |
| 8 | Finalise the project report and presentation. | Report + presentation | — |

### 1.5 Expected outcomes [P]

- A method for running accurate MD of SCO materials with MLIPs, specifically MACE and mMACE.
- If it works, larger systems and longer timescales become computationally feasible. That supports further SCO research and the design of future devices.

---

## Part 2: Theory

### 2.1 Spin crossover: the physics

**Ligand-field picture.** Take an Fe(II) ion (d⁶) in an octahedral complex. The ligands split the five d orbitals into a lower, three-fold **t₂g** set and a higher, two-fold **e_g** set, separated by the ligand-field splitting Δ₀. The spin state depends on how Δ₀ compares with the spin-pairing energy *P* [C, T]:

| | Condition | Configuration | Total spin | Magnetism |
|---|---|---|---|---|
| **LS** | Δ₀ > *P* (strong field) | t₂g⁶ | *S* = 0 (singlet) | diamagnetic |
| **HS** | Δ₀ < *P* (weak field) | t₂g⁴ e_g² | *S* = 2 (quintet), 4 unpaired e⁻ | paramagnetic |

When Δ₀ ≈ *P*, both states are thermally accessible and the material can undergo SCO. It can be triggered by temperature, pressure, magnetic field or light [C, T]. SCO is generally seen for d⁴–d⁷ metal centres [C].

**Structural signature.** In the HS state, electrons occupy the antibonding e_g orbitals. This makes the metal–ligand bonds longer:
- Fe–N lengthens by about 10% (≈ 0.19 ± 0.05 Å for [FeN₆] cores), with a 3.8–6.0% volume expansion [C, T].
- The metal–ligand stretching frequencies soften in the HS state [C, T].

The mean Fe–N bond length is therefore a natural **order parameter** for the spin state [C].

**Cooperativity.** In solids, local spin-state changes are passed between molecules by long-range elastic interactions, carried mainly by **low-frequency (acoustic) phonons** [T]. This controls the *shape* of the transition, as shown in [C] (Fig. 2):
- **gradual:** typical in solution
- **abrupt:** typical in solids
- **hysteretic:** T₁/₂↑ ≠ T₁/₂↓. A wide hysteresis loop is a design goal for memory applications.

Multistep transitions arise from longer-range elastic interactions in ordered lattices [C]. A single-molecule model cannot capture hysteresis or multistep behaviour [T].

### 2.2 Thermodynamics of thermal SCO

The key observable is the **transition temperature** T₁/₂, where HS and LS populations are equal [C, T]:

$$\Delta G_{HL} = G_{HS} - G_{LS} = \Delta H - T\Delta S, \qquad \Delta G(T_{1/2}) = 0 \;\Rightarrow\; T_{1/2} = \frac{\Delta H}{\Delta S}$$

Both ΔH and ΔS have an electronic part and a vibrational part: $\Delta H = \Delta H_{elec} + \Delta H_{vib}$ and $\Delta S = \Delta S_{elec} + \Delta S_{vib}$ [C, T].

**Electronic terms** are approximately temperature-independent [C]:
- $\Delta H_{elec}$ comes directly from electronic-structure calculations. It must be > 0 so that LS is the ground state at low temperature [T].
- $\Delta S_{elec}$ comes from the change in orbital and spin degeneracy:

$$\Delta S_{elec} = k_B\ln\frac{2L_{HS}+1}{2L_{LS}+1} + k_B\ln\frac{2S_{HS}+1}{2S_{LS}+1} = k_B\ln 3 + k_B\ln 5 = k_B \ln 15 \approx 0.23\ \text{meV K}^{-1}$$

  per metal centre for octahedral d⁶ [C, T].

**Vibrational terms** (harmonic, summed over modes *i*) [C, T]:

$$H_{vib}(T) = \sum_i \left[\frac{\hbar\omega_i}{2} + \frac{\hbar\omega_i\, e^{-\hbar\omega_i/k_BT}}{1-e^{-\hbar\omega_i/k_BT}}\right]$$

$$S_{vib}(T) = \sum_i \left[\frac{\hbar\omega_i}{2T}\coth\!\left(\frac{\hbar\omega_i}{2k_BT}\right) - k_B\ln\!\left(2\sinh\frac{\hbar\omega_i}{2k_BT}\right)\right]$$

- The first term of $H_{vib}$ is the **zero-point energy (ZPE)**, $\sum_i \hbar\omega_i/2$.
- For a molecule the sum runs over 3N−6 modes. For a crystal it runs over **all phonon modes**, with Bose–Einstein occupation [C].
- ΔS_vib is usually 2–4× larger than ΔS_elec in Fe(II) compounds: about 0.42–0.83 meV K⁻¹ for N-coordinated octahedral Fe(II) [C, T]. The softer HS modes are what make entropy favour HS.

**Quasi-harmonic approximation (QHA).** QHA lets the phonon frequencies depend on volume, ω_j(V). That captures thermal expansion [T]:

$$F(V,T) = E(V) + \sum_j \frac{\hbar\omega_j(V)}{2} + k_BT\sum_j \ln\!\left(1 - e^{-\hbar\omega_j(V)/k_BT}\right)$$

- Minimising F over V at each T gives $V_{eq}(T)$ and the thermal expansion coefficient $\alpha_V = \frac{1}{V_{eq}}\left(\frac{\partial V_{eq}}{\partial T}\right)_P$ [T].
- T₁/₂ is then where the minima of $F_{HS}$ and $F_{LS}$ cross [C, T].
- QHA needs phonons at several volumes. This is prohibitively expensive with DFT for large SCO cells, and cheap with an MLIP [C, T].

> **Key lesson from [T]: zero-point energy matters.**
> - With MLFF-based QHA on [Fe(ptz)₆](BF₄)₂, the LS ZPE came out **101.8 meV higher** than the HS ZPE. That is larger than the electronic splitting (~85 meV), so HS became the predicted ground state and thermal SCO could not occur.
> - The cause was the DFT+U reference: U had been tuned to ΔH_elec **without** vibrational corrections.
> - [C] redid the tuning against T₁/₂ *including* vibrational corrections to ΔH(T) and ΔS(T), and got **U = 1.7 eV** (instead of 1.9 eV).
> - **Implication for this project:** always compare free energies, not bare electronic energies, when predicting T₁/₂.

### 2.3 The reference method: DFT and DFT+U

- **DFT** replaces the many-body wavefunction with the electron density n(r) (Hohenberg–Kohn). In the Kohn–Sham scheme, all the unknown physics sits in the exchange–correlation functional $E_{XC}$ [C, T].
- **Problem for SCO.** Approximate $E_{XC}$ leaves a **self-interaction error** that over-delocalises the Fe d electrons. This systematically mis-estimates the HS–LS splitting [C, T].
  - Hybrid functionals (B3LYP*, TPSSh, M06-L) do better, but they are expensive for cells of hundreds of atoms and partly empirical [C, T].
- **DFT+U (Dudarev)** adds an on-site penalty on partial d-orbital occupation, which increases localisation [C, T]:

$$E_{DFT+U} = E_{DFT} + \frac{U-J}{2}\sum_\sigma \left[\sum_m n_{m,\sigma} - \sum_{m,m'} n_{m,m',\sigma}\,n_{m',m,\sigma}\right], \qquad U_{\text{eff}} = U - J$$

  - U is material-specific and must be benchmarked.
  - Ohlrich & Powell found U_eff ≈ 1.6 eV (PBE+U) works well across seven Fe SCO materials [T].
- **Dispersion.** Semi-local functionals miss van der Waals interactions, which differ between the compact LS and the expanded HS state. D3(BJ) is the common correction [T].
- **Metastable states.** Small distortions in the HS molecule break the t₂g degeneracy. Different d-orbital occupations must be tested (occupation matrix control) so the calculation does not get stuck in a metastable state [C].

**Reference method used for [Fe(ptz)₆](BF₄)₂ in this group** [T, C]:
- VASP, PBE-D3(BJ)+U (Dudarev) on Fe.
- 500 eV cutoff, Γ-point only, symmetry off.
- LS is non-spin-polarised. HS is constrained to S = 2.
- U_eff = 1.9 eV [T], later revised to 1.7 eV [C].

### 2.4 Phonons

- In a crystal, vibrations are collective **phonons**. For M atoms in the cell there are 3M modes at each wave vector q [C, T].
- **Finite-displacement method** [C, T, V]:
  1. Displace each atom by ±δ (0.015 Å in both [T] and [V]).
  2. Compute the forces.
  3. Build the force-constant (Hessian) matrix.
  4. Mass-weight it into the dynamical matrix and diagonalise to get the frequencies.

  **phonopy** automates this.
- **Cost.**
  - For a 321-atom cell with no symmetry, this means thousands of force evaluations. The DFT HS phonon calculation in [T] took **6 days 11 hours on 96 CPUs**; the MLFF took about **10 minutes**.
  - Symmetry reduces the number of displacements [V]. It is off in the SCO cells [T].
- **Which modes matter for SCO.** Low-frequency modes (1–10 THz) carry cooperativity. Check these carefully when benchmarking a model's phonon density of states (pDOS) [T].
- **Long-range electrostatics** (the LO–TO splitting in polar crystals) needs a non-analytic correction from Born effective charges [V]. This is probably less important for a molecular crystal at Γ, but worth knowing.

### 2.5 Molecular dynamics

- MD integrates Newton's equations, for example with the Verlet algorithm [C]:

$$\mathbf r_i(t+\Delta t) \approx 2\mathbf r_i(t) - \mathbf r_i(t-\Delta t) + \frac{\mathbf F_i(t)}{m_i}\Delta t^2$$

- **Ensembles** [V, T]:
  - **NVT** uses a thermostat (e.g. Langevin).
  - **NPT** also uses a barostat, so the cell can respond to temperature. This is essential for SCO, because the volume changes on switching.
  - ASE's `NPT` combines Nosé–Hoover and Parrinello–Rahman dynamics. Typical time constants are ttime ≈ 25 fs and ptime ≈ 75 fs. Values that are too small cause instabilities; values that are too large cause slow oscillations [V].
- **Timescale problem** [C]:
  - Phonons of about 1 THz need timesteps of ≤ 10 fs (2 fs is typical).
  - Spin switching in solids takes ns–µs, which is about 10⁵–10⁸ MD steps. That is impossible with AIMD for large cells, and it is the main motivation for MLIPs.
- **Hydrogen.** [T] raised the H mass to 4 amu to keep a 2 fs timestep stable. Do the same in ASE (edit `atoms.set_masses`), or use a shorter timestep.
- **Detecting SCO in MD.** Track the mean Fe–N distance and the volume as the temperature is ramped. A jump shows the transition, as with the monoclinic→tetragonal volume drop in ZrO₂ in [V] Ex. 2.2 [T, C, V].
- **Ramping.** Heat slowly, or run long simulations near the transition. Fast ramps well above T_c are crude and only for illustration [V].

### 2.6 Machine-learned interatomic potentials

**General workflow** [C]:
1. Compute DFT energies, forces and stresses for training structures.
2. Encode each atomic environment in a symmetry-respecting representation.
3. Learn the structure → energy mapping.

The total energy is a sum of local atomic energies, and forces are its negative gradient.

**VASP on-the-fly kernel MLFF (how the existing SCO data was made)** [C, T]:
- **Descriptors:** rotationally invariant 2-body (radial) and 3-body (angular) atomic distribution functions.
- **Model:** kernel regression $K(\mathbf x_i,\mathbf x_j)$, with weights from least squares, $\mathbf Y = \Phi\mathbf w$.
- **On-the-fly learning:**
  - During MD, a Bayesian error estimate of the forces (**BEEF**) is compared with a threshold (**CTIFOR**).
  - When BEEF is above CTIFOR, VASP runs DFT and may add the structure to the training set. Otherwise it propagates with the MLFF.
  - Up to 98% of AIMD steps are skipped.
- **Sampling in [T]:** NPT with a Langevin thermostat at 300 K, then 100 K. That gave 393 LS and 347 HS configurations of the 321-atom cell.
- **Accuracy in [T]:** final errors ≈ 27 meV/Å (forces) and ≈ 24 meV per cell (energy). ΔH_HL came out at 80.2 meV vs 77.0 meV from DFT. The MLFF relaxations were actually closer to experiment than DFT.
- **Output format:** VASP writes the training set to `ML_AB` files. These are the raw data in `data/raw/`.

**MACE** [V, M]:
- An **equivariant message-passing neural network**.
- **Messages:** each atom's features are updated from neighbours within a cutoff *r_max*, over `num_interactions` layers (2 is recommended). The **receptive field** is `num_interactions × r_max`, e.g. 2 × 4 Å = 8 Å [V].
- **Many-body basis:** messages are built from higher body-order products of neighbour features. `correlation = 3` gives body order 4.
- **Feature symmetry:**
  - `max_L = 0`: only invariant (scalar) features. Small and fast.
  - `max_L = 1` or `2`: also vector/tensor features. More accurate, costlier [V].
- **Loss:** weighted energies, forces and stresses (Huber loss). **Stage two / SWA** increases the energy weight for the last epochs [V].
- **Isolated-atom energies (`E0s`):** set explicitly, or estimate them with `--E0s="average"` [V].
- **Example accuracy on ZrO₂ (592 configs)** [V]:
  - small model (64×0e): 1.8 meV/atom, 94 meV/Å on the test set.
  - "medium" model (`max_L=1`, 64×0e+64×1o): 1.1 meV/atom, 57 meV/Å, with visibly better phonons for the unstable cubic phase.
  - A more diverse SiO₂ dataset (crystal + liquid + amorphous) gives much larger errors (~35 meV/atom).

**Foundation models and fine-tuning:**
- Foundation models are pre-trained on large, chemically diverse datasets and work out of the box [P].
- [M] shows that fine-tuning a pre-trained model (MATPES-PBE) with **very little targeted data** (FeNi: 147 training configurations) recovers properties that the foundation model gets wrong. Fine-tuning settings in [M]: learning rate reduced 0.005 → 0.0005, gradient clip reduced 100 → 10.
- [M] also warns that fine-tuning on a very small number of *parent* structures can overfit from epoch 1. **Watch the validation error.**

### 2.7 mMACE: magnetic MACE [M]

**What it adds.**
- mMACE adds **atomic magnetic moments m_i ∈ ℝ³** as explicit degrees of freedom alongside positions and species.
- The energy becomes $E(\{\mathbf r_i,\mathbf m_i,z_i\})$. This gives ordinary forces $\mathbf F_i = -\nabla_{\mathbf r_i}E$ and also **magnetic forces** $\mathbf F^{mag}_i = -\nabla_{\mathbf m_i}E$.
- A one-body term $E_0(|\mathbf m_i|)$ fixes the isolated-atom limit for each moment magnitude.

**Symmetry.**
- Energy must be invariant under a *joint* rotation of positions and moments (O(3)).
- **Without spin–orbit coupling (SOC)**, positions and moments can also be rotated *independently* (O(3)×O(3)). mMACE enforces this approximately through data augmentation, or exactly with an alternative architecture.
- **With SOC**, the independent symmetry is broken. That breaking is what lets mMACE capture magnetocrystalline anisotropy.

**Architecture.**
- Edge radial bases depend on |r_ji| and |m_j| (Bessel ⊕ Chebyshev).
- Neighbour moment directions enter through solid harmonics $R_{l}(\mathbf m_j)$, which are smooth as |m| → 0.
- The central atom's own moment is correlated with its many-body environment.
- Otherwise it is the standard MACE structure, at modest extra cost.

**Results relevant here.**
- **Standard MACE cannot tell spin states apart at fixed geometry.**
  - On collinear CrN it "fails to resolve energy differences associated with different spin states"; mMACE resolves them.
  - Energy RMSE: 31.2 (MACE) → 1.21 meV/atom (mMACE).
  - Force RMSE: 219 → 24 meV/Å.
- Pre-trained mMACE (MATPES-PBE) improves on magnetic configurations **without degrading** non-magnetic ones.
- Magnetic moments can be relaxed self-consistently by minimising E with respect to m, the analogue of an SCF cycle.

**Why this matters for SCO (interpretation, not from [M]).**
- Fe in the LS state has |m| ≈ 0; in the HS state |m| ≈ 4 µ_B.
- A standard MACE model has to *infer* the spin state from geometry alone. That works deep in the LS or HS basin, where Fe–N ≈ 1.99 vs 2.18 Å [T]. It is ambiguous in the **intermediate region (Q ≈ 2.1 Å)**, which is exactly where [C] found that a combined LS+HS kernel MLFF was "not yet optimal".
- An mMACE model with |m_Fe| as an input can represent **two energy surfaces at the same geometry**. That is what a single continuous model of thermal SCO needs.
- This is a key hypothesis to test in weeks 4–6.

**Practical caveat.**
- mMACE code is released separately (ref. [18] in [M]). Its interface is **not** described in these documents, so check the repository before planning around specific commands.
- The mMACE hyperparameters in [M] (App. F) are a reasonable starting point:
  - 128×0e + 128×1o, correlation 3, r_max = 6 Å
  - $l^{pos}_{max}=3$, $l^{mag}_{max}=1$
  - loss weights E:F:S = 1:10:5
  - Adam with learning rate 0.005, AMSGrad and EMA
  - M_max,Fe = 4.0 µ_B

### 2.8 The benchmark material: [Fe(ptz)₆](BF₄)₂

Values from [T] and [C]:

| Property | Experiment | DFT (PBE-D3(BJ)+U, U_eff = 1.9 eV) | VASP MLFF |
|---|---|---|---|
| Space group / Z | R-3 / 3 | — | — |
| Atoms per cell | 321 (Fe, N, C, H, B, F) | | |
| a (Å) LS / HS | 10.70 / 10.88 | 10.66 / 10.89 | 10.62 / 10.88 |
| c (Å) LS / HS | 31.90 / 31.48 | 31.68 / 30.82 | 31.78 / 30.95 |
| V (Å³) LS / HS | 3163.5 / 3227.8 | 3117.9 / 3163.0 | 3107.2 / 3172.1 |
| d(Fe–N) (Å) LS / HS | 1.99 / 2.18 | 1.98 / 2.19 | 1.98 / 2.19 |
| ΔH_HL (meV) | 63.2 | 77.0 | 80.2 |
| T₁/₂ | 135 K, abrupt, no hysteresis (fast-cooled) | | |
| Other | LIESST at 10 K, T_LIESST ≈ 60 K; slow cooling gives an R-3 → P-1 structural phase transition | | |

Experimental structures come from Kusz *et al.* (CSD JANSAS06 = LS, JANSAS05 = HS) [T].

> **Accuracy target (derived).**
> - ΔH ≈ 65 meV per Fe. With Z = 3 that is about **0.2 eV per 321-atom cell, or ~0.6 meV/atom**.
> - An MLIP must have energy errors well below ~0.5 meV/atom *and* systematic, not random, errors between states to resolve the spin-state energetics.
> - The tutorial ZrO₂ models reach 1–2 meV/atom. SCO is a much more demanding energy target than typical MLIP benchmarks, so report ΔE_HL directly rather than relying on RMSE per atom.

---

## Part 3: Practical Examples

All code below is adapted from the tutorial scripts [V] and the Tan MD script [T]. It has **not been run** on the SCO data yet. Treat it as a starting template.

The environment in this repo is `.venv-MACE` (Python 3.14, mace-torch 0.3.16, ASE, torch, pymlff). The tutorial also installs `phonopy` and `h5py`, which may need `pip install phonopy h5py`.

**Project data** (in `data/raw/`):

| File | Contents |
|---|---|
| `ML_AB` | Combined LS+HS training set: 790 configurations, elements Fe N C H F B, up to 321 atoms |
| `ML_AB_LS`, `ML_AB_HS` | Separate LS and HS training sets |
| `ML_AB_LS_Test`, `ML_AB_HS_Test` | Held-out test sets |
| `POSCAR_LS` | LS starting structure |

### 3.1 Convert VASP `ML_AB` → extended XYZ

MACE trains on extxyz files: per-frame energy and stress, per-atom forces [V].

```python
# scripts/convert_data.py (extended from the existing script)
from pymlff import MLAB

for name in ["ML_AB", "ML_AB_LS", "ML_AB_HS", "ML_AB_LS_Test", "ML_AB_HS_Test"]:
    ab = MLAB.from_file(f"data/raw/{name}")
    ab.write_extxyz(f"data/converted/{name}.xyz", "kbar")
```

Then **inspect the output before training**:

```python
from ase.io import read
frames = read("data/converted/ML_AB.xyz", ":")
a = frames[0]
print(len(frames), len(a), a.info.keys(), a.arrays.keys())
print(a.info)                       # which key holds energy / stress?
```

Check before training:
- **Key names.** Pass the names you find to `--energy_key`, `--forces_key` and `--stress_key`.
- **Stress units and sign.** MACE/ASE expect **eV/Å³** [V], while VASP reports kBar (1 kBar = 1/1602.18 eV/Å³ ≈ 6.24×10⁻⁴ eV/Å³). Check what `write_extxyz(..., "kbar")` actually writes.
- **Spin-state labels.** If you train one combined model, add a per-frame `config_type` (LS/HS) so errors can be reported per spin state. `ML_AB` concatenates both states.
- **Reference energies.** The reference atomic energies in `ML_AB` are all 0, so there are no DFT isolated-atom energies. Use `--E0s="average"` [V] unless isolated-atom calculations are done.

### 3.2 Foundation model: single point and relaxation (weeks 1–3)

*External, verify.* The documents do not describe how to load MACE foundation models. In mace-torch the usual entry point is `mace.calculators.mace_mp(model=...)`. Check the MACE docs (https://mace-docs.readthedocs.io) for current model names.

This script follows `relax.py` [V] (FrechetCellFilter + BFGS, fmax = 0.01), adapted to report the SCO order parameter:

```python
# relax_sco.py  —  usage: python relax_sco.py POSCAR_LS LS
import sys, numpy as np, torch
from ase.io import read, write
from ase.optimize import BFGS
from ase.filters import FrechetCellFilter
from mace.calculators import mace_mp            # external, verify
# from mace.calculators import MACECalculator  # for a trained/fine-tuned model

torch.set_num_threads(8)
path, label = sys.argv[1], sys.argv[2]
atoms = read(path)
atoms.calc = mace_mp(model="medium", device="cpu", default_dtype="float64")
# atoms.calc = MACECalculator(model_paths="sco_MACE_stagetwo.model", device="cpu", default_dtype="float64")

def mean_fe_n(a, cutoff=2.6):
    """Mean Fe–N distance over the 6 nearest N of each Fe (the SCO order parameter)."""
    fe = [i for i, s in enumerate(a.get_chemical_symbols()) if s == "Fe"]
    n  = [i for i, s in enumerate(a.get_chemical_symbols()) if s == "N"]
    out = []
    for i in fe:
        d = np.sort(a.get_distances(i, n, mic=True))[:6]
        out.append(d.mean())
    return np.array(out)

print("start  Fe–N:", mean_fe_n(atoms))
opt = BFGS(FrechetCellFilter(atoms), trajectory=f"{label}-relax.traj", logfile=f"{label}-relax.log")
opt.run(fmax=0.01)
print("final  Fe–N:", mean_fe_n(atoms), " V =", atoms.get_volume(), " E =", atoms.get_potential_energy())
write(f"{label}-relax.cif", atoms)
```

Notes:
- Use `float64` for relaxations and phonons. The tutorial uses `float32` for speed, but SCO energy differences are sub-meV/atom (§2.8).
- Compare the relaxed structures with the table in §2.8: a, c, V and d(Fe–N) per state, then **ΔE_HL = E_HS − E_LS** per Fe (divide the cell difference by 3).
- A HS starting structure is needed, e.g. a relaxed HS frame from `ML_AB_HS` or the JANSAS05 CIF [T].
- **Key question for the foundation model:** starting from the HS geometry, does it keep a HS-like Fe–N ≈ 2.18 Å, or collapse to LS? A non-magnetic model sees only geometry.

### 3.3 Train or fine-tune a MACE model (week 4)

Training from scratch, adapted from `run1.sh` [V]. Remove the SLURM header when running locally.

```bash
mace_run_train \
  --name="sco_MACE" \
  --train_file="data/converted/ML_AB.xyz" \
  --valid_fraction=0.1 \
  --test_file="data/converted/ML_AB_LS_Test.xyz" \
  --energy_key="energy" --forces_key="forces" --stress_key="stress" \
  --compute_stress=True \
  --loss="huber" \
  --E0s="average" \
  --model="ScaleShiftMACE" \
  --num_interactions=2 --num_channels=64 --max_L=1 --correlation=3 \
  --r_max=5.0 \
  --batch_size=2 --max_num_epochs=100 \
  --ema --ema_decay=0.99 --amsgrad \
  --swa \
  --default_dtype="float64" \
  --device=cpu --seed=123
```

- **Starting settings.**
  - `max_L=1` follows the "medium" model [V].
  - The small `batch_size` is because each frame has 321 atoms, versus 12 for ZrO₂.
  - Use `--device=cuda` on a GPU node [V].
  - Run `mace_run_train --help` for every option [V].
- **Hyperparameters to sweep** [V]:
  - energy/forces/stress weights (`--energy_weight` etc.). Up-weight energy, since ΔE_HL is the critical quantity. [T] also raised the energy weight 10× when refitting the VASP MLFFs.
  - `max_L`, `num_channels` (64/128/256), `r_max`, number of epochs.
- **Separate vs combined models.** Train LS-only, HS-only and combined models, and compare them on the LS and HS test sets. This mirrors [C] and [T].
- **Final model.** Once the settings are chosen, retrain on all data with `--valid_fraction=0` [V].

*Fine-tuning a foundation model (external, verify).* mace-torch provides `--foundation_model=<name or path>` (and related multihead fine-tuning options) in `mace_run_train`. Following [M], use a **~10× lower learning rate** (e.g. `--lr=0.0005`) and a smaller gradient clip, and watch the validation curve for early overfitting.

*mMACE fine-tuning:* the training data needs per-atom magnetic moments (Fe: ~0 for LS, ~4 µ_B for HS). Check whether they can be recovered from the VASP data (the `ML_AB` format may not store them) or must be assigned by spin label.

### 3.4 Evaluate: parity plots and errors (weeks 4 and 7)

Adapted from `run2.sh` and `plot-parity.py` [V]:

```bash
mace_eval_configs --configs data/converted/ML_AB_LS_Test.xyz --model sco_MACE_stagetwo.model \
                  --output ls_test_eval.xyz --device=cpu --compute_stress --default_dtype="float64"
mace_eval_configs --configs data/converted/ML_AB_HS_Test.xyz --model sco_MACE_stagetwo.model \
                  --output hs_test_eval.xyz --device=cpu --compute_stress --default_dtype="float64"
```

- Then run `plot-parity.py` [V, Appendix 1] with the `datasets` dict pointed at these files. Predictions are stored under `MACE_energy`, `MACE_forces` and `MACE_stress`.
- **Report:**
  - RMSE E (meV/atom *and* meV/cell), RMSE F (meV/Å) and RMSE stress, for **LS and HS separately**.
  - For context: the VASP kernel MLFF reached ≈ 27 meV/Å and ≈ 24 meV/cell [T].
  - **ΔE_HL** on matched relaxed LS/HS pairs.
  - **Interpolation test [C]:** linearly interpolate between the relaxed LS and HS structures, compute E along the path with each model, and compare with DFT if it is available. The intermediate region (Q ≈ 2.1 Å) is the hard case.

For week 7 ("larger existing datasets"), reuse the same eval + parity pipeline on each dataset with the foundation model, and tabulate errors per material.

### 3.5 Phonons and pDOS (weeks 2–3, 5–6)

Adapted from the tutorial's `phonons.py` (Ex. 1 and 3) [V]. The settings follow [T]: Γ-only (1×1×1 supercell, since the 321-atom cell is already large), 0.015 Å displacements.

```python
# phonons_sco.py  —  usage: python phonons_sco.py LS
import sys, numpy as np, torch
from ase import Atoms
from ase.io import read
from mace.calculators import MACECalculator
from phonopy import Phonopy
from phonopy.structure.atoms import PhonopyAtoms

torch.set_num_threads(8)
label = sys.argv[1]
atoms = read(f"{label}-relax.cif")
calc = MACECalculator(model_paths="sco_MACE_stagetwo.model", device="cpu", default_dtype="float64")

unitcell = PhonopyAtoms(symbols=atoms.get_chemical_symbols(),
                        cell=atoms.cell.array, scaled_positions=atoms.get_scaled_positions())
phonon = Phonopy(unitcell, supercell_matrix=np.diag([1, 1, 1]), primitive_matrix="auto", symprec=1e-4)
phonon.generate_displacements(distance=0.015)
scs = phonon.supercells_with_displacements
print(f"{len(scs)} displaced supercells")        # expect up to ~6×321 without symmetry

forces = []
for i, sc in enumerate(scs):
    a = Atoms(symbols=sc.symbols, cell=sc.cell, scaled_positions=sc.scaled_positions, pbc=True)
    a.calc = calc
    forces.append(a.get_forces())

phonon.produce_force_constants(forces=forces)
phonon.symmetrize_force_constants()
phonon.save(filename=f"{label}-phonopy.yaml")

# pDOS (compare 1–10 THz region between LS and HS; [T] Fig. 4.4)
phonon.run_mesh([1, 1, 1])
phonon.run_total_dos(sigma=0.5)
phonon.write_total_dos(filename=f"{label}-total_dos.dat")

# Harmonic thermodynamics → F_vib(T), S_vib(T)
phonon.run_thermal_properties(t_min=0, t_max=300, t_step=5)
tp = phonon.get_thermal_properties_dict()       # free_energy in kJ/mol (per cell), entropy in J/K/mol
np.savetxt(f"{label}-thermal.dat",
           np.c_[tp["temperatures"], tp["free_energy"], tp["entropy"], tp["heat_capacity"]],
           header="T(K) F_vib(kJ/mol) S_vib(J/K/mol) Cv(J/K/mol)")
```

**Checks:**
- **Imaginary modes.** A few near zero at Γ are acoustic. Large imaginary modes mean the structure is not relaxed or the model is unstable.
- **LS vs HS pDOS.** HS should be softer at low frequency [C, T].
- **ZPE.** `F_vib(T=0)` is the ZPE. Compare ΔZPE_HL with the 101.8 meV found in [T] (§2.2).
- **Speed.** [T] recorded DFT vs MLFF timings. Record the MACE timing too, as one of the project's headline results.

### 3.6 Gibbs / Helmholtz free energy and T₁/₂ (weeks 2–3, 5–6)

**Harmonic estimate at the relaxed volumes.** Combine §3.2 and §3.5:

```python
import numpy as np
kJmol_to_eV = 1.0 / 96.485
Z = 3                                           # Fe per cell
E = {"LS": E_LS_cell, "HS": E_HS_cell}          # eV, from relax_sco.py
T, F_LS = np.loadtxt("LS-thermal.dat", usecols=(0, 1), unpack=True)
_, F_HS = np.loadtxt("HS-thermal.dat", usecols=(0, 1), unpack=True)
kB = 8.617333e-5
dG = ((E["HS"] - E["LS"]) + (F_HS - F_LS) * kJmol_to_eV) / Z - T * kB * np.log(15)   # eV per Fe
# add ΔS_elec = kB ln 15 per Fe (§2.2); T_1/2 where dG crosses zero
i = np.where(np.diff(np.sign(dG)))[0]
print("T_1/2 ≈", T[i] if len(i) else "no crossing (check ZPE / reference method)")
```

**Quasi-harmonic version,** following the procedure in [T] §3.5:
1. Scale the relaxed cell isotropically by factors of about 0.99–1.03 (5 volumes).
2. Relax at fixed volume (positions + cell shape). ASE: `FrechetCellFilter(atoms, constant_volume=True)`.
3. Compute phonons + `run_thermal_properties` at each volume.
4. Fit F(V) at each T with an equation of state. phonopy's `PhonopyQHA` (external, verify) takes `volumes`, `electronic_energies`, `temperatures`, `free_energy`, `cv` and `entropy`, and returns V_eq(T), α_V(T) and G(T).
5. Find T₁/₂ where G_HS(T) = G_LS(T) per Fe, after adding ΔS_elec.

Compare α_V(T) with [T] Fig. 4.7: LS should expand more than HS.

### 3.7 MD: equilibrium NPT run (weeks 5–6)

Adapted from `md-npt1.py` [V], with the SCO observables from [T] A.7:

```python
# md_npt_sco.py  —  usage: python md_npt_sco.py LS 100
import sys, numpy as np, torch
from ase import units
from ase.io import read, write
from ase.md.npt import NPT
from ase.md.velocitydistribution import MaxwellBoltzmannDistribution, Stationary
from mace.calculators import MACECalculator

torch.set_num_threads(8)
label, T = sys.argv[1], float(sys.argv[2])
dt, tot_ps = 1.0, 20.0                   # fs, ps ([T] used 2 fs with m_H = 4 amu)
atoms = read(f"{label}-relax.cif"); atoms.pbc = True
# optional: heavier H to allow 2 fs  ([T] §3.4)
# m = atoms.get_masses(); m[np.array(atoms.get_chemical_symbols()) == "H"] = 4.0; atoms.set_masses(m)
atoms.calc = MACECalculator(model_paths="sco_MACE_stagetwo.model", device="cpu", default_dtype="float32")

MaxwellBoltzmannDistribution(atoms, temperature_K=T, rng=np.random.default_rng(123)); Stationary(atoms)
ptime = 75 * units.fs
dyn = NPT(atoms, timestep=dt * units.fs, temperature_K=T, externalstress=0.0,
          ttime=25 * units.fs, pfactor=ptime**2 * 1.0)   # B=1.0 as in [V]; tune

fe = [i for i, s in enumerate(atoms.get_chemical_symbols()) if s == "Fe"]
n  = [i for i, s in enumerate(atoms.get_chemical_symbols()) if s == "N"]
log = open(f"{label}-T{T}.log", "w")
log.write("# t(ps) T(K) Epot/atom(eV) V(A^3) FeN_mean(A) FeN_per_site...\n")
def write_log():
    q = [np.sort(atoms.get_distances(i, n, mic=True))[:6].mean() for i in fe]
    log.write(f"{dyn.get_time()/units.fs/1000:10.4f} {atoms.get_temperature():9.2f} "
              f"{atoms.get_potential_energy()/len(atoms):14.8f} {atoms.get_volume():11.3f} "
              f"{np.mean(q):8.4f} " + " ".join(f"{x:7.4f}" for x in q) + "\n")
dyn.attach(write_log, interval=20)
dyn.attach(lambda: write(f"{label}-T{T}.xyz", atoms, append=True), interval=200)
dyn.run(int(tot_ps * 1000 / dt))
log.close()
```

- **Stability check [V]:** T, E, V and lattice parameters should fluctuate about steady values. Average the lattice parameters after discarding the equilibration part, as in `lattice-par.py` [V].
- **Per-site Fe–N distances** show whether individual Fe centres switch. Such switches can only happen with a combined or magnetic model.
- **Visualisation:** open the `.xyz` in OVITO [V].

### 3.8 MD: heating/cooling ramp to look for SCO (weeks 5–6)

This combines the temperature ramp in `md-npt2.py` [V] with the protocol proposed in [T] §5.1.1:
1. Equilibrate the MLIP-relaxed LS structure at **50 K for 20 ps**.
2. **Heat 50 → 250 K in 10 K steps**, 20 ps per step.
3. Cool back down the same way.
4. Record V and mean Fe–N every ~200 fs. T₁/₂ is where the mean Fe–N (or V) jumps. Compare heating and cooling curves to look for hysteresis (§2.1).

```python
# replace the single dyn.run(...) in §3.7 with:
for T_target in list(range(50, 260, 10)) + list(range(240, 40, -10)):
    dyn.set_temperature(temperature_K=float(T_target))
    dyn.run(int(20.0 * 1000 / dt))
```

**Important caveats:**
- **Separate LS and HS models cannot switch.** Only a **combined** model (one surface covering both basins) or an **mMACE** model with a spin degree of freedom can show SCO in MD. This is the central scientific test of weeks 5–6.
- **Tan's mixed-calculator approach is not a way round this.** [T] proposed mixing LS/IS/HS MLFFs with an SOC Hamiltonian at each step, but it was impractically slow through VASP, estimated at more than 7 years for the full protocol. A MACE model evaluated in-process avoids the VASP restart overhead that caused this.
- **Finite-size and timescale limits.** Real switching takes ns–µs [C]. With one 321-atom cell (3 Fe), expect noisy, finite-size-dominated behaviour. Consider supercells (e.g. `atoms.repeat((2,2,1))`) once the model is stable, and use slow ramps, as recommended for the ZrO₂ phase transition in [V].

---

## Part 4: Open questions to raise with the supervisor

These come out of the documents above; the proposal does not settle them.

1. **Which reference data?** `data/raw/ML_AB*` has 790 configurations. [C] mentions ~600 structures recomputed with U = 1.7 eV, and [T] used U = 1.9 eV. Which DFT+U setting produced this dataset?
2. **Magnetic moments.** Are per-atom Fe moments available for mMACE training, or should they be assigned from the LS/HS label?
3. **Which foundation models** (MACE-MP variants, MATPES-PBE mMACE from [M])? Their reference functional (PBE / r²SCAN, no +U) differs from the project's PBE-D3(BJ)+U. That difference should show up as a systematic ΔE_HL error.
4. **"Larger set of SCO materials" (week 7).** Which existing datasets?
5. **Compute.** Is a GPU available? The tutorial trained on 28 CPU threads, and 321-atom frames make training noticeably heavier.
