# TCAT2Phase

One-dimensional simulators for two-fluid flow in porous media. The package implements and compares:

| Model | What it is | Driver prefix |
|---|---|---|
| **RE** | Richards' equation (pressure formulation) | `re_` |
| **2P-TM** | Traditional two-phase model (saturation-pressure or pressure-pressure formulations) | `2p_` |
| **ERE / E2P** | Richards' equation and the two-phase model extended with dynamic (non-equilibrium) capillary pressure | `ere_`, `e2p_` |
| **TCAT CGP** | Thermodynamically Constrained Averaging Theory model with constant gas-phase pressure (six equations, six unknowns) | `tcat_cgp_` |
| **Full TCAT** | Full two-phase TCAT model | `tcat_full_` |

All models are solved with [PETSc](https://petsc.org) through `petsc4py`: finite-volume discretization in space, implicit BDF time stepping, Newton's method (SNES), and GMRES with SOR preconditioning.

The eight test cases in `applications/` are the ones in T. M. Nelson, *On the Macroscale Modeling of Two-Fluid Flow in Porous Medium Systems Using the Thermodynamically Constrained Averaging Theory*, PhD dissertation, UNC Chapel Hill, 2025 (Chapters 4 and 5).

---

## 1. Install

You need a terminal, `git`, and a conda distribution ([Miniforge](https://github.com/conda-forge/miniforge) is recommended; Anaconda also works). PETSc is hard to build by hand, so install it from conda-forge rather than with `pip`.

```bash
# Get the code
git clone https://github.com/TLN070405/TCAT_2P_Code.git
cd TCAT_2P_Code

# Create an environment with PETSc and the other dependencies
conda create -n tcat2p -c conda-forge python=3.9 petsc4py mpi4py "numpy<2" h5py matplotlib pytest
conda activate tcat2p

# Install this package in editable mode, so edits to src/ take effect immediately
pip install -e .
```

Check that it worked:

```bash
python -c "import petsc4py, TCAT2Phase; print('petsc4py', petsc4py.__version__, '- TCAT2Phase OK')"
```

> **Tested setup:** Python 3.9, petsc4py 3.16, numpy 1.26, h5py 3.13, matplotlib 3.5 on macOS. Python 3.9 or newer is required. Newer package versions should work but have not been tested.

Run `conda activate tcat2p` in every new terminal before using the code.

## 2. Run your first simulation

Each driver script is self-contained and can be run from any folder:

```bash
python applications/Testcase1_fully_saturated_system/re_fully_saturated.py
```

It should finish in under 10 seconds. The last line printed is where it saved its results:

```
Saving file to: .../outputs/Testcase1_fully_saturated_system/outputs/RE_test_case_1.h5
```

From VS Code: open the repo folder, choose the `tcat2p` interpreter (Command Palette → *Python: Select Interpreter*), open a driver, and click the ▷ Run button in the upper right.

## 3. Test cases

Every row ran to completion on the tested setup except the Test Case 4 TCAT CGP driver (see [Known issues](#8-known-issues)). That was with the legacy −2K Gaussian-curvature term. With the corrected +2K, now the default, the regular Test Case 5 and 6 TCAT drivers fail after a few seconds of simulated time (same section). Pass `gaussian_curvature_sign=-1` to `TCATInterface` to get the legacy behaviour. Times are wall-clock on a quad-core Intel Core i5 Mac.

| Test case (folder in `applications/`) | Drivers | Approx. run time |
|---|---|---|
| 1 Fully saturated system | `re_fully_saturated.py`, `2p_fully_saturated.py`, `tcat_cgp_fully_saturated.py` | < 10 s each |
| 2 Hydrostatic equilibrium | `re_…`, `2p_…`, `tcat_cgp_hydrostatic_equi_vertical.py` | 3–25 s |
| 3 Hydrostatic equilibrium with perturbation | `re_equilibrium_perturbation.py`, `2p_…`, `tcat_cgp_…_perturbation.py` | 20–75 s |
| 4 Infiltration | `re_infiltration_horizontal.py`, `2p_…`, `tcat_cgp_…` | 10–25 s; TCAT CGP stops at 0.52 s after ~3 min |
| 5 Primary drainage | `re_drainage_pdc.py`, `2p_drainage_pdc.py`, `tcat_cgp_drainage.py`, `tcat_full_drainage.py` | 30 s – 1.5 min; full TCAT ≈ 7 min |
| 6 Main imbibition | `re_imbibition_mic.py`, `2p_imbibition_mic.py`, `tcat_cgp_imbibition.py`, `tcat_full_imbibition.py` | 2–5 min; full TCAT ≈ 8 min |
| 7 Non-equilibrium capillary pressure | `ere_malgorzata.py`, `e2p_malgorzata.py` (each runs three values of τ) | 7–14 min |
| 8 Hysteresis and entrapment | `re_malgorzata_hysteresis_entrapment_pdc.py`, `2p_malgorzata_hysteresis_entrapment_pdc.py` | **very long** (see below) |

Start with Test Cases 1–4 to confirm everything works before trying the longer runs.

**Parameters follow the dissertation tables.** The regular drivers use the values in the dissertation's Chapter 4 and 5 tables:

- **Water compressibility:** βw = 4.5×10⁻¹⁰ Pa⁻¹ in every driver (Tables 4.2 and 5.6).
- **Test Case 1, TCAT CGP:** R̂ = ε²μ/k = 5.10×10⁴ (Table 4.7, Eqn 4.19).
- **Test Case 4:** a horizontal 10 m domain with pw = −900 / −1000 Pa, run to 1.5×10⁵ s (Tables 4.13–4.16). RE uses the MIC van Genuchten parameters (Table 4.3).
- **Test Case 5, RE:** drains from saturation to εw = 0.0254 (Tables 4.17–4.18; output times as in Fig 4.5).
- **Test Case 6, RE:** imbibes from εw = 0.0254 to 0.337, with S_ws = 0.913 on the MIC (Tables 4.23–4.24; Fig 4.16).
- **Test Cases 5 and 6, TCAT CGP:** 201 nodes, the resolution the dissertation found sufficient.

Two exceptions:

- The `*_defense.py` drivers keep the values of the original codes, so the defense slides still reproduce.
- The Tocci and Helmig validation drivers use their papers' values.

Remaining gaps are listed under [Known issues](#8-known-issues).

**Defense scenarios (Test Cases 5 and 6).** The regular Test Case 5 and 6 drivers run a short 300 s case. The scenarios shown in the May 2025 defense and in the dissertation (drainage from εw = 0.337 to 0.240 over 8000 s; imbibition from 0.325 to 0.337 over 250 s) are in the `*_defense.py` drivers:

| Slides | Drivers | Plot script (in `outputs/<case>/postprocessing/`) | Matches the original code to |
|---|---|---|---|
| 9–12 (RE vs TCAT CGP) | `re_drainage_defense.py`, `tcat_cgp_drainage_defense.py`, `re_imbibition_defense.py`, `tcat_cgp_imbibition_defense.py` | `plot_results_defense_re_cgp.py` | RE about 1e-12; TCAT CGP within 0.3% of each variable's change |
| 16–21 (2P-TM vs full TCAT) | `2p_drainage_defense.py`, `tcat_full_drainage_defense.py`, `2p_imbibition_defense.py`, `tcat_full_imbibition_defense.py` | `plot_results_defense_2p_tcat.py` | 2P-TM 1e-10 to 1e-7; full TCAT about 1e-9 |

The defense results were produced with earlier standalone Cython codes. The new drivers copy two quirks of those codes on purpose, so the slides can be reproduced. Each is flagged at the top of the driver:

- `REPRODUCE_DEFENSE_KRW = True` in the RE drivers: the original code evaluated the RE relative permeability at the volume fraction instead of the capillary pressure, which made k_rw = 1. Set it to `False` for the correct Van Genuchten k_rw.
- `MONITOR_NODE` in the 2P/full-TCAT drivers: because of an indexing slip, the original code recorded its normalized time history at x = 0.25 m (2P-TM) and x = 0.375 m (full TCAT), not the midpoint.

Run times on the tested Mac: RE under 1 min, TCAT CGP 2–3 min, 2P-TM 1–35 min, full TCAT 8–15 min. Each needs under 100 MB of memory.

**Test Case 8** simulates 3000 s of repeated drainage and imbibition, with Parker–Lenhard hysteresis, van Genuchten closure relations, and Land's model for entrapment. The solver takes very small time steps (about 0.002 s early in the run), so a full run takes well over 15 minutes (it was not run to completion when these instructions were tested), and its output file is large (the 2P result is about 160 MB). To try it quickly, shorten the run by editing `time_values` and `pw_values` in `pw0_bc()` and the simulation times passed to `model.set_simulation_time(...)`.

## 4. Results and plots

Each driver writes an HDF5 file to

```
outputs/<TestCase folder>/outputs/<MODEL>_test_case_<N>.h5
```

The folder is created automatically. Running a driver again **overwrites** its previous file, so rename or copy results you want to keep.

Plotting scripts live in `outputs/<TestCase folder>/postprocessing/` (`plot_results_re.py`, `plot_results_2p.py`, and so on). Run them after the matching driver has written its `.h5` file, for example:

```bash
cd outputs/Testcase8_complex_displacement_patterns/postprocessing
python plot_results_re.py
```

Each opens its figures in a window. To read results in your own code:

```python
from TCAT2Phase.io.hdf5_io import load_solutions_from_hdf5
```

> **Note:** the `plot_all_results*.py` scripts compare several grid resolutions and expect files named like `RE_test_case_1_101.h5` (node count appended). The drivers currently write files without the node count, so these scripts need their filenames edited before they will run.

## 5. Changing a simulation

Everything that defines a run is set in the driver script itself; there are no input files or command-line options. Near the top of each driver you will find:

- **Domain and grid:** `domain_size` and `nodes`
- **Material properties:** porosity, permeability, and fluid densities and viscosities
- **Closure relations:** van Genuchten parameters (`VanGenuchten(...)`) and, for Test Case 8, the Parker–Lenhard settings
- **Initial and boundary conditions:** the `*_ic` and `*_bc` functions
- **Output times:** the array passed to `model.set_simulation_time(...)`
- **Solver settings:** the `solver_options.solver_options(...)` call (BDF order, preconditioner, maximum number of steps). These are defined in `src/TCAT2Phase/numerics/solver_options.py`; pass `monitor=False` to silence the per-step output. Any other PETSc option can be set with `PETSc.Options().setValue(name, value)` before `setup_solver` is called.

## 6. Repository layout

```
src/TCAT2Phase/     the package
  numerics/           model setup, residuals for each model, PETSc solver driver
  properties/         domain, phase, and interface objects
  state_equations/    equations of state, van Genuchten, Parker–Lenhard hysteresis
  mesh/               1D mesh
  io/                 HDF5 save/load
applications/       one folder per test case, with its driver scripts
outputs/            results (.h5) and plotting scripts, one folder per test case
tests/              pytest unit tests
HysteresisCharts/   scripts for the hysteresis figures in the dissertation
```

## 7. Tests

From the repo root, with the environment active:

```bash
pytest            # unit and solver tests, about 15 seconds
pytest -m slow    # also runs the Test Case 1-4 driver scripts, about 5-10 minutes
```

| File | What it checks |
|---|---|
| `test_mesh.py`, `test_utils.py` | Mesh geometry and connectivity; face gradients and interpolation |
| `test_phase_functions.py` | Densities and their derivatives, phase objects, Darcy flux |
| `test_vg_functions.py`, `test_bc_functions.py` | Van Genuchten and Brooks–Corey: known values, analytic formulas, inverse consistency, derivatives against finite differences, limits and monotonicity |
| `test_tcat_functions.py` | TCAT geometric closure relations |
| `test_hysteresis.py` | Parker–Lenhard hysteresis and Land's entrapment, as used in Test Case 8 |
| `test_interface.py`, `test_model.py` | Interface wiring; every model/formulation selects the right unknowns and residual |
| `test_hdf5_io.py` | Saving and loading results |
| `test_solver.py` | Small RE and two-phase problems solved with PETSc, checked against hydrostatic equilibrium |
| `test_drivers.py` (slow) | Runs the real driver scripts in a temporary folder, so `outputs/` is never overwritten |
| `test_benchmarks.py` | Literature benchmarks. Fast: the McWhorter & Sunada (1990) exact two-phase solution is checked against its own integral equation and an independent finite-difference solve. Slow: 2P-TM vs that exact solution for Szymkiewicz (2007) Example 1 (error and grid convergence), and RE for Tocci et al. (1997) Problem A (front position and Celia-style mass balance) |

To document a known bug without fixing it yet, mark its test `@pytest.mark.xfail(strict=True, reason=...)`. Once the bug is fixed, pytest reports the test as an unexpected pass, which is the reminder to remove the marker.

Add a test whenever you change the library. In particular, compare any new analytic derivative against a finite difference, as the existing tests do.

## 8. Known issues

- **Gaussian-curvature term in the TCAT curvature equation: sign corrected to +2K; with it, most TCAT CGP runs become unstable.** The mean-curvature equation is now εwn ∂J/∂t + J ∂εwn/∂t **+** 2K ∂εw/∂t + … = 0, with J the non-wetting-side curvature J_n (p_c = γJ > 0, as the drivers set it). This follows from Gray, Shepherd & Miller, *Toward the Closure of a New Generation of Two-Fluid Flow Models: Curvature Evolution* (manuscript v0.83, 2021): its Eqn (87) has −2K ∂εⁿ/∂t, and ∂εⁿ/∂t = −∂εw/∂t; its Eqn (89) has −2K ∂εw/∂t written in J_w, and J_w = −J_n flips the other terms but not K. Dissertation Eq. 3.49 also has +2K. The other terms in that row (J ∂εwn/∂t, the advection term, and (2/3)∂²/∂x²(∂εw/∂t) with ∂εw/∂t taken from the volume-fraction equation) were already consistent with J = J_n. The original code, and so the dissertation and defense results, used −2K. `TCATInterface(..., gaussian_curvature_sign=+1)` is the default (theory). `-1` gives the legacy term, and the four `*_defense.py` drivers set it so their results are unchanged.
  With +2K, Test Cases 1, 2 and 3 (to the driver's 0.01 s) still run. Test Cases 5 and 6 (regular drivers) fail at t ≈ 2.2 s and 1.16 s, and the Test Case 5 defense setup fails at t ≈ 108 s. The failure mechanism is the same in each case. Combining the volume-fraction equation, ∂εw/∂t ≈ ĉ(p_w + γJ), with the interfacial-area and curvature equations gives, locally, εwn ∂J/∂t ≈ (J² − 2sK) ∂εw/∂t, where s is the sign. The mismatch φ = p_w + γJ then grows at rate ≈ ĉγ(J² − 2sK)/εwn whenever J² − 2sK > 0. For a real surface, J² − 2K = κ₁² + κ₂² ≥ 0, so with the theory sign the closed equations amplify the mismatch: J rises as εw rises, εwn falls (∂εwn/∂t ≈ −J ∂εw/∂t) toward zero, and J and K blow up. The K from the linear geometric state equation (Eqn 3.51) is ≈ −2.7×10⁷ m⁻² at the Test Case 5/6 saturations, so J² − 2K > 0 everywhere there. The legacy −2K gives J² + 2K < 0 for those states, which is why the published runs were stable. It is unstable instead where K > 0, for example the drained part of the Test Case 3 column (K ≈ 2.5×10⁸ m⁻²). Restoring the right behaviour (J_n increasing in drainage) probably needs the pore-geometry information that the manuscript drops when it neglects the common-curve terms (between its Eqns 86 and 88). This is an open modelling question, not something to tune.
- **Incompressible two-phase runs with no-flow boundaries for both phases fail on the first time step** (2P-TM, both formulations). Giving the air phase its physical compressibility makes the `pp` formulation run; `sp` still fails. Found by the Szymkiewicz benchmark.
- **The slow Tocci Problem A benchmarks have not yet been run to completion**, so their tolerances are unverified.
- **Relative permeability is not hysteretic.** Test Case 8 uses the main drainage curve's k_r on scanning curves as well.
- **Face mobility is averaged, not upwinded**, which matches the original code but may cause trouble at sharp fronts.
- **The TCAT momentum resistance R̂ʷ_w is a constant (3.75×10⁵)**, the value of the original codes. Dissertation Tables 4.11–4.25 cite Eqn 4.12, a state equation in εw, the interfacial areas and the curvatures. Its wetting-phase coefficients e₁–e₄ are not published: Bruning & Miller (2019, Table 2) fit only the non-wetting-phase coefficients, in lattice-Boltzmann units. Only Test Case 1 (Table 4.7) gives a number, 5.10×10⁴, which that driver uses.
- **Test Case 4 TCAT CGP stalls at t = 0.52 s** (the time step collapses to ~10⁻¹⁶ s), before and after the velocity fix. Figs 4.2–4.4 were made with the older Cython code in `OtherApps/Testcase4_infiltration_problem`, which used different, pre-Finney-pack parameters. Its εw boundary values (0.359 / 0.368) also differ from Table 4.16, which gives εw = ε. The sign of the K term and R̂ (3.75×10⁵ or 5.10×10⁴) only move the stall (t = 0.18–0.52 s). It always happens at the node next to the inlet: εw rises above the porosity (nothing bounds it), εwn falls from 100 m⁻¹ to zero by ∂εwn/∂t ≈ −J ∂εw/∂t (with J ≈ 1.4×10⁴ m⁻¹, a change of only 0.007 in εw uses up all of εwn), and J diverges. The input data (εw IC almost at ε while p_c = 1000 Pa, εwn = 100 m⁻¹) are the root cause. The slow driver test marks this case as an expected failure.
- **Test Case 3 TCAT CGP: no 16 Pa limit with the current code, but no return to equilibrium either.** The driver runs to t = 0.01 s with perturbations of 10, 11, 16, 20, 50, 100 and 500 Pa, with either K sign and with R̂ = 5.10×10⁴, so the 16 Pa limit reported in the dissertation is not reproduced. By 0.01 s, however, p_w + γJ is still about equal to the perturbation, so the system has not re-equilibrated. The dissertation does not give the end time of its Test Case 3 runs; the older Cython driver in `OtherApps/` ran to 5 s, and an earlier version of this driver to 50 s. Run longer (output to 100 s), the time step collapses with any perturbation, including none at all. With −2K this happens at t ≈ 0.05 s (10 Pa), 0.11 s (10⁻³ Pa) and 0.16 s (0 Pa). With +2K it happens at t ≈ 4.9 s (10 Pa), 11.3 s (10⁻³ Pa) and 9.6 s (0 Pa). So the column's starting state, built with εwn = 800·εw, is not a stable equilibrium of the model; this is the instability described in the first item. The failure does not depend on the size of the perturbation. (Test Case 2 uses εwn = εw, not 800·εw, and BDF1 with a single 100 s step, which damps such modes.) The steady state of the εw and εwn equations forces εwn = εwn_eq(x), but it leaves εw free (K absorbs it), so a "better" εwn_eq alone cannot fix this. Also, the driver's εwn_eq = 800·εw(x) is nonzero in the saturated zone.
- **No 2P-TM drivers for the Chapter 5 validation cases** (Kees & Miller drainage, Tables 5.2–5.3; Szymkiewicz imbibition, Tables 5.4–5.5). `2p_drainage_pdc.py` and `2p_imbibition_mic.py` run Finney-pack cases instead. The Szymkiewicz setup is closest to the benchmark in `tests/benchmarks/countercurrent_2p.py`.

## 9. Troubleshooting

| Symptom | Cause and fix |
|---|---|
| `ModuleNotFoundError: No module named 'petsc4py'` | The conda environment isn't active. Run `conda activate tcat2p`. |
| `ModuleNotFoundError: No module named 'TCAT2Phase'` | Run `pip install -e .` from the repo root with the environment active. |
| `FileNotFoundError` from a plotting script | Run the matching driver first, and check that the filename in the script matches the `.h5` file in the `outputs/` folder. |
| A run seems frozen | Watch the `TS dt ... time ...` lines it prints. If time is still increasing, it is working, just slowly (see Test Case 8). |
