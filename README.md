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

> **Tested setup:** Python 3.9, petsc4py 3.16, numpy 1.26, mpi4py 3.1, h5py 3.13, matplotlib 3.5 on macOS. Newer versions may work but have not been tested; numpy 2 in particular prints deprecation warnings from `src/TCAT2Phase/numerics/utils.py`.

Run `conda activate tcat2p` in every new terminal before using the code.

## 2. Run your first simulation

Each driver script is self-contained: run it **from inside its own folder**, because it imports the `solver_options.py` that sits next to it.

```bash
cd applications/Testcase1_fully_saturated_system
python re_fully_saturated.py
```

It should finish in under 10 seconds. The last line printed is where it saved its results:

```
Saving file to: .../outputs/Testcase1_fully_saturated_system/outputs/RE_test_case_1.h5
```

From VS Code: open the repo folder, choose the `tcat2p` interpreter (Command Palette → *Python: Select Interpreter*), open a driver, and click the ▷ Run button in the upper right. VS Code runs the file from its own folder, so the import works.

## 3. Test cases

Every row ran to completion on the tested setup. Times are wall-clock on a quad-core Intel Core i5 Mac.

| Test case (folder in `applications/`) | Drivers | Approx. run time |
|---|---|---|
| 1 Fully saturated system | `re_fully_saturated.py`, `2p_fully_saturated.py`, `tcat_cgp_fully_saturated.py` | < 10 s each |
| 2 Hydrostatic equilibrium | `re_…`, `2p_…`, `tcat_cgp_hydrostatic_equi_vertical.py` | 3–25 s |
| 3 Hydrostatic equilibrium with perturbation | `re_equilibrium_perturbation.py`, `2p_…`, `tcat_cgp_…_perturbation.py` | 20–75 s |
| 4 Infiltration | `re_infiltration_horizontal.py`, `2p_…`, `tcat_cgp_…` | 10–25 s |
| 5 Primary drainage | `re_drainage_pdc.py`, `2p_drainage_pdc.py`, `tcat_cgp_drainage.py`, `tcat_full_drainage.py` | 30 s – 1.5 min; full TCAT ≈ 7 min |
| 6 Main imbibition | `re_imbibition_mic.py`, `2p_imbibition_mic.py`, `tcat_cgp_imbibition.py`, `tcat_full_imbibition.py` | 1–4 min; full TCAT ≈ 8 min |
| 7 Non-equilibrium capillary pressure | `ere_malgorzata.py`, `e2p_malgorzata.py` (each runs three values of τ) | 7–14 min |
| 8 Hysteresis and entrapment | `re_malgorzata_hysteresis_entrapment_pdc.py`, `2p_malgorzata_hysteresis_entrapment_pdc.py` | **very long** (see below) |

Start with Test Cases 1–4 to confirm everything works before trying the longer runs.

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
- **Solver settings:** the `solver_options.solver_options(...)` call (BDF order, preconditioner, maximum number of steps). Any other PETSc option can be added in that folder's `solver_options.py`.

## 6. Repository layout

```
src/TCAT2Phase/     the package
  numerics/           model setup, residuals for each model, PETSc solver driver
  properties/         domain, phase, and interface objects
  state_equations/    equations of state, van Genuchten, Parker–Lenhard hysteresis
  mesh/               1D mesh
  io/                 HDF5 save/load
applications/       one folder per test case, each with driver scripts and solver_options.py
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

One test is marked **xfail** (expected to fail): it documents a known inconsistency in the TCAT closure relations that the current solvers don't depend on; the reason is printed with `pytest -rx`. If one of them is fixed, pytest reports it as an unexpected pass, and its `xfail` marker should be removed.

Add a test whenever you change the library. In particular, compare any new analytic derivative against a finite difference, as the existing tests do.

## 8. Troubleshooting

| Symptom | Cause and fix |
|---|---|
| `ModuleNotFoundError: No module named 'petsc4py'` | The conda environment isn't active. Run `conda activate tcat2p`. |
| `ModuleNotFoundError: No module named 'TCAT2Phase'` | Run `pip install -e .` from the repo root with the environment active. |
| `ModuleNotFoundError: No module named 'solver_options'` | You ran a driver from a different folder. `cd` into its test-case folder first. |
| `FileNotFoundError` from a plotting script | Run the matching driver first, and check that the filename in the script matches the `.h5` file in the `outputs/` folder. |
| A run seems frozen | Watch the `TS dt ... time ...` lines it prints. If time is still increasing, it is working, just slowly (see Test Case 8). |
