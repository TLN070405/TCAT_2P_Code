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

Every row ran to completion on the tested setup except the Test Case 4 TCAT CGP driver (see [Known issues](#8-known-issues)). Times are wall-clock on a quad-core Intel Core i5 Mac.

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

## 6. Closure relations: where they live and how to swap one

This section is for anyone reusing the solver with different closures, for example a new momentum-transfer or resistance model.

### How a closure reaches the equations

1. The driver builds three kinds of objects: a `Solid`, one `Fluid` per phase (`properties/phase.py`), and an interface (`properties/interface.py`). The interface wraps a closure class from `state_equations/`.
2. `model.set_model_phases(s_phase, w_phase, n_phase, interface)` attaches them to the model.
3. The residual class for the chosen model (`numerics/residual_*.py`) reads them as `self.model.solid`, `self.model.w_phase`, `self.model.n_phase` and `self.model.interface`.

Closure classes are not registered anywhere. A residual simply calls methods by name. A new class with the same method names and signatures therefore drops in without touching the solver.

### Where each closure is set and used

| Closure | Defined in | Set in the driver by | Used in |
|---|---|---|---|
| Fluid density (equation of state) | `state_equations/state_equations.py`: `density_exp_pressure`, `density_linear_pressure` and their derivatives | `Fluid(density_function=..., d_density_function=..., rho0=..., beta=...)` | every residual |
| Darcy flux, traditional models | `state_equations.darcy_velocity` | `Solid(permeability=...)`, `Fluid(viscosity=...)` | `BaseResidual.compute_phases_fluxes` (RE, 2P-TM, ERE, E2P) |
| Capillary pressure and relative permeability | `state_equations_interface.py`: `VanGenuchten`, `BrooksCorey` | `TraditionalInterface(VanGenuchten(...))` | traditional models |
| Hysteresis and entrapment | `state_equations_hysteresis.py`: `ParkerLenhard` (with Land's model) | `ParkerLenhard(model, imbibition, drainage, ...)` | Test Case 8 |
| Dynamic capillary pressure coefficient τ | (a constant) | `Solid(dynamic_capillary_pressure_parameter=...)` | `residual_ere_sp.py`, `residual_e2p_spp.py` |
| **TCAT momentum resistance R̂** | (a constant) | `Fluid(momentum_resistance=...)` | `compute_phases_fluxes` in `residual_tcat_cgp.py` and `residual_full_tcat.py` |
| TCAT mobility factor | `TCAT.w_relative_permeability` (εw²), `TCAT.n_relative_permeability` (εn²) | `TCAT(...)` | the same two flux functions |
| Geometric state equation | `TCAT.gaussian_curvature`, `TCAT.volume_fraction_w` | `TCAT(b1, b2, b3, porosity, e_ss, js_ns, kn_ns)` | `od_k` row of the TCAT residuals |
| Interfacial-area relaxation and dynamics | (constants and a function of x) | `TCATInterface(tcat, interfacial_dynamics, interfacial_tension, interfacial_area_relaxation, interfacial_area_equilibrium)` | `compute_spatial_operator` in the TCAT residuals |

In the TCAT models the wetting-phase flux is

```
q_w = -(εw² / R̂) (∂p_w/∂x − ρ_w g)
```

`compute_phases_fluxes` builds it as `darcy_velocity(mesh, 1/R̂, k_r·ρ, 1.0, ...)`, with `k_r = interface.w_relative_permeability(e_w)`. R̂ is a single number for the whole run.

### Swapping a closure

**A closure that depends only on εw** needs no library change. Write a subclass and pass it to the interface in your driver. For example, to make R̂ a function of εw, fold it into the mobility factor and set `momentum_resistance=1.0`:

```python
class TCATVariableR(state_equations_interface.TCAT):
    def w_relative_permeability(self, e_w):
        return e_w**2 / my_resistance(e_w)  # your R̂(εw)

tcat = TCATVariableR(b1=..., b2=..., b3=..., porosity=..., e_ss=..., js_ns=..., kn_ns=...)
w_phase = TCAT2Phase.properties.phase.Fluid('w', viscosity=None, momentum_resistance=1.0, ...)
```

**A closure that depends on other unknowns** (εwn, Jwn, Kwn, a velocity difference) needs a residual subclass. Dissertation Eqn 4.12 and the Shepherd, Bowers & Miller (2026) resistance R̂(εs, ε_ws J_ws) are of this kind.

1. Subclass the residual in a new file under `numerics/` and override `compute_phases_fluxes`. Its `primary_variables` argument is a dict of node arrays (`'e_w'`, `'p_w'`, `'e_wn'`, `'j_wn'`, …), so R̂ can be computed node by node and interpolated to faces with `utils.interpolate`. The same dict is also stored on `self.model.interface.primary_variables`.
2. Add a model type for it in `numerics/model_types.py`, with its unknowns in `u_names`, and a branch in `get_residual_class`. The solver picks the residual class through `get_residual_class`.
3. Add a test in `tests/test_model.py` that the new model selects the new residual. Also compare any analytic derivative with a finite difference, as `tests/test_vg_functions.py` does.

### Writing a new model

Each residual implements four methods. `BaseResidual` handles the rest: the loop over cells, boundary conditions and the PETSc interface.

| Method | Returns |
|---|---|
| `compute_primary_variables(u)` | dict of node arrays, one per unknown, plus any derived quantities |
| `compute_phases_fluxes(primary_variables)` | dict of face fluxes (`q_w`, …) and cell-centered densities |
| `compute_spatial_operator(primary_variables, fluxes, cell)` | list `od`, one entry per equation, for cell `n = cell.ID` |
| `compute_time_matrix(primary_variables, fluxes)` | fills `self.time_mat` (num_u × num_u) for `self.model.current_cell` |

The residual for each cell is `time_mat · u̇ − od` (`BaseResidual.compute_residual_equations`). Equations with no time derivative, such as the geometric state equation, have a zero row in `time_mat`.

Assumptions built into the current code:
- the domain is one-dimensional, with a uniform mesh;
- the solid is rigid and does not move (`Solid` holds porosity and permeability, but no velocity);
- face mobilities are averaged, not upwinded.

A model with a moving solid phase would need a solid velocity unknown and its own momentum equation in a new residual class. The mesh, the PETSc setup, the time stepping and the HDF5 output can be reused as they are.

## 7. Repository layout

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

## 8. Tests

From the repo root, with the environment active:

```bash
pytest            # unit and solver tests, about 15 seconds
pytest -m slow    # also runs the Test Case 1-4 driver scripts and the analytic checks, about 5-10 minutes
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
| `test_verification.py` (slow) | Test Cases 1–2 against their analytic solutions, using the dissertation's relative L2 error (Sections 4.4.3–4.4.4). TCAT CGP reproduces the dissertation's values (e.g. p_w 4.55×10⁻⁹ in Test Case 1). The RE errors are about 22 times smaller than in the dissertation, exactly the ratio of the old to the new βw (1×10⁻⁸ / 4.5×10⁻¹⁰): the analytic solutions assume incompressible water |
| `test_benchmarks.py` | Literature benchmarks. Fast: the McWhorter & Sunada (1990) exact two-phase solution is checked against its own integral equation and an independent finite-difference solve. Slow: 2P-TM vs that exact solution for Szymkiewicz (2007) Example 1 (error and grid convergence), and RE for Tocci et al. (1997) Problem A (front position and Celia-style mass balance) |

To document a known bug without fixing it yet, mark its test `@pytest.mark.xfail(strict=True, reason=...)`. Once the bug is fixed, pytest reports the test as an unexpected pass, which is the reminder to remove the marker.

Add a test whenever you change the library. In particular, compare any new analytic derivative against a finite difference, as the existing tests do.

## 9. Known issues

- **Sign of the Gaussian-curvature term in the TCAT curvature equation.** The code has −2K·∂εw/∂t (`residual_tcat_cgp.py`, `residual_full_tcat.py`), as did the original code behind the published results. The Overleaf design notes and dissertation Eq. 3.49 give +2K when written with the non-wetting-side curvature Jn that the code uses. So does the source derivation (Gray, Shepherd & Miller, *Toward the Closure of a New Generation of Two-Fluid Flow Models: Curvature Evolution*, manuscript v0.83, 2021). The code follows its Eqn (89), which is written in J_w. Substituting J_w = −J_n gives +2K·∂εw/∂t, as does its Eqn (87). With +2K, the Test Case 5 defense drainage run fails at t ≈ 108 s, so this is unresolved rather than a simple fix.
- **Incompressible two-phase runs with no-flow boundaries for both phases fail on the first time step** (2P-TM, both formulations). Giving the air phase its physical compressibility makes the `pp` formulation run; `sp` still fails. Found by the Szymkiewicz benchmark.
- **The slow Tocci Problem A benchmarks have not yet been run to completion**, so their tolerances are unverified.
- **Relative permeability is not hysteretic.** Test Case 8 uses the main drainage curve's k_r on scanning curves as well.
- **Face mobility is averaged, not upwinded**, which matches the original code but may cause trouble at sharp fronts.
- **The TCAT momentum resistance R̂ʷ_w is a constant (3.75×10⁵)**, the value of the original codes. Dissertation Tables 4.11–4.25 cite Eqn 4.12, a state equation in εw, the interfacial areas and the curvatures. Its wetting-phase coefficients e₁–e₄ are not published: Bruning & Miller (2019, Table 2) fit only the non-wetting-phase coefficients, in lattice-Boltzmann units. Only Test Case 1 (Table 4.7) gives a number, 5.10×10⁴, which that driver uses.
- **Test Case 4 TCAT CGP stalls at t = 0.52 s** (the time step collapses to ~10⁻¹⁶ s), before and after the velocity fix. Figs 4.2–4.4 were made with the older Cython code in `OtherApps/Testcase4_infiltration_problem`, which used different, pre-Finney-pack parameters. Its εw boundary values (0.359 / 0.368) also differ from Table 4.16, which gives εw = ε. The slow driver test marks this case as an expected failure.
- **TCAT CGP diverges in Test Case 3 for pressure perturbations above about 16 Pa** (dissertation Section 4.4.5); the driver uses 10 Pa. RE handles larger perturbations. The dissertation attributes this to the approximate equilibrium interfacial area εwn_eq. A reformulation that removes εwn_eq was started in Sept–Oct 2025 (prototypes in `OtherApps/20251011_NewForm`) but has no results yet. It uses +2K, see the first item.
- **Grid convergence of TCAT CGP is not established for Test Cases 4–6.** The errors between successive grids in the dissertation drafts did not behave as expected: for Test Case 4 (old Cython code, 51/101/201 nodes), the εwn, Jwn, Kwn and εw errors grew with refinement, and for Test Cases 5–6 (101/201/401 nodes) every error jumped at t = 300 s. Neither was explained, and the final dissertation reports no error tables for these cases. Also, the dissertation's Test Case 5–6 RE and TCAT CGP runs do not start and end at the same εw (for drainage, RE goes 0.369 → 0.0254 and TCAT CGP 0.300 → 0.2746); only the normalized figures use matched runs.
- **No 2P-TM drivers for the Chapter 5 validation cases** (Kees & Miller drainage, Tables 5.2–5.3; Szymkiewicz imbibition, Tables 5.4–5.5). `2p_drainage_pdc.py` and `2p_imbibition_mic.py` run Finney-pack cases instead. The Szymkiewicz setup is closest to the benchmark in `tests/benchmarks/countercurrent_2p.py`.

## 10. Troubleshooting

| Symptom | Cause and fix |
|---|---|
| `ModuleNotFoundError: No module named 'petsc4py'` | The conda environment isn't active. Run `conda activate tcat2p`. |
| `ModuleNotFoundError: No module named 'TCAT2Phase'` | Run `pip install -e .` from the repo root with the environment active. |
| `FileNotFoundError` from a plotting script | Run the matching driver first, and check that the filename in the script matches the `.h5` file in the `outputs/` folder. |
| A run seems frozen | Watch the `TS dt ... time ...` lines it prints. If time is still increasing, it is working, just slowly (see Test Case 8). |
