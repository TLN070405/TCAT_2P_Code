"""
Smoke tests: run the real driver scripts in applications/ and check each writes
a readable result file. These catch drift between the drivers and the library
API. They take a few minutes, so they are marked slow and skipped by default:

    pytest -m slow

Each test case folder is copied into a temporary directory first, so the
drivers write their .h5 files there and never overwrite results in outputs/.
Test Cases 5-7 and the Test Case 8 2P run are left out because they take many minutes; of the
Chapter 5 validation drivers only the Kees & Miller drainage run is included.
"""
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

from TCAT2Phase.io.hdf5_io import load_solutions_from_hdf5

REPO = Path(__file__).resolve().parent.parent

DRIVERS = [
    ("Testcase1_fully_saturated_system", "re_fully_saturated.py", "RE_test_case_1.h5"),
    ("Testcase1_fully_saturated_system", "2p_fully_saturated.py", "2P_test_case_1.h5"),
    ("Testcase1_fully_saturated_system", "tcat_cgp_fully_saturated.py", "tcat_cgp_test_case_1.h5"),
    ("Testcase2_hydrostatic_equilibrium", "re_hydrostatic_equi_vertical.py", "RE_test_case_2.h5"),
    ("Testcase2_hydrostatic_equilibrium", "2p_hydrostatic_equi_vertical.py", "2P_test_case_2.h5"),
    ("Testcase2_hydrostatic_equilibrium", "tcat_cgp_hydrostatic_equi_vertical.py", "tcat_cgp_test_case_2.h5"),
    ("Testcase3_hydrostatic_equilibrium_perturabation", "re_equilibrium_perturbation.py", "RE_test_case_3.h5"),
    ("Testcase3_hydrostatic_equilibrium_perturabation", "2p_equilibrium_perturbation.py", "2P_test_case_3.h5"),
    ("Testcase3_hydrostatic_equilibrium_perturabation",
     "tcat_cgp_hydrostatic_equi_vertical_perturbation.py", "tcat_cgp_test_case_3.h5"),
    ("Testcase4_infiltration_problem", "re_infiltration_horizontal.py", "RE_test_case_4.h5"),
    ("Testcase4_infiltration_problem", "2p_infiltration_horizontal.py", "2P_test_case_4.h5"),
    pytest.param("Testcase4_infiltration_problem", "tcat_cgp_infiltration_horizontal.py", "tcat_cgp_test_case_4.h5",
                 marks=pytest.mark.xfail(strict=True, reason="TC4 TCAT CGP stalls at t ~ 0.18 s (README, Known issues)")),
    # Chapter 5 validation, about 3 min of CPU (2p_szymkiewicz_imbibition.py takes about 9 min and is
    # covered by the Szymkiewicz benchmark in test_benchmarks.py)
    ("Chapter5_validation", "2p_kees_miller_drainage.py", "2P_kees_miller_drainage.h5"),
]


@pytest.mark.slow
@pytest.mark.parametrize("case,driver,output", DRIVERS, ids=[(d.values if hasattr(d, "values") else d)[1] for d in DRIVERS])
def test_driver_runs_and_writes_results(tmp_path, case, driver, output):
    workdir = tmp_path / "applications" / case
    shutil.copytree(REPO / "applications" / case, workdir)

    env = dict(os.environ, MPLBACKEND="Agg")
    result = subprocess.run(
        [sys.executable, driver], cwd=workdir, env=env,
        capture_output=True, text=True, timeout=600,
    )
    assert result.returncode == 0, result.stderr[-2000:]

    out_dir = tmp_path / "outputs" / case / "outputs"
    solutions, points, _ = load_solutions_from_hdf5(out_dir, output)
    assert len(points) > 0
    assert len(solutions) > 0
    for variables in solutions.values():
        for values in variables.values():
            assert len(values) == len(points)
            assert all(v == v for v in values)  # no NaNs


@pytest.mark.slow
def test_tc8_re_hysteresis_driver(tmp_path):
    """Test Case 8 RE (Parker-Lenhard hysteresis, about 1.5 min): the inlet node follows the
    boundary schedule through its 5 reversals and p_w stays within the boundary values."""
    import numpy as np
    from TCAT2Phase.state_equations.state_equations_hysteresis import find_reversal_indices

    case = "Testcase8_complex_displacement_patterns"
    workdir = tmp_path / "applications" / case
    shutil.copytree(REPO / "applications" / case, workdir)
    result = subprocess.run(
        [sys.executable, "re_malgorzata_hysteresis_entrapment_pdc.py"], cwd=workdir,
        env=dict(os.environ, MPLBACKEND="Agg"), capture_output=True, text=True, timeout=600,
    )
    assert result.returncode == 0, result.stderr[-2000:]

    solutions, points, metadata = load_solutions_from_hdf5(
        tmp_path / "outputs" / case / "outputs", "RE_test_case_8.h5")
    final = solutions[max(solutions)]
    assert max(solutions) == pytest.approx(3000.0)
    p_w = np.asarray(final["p_w"])
    assert len(p_w) == len(points) and np.all(np.isfinite(p_w))
    assert p_w.min() >= -1800.0 - 1e-6 and p_w.max() <= 1e-6
    counter = metadata["counter"]
    pc_inlet = np.asarray(final["p_c_hysteresis"])[0, 1:counter]
    assert len(find_reversal_indices(pc_inlet, tol=1.0)) == 5
