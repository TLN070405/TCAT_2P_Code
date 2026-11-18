"""
Verification against analytic solutions: Test Cases 1 and 2 (dissertation Sections 4.4.3-4.4.4).

Runs the RE and TCAT CGP drivers (101 nodes) in a temporary folder and computes the
relative L2 error used in the dissertation and in outputs/*/postprocessing/plot_all_results.py:

    L2 = sqrt( (1/N) * sum( ((exact - numerical) / exact)**2 ) )

Dissertation values (computed with beta_w = 1e-8):
    TC1  TCAT CGP  e_w 1.31e-14  p_w 4.55e-9    RE  e_w 0        p_w 3.87e-6
    TC2  TCAT CGP  e_w 1.04e-13  p_w 1.20e-13   RE  e_w 9.76e-6  p_w 1.08e-6

The TCAT CGP values reproduce. The RE errors are now about 22 times smaller,
exactly the ratio 1e-8 / 4.5e-10: the analytic solutions assume incompressible
water, so the RE error comes from compressibility, and the drivers now use Table 4.2's
beta_w = 4.5e-10. The thresholds below are about 3 times the current errors.
"""
import os
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from TCAT2Phase.io.hdf5_io import load_solutions_from_hdf5

REPO = Path(__file__).resolve().parent.parent

# Van Genuchten primary drainage curve (Table 4.3), used by the Test Case 2 drivers
VG_N, VG_ALPHA, POROSITY = 37.8, 2.28e-3, 0.369
E_WR = 0.0687 * POROSITY


def tc1_exact(x):
    return np.full_like(x, POROSITY), 2500.0 - 500.0 * x


def tc2_exact(x):
    p_w = -1.0 - 1000.0 * 9.81 * x
    p_c = -p_w
    m = 1.0 - 1.0 / VG_N
    s_e = np.where(p_c > 0.0, (1.0 + (VG_ALPHA * np.maximum(p_c, 0.0)) ** VG_N) ** (-m), 1.0)
    return s_e * (POROSITY - E_WR) + E_WR, p_w


CASES = [
    # case folder, driver, output file, exact solution, max L2 for e_w, max L2 for p_w
    ("Testcase1_fully_saturated_system", "re_fully_saturated.py", "RE_test_case_1.h5", tc1_exact, 1e-14, 5e-7),
    ("Testcase1_fully_saturated_system", "tcat_cgp_fully_saturated.py", "tcat_cgp_test_case_1.h5", tc1_exact, 1e-12, 1e-8),
    ("Testcase2_hydrostatic_equilibrium", "re_hydrostatic_equi_vertical.py", "RE_test_case_2.h5", tc2_exact, 1.5e-6, 1.5e-7),
    ("Testcase2_hydrostatic_equilibrium", "tcat_cgp_hydrostatic_equi_vertical.py", "tcat_cgp_test_case_2.h5", tc2_exact, 1e-12, 1e-12),
]


def relative_l2(exact, numerical):
    return float(np.sqrt(np.mean(((exact - numerical) / exact) ** 2)))


@pytest.mark.slow
@pytest.mark.parametrize("case,driver,output,exact,tol_e_w,tol_p_w", CASES, ids=[c[1] for c in CASES])
def test_l2_error_against_analytic_solution(tmp_path, case, driver, output, exact, tol_e_w, tol_p_w):
    workdir = tmp_path / "applications" / case
    shutil.copytree(REPO / "applications" / case, workdir)
    result = subprocess.run(
        [sys.executable, driver], cwd=workdir, env=dict(os.environ, MPLBACKEND="Agg"),
        capture_output=True, text=True, timeout=600,
    )
    assert result.returncode == 0, result.stderr[-2000:]

    solutions, points, _ = load_solutions_from_hdf5(tmp_path / "outputs" / case / "outputs", output)
    x = np.array([p[0] for p in points])
    final = solutions[max(solutions)]
    e_w_exact, p_w_exact = exact(x)

    assert relative_l2(e_w_exact, final["e_w"]) <= tol_e_w
    assert relative_l2(p_w_exact, final["p_w"]) <= tol_p_w
