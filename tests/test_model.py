"""Tests for model selection and bookkeeping (numerics/model.py)."""
import numpy as np
import pytest

import TCAT2Phase

from conftest import build_mesh

EXPECTED_UNKNOWNS = [
    ("Richards", "pr", ["p_w"]),
    ("extended_Richards", "sp", ["e_w", "p_w"]),
    ("two_phase", "sp", ["e_w", "p_w"]),
    ("two_phase", "pp", ["p_w", "p_n"]),
    ("extended_two_phase", "spp", ["e_w", "p_w", "p_n"]),
    ("cgp-TCAT", "spijvk", ["e_w", "p_w", "e_wn", "j_wn", "w_wn", "k_wn"]),
    ("full-TCAT", "sppijgvk", ["e_w", "p_w", "p_n", "e_wn", "j_wn", "g_wn", "w_wn", "k_wn"]),
]


def make_model(model_type, form, nodes=5):
    domain, mesh = build_mesh(nodes)
    return TCAT2Phase.numerics.model.Model(
        model_type=model_type, form=form, domain=domain, nodes=nodes, mesh=mesh)


@pytest.mark.parametrize("model_type,form,u_names", EXPECTED_UNKNOWNS)
def test_model_selects_unknowns(model_type, form, u_names):
    model = make_model(model_type, form)
    assert model.model.u_names == u_names
    assert model.model.num_u == len(u_names)


@pytest.mark.parametrize("model_type,form,_", EXPECTED_UNKNOWNS)
def test_every_model_has_a_residual_class(model_type, form, _):
    model = make_model(model_type, form)
    residual_class = TCAT2Phase.numerics.model_types.get_residual_class(model)
    assert issubclass(residual_class, TCAT2Phase.numerics.base_residual.BaseResidual)


def test_boundary_conditions_are_indexed_by_unknown():
    model = make_model("two_phase", "pp")
    model.set_boundary_conditions({"p_w": [[-100.0, -200.0]], "p_n": [[0.0, 0.0]]})
    assert model.boundary_conditions == {0: [[-100.0, -200.0]], 1: [[0.0, 0.0]]}


def test_initial_conditions_and_boundary_types_are_keyed_by_name():
    model = make_model("Richards", "pr")
    ic = lambda x: -1.0 - x
    model.set_initial_conditions({"p_w": ic})
    model.set_boundary_type({"p_w": [[0, 0]]})
    assert model.initial_conditions["p_w"] is ic
    assert model.boundary_type == {"p_w": [[0, 0]]}


def test_clean_solution_deinterleaves_unknowns():
    model = make_model("two_phase", "pp", nodes=3)
    petsc_array = np.array([1.0, 10.0, 2.0, 20.0, 3.0, 30.0])  # [pw0, pn0, pw1, pn1, ...]
    model.clean_solution(petsc_array, time=5.0)
    np.testing.assert_array_equal(model.solution[5.0]["p_w"], [1.0, 2.0, 3.0])
    np.testing.assert_array_equal(model.solution[5.0]["p_n"], [10.0, 20.0, 30.0])


def test_clean_solution_copies_data():
    model = make_model("Richards", "pr", nodes=3)
    petsc_array = np.array([1.0, 2.0, 3.0])
    model.clean_solution(petsc_array, time=0.0)
    petsc_array[:] = 0.0
    np.testing.assert_array_equal(model.solution[0.0]["p_w"], [1.0, 2.0, 3.0])


def test_set_model_phases():
    model = make_model("Richards", "pr")
    model.set_model_phases("s", "w", "n", "iface")
    assert (model.solid, model.w_phase, model.n_phase, model.interface) == ("s", "w", "n", "iface")
