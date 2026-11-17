"""
Tests for Parker-Lenhard hysteresis with Land's entrapment model
(state_equations_hysteresis.ParkerLenhard), as used by Test Case 8.

The setup mirrors the Test Case 8 drivers: every node starts on the primary
drainage curve (PDC), with the reversal point at full saturation and the
endpoint at residual saturation.
"""
import numpy as np
import pytest

from TCAT2Phase.properties.interface import TraditionalInterface
from TCAT2Phase.state_equations import state_equations_hysteresis, state_equations_interface

from conftest import fake_model

POROSITY, E_WR, E_NR = 0.4, 0.02, 0.05
MAX_REVERSALS = 20


def apparent_saturation(e_w):
    return (e_w - E_WR) / (POROSITY - E_WR)


def make_parker_lenhard(e_nr=E_NR):
    model = fake_model(POROSITY, E_WR, e_nr)
    drainage = TraditionalInterface(state_equations_interface.VanGenuchten(
        m=1 - 1 / 6.0, n=6.0, alpha=1 / 1500.0, porosity=POROSITY, e_wr=E_WR, e_nr=0.0))
    imbibition = TraditionalInterface(state_equations_interface.VanGenuchten(
        m=1 - 1 / 9.0, n=9.0, alpha=1 / 800.0, porosity=POROSITY, e_wr=E_WR, e_nr=0.0))

    ew_rev, ew_end = POROSITY, E_WR + 1e-5
    full = lambda value: np.full((1, MAX_REVERSALS), value)
    params = dict(
        drainage_process_new=np.array([True]),
        pc_reversal=full(drainage.pc_function(ew_rev)),
        pc_endpoint=full(drainage.pc_function(ew_end)),
        ew_reversal=full(ew_rev),
        ew_endpoint=full(ew_end),
        sa_reversal=full(apparent_saturation(ew_rev)),
        sa_endpoint=full(apparent_saturation(ew_end)),
        snte_reversal=full(0.0),
        snte_endpoint=full(0.0),
        path_is_pdc=np.array([True]),
        switch_condition="p_c",
        hysteresis_tol=1e-6,
        max_reversals=MAX_REVERSALS,
        max_time_steps=10,
    )
    return state_equations_hysteresis.ParkerLenhard(model, imbibition, drainage, **params), drainage


def reverse_to_imbibition(pl, pc_reversal):
    """Reverse from primary drainage to imbibition at pc_reversal."""
    sa = pl.apparent_saturation_function(pc_reversal)
    pl.data.drainage_process_new[0] = False
    pl.update_scaling_points_reversal_case(sa, 0.0, pc_reversal)
    return sa


@pytest.fixture
def pl():
    return make_parker_lenhard()[0]


# --- Construction --------------------------------------------------------------

def test_missing_parameter_raises():
    with pytest.raises(ValueError, match="Missing required parameter"):
        state_equations_hysteresis.ParkerLenhard(fake_model(POROSITY, E_WR, E_NR), None, None)


def test_storage_arrays_sized_from_model(pl):
    assert pl.data.ew_hysteresis.shape == (1, 10)
    assert pl.data.reversals_number.shape == (1,)


# --- Primary drainage curve ----------------------------------------------------

@pytest.mark.parametrize("p_c", [300.0, 1500.0, 4000.0])
def test_follows_drainage_curve_initially(p_c):
    pl, drainage = make_parker_lenhard()
    assert pl.volume_fraction_function(p_c) == pytest.approx(drainage.volume_fraction_function(p_c), rel=1e-12)


@pytest.mark.parametrize("p_c", [300.0, 1500.0, 4000.0])
def test_d_volume_fraction_matches_finite_difference_on_pdc(pl, fd, p_c):
    assert pl.d_volume_fraction_function(p_c) == pytest.approx(
        fd(pl.volume_fraction_function, p_c, 1e-2), rel=1e-6)


@pytest.mark.parametrize("p_c", [500.0, 1500.0])
def test_pc_function_inverts_volume_fraction_on_pdc(pl, p_c):
    assert pl.pc_function(pl.volume_fraction_function(p_c)) == pytest.approx(p_c, rel=1e-10)


def test_d_pc_function_matches_finite_difference_on_pdc(pl, fd):
    e_w = pl.volume_fraction_function(1500.0)
    assert pl.d_pc_function(e_w) == pytest.approx(fd(pl.pc_function, e_w, 1e-7), rel=1e-5)


# --- Land's entrapment model ---------------------------------------------------

def test_lands_formula_after_reversal_from_pdc(pl):
    sa_r = reverse_to_imbibition(pl, 1500.0)
    r_nw = (POROSITY - E_WR) / E_NR - 1
    expected = (1 - sa_r) / (1 + r_nw * (1 - sa_r))
    assert pl.data.snte_endpoint[0][0] == pytest.approx(expected, rel=1e-12)
    assert pl.data.snte_reversal[0][0] == 0.0


def test_no_entrapment_when_residual_nonwetting_is_zero():
    pl, _ = make_parker_lenhard(e_nr=0.0)
    assert pl.lands_residual_effective_saturation_nw() == 0.0


# --- Imbibition scanning curve after a reversal --------------------------------

def test_reversal_updates_scaling_points(pl):
    sa_r = reverse_to_imbibition(pl, 1500.0)
    assert pl.data.reversals_number[0] == 1
    assert pl.data.pc_reversal[0][0] == 1500.0
    assert pl.data.pc_endpoint[0][0] == pytest.approx(0.0)
    assert pl.data.sa_reversal[0][0] == pytest.approx(sa_r)
    assert pl.data.sa_endpoint[0][0] == pytest.approx(1.0)


def test_scanning_curve_is_continuous_at_reversal(pl):
    e_w_at_reversal = pl.volume_fraction_function(1500.0)
    reverse_to_imbibition(pl, 1500.0)
    assert pl.volume_fraction_function(1500.0) == pytest.approx(e_w_at_reversal, rel=1e-12)


def test_scanning_curve_ends_at_land_entrapment(pl):
    reverse_to_imbibition(pl, 1500.0)
    snr = pl.data.snte_endpoint[0][0]
    expected = (1 - snr) * (POROSITY - E_WR) + E_WR
    assert pl.volume_fraction_function(1e-9) == pytest.approx(expected, rel=1e-10)
    assert expected < POROSITY  # some non-wetting phase stays trapped


def test_imbibition_increases_volume_fraction(pl):
    reverse_to_imbibition(pl, 1500.0)
    e_w = [pl.volume_fraction_function(pc) for pc in np.linspace(1500.0, 1.0, 50)]
    assert np.all(np.diff(e_w) > 0)


@pytest.mark.parametrize("p_c", [1200.0, 800.0])
def test_d_volume_fraction_matches_finite_difference_on_scanning_curve(pl, fd, p_c):
    reverse_to_imbibition(pl, 1500.0)
    assert pl.d_volume_fraction_function(p_c) == pytest.approx(
        fd(pl.volume_fraction_function, p_c, 1e-2), rel=1e-6)


@pytest.mark.parametrize("p_c", [1400.0, 800.0, 200.0])
def test_pc_function_inverts_volume_fraction_on_scanning_curve(pl, p_c):
    reverse_to_imbibition(pl, 1500.0)
    e_w = pl.volume_fraction_function(p_c)
    assert pl.pc_function(e_w) == pytest.approx(p_c, rel=1e-8)


@pytest.mark.parametrize("p_c", [1200.0, 800.0])
def test_d_pc_function_matches_finite_difference_on_scanning_curve(pl, fd, p_c):
    reverse_to_imbibition(pl, 1500.0)
    e_w = pl.volume_fraction_function(p_c)
    assert pl.d_pc_function(e_w) == pytest.approx(fd(pl.pc_function, e_w, 1e-7), rel=1e-5)


def test_entrapment_in_ew_matches_entrapment_in_sa(pl):
    reverse_to_imbibition(pl, 1500.0)
    sa = pl.apparent_saturation_function(800.0)
    e_w = pl.volume_fraction_function(800.0)
    assert pl.linear_effective_entrapped_saturation_nw_ew(e_w) == pytest.approx(
        pl.linear_effective_entrapped_saturation_nw(sa), rel=1e-12)


# --- Process switching ---------------------------------------------------------

def set_state(pl, p_c):
    pl.p_c[0] = p_c
    pl.s_a[0] = pl.apparent_saturation_function(p_c)
    pl.s_nt_e[0] = pl.linear_effective_entrapped_saturation_nw(pl.s_a[0])


def test_small_switch_condition_keeps_drainage(pl):
    set_state(pl, 1500.0)
    pl.determine_hysteresis_state(switch_condition=1e-9)
    assert pl.data.drainage_process_new[0]
    assert pl.data.reversals_number[0] == 0


def test_positive_switch_condition_reverses_to_imbibition(pl):
    set_state(pl, 1500.0)
    pl.determine_hysteresis_state(switch_condition=1.0)  # p_c dropped: imbibition
    assert not pl.data.drainage_process_new[0]
    assert pl.data.reversals_number[0] == 1
    assert pl.data.pc_reversal[0][0] == 1500.0


def test_store_hysteresis_results_records_process(pl):
    pl.store_hysteresis_results(ew=0.3, pc=1000.0)
    k = pl.data.counter_hysteresis
    assert pl.data.ew_hysteresis[0, k] == 0.3
    assert pl.data.pc_hysteresis[0, k] == 1000.0
    assert pl.data.process_hysteresis[0, k] == -1  # drainage
