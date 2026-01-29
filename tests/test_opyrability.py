import pytest
import numpy as np
from opyrability import (multimodel_rep, OI_eval, AIS2AOS_map, nlp_based_approach,
                         dynamic_operability, dynamic_OI_eval,
                         compute_dynamic_operability_envelope)
from shower import shower2x2, inv_shower2x2
from dma_mr import dma_mr_design

# -----------------------------------------------------------------------------
# Tests for opyrability
# Author: Victor Alves
# Control, Optimization and Design for Energy and Sustainability,
# CODES Group, West Virginia University (2023)
# -----------------------------------------------------------------------------

# Tolerances and to see or not to see the operability plots.
plot_flag = False
abs_tol = 1e-7
rel_tol = 1e-7


# Multimodel approach tests.
def test_shower2x2():
    DOS_bounds = np.array([[10, 20],
                           [70, 100]])

    AIS_bounds = np.array([[1, 10],
                           [1, 10]])

    AIS_resolution = [5, 5]

    AIS, AOS = AIS2AOS_map(shower2x2,
                           AIS_bounds,
                           AIS_resolution,
                           plot=plot_flag)

    AOS_region = multimodel_rep(shower2x2,
                                AIS_bounds,
                                AIS_resolution,
                                plot=plot_flag)

    OI = OI_eval(AOS_region,
                 DOS_bounds,
                 plot=plot_flag)

    assert OI == pytest.approx(60.23795007653283, abs=abs_tol, rel=rel_tol)

    

def test_shower_analytical_inverse():
    
    AOS_bounds = np.array([[10, 20],
                           [70, 100]])

    AOS_resolution = [5, 5]

    model = inv_shower2x2

    DIS_bounds = np.array([[0, 10.00],
                           [0, 10.00]])

    AIS, AOS = AIS2AOS_map(model,
                           AOS_bounds,
                           AOS_resolution,
                           plot=plot_flag)

    AIS_region = multimodel_rep(model,
                                AOS_bounds,
                                AOS_resolution,
                                polytopic_trace='polyhedra',
                                plot=plot_flag)

    OI = OI_eval(AIS_region,
                 DIS_bounds,
                 plot=plot_flag)

    assert OI == pytest.approx(40, abs=abs_tol, rel=rel_tol)
    
    
def test_dma_mr_design():
    
    DOS_bounds = np.array([[20, 25],
                           [35, 45]])

    AIS_bounds = np.array([[10, 150],
                           [0.5, 2]])

    AIS_resolution = [5, 5]

    AOS_region = multimodel_rep(dma_mr_design,
                                AIS_bounds,
                                AIS_resolution,
                                plot=plot_flag)

    OI = OI_eval(AOS_region,
                 DOS_bounds,
                 plot=plot_flag)

    assert OI == pytest.approx(23.373846526953766, abs=abs_tol, rel=rel_tol)


# NLP-based approach tests

def test_shower_inverse_nlp_2x2():
    
    u0 = np.array([0, 10])
    lb = np.array([0, 0])
    ub = np.array([100, 100])

    DOS_bound = np.array([[17.5, 21.0],
                          [80.0, 100.0]])

    DOSresolution = [5, 5]

    fDIS, fDOS, message = nlp_based_approach(shower2x2,
                                             DOS_bound,
                                             DOSresolution,
                                             u0,
                                             lb,
                                             ub,
                                             method='ipopt',
                                             plot=plot_flag,
                                             ad=False,
                                             warmstart=True)

    norm_fDIS = np.linalg.norm(fDIS)
    norm_fDOS = np.linalg.norm(fDOS)

    asserted_fDIS = 70.0683253828694
    asserted_fDOS = 461.57593383905106

    assert norm_fDIS == pytest.approx(asserted_fDIS, abs=abs_tol, rel=rel_tol)
    assert norm_fDOS == pytest.approx(asserted_fDOS, abs=abs_tol, rel=rel_tol)


# Dynamic operability tests (ODE-based paradigm)

def ode_linear_system(t, y, u):
    """
    Simple ODE system for testing dynamic operability.

    dy1/dt = -y1 + u1
    dy2/dt = -0.5*y2 + u2
    """
    dy1_dt = -y[0] + u[0]
    dy2_dt = -0.5 * y[1] + u[1]
    return np.array([dy1_dt, dy2_dt])


def test_dynamic_operability_basic():
    """Test basic dynamic operability computation with ODE model."""
    y0 = np.array([0.0, 0.0])
    AIS_bounds = np.array([[1, 5],
                           [2, 8]])
    AIS_resolution = [3, 3]
    time_span = np.array([0, 5])
    time_eval = np.linspace(0, 5, 3)

    results = dynamic_operability(
        model=ode_linear_system,
        y0=y0,
        AIS_bounds=AIS_bounds,
        AIS_resolution=AIS_resolution,
        time_span=time_span,
        time_eval=time_eval,
        polytopic_trace='simplices',
        plot=plot_flag
    )

    # Check that results contain expected keys
    assert 'AOS_regions' in results
    assert 'AOS_vertices' in results
    assert 'time_points' in results
    assert 'polytopes_by_time' in results
    assert 'trajectories' in results

    # Check dimensions
    assert len(results['AOS_regions']) == len(time_eval)
    assert len(results['time_points']) == len(time_eval)
    assert results['time_points'][0] == 0
    assert results['time_points'][-1] == 5


def test_dynamic_operability_polyhedra():
    """Test dynamic operability with polyhedra trace."""
    y0 = np.array([0.0, 0.0])
    AIS_bounds = np.array([[1, 3],
                           [1, 3]])
    AIS_resolution = [3, 3]
    time_span = np.array([0, 2])
    time_eval = np.linspace(0, 2, 2)

    results = dynamic_operability(
        model=ode_linear_system,
        y0=y0,
        AIS_bounds=AIS_bounds,
        AIS_resolution=AIS_resolution,
        time_span=time_span,
        time_eval=time_eval,
        polytopic_trace='polyhedra',
        plot=plot_flag
    )

    assert len(results['AOS_regions']) == len(time_eval)


def test_dynamic_OI_eval_constant_DOS():
    """Test dynamic OI evaluation with constant DOS."""
    y0 = np.array([0.0, 0.0])
    AIS_bounds = np.array([[1, 5],
                           [2, 8]])
    AIS_resolution = [4, 4]
    time_span = np.array([0, 4])
    time_eval = np.linspace(0, 4, 3)

    results = dynamic_operability(
        model=ode_linear_system,
        y0=y0,
        AIS_bounds=AIS_bounds,
        AIS_resolution=AIS_resolution,
        time_span=time_span,
        time_eval=time_eval,
        plot=plot_flag
    )

    DOS_bounds = np.array([[1, 6],
                           [2, 10]])

    OI_results = dynamic_OI_eval(
        results,
        DOS_bounds,
        plot=plot_flag
    )

    # Check results
    assert 'OI_values' in OI_results
    assert 'time_points' in OI_results
    assert len(OI_results['OI_values']) == len(time_eval)

    # OI should be between 0 and 100
    assert all(0 <= oi <= 100 for oi in OI_results['OI_values'])


def test_dynamic_OI_eval_time_varying_DOS():
    """Test dynamic OI with time-varying DOS."""
    y0 = np.array([0.0, 0.0])
    AIS_bounds = np.array([[1, 5],
                           [2, 8]])
    AIS_resolution = [3, 3]
    time_span = np.array([0, 2])
    time_eval = np.linspace(0, 2, 2)

    results = dynamic_operability(
        model=ode_linear_system,
        y0=y0,
        AIS_bounds=AIS_bounds,
        AIS_resolution=AIS_resolution,
        time_span=time_span,
        time_eval=time_eval,
        plot=plot_flag
    )

    def time_varying_DOS(t):
        return np.array([[1 + 0.1*t, 6 + 0.1*t],
                         [2, 10]])

    OI_results = dynamic_OI_eval(
        results,
        None,
        time_varying_DOS=time_varying_DOS,
        plot=plot_flag
    )

    assert len(OI_results['OI_values']) == len(time_eval)
    assert all(0 <= oi <= 100 for oi in OI_results['OI_values'])


def test_dynamic_operability_envelope():
    """Test envelope computation."""
    y0 = np.array([0.0, 0.0])
    AIS_bounds = np.array([[1, 3],
                           [1, 3]])
    AIS_resolution = [3, 3]
    time_span = np.array([0, 3])
    time_eval = np.linspace(0, 3, 3)

    results = dynamic_operability(
        model=ode_linear_system,
        y0=y0,
        AIS_bounds=AIS_bounds,
        AIS_resolution=AIS_resolution,
        time_span=time_span,
        time_eval=time_eval,
        plot=plot_flag
    )

    envelope = compute_dynamic_operability_envelope(results)

    assert 'envelope_volume' in envelope
    assert 'individual_volumes' in envelope
    assert len(envelope['individual_volumes']) == len(time_eval)

    # Envelope volume should be >= max individual volume
    assert envelope['envelope_volume'] >= max(envelope['individual_volumes'])


def test_dynamic_operability_callable_y0():
    """Test dynamic operability with input-dependent initial conditions."""
    def y0_from_input(u):
        # Steady state initial conditions
        return np.array([u[0], 2*u[1]])

    AIS_bounds = np.array([[1, 3],
                           [1, 3]])
    AIS_resolution = [3, 3]
    time_span = np.array([0, 2])
    time_eval = np.linspace(0, 2, 2)

    results = dynamic_operability(
        model=ode_linear_system,
        y0=y0_from_input,  # Callable
        AIS_bounds=AIS_bounds,
        AIS_resolution=AIS_resolution,
        time_span=time_span,
        time_eval=time_eval,
        plot=plot_flag
    )

    assert len(results['trajectories']) == 9  # 3x3 = 9 input points


if __name__ == '__main__':
    pytest.main()
    