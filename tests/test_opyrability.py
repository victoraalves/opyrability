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


# Dynamic operability tests

def time_varying_linear_model(u, t):
    """Simple time-varying model for testing."""
    y1 = u[0] + 0.5 * u[1] + 0.2 * t
    y2 = 0.3 * u[0] + u[1] - 0.1 * t
    return np.array([y1, y2])


def test_dynamic_operability_basic():
    """Test basic dynamic operability computation."""
    AIS_bounds = np.array([[0, 10],
                           [0, 10]])
    AIS_resolution = [3, 3]
    time_bounds = np.array([0, 5])
    time_resolution = 3

    results = dynamic_operability(
        time_varying_linear_model,
        AIS_bounds,
        AIS_resolution,
        time_bounds,
        time_resolution,
        polytopic_trace='simplices',
        plot=plot_flag
    )

    # Check that results contain expected keys
    assert 'AOS_regions' in results
    assert 'AOS_vertices' in results
    assert 'time_points' in results
    assert 'polytopes_by_time' in results

    # Check dimensions
    assert len(results['AOS_regions']) == time_resolution
    assert len(results['time_points']) == time_resolution
    assert results['time_points'][0] == 0
    assert results['time_points'][-1] == 5


def test_dynamic_operability_polyhedra():
    """Test dynamic operability with polyhedra trace."""
    AIS_bounds = np.array([[0, 5],
                           [0, 5]])
    AIS_resolution = [3, 3]
    time_bounds = np.array([0, 2])
    time_resolution = 2

    results = dynamic_operability(
        time_varying_linear_model,
        AIS_bounds,
        AIS_resolution,
        time_bounds,
        time_resolution,
        polytopic_trace='polyhedra',
        plot=plot_flag
    )

    assert len(results['AOS_regions']) == time_resolution


def test_dynamic_OI_eval_constant_DOS():
    """Test dynamic OI evaluation with constant DOS."""
    AIS_bounds = np.array([[0, 10],
                           [0, 10]])
    AIS_resolution = [4, 4]
    time_bounds = np.array([0, 4])
    time_resolution = 3

    results = dynamic_operability(
        time_varying_linear_model,
        AIS_bounds,
        AIS_resolution,
        time_bounds,
        time_resolution,
        plot=plot_flag
    )

    DOS_bounds = np.array([[5, 15],
                           [3, 12]])

    OI_results = dynamic_OI_eval(
        results,
        DOS_bounds,
        plot=plot_flag
    )

    # Check results
    assert 'OI_values' in OI_results
    assert 'time_points' in OI_results
    assert len(OI_results['OI_values']) == time_resolution

    # OI should be between 0 and 100
    assert all(0 <= oi <= 100 for oi in OI_results['OI_values'])


def test_dynamic_OI_eval_time_varying_DOS():
    """Test dynamic OI with time-varying DOS."""
    AIS_bounds = np.array([[0, 10],
                           [0, 10]])
    AIS_resolution = [3, 3]
    time_bounds = np.array([0, 2])
    time_resolution = 2

    results = dynamic_operability(
        time_varying_linear_model,
        AIS_bounds,
        AIS_resolution,
        time_bounds,
        time_resolution,
        plot=plot_flag
    )

    def time_varying_DOS(t):
        return np.array([[5 + 0.1*t, 15 + 0.1*t],
                         [3, 12]])

    OI_results = dynamic_OI_eval(
        results,
        None,
        time_varying_DOS=time_varying_DOS,
        plot=plot_flag
    )

    assert len(OI_results['OI_values']) == time_resolution
    assert all(0 <= oi <= 100 for oi in OI_results['OI_values'])


def test_dynamic_operability_envelope():
    """Test envelope computation."""
    AIS_bounds = np.array([[0, 5],
                           [0, 5]])
    AIS_resolution = [3, 3]
    time_bounds = np.array([0, 3])
    time_resolution = 3

    results = dynamic_operability(
        time_varying_linear_model,
        AIS_bounds,
        AIS_resolution,
        time_bounds,
        time_resolution,
        plot=plot_flag
    )

    envelope = compute_dynamic_operability_envelope(results)

    assert 'envelope_volume' in envelope
    assert 'individual_volumes' in envelope
    assert len(envelope['individual_volumes']) == time_resolution

    # Envelope volume should be >= max individual volume
    assert envelope['envelope_volume'] >= max(envelope['individual_volumes'])


if __name__ == '__main__':
    pytest.main()
    