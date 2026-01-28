import numpy as np
from opyrability import (dynamic_operability,
                         dynamic_OI_eval,
                         plot_dynamic_operability,
                         plot_dynamic_operability_with_DOS,
                         compute_dynamic_operability_envelope)

# -----------------------------------------------------------------------------
# Examples: Dynamic Operability Mapping
# Author: Victor Alves
# Control, Optimization and Design for Energy and Sustainability,
# CODES Group, West Virginia University (2024)
# -----------------------------------------------------------------------------

"""
This script presents illustrative examples that showcase the application of
Dynamic Operability Analysis, which evaluates how process operability sets
evolve over time.

The methodology is based on:
S. Dinh and F. V. Lima. "Dynamic Operability Analysis for Process Design and
Control of Modular Natural Gas Utilization Systems." Ind. & Eng. Chem. Res.
2023. https://doi.org/10.1021/acs.iecr.2c03543

Key concepts demonstrated:
1. Time-varying Achievable Output Sets (AOS)
2. Connected polytopes showing temporal evolution
3. Dynamic Operability Index (OI) evaluation
4. Visualization with time as an axis

"""

# %% Example 1: Simple Time-Varying Linear System (2D outputs)
# This example demonstrates a basic time-varying process where the output
# ranges shift over time.

print("=" * 70)
print("Example 1: Simple Time-Varying Linear System")
print("=" * 70)

def time_varying_linear_model(u, t):
    """
    A simple time-varying model where outputs shift over time.

    Parameters
    ----------
    u : array-like
        Input vector [u1, u2]
    t : float
        Time parameter

    Returns
    -------
    y : ndarray
        Output vector [y1, y2]
    """
    y1 = u[0] + 0.5 * u[1] + 0.2 * t
    y2 = 0.3 * u[0] + u[1] - 0.1 * t
    return np.array([y1, y2])


# Define bounds and resolution
AIS_bounds = np.array([[0, 10],
                       [0, 10]])

AIS_resolution = [5, 5]

time_bounds = np.array([0, 10])
time_resolution = 5

# Compute dynamic operability
print("\nComputing dynamic operability for time-varying linear system...")
results = dynamic_operability(
    time_varying_linear_model,
    AIS_bounds,
    AIS_resolution,
    time_bounds,
    time_resolution,
    polytopic_trace='simplices',
    plot=True,
    labels=['$y_1$', '$y_2$'],
    time_label='Time [s]'
)

print(f"\nNumber of time points: {len(results['time_points'])}")
print(f"Time range: [{results['time_points'][0]}, {results['time_points'][-1]}]")


# %% Example 2: Dynamic OI Evaluation with Constant DOS
# Evaluate how the Operability Index changes over time against a fixed DOS

print("\n" + "=" * 70)
print("Example 2: Dynamic OI with Constant DOS")
print("=" * 70)

DOS_bounds = np.array([[5, 15],
                       [3, 12]])

print("\nEvaluating dynamic Operability Index...")
OI_results = dynamic_OI_eval(
    results,
    DOS_bounds,
    plot=True,
    time_label='Time [s]'
)

print(f"\nOI values at each time point:")
for t, oi in zip(OI_results['time_points'], OI_results['OI_values']):
    print(f"  t = {t:.2f}: OI = {oi:.2f}%")


# %% Example 3: Dynamic OI with Time-Varying DOS
# The desired output set also changes with time

print("\n" + "=" * 70)
print("Example 3: Dynamic OI with Time-Varying DOS")
print("=" * 70)

def time_varying_DOS(t):
    """
    DOS that shifts over time following the process dynamics.
    """
    return np.array([[5 + 0.15*t, 15 + 0.15*t],
                     [3 - 0.05*t, 12 - 0.05*t]])

print("\nEvaluating dynamic OI with time-varying DOS...")
OI_results_varying = dynamic_OI_eval(
    results,
    None,  # Not used when time_varying_DOS is provided
    time_varying_DOS=time_varying_DOS,
    plot=True,
    time_label='Time [s]'
)


# %% Example 4: Visualization with DOS overlay
# Show both AOS evolution and DOS region in the same plot

print("\n" + "=" * 70)
print("Example 4: Visualization with DOS Overlay")
print("=" * 70)

print("\nPlotting AOS evolution with constant DOS overlay...")
plot_dynamic_operability_with_DOS(
    results,
    DOS_bounds,
    alpha=0.5,
    colormap='plasma'
)


# %% Example 5: Time-Varying Shower Problem
# Adaptation of the classic shower problem with time-varying parameters

print("\n" + "=" * 70)
print("Example 5: Time-Varying Shower Problem")
print("=" * 70)

def time_varying_shower(u, t):
    """
    Time-varying shower problem where hot water temperature
    changes over time (e.g., heating up).

    Parameters
    ----------
    u : array-like
        Input vector [cold_flow, hot_flow]
    t : float
        Time parameter

    Returns
    -------
    y : ndarray
        Output vector [total_flow, temperature]
    """
    cold_temp = 60   # Cold water temperature [F]
    hot_temp = 120 + 5 * np.sin(0.5 * t)  # Time-varying hot temp

    total_flow = u[0] + u[1]

    if total_flow > 0:
        temperature = (u[0] * cold_temp + u[1] * hot_temp) / total_flow
    else:
        temperature = (cold_temp + hot_temp) / 2

    return np.array([total_flow, temperature])


# Define bounds and resolution
AIS_bounds_shower = np.array([[0, 10],
                              [0, 10]])

AIS_resolution_shower = [6, 6]

time_bounds_shower = np.array([0, 4 * np.pi])  # Full cycle
time_resolution_shower = 8

# Compute dynamic operability for shower problem
print("\nComputing dynamic operability for time-varying shower problem...")
results_shower = dynamic_operability(
    time_varying_shower,
    AIS_bounds_shower,
    AIS_resolution_shower,
    time_bounds_shower,
    time_resolution_shower,
    polytopic_trace='simplices',
    plot=True,
    labels=['Total Flow [gal/min]', 'Temperature [F]'],
    time_label='Time [s]'
)

# Evaluate OI
DOS_bounds_shower = np.array([[10, 20],
                              [70, 100]])

print("\nEvaluating OI for shower problem over time...")
OI_results_shower = dynamic_OI_eval(
    results_shower,
    DOS_bounds_shower,
    plot=True,
    time_label='Time [s]'
)


# %% Example 6: Compute Operability Envelope
# Find the total reachable output space over the entire time horizon

print("\n" + "=" * 70)
print("Example 6: Operability Envelope Computation")
print("=" * 70)

print("\nComputing the total operability envelope...")
envelope = compute_dynamic_operability_envelope(results)

print(f"\nIndividual AOS volumes at each time point:")
for t, vol in zip(envelope['time_points'], envelope['individual_volumes']):
    print(f"  t = {t:.2f}: Volume = {vol:.4f}")

print(f"\nTotal envelope volume: {envelope['envelope_volume']:.4f}")


# %% Example 7: Process with Decaying Dynamics
# A process where the achievable range shrinks over time (e.g., catalyst decay)

print("\n" + "=" * 70)
print("Example 7: Process with Decaying Dynamics")
print("=" * 70)

def decaying_process(u, t):
    """
    Process with exponentially decaying gain (e.g., catalyst deactivation).

    Parameters
    ----------
    u : array-like
        Input vector [u1, u2]
    t : float
        Time parameter

    Returns
    -------
    y : ndarray
        Output vector [y1, y2]
    """
    decay_factor = np.exp(-0.1 * t)

    y1 = decay_factor * (2 * u[0] + u[1]) + 5
    y2 = decay_factor * (u[0] + 1.5 * u[1]) + 3

    return np.array([y1, y2])


# Define bounds and resolution
AIS_bounds_decay = np.array([[0, 5],
                             [0, 5]])

AIS_resolution_decay = [5, 5]

time_bounds_decay = np.array([0, 20])
time_resolution_decay = 6

# Compute dynamic operability
print("\nComputing dynamic operability for decaying process...")
results_decay = dynamic_operability(
    decaying_process,
    AIS_bounds_decay,
    AIS_resolution_decay,
    time_bounds_decay,
    time_resolution_decay,
    polytopic_trace='simplices',
    plot=True,
    labels=['$y_1$', '$y_2$'],
    time_label='Time [h]'
)

# Define a DOS that was achievable at t=0
DOS_bounds_decay = np.array([[10, 18],
                             [8, 15]])

print("\nEvaluating OI decay over time...")
OI_results_decay = dynamic_OI_eval(
    results_decay,
    DOS_bounds_decay,
    plot=True,
    time_label='Time [h]'
)

print("\nObserve how the OI decreases over time as the process gain decays.")


# %% Summary
print("\n" + "=" * 70)
print("SUMMARY: Dynamic Operability Analysis Examples")
print("=" * 70)
print("""
Key functions demonstrated:

1. dynamic_operability()
   - Computes AOS at multiple time points
   - Returns polytope regions for each time instance
   - Generates 3D visualization with time axis

2. dynamic_OI_eval()
   - Evaluates Operability Index over time
   - Supports both constant and time-varying DOS
   - Plots OI evolution

3. plot_dynamic_operability()
   - Creates 3D visualization of connected polytopes
   - Color-coded by time

4. plot_dynamic_operability_with_DOS()
   - Shows AOS evolution relative to DOS

5. compute_dynamic_operability_envelope()
   - Computes the union of all AOS over time
   - Useful for understanding total reachable space

These tools enable comprehensive analysis of how process operability
evolves dynamically, supporting design decisions for time-varying
or batch processes.
""")
