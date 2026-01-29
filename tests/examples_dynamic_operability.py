import numpy as np
from opyrability import (dynamic_operability,
                         dynamic_OI_eval,
                         plot_dynamic_operability,
                         plot_dynamic_operability_with_DOS,
                         compute_dynamic_operability_envelope,
                         solve_experiment_experiment)

# -----------------------------------------------------------------------------
# Examples: Dynamic Operability Mapping with ODE Systems
# Author: Victor Alves
# Control, Optimization and Design for Energy and Sustainability,
# CODES Group, West Virginia University (2024)
# -----------------------------------------------------------------------------

"""
This script presents illustrative examples that showcase the application of
Dynamic Operability Analysis using ODE-based dynamic systems.

The methodology is based on:
S. Dinh and F. V. Lima. "Dynamic Operability Analysis for Process Design and
Control of Modular Natural Gas Utilization Systems." Ind. & Eng. Chem. Res.
2023. https://doi.org/10.1021/acs.iecr.2c03543

Key paradigm:
- Model is specified as an ODE system: dydt = f(t, y, u)
- The system is solved using scipy.integrate.solve_ivp
- For each input u in the AIS, we solve the ODE and track outputs over time
- Connected polytopes show how the AOS evolves as the dynamic system evolves

This differs from steady-state operability where the model is y = f(u).
Here we capture the transient behavior of the system.

"""

# %% Example 1: Simple First-Order System
# A simple first-order dynamic system with two states

print("=" * 70)
print("Example 1: First-Order Dynamic System")
print("=" * 70)


def first_order_system(t, y, u):
    """
    Simple first-order dynamic system.

    States: y = [y1, y2]
    Inputs: u = [u1, u2]

    dy1/dt = -y1 + u1
    dy2/dt = -0.5*y2 + u2

    This represents two decoupled first-order processes with different
    time constants.
    """
    dy1_dt = -y1 + u[0]
    dy2_dt = -0.5 * y[1] + u[1]
    return np.array([dy1_dt, dy2_dt])


# Initial conditions (states start at zero)
y0 = np.array([0.0, 0.0])

# Input bounds and resolution
AIS_bounds = np.array([[1, 5],    # u1 bounds
                       [2, 8]])   # u2 bounds
AIS_resolution = [4, 4]

# Time span and evaluation points
time_span = np.array([0, 10])
time_eval = np.linspace(0, 10, 6)  # Evaluate at 6 time points

print("\nSolving dynamic system and computing AOS over time...")
results = dynamic_operability(
    model=first_order_system,
    y0=y0,
    AIS_bounds=AIS_bounds,
    AIS_resolution=AIS_resolution,
    time_span=time_span,
    time_eval=time_eval,
    output_func=None,  # States are outputs
    polytopic_trace='simplices',
    solver_method='RK45',
    plot=True,
    labels=['$y_1$', '$y_2$'],
    time_label='Time [s]'
)

print(f"\nNumber of time points: {len(results['time_points'])}")
print(f"Number of trajectories computed: {len(results['trajectories'])}")


# %% Example 2: CSTR (Continuous Stirred Tank Reactor)
# A classic chemical engineering example

print("\n" + "=" * 70)
print("Example 2: CSTR Dynamic Model")
print("=" * 70)


def cstr_model(t, y, u):
    """
    Isothermal CSTR with first-order reaction A -> B.

    States: y = [Ca, Cb] - Concentrations of A and B
    Inputs: u = [F, Ca_in] - Flow rate and inlet concentration

    Parameters (fixed):
    - V = 1.0 m^3 (reactor volume)
    - k = 0.5 1/min (reaction rate constant)

    Mass balances:
    dCa/dt = (F/V)*(Ca_in - Ca) - k*Ca
    dCb/dt = -(F/V)*Cb + k*Ca
    """
    Ca, Cb = y
    F, Ca_in = u

    V = 1.0   # Reactor volume [m^3]
    k = 0.5   # Rate constant [1/min]

    dCa_dt = (F / V) * (Ca_in - Ca) - k * Ca
    dCb_dt = -(F / V) * Cb + k * Ca

    return np.array([dCa_dt, dCb_dt])


# Initial conditions (reactor starts empty of product)
y0_cstr = np.array([0.5, 0.0])  # [Ca0, Cb0]

# Input bounds: flow rate and inlet concentration
AIS_bounds_cstr = np.array([[0.5, 2.0],    # F [m^3/min]
                            [1.0, 3.0]])    # Ca_in [mol/m^3]
AIS_resolution_cstr = [5, 5]

# Time span
time_span_cstr = np.array([0, 20])  # 20 minutes
time_eval_cstr = np.linspace(0, 20, 8)

print("\nSolving CSTR dynamic model...")
results_cstr = dynamic_operability(
    model=cstr_model,
    y0=y0_cstr,
    AIS_bounds=AIS_bounds_cstr,
    AIS_resolution=AIS_resolution_cstr,
    time_span=time_span_cstr,
    time_eval=time_eval_cstr,
    polytopic_trace='simplices',
    solver_method='RK45',
    plot=True,
    labels=['$C_A$ [mol/m³]', '$C_B$ [mol/m³]'],
    time_label='Time [min]'
)

# Evaluate OI against a DOS (desired product concentration)
DOS_bounds_cstr = np.array([[0.1, 0.4],    # Ca desired
                            [0.3, 0.8]])    # Cb desired

print("\nEvaluating dynamic OI for CSTR...")
OI_results_cstr = dynamic_OI_eval(
    results_cstr,
    DOS_bounds_cstr,
    plot=True,
    time_label='Time [min]'
)


# %% Example 3: Heated Tank with Temperature Control
# Temperature dynamics with heat input

print("\n" + "=" * 70)
print("Example 3: Heated Tank System")
print("=" * 70)


def heated_tank(t, y, u):
    """
    Heated tank with liquid flow and heat input.

    States: y = [T, h] - Temperature and liquid level
    Inputs: u = [Q, F_in] - Heat input and inlet flow rate

    Energy balance: rho*Cp*V*dT/dt = Q - F_out*rho*Cp*(T - T_ref)
    Mass balance: A*dh/dt = F_in - F_out

    Simplified (assuming F_out proportional to h):
    dT/dt = Q/(rho*Cp*V) - k_out*h*(T - T_ref)/(rho*Cp*V)
    dh/dt = F_in/A - k_out*h/A
    """
    T, h = y
    Q, F_in = u

    # Parameters
    rho = 1000      # Density [kg/m^3]
    Cp = 4.18       # Heat capacity [kJ/kg-K]
    A = 1.0         # Tank cross-section [m^2]
    k_out = 0.5     # Outlet coefficient
    T_ref = 20      # Reference temperature [°C]
    V = A * max(h, 0.1)  # Volume based on level

    dT_dt = Q / (rho * Cp * V) - k_out * h * (T - T_ref) / (rho * Cp * V)
    dh_dt = F_in / A - k_out * h / A

    return np.array([dT_dt, dh_dt])


# Initial conditions
y0_tank = np.array([25.0, 1.0])  # [T0, h0]

# Input bounds
AIS_bounds_tank = np.array([[1000, 5000],   # Q [kJ/min]
                            [0.2, 1.0]])     # F_in [m^3/min]
AIS_resolution_tank = [4, 4]

# Time span
time_span_tank = np.array([0, 30])
time_eval_tank = np.linspace(0, 30, 10)

print("\nSolving heated tank dynamics...")
results_tank = dynamic_operability(
    model=heated_tank,
    y0=y0_tank,
    AIS_bounds=AIS_bounds_tank,
    AIS_resolution=AIS_resolution_tank,
    time_span=time_span_tank,
    time_eval=time_eval_tank,
    solver_method='LSODA',  # Good for potentially stiff systems
    plot=True,
    labels=['Temperature [°C]', 'Level [m]'],
    time_label='Time [min]'
)


# %% Example 4: Input-Dependent Initial Conditions
# Sometimes initial conditions depend on the input

print("\n" + "=" * 70)
print("Example 4: Input-Dependent Initial Conditions")
print("=" * 70)


def linear_system(t, y, u):
    """Simple linear system."""
    dy1_dt = -0.5 * y[0] + 0.3 * u[0]
    dy2_dt = -0.3 * y[1] + 0.5 * u[1]
    return np.array([dy1_dt, dy2_dt])


def y0_from_input(u):
    """
    Initial conditions that depend on the input.
    This could represent a system that starts at steady state
    for each input condition.
    """
    # Steady state: dy/dt = 0 => y_ss = f(u)
    y1_ss = 0.3 * u[0] / 0.5  # From -0.5*y1 + 0.3*u1 = 0
    y2_ss = 0.5 * u[1] / 0.3  # From -0.3*y2 + 0.5*u2 = 0
    return np.array([y1_ss, y2_ss])


AIS_bounds_lin = np.array([[1, 5], [1, 5]])
AIS_resolution_lin = [4, 4]

time_span_lin = np.array([0, 15])
time_eval_lin = np.linspace(0, 15, 6)

print("\nSolving with input-dependent initial conditions...")
results_lin = dynamic_operability(
    model=linear_system,
    y0=y0_from_input,  # Callable for input-dependent IC
    AIS_bounds=AIS_bounds_lin,
    AIS_resolution=AIS_resolution_lin,
    time_span=time_span_lin,
    time_eval=time_eval_lin,
    plot=True,
    labels=['$y_1$', '$y_2$'],
    time_label='Time [s]'
)


# %% Example 5: Output Function (Select Specific Outputs)
# Use an output function when only some states are outputs of interest

print("\n" + "=" * 70)
print("Example 5: Using Output Function")
print("=" * 70)


def reactor_with_intermediate(t, y, u):
    """
    Reactor with intermediate species: A -> B -> C

    States: y = [Ca, Cb, Cc] - Three species
    Inputs: u = [F, k1_factor] - Flow and reaction rate factor
    """
    Ca, Cb, Cc = y
    F, k1_factor = u

    V = 1.0
    k1 = 0.5 * k1_factor  # Rate A -> B
    k2 = 0.2              # Rate B -> C

    dCa_dt = F * (1.0 - Ca) / V - k1 * Ca
    dCb_dt = -F * Cb / V + k1 * Ca - k2 * Cb
    dCc_dt = -F * Cc / V + k2 * Cb

    return np.array([dCa_dt, dCb_dt, dCc_dt])


def output_function(y, u):
    """
    We only care about product C and conversion of A.
    """
    Ca, Cb, Cc = y
    conversion = 1.0 - Ca  # Conversion of A
    return np.array([conversion, Cc])


y0_reactor = np.array([1.0, 0.0, 0.0])  # Pure A initially

AIS_bounds_reactor = np.array([[0.5, 2.0],    # F
                               [0.5, 2.0]])    # k1_factor
AIS_resolution_reactor = [4, 4]

time_span_reactor = np.array([0, 25])
time_eval_reactor = np.linspace(0, 25, 8)

print("\nSolving reactor with output function...")
results_reactor = dynamic_operability(
    model=reactor_with_intermediate,
    y0=y0_reactor,
    AIS_bounds=AIS_bounds_reactor,
    AIS_resolution=AIS_resolution_reactor,
    time_span=time_span_reactor,
    time_eval=time_eval_reactor,
    output_func=output_function,  # Select outputs
    plot=True,
    labels=['Conversion', 'Product $C_C$'],
    time_label='Time [min]'
)


# %% Example 6: Stiff System (Using BDF Solver)
# For stiff systems, use appropriate solver

print("\n" + "=" * 70)
print("Example 6: Stiff System with BDF Solver")
print("=" * 70)


def stiff_system(t, y, u):
    """
    Stiff system with fast and slow dynamics.

    The fast dynamics (epsilon term) create stiffness.
    """
    epsilon = 0.01  # Small parameter creating stiffness

    dy1_dt = y[1]
    dy2_dt = (-y[0] + u[0]) / epsilon - y[1] / epsilon

    return np.array([dy1_dt, dy2_dt])


y0_stiff = np.array([0.0, 0.0])

AIS_bounds_stiff = np.array([[1, 3]])  # Single input
AIS_resolution_stiff = [6]

time_span_stiff = np.array([0, 2])
time_eval_stiff = np.linspace(0, 2, 8)

print("\nSolving stiff system with BDF solver...")
results_stiff = dynamic_operability(
    model=stiff_system,
    y0=y0_stiff,
    AIS_bounds=AIS_bounds_stiff,
    AIS_resolution=AIS_resolution_stiff,
    time_span=time_span_stiff,
    time_eval=time_eval_stiff,
    solver_method='BDF',  # Better for stiff systems
    solver_options={'rtol': 1e-8, 'atol': 1e-10},
    plot=True,
    labels=['$y_1$', '$y_2$'],
    time_label='Time [s]'
)


# %% Example 7: Operability Envelope Over Time
# Compute the total reachable space over the time horizon

print("\n" + "=" * 70)
print("Example 7: Computing Operability Envelope")
print("=" * 70)

print("\nComputing the operability envelope for the CSTR...")
envelope = compute_dynamic_operability_envelope(results_cstr)

print(f"\nIndividual AOS volumes at each time point:")
for t, vol in zip(envelope['time_points'], envelope['individual_volumes']):
    print(f"  t = {t:.1f} min: Volume = {vol:.6f}")

print(f"\nTotal envelope volume: {envelope['envelope_volume']:.6f}")
print("The envelope represents all outputs achievable at any point in time.")


# %% Example 8: Single Experiment Simulation
# Using the helper function to simulate a single trajectory

print("\n" + "=" * 70)
print("Example 8: Single Experiment with Time-Varying Input")
print("=" * 70)


def input_trajectory(t):
    """Time-varying input: step change at t=5."""
    if t < 5:
        return np.array([1.0, 2.0])
    else:
        return np.array([1.5, 2.5])


print("\nSimulating single experiment with step change...")
experiment_results = solve_experiment_experiment(
    model=cstr_model,
    y0=y0_cstr,
    u_trajectory=input_trajectory,
    time_span=np.array([0, 20]),
    time_eval=np.linspace(0, 20, 100),
    solver_method='RK45'
)

print(f"Simulation completed. Time points: {len(experiment_results['t'])}")
print(f"Final state: Ca = {experiment_results['y'][-1, 0]:.4f}, "
      f"Cb = {experiment_results['y'][-1, 1]:.4f}")


# %% Summary
print("\n" + "=" * 70)
print("SUMMARY: Dynamic Operability Analysis with ODE Systems")
print("=" * 70)
print("""
Key Paradigm:
-------------
The model is specified as an ODE system: dydt = model(t, y, u)
where:
  - t: time
  - y: state vector
  - u: input vector (from AIS)

The system is solved using scipy.integrate.solve_ivp for each
input point in the discretized AIS. This captures the transient
behavior of the system, not just steady-state.

Key Functions:
--------------
1. dynamic_operability(model, y0, AIS_bounds, ...)
   - model: ODE right-hand side dydt = f(t, y, u)
   - y0: Initial conditions (array or callable)
   - Solves ODE for each input, builds AOS polytopes over time

2. dynamic_OI_eval(results, DOS_bounds, ...)
   - Evaluates OI at each time point

3. solve_experiment_experiment(model, y0, u_trajectory, ...)
   - Simulates a single trajectory with time-varying input

4. compute_dynamic_operability_envelope(results)
   - Computes union of all AOS over time

Solver Options:
---------------
- 'RK45': Default, good for non-stiff problems
- 'BDF', 'Radau': Better for stiff problems
- 'LSODA': Auto-switches between stiff/non-stiff

This framework enables analysis of:
- Startup/shutdown dynamics
- Batch process operability
- Transient response to disturbances
- Time-varying achievability of specifications
""")
