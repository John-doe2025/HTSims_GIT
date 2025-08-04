"""
================================================================================
Thermal Analysis of a Nacelle
================================================================================
Purpose:
This script performs a transient thermal analysis of a nacelle containing
batteries, an ESC (Electronic Speed Controller), and shell components.
It solves a system of ordinary differential equations (ODEs) to model how the
temperature of each component evolves over time due to heat generation,
conduction, convection, and radiation.

Key Features:
- Models multiple interconnected thermal nodes.
- Accounts for heat generation in batteries and ESC.
- Calculates heat transfer via conduction, natural/forced convection, and radiation.
- Uses temperature-dependent properties for air, interpolated from an Excel file.
- Employs a highly efficient, professional ODE solver from SciPy suitable for
  "stiff" systems, which are common in thermal simulations.

Conditions:
- Ambient Temperature: 293 K
- Altitude: Ground Level (1 atm)
- HAP (High-Altitude Platform) Velocity: 0 m/s (static case)
================================================================================
"""

# -------------------------
# 1. Import Necessary Libraries
# -------------------------
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

# Start a timer to measure the total execution time of the script.
start_time = time.time()

print('Conditions: Ambient Temperature = 293K, Ground Level (1 atm), HAP velocity = 0 m/s')

# -------------------------
# 2. Load and Prepare Air Properties
# -------------------------
# Load temperature-dependent air properties from an Excel spreadsheet.
# This is done only once at the beginning to avoid slow file I/O during the simulation.
path = "./298.xlsx"
df_air = pd.read_excel(path, sheet_name=1)

# Extract data into lists for use with NumPy's fast interpolation function.
temperatures = df_air['Temperature (K)'].tolist()
rho_values   = df_air['Density (rho) kg/m3'].tolist()
cp_values    = df_air['Specific Heat (cp) J/kg.K'].tolist()
k_values     = df_air['Thermal Conductivity (k) W/m.K'].tolist()
mu_values    = df_air['Dynamic Viscosity (m)  kg/m.s'].tolist()
nu_values    = df_air['Kinematic Viscosity (n)  m2/s'].tolist()
Pr_values    = df_air['Prandtl Number (Pr)'].tolist()

def prop(T):
    """
    Interpolates air properties at a given temperature T.
    Uses np.interp for fast linear interpolation based on the data loaded from Excel.

    Args:
        T (float): Temperature in Kelvin.

    Returns:
        tuple: A tuple containing (rho, cp, k, mu, nu, Pr) at temperature T.
    """
    r  = np.interp(T, temperatures, rho_values)
    cp = np.interp(T, temperatures, cp_values)
    k  = np.interp(T, temperatures, k_values)
    mu = np.interp(T, temperatures, mu_values)
    nu = np.interp(T, temperatures, nu_values)
    Pr = np.interp(T, temperatures, Pr_values)
    return r, cp, k, mu, nu, Pr

# -------------------------
# 3. Define Helper Functions for Physics Calculations
# -------------------------
def T_power4(T):
    """
    Safely computes T^4 for radiation calculations.
    Clipping prevents numerical errors if a temperature temporarily becomes negative
    during the solver's iterations, which would raise an error.
    """
    return np.clip(T, 1e-6, 1e6)**4

def rad_coeff(e1, e2, a1, a2, vf):
    """
    Calculates the radiation heat transfer coefficient between two surfaces.

    Args:
        e1, e2 (float): Emissivity of surface 1 and 2.
        a1, a2 (float): Area of surface 1 and 2 (m^2).
        vf (float): View factor from surface 1 to 2.

    Returns:
        float: The radiation coefficient (W/K^4).
    """
    sigma = 5.67e-8  # Stefan-Boltzmann constant (W/m^2·K^4)
    C_rad = sigma / ((1 - e1) / (e1 * a1) + 1 / (a1 * vf) + (1 - e2) / (e2 * a2))
    return C_rad

# -------------------------
# 4. Define System Constants and Geometry
# -------------------------
# All constants are defined here for clarity and easy modification.

# Masses (kg)
m_B_total = 32.4
m_Bf = m_Bm = m_Br = m_B_total / 3  # Battery mass distributed across 3 nodes
m_ESC = 0.12
m_TS  = 0.9
m_BS  = 0.9

# Specific Heat Capacities (J/kg·K)
C_B   = 1100
C_ESC = 100
C_TS  = 1040
C_BS  = 1040

# Internal Heat Generation (W)
Q_B_total = 12.6
Q_B_front = Q_B_middle = Q_B_rear = Q_B_total / 3
Q_ESC = 100
Q_S = 0  # Solar heat load
Q_A = 0  # Albedo heat load
Q_P = 0  # Planetary heat load

# Geometry and Conduction Coefficients
A_cross = 0.252 * 0.154          # Cross-sectional area for battery conduction
k_eff = 2.5                      # Effective thermal conductivity between battery nodes
C_cond = k_eff * A_cross / 0.280 # Conduction coeff between battery nodes (W/K)

A_Bf_conv = 0.26617  # Convective areas for each component (m^2)
A_Bm_conv = 0.22736
A_Br_conv = 0.26617
A_ESC_conv = 0.01348
A_TS = 0.54168
A_BS = 0.54168

# Radiation Coefficients (W/K^4) - Pre-calculated for efficiency
C_Bf_TS_int_rad = rad_coeff(0.9, 0.2, 0.0427, 0.54168, 1)
C_TS_BS_rad = rad_coeff(0.2, 0.2, 0.54168, 0.54168, 0.5)
C_ESC_TS_rad = rad_coeff(0.8, 0.2, 0.0013959, 0.54168, 1)
C_Bf_ESC_rad = rad_coeff(0.2, 0.8, 0.038808, 0.00206415, 1)

# Characteristic Lengths (m) for convection calculations
LC_B_horiz = 0.277; LC_B_vert = 0.252; LC_ESC = 0.0695; LC_TS = LC_BS = 0.84

# Mounting Conduction (e.g., ESC to battery mount)
k_mount = 3; A_contact = 0.005; L_mount = 0.005
C_ESC_Bf_cond = k_mount * A_contact / L_mount # Conduction coeff (W/K)

# Skin Conduction (internal to external shell surface)
k_cfrp = 0.015; A_cfrp = 0.54168; t_cfrp = 0.005
C_cond_cfrp = k_cfrp * A_cfrp / t_cfrp # Conduction coeff (W/K)

# Environmental Conditions
g = 9.81         # Gravitational acceleration (m/s^2)
velocity = 0.0     # External air velocity (m/s)
T_E = 193.15     # Ambient external temperature (K)

# -------------------------
# 5. Define Convection Models (NEW - Based on Incropera & DeWitt)
# -------------------------

def natural_convection_h(p_film, T_surface, T_fluid, L_char, is_vertical):
    """
    Calculates h for NATURAL convection using cited correlations from Incropera 6th Ed.
    """
    k, Pr, nu_val = p_film[2], p_film[5], p_film[4]
    if abs(T_surface - T_fluid) < 1e-4: return 0.0

    # Film temperature for beta calculation
    T_film = (T_surface + T_fluid) / 2
    beta = 1.0 / T_film

    # Calculate Grashof and Rayleigh numbers
    Gr = (g * beta * abs(T_surface - T_fluid) * L_char**3) / (nu_val**2)
    Ra = Gr * Pr
    Nu = 1.0  # Default fallback

    if is_vertical:
        # Source: Incropera, 6th Ed., Eq. 9.26 (Churchill and Chu for vertical plate)
        Nu = (0.825 + (0.387 * Ra**(1/6)) / (1 + (0.492 / Pr)**(9/16))**(8/27))**2
    else:  # Horizontal plate
        if T_surface > T_fluid:  # Hot Surface Facing Up
            # Source: Incropera, 6th Ed., Eqs. 9.30 & 9.31
            if 1e4 <= Ra <= 1e7: Nu = 0.54 * Ra**(1/4)
            elif Ra > 1e7: Nu = 0.15 * Ra**(1/3)
        else:  # Cold Surface Facing Up (or Hot Surface Facing Down)
            # Source: Incropera, 6th Ed., Eq. 9.32
            if 1e5 <= Ra <= 1e10: Nu = 0.27 * Ra**(1/4)

    return Nu * k / L_char

def forced_convection_h(p_film, L_char, velocity):
    """
    Calculates h for FORCED convection using cited flat plate correlations.
    """
    k, Pr, rho, mu = p_film[2], p_film[5], p_film[0], p_film[3]

    # Calculate Reynolds number
    Re_L = rho * velocity * L_char / max(mu, 1e-12)
    if Re_L < 100: return 0.0  # No meaningful forced flow

    # Source: Incropera & DeWitt, 6th Ed., Chapter 7
    Re_crit = 5e5
    if Re_L <= Re_crit:
        # Laminar Flow Correlation: Eq. 7.30
        Nu = 0.664 * (Re_L**0.5) * (Pr**(1/3))
    else:
        # Mixed Flow (Laminar + Turbulent) Correlation: Eq. 7.41
        # The constant 871 is valid for a critical Reynolds number of 5e5.
        Nu = (0.037 * (Re_L**0.8) - 871) * (Pr**(1/3))

    return Nu * k / L_char

def get_external_surface_h(p_film, T_surface, T_fluid, L_char, velocity):
    """
    Determines the appropriate convection coefficient for an external surface.
    Switches between forced convection for moving cases and natural convection
    for static cases to ensure physical accuracy.
    """
    # If velocity is significant, forced convection dominates.
    if velocity > 0.1: # m/s
        return forced_convection_h(p_film, L_char, velocity)
    # If velocity is zero or negligible, heat transfer is by natural convection.
    else:
        # Assuming the external nacelle surface is a horizontal plate for natural convection.
        return natural_convection_h(p_film, T_surface, T_fluid, L_char, is_vertical=False)


# -------------------------
# 6. Define the System of ODEs
# -------------------------
# This function `f(t, x)` defines the derivatives dT/dt for each component.
# It is the core of the thermal model and is called repeatedly by the ODE solver.

def f(t, x):
    """
    Calculates the rate of change of temperature for each node.

    Args:
        t (float): Current time (s).
        x (np.array): Array of current temperatures [T_Bf, T_Bm, ..., T_air] in K.

    Returns:
        np.array: Array of temperature derivatives [dT_Bf/dt, ..., dT_air/dt].
    """
    # Unpack the temperature array for readability
    T_Bf, T_Bm, T_Br, T_ESC, T_TS_int, T_TS_ext, T_BS_int, T_BS_ext, T_air = x

    # --- Optimization: Calculate properties ONCE per function call ---
    p_air = prop(T_air)
    p_batt_film = prop((T_Bf + T_air) / 2) # Representative film temp for all batteries
    p_esc_film = prop((T_ESC + T_air) / 2)
    p_ts_int_film = prop((T_TS_int + T_air) / 2)
    p_bs_int_film = prop((T_BS_int + T_air) / 2)
    p_ts_ext_film = prop((T_TS_ext + T_E) / 2)
    p_bs_ext_film = prop((T_BS_ext + T_E) / 2)

    # --- Calculate Heat Flows (Q) for Each Component ---

    # BATTERY NODES (Internal Natural Convection)
    # The battery pack is modeled with top, bottom, and vertical surfaces.
    h_batt_top = natural_convection_h(p_batt_film, T_surface=T_Bf, T_fluid=T_air, L_char=LC_B_horiz, is_vertical=False)
    h_batt_bottom = natural_convection_h(p_batt_film, T_surface=T_Bf, T_fluid=T_air, L_char=LC_B_horiz, is_vertical=False)
    h_batt_vert = natural_convection_h(p_batt_film, T_surface=T_Bf, T_fluid=T_air, L_char=LC_B_vert, is_vertical=True)
    h_batt_avg = (h_batt_top + h_batt_bottom + 2 * h_batt_vert) / 4.0 # Weighted average h

    Q_conv_Bf = h_batt_avg * A_Bf_conv * (T_air - T_Bf)
    Q_conv_Bm = h_batt_avg * A_Bm_conv * (T_air - T_Bm)
    Q_conv_Br = h_batt_avg * A_Br_conv * (T_air - T_Br)

    Q_cond_Bf = C_cond * (T_Bm - T_Bf)
    Q_cond_Bm = C_cond * ((T_Bf - T_Bm) + (T_Br - T_Bm))
    Q_cond_Br = C_cond * (T_Bm - T_Br)

    T4_Bf, T4_Bm, T4_Br, T4_ESC, T4_TS_int, T4_BS_int = T_power4(np.array([T_Bf, T_Bm, T_Br, T_ESC, T_TS_int, T_BS_int]))
    Q_rad_Bf = (C_Bf_TS_int_rad * (T4_TS_int - T4_Bf) + C_Bf_ESC_rad * (T4_ESC - T4_Bf) + C_Bf_TS_int_rad * (T4_BS_int - T4_Bf))
    Q_rad_Bm = (C_Bf_TS_int_rad * (T4_TS_int - T4_Bm) + C_Bf_TS_int_rad * (T4_BS_int - T4_Bm))
    Q_rad_Br = (C_Bf_TS_int_rad * (T4_TS_int - T4_Br) + C_Bf_TS_int_rad * (T4_BS_int - T4_Br))

    # ESC (Internal Natural Convection)
    # The ESC is modeled as a horizontal plate.
    h_ESC = natural_convection_h(p_esc_film, T_surface=T_ESC, T_fluid=T_air, L_char=LC_ESC, is_vertical=False)
    Q_conv_ESC = h_ESC * A_ESC_conv * (T_air - T_ESC)
    Q_rad_ESC = (C_Bf_ESC_rad * (T4_Bf - T4_ESC) + C_ESC_TS_rad * (T4_TS_int - T4_ESC) + C_ESC_TS_rad * (T4_BS_int - T4_ESC))
    Q_cond_ESC_Bf = C_ESC_Bf_cond * (T_ESC - T_Bf) # Conduction from ESC to Battery Front

    # NACELLE SHELLS (Top and Bottom)
    Q_cond_TS_ext = C_cond_cfrp * (T_TS_int - T_TS_ext) # Conduction through top shell
    Q_cond_BS_ext = C_cond_cfrp * (T_BS_int - T_BS_ext) # Conduction through bottom shell

    # Internal convection to inner shell surfaces (natural)
    h_TS_in = natural_convection_h(p_ts_int_film, T_surface=T_TS_int, T_fluid=T_air, L_char=LC_TS, is_vertical=False)
    Q_conv_TS_in = h_TS_in * A_TS * (T_air - T_TS_int)
    h_BS_in = natural_convection_h(p_bs_int_film, T_surface=T_BS_int, T_fluid=T_air, L_char=LC_BS, is_vertical=False)
    Q_conv_BS_in = h_BS_in * A_BS * (T_air - T_BS_int)

    # External convection from outer shell surfaces (forced or natural)
    h_TS_ext = get_external_surface_h(p_ts_ext_film, T_surface=T_TS_ext, T_fluid=T_E, L_char=LC_TS, velocity=velocity)
    Q_conv_TS_ext = h_TS_ext * A_TS * (T_E - T_TS_ext)
    h_BS_ext = get_external_surface_h(p_bs_ext_film, T_surface=T_BS_ext, T_fluid=T_E, L_char=LC_BS, velocity=velocity)
    Q_conv_BS_ext = h_BS_ext * A_BS * (T_E - T_BS_ext)

    Q_rad_TS = (C_Bf_TS_int_rad * ((T4_Bf - T4_TS_int) + (T4_Bm - T4_TS_int) + (T4_Br - T4_TS_int)) + C_TS_BS_rad * (T4_BS_int - T4_TS_int))
    Q_rad_BS = (C_Bf_TS_int_rad * ((T4_Bf - T4_BS_int) + (T4_Bm - T4_BS_int) + (T4_Br - T4_BS_int)) + C_TS_BS_rad * (T4_TS_int - T4_BS_int))

    # INTERNAL AIR
    # Total heat transferred TO the air is the sum of heat FROM all components.
    Q_conv_air = -(Q_conv_Bf + Q_conv_Bm + Q_conv_Br + Q_conv_ESC + Q_conv_TS_in + Q_conv_BS_in)
    V_a = 0.11; m_a = p_air[0] * V_a # Mass of internal air

    # --- Calculate Temperature Derivatives (dT/dt = ΣQ / (m*C)) ---
    dT_Bf_dt     = (Q_B_front + Q_cond_Bf + Q_conv_Bf + Q_rad_Bf + Q_cond_ESC_Bf) / (m_Bf * C_B)
    dT_Bm_dt     = (Q_B_middle + Q_cond_Bm + Q_conv_Bm + Q_rad_Bm) / (m_Bm * C_B)
    dT_Br_dt     = (Q_B_rear + Q_cond_Br + Q_conv_Br + Q_rad_Br) / (m_Br * C_B)
    dT_ESC_dt    = (Q_ESC + Q_conv_ESC + Q_rad_ESC - Q_cond_ESC_Bf) / (m_ESC * C_ESC)
    dT_TS_int_dt = (-Q_cond_TS_ext + Q_conv_TS_in + Q_rad_TS) / (m_TS * C_TS)
    dT_TS_ext_dt = (Q_cond_TS_ext + Q_conv_TS_ext + Q_S) / (m_TS * C_TS)
    dT_BS_int_dt = (-Q_cond_BS_ext + Q_conv_BS_in + Q_rad_BS) / (m_BS * C_BS)
    dT_BS_ext_dt = (Q_cond_BS_ext + Q_conv_BS_ext + Q_A + Q_P) / (m_BS * C_BS)
    dT_air_dt    = Q_conv_air / (m_a * p_air[1])

    return np.array([dT_Bf_dt, dT_Bm_dt, dT_Br_dt, dT_ESC_dt, dT_TS_int_dt, dT_TS_ext_dt, dT_BS_int_dt, dT_BS_ext_dt, dT_air_dt])

# -------------------------
# 7. Set Up and Run the Simulation
# -------------------------

# Initial conditions: all components start at 298.15 K (25 °C).
x0 = np.array([298.15] * 9)
print("Initial Temperatures (K):", x0)

# Simulation time span
t0 = 0
T_total = 10000  # Total simulation time in seconds

# --- Progress Monitor for the Solver ---
# This class provides feedback during long simulations.
class ProgressMonitor:
    def __init__(self, t_total):
        self.t_total = t_total
        self.last_t = -1
    def __call__(self, t, y):
        # Print progress every 100 simulated seconds.
        if t - self.last_t >= 100:
            print(f"  ... Solving at time t = {int(t)} s ({100*t/self.t_total:.0f}%)")
            self.last_t = t
        return 0 # Return 0 to tell the solver to continue

progress_monitor = ProgressMonitor(T_total)
print(f"\nStarting solver for t = {t0} to {T_total} s...")

# --- Run the ODE Solver ---
# We use 'BDF' (Backward Differentiation Formula), which is excellent for "stiff"
# problems like this one, where some components (air) change temperature much
# faster than others (batteries). This is the key to a fast and stable solution.
sol = solve_ivp(
    fun=f,                      # The function defining the ODEs
    t_span=[t0, T_total],       # The time interval to solve over
    y0=x0,                      # The initial temperatures
    method='BDF',               # The stiff-aware solver method
    dense_output=True,          # Generates a smooth solution for plotting
    events=progress_monitor     # The progress monitor callback
)
print("... Solver finished.")

# -------------------------
# 8. Process and Display Results
# -------------------------

# Extract the final temperatures from the solution object.
final_temps = sol.y[:, -1] # .y is (n_vars, n_times), so we take the last column
labels = ['Battery Front', 'Battery Middle', 'Battery Rear', 'ESC', 'Top Shell Internal',
          'Top shell external', 'Bottom shell internal', 'Bottom Shell external', 'Internal Air']

print("\nFinal Temperatures (K):")
for lab, temp in zip(labels, final_temps):
    print(f"{lab:<25}: {temp:.2f} K")

end_time = time.time()
print(f"\nTotal execution time: {end_time - start_time:.2f} seconds")

# -------------------------
# 9. Plot the Results
# -------------------------
plt.figure(figsize=(14, 9))

# Use the 'dense_output' to create a smooth plot with many points.
t_plot = np.linspace(t0, T_total, 1000)
x_plot = sol.sol(t_plot).T # .sol(t) evaluates the solution at times t

# Plot the temperature evolution for each component.
for i, lab in enumerate(labels):
    plt.plot(t_plot, x_plot[:, i], label=lab)

# Formatting the plot for clarity.
plt.xlabel('Time (s)', fontsize=12)
plt.ylabel('Temperature (K)', fontsize=12)
plt.title('Transient Temperature Evolution of Nacelle Components', fontsize=16)
plt.legend(loc='best', fontsize=10)
plt.grid(True, which='both', linestyle='--', linewidth=0.5)
plt.tight_layout()
plt.show()