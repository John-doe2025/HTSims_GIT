"""
================================================================================
Thermal Analysis of a Nacelle - V7 (Clean, Validated, Modular)
================================================================================
Purpose:
This script performs a transient thermal analysis of a nacelle. This version
is formatted for maximum clarity and uses textbook-validated models for
external convection, with internal natural convection.

Key Features:
- Clean, systematic, and well-commented code structure.
- All code is formatted for readability, one logical calculation per line.
- Implements external convection models strictly compliant with cited
  correlations from Incropera & DeWitt's "Fundamentals of Heat and
  Mass Transfer, 6th Ed." for both forced and natural convection.
- Optional Day/Night Cycle: Solar, albedo, and planetary radiation can be
  enabled or disabled with a single, clear switch.

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

start_time = time.time()

# -------------------------
# 2. Load and Prepare Air Properties
# -------------------------
# NOTE: This Excel file should contain air properties over a wide temperature range
# (e.g., 150K - 450K) to ensure solver stability in extreme conditions.
path = "./298.xlsx"
df_air = pd.read_excel(path, sheet_name=1)

temperatures = df_air['Temperature (K)'].tolist()
rho_values   = df_air['Density (rho) kg/m3'].tolist()
cp_values    = df_air['Specific Heat (cp) J/kg.K'].tolist()
k_values     = df_air['Thermal Conductivity (k) W/m.K'].tolist()
mu_values    = df_air['Dynamic Viscosity (m)  kg/m.s'].tolist()
nu_values    = df_air['Kinematic Viscosity (n)  m2/s'].tolist()
Pr_values    = df_air['Prandtl Number (Pr)'].tolist()

def prop(T):
    """Interpolates air properties at a given temperature T, clipping to data bounds."""
    T_clipped = np.clip(T, temperatures[0], temperatures[-1])
    r  = np.interp(T_clipped, temperatures, rho_values)
    cp = np.interp(T_clipped, temperatures, cp_values)
    k  = np.interp(T_clipped, temperatures, k_values)
    mu = np.interp(T_clipped, temperatures, mu_values)
    nu = np.interp(T_clipped, temperatures, nu_values)
    Pr = np.interp(T_clipped, temperatures, Pr_values)
    return r, cp, k, mu, nu, Pr

# -------------------------
# 3. Define Helper Functions for Physics
# -------------------------
def T_power4(T):
    """Safely computes T^4 for radiation calculations."""
    return np.clip(T, 1e-6, 1e6)**4

def rad_coeff(e1, e2, a1, a2, vf):
    """Calculates the radiation heat transfer coefficient between two surfaces."""
    sigma = 5.67e-8
    return sigma / ((1 - e1) / (e1 * a1) + 1 / (a1 * vf) + (1 - e2) / (e2 * a2))

# -------------------------
# 4. Define System Constants and Geometry
# -------------------------
# --- Simulation Control & Environment ---
T_E = 216.7      # Ambient Temp (K) - e.g., 216.7K for high altitude, 293.15K for ground
velocity = 20.0  # Nacelle Velocity (m/s)
g = 9.81         # Gravitational acceleration (m/s^2)

# --- Optional Day/Night Cycle Control ---
ENABLE_DAY_NIGHT_CYCLE = False # Set to True to turn on solar/albedo/planetary radiation
DAY_DURATION_S = 86400; SOLAR_FLUX_PEAK = 1361; ALBEDO_FLUX_PEAK = 408
PLANETARY_IR_FLUX = 240; SOLAR_ABSORPTIVITY_TS = 0.8; SOLAR_ABSORPTIVITY_BS = 0.8
IR_EMISSIVITY_BS = 0.8

# --- Geometry and Material Properties ---
L_NACELLE_FORCED = 0.840   # Characteristic length for forced convection (m)
H_NACELLE_NATURAL = 0.320  # Characteristic length for natural convection (m)
A_TS = L_NACELLE_FORCED * H_NACELLE_NATURAL # External area of Top Shell (Use actual area)
A_BS = L_NACELLE_FORCED * H_NACELLE_NATURAL # External area of Bottom Shell (Use actual area)
k_cfrp = 4.0; t_cfrp = 0.0006; C_cond_cfrp = k_cfrp * A_TS / t_cfrp

# --- Component Properties ---
m_Bf=21.6; m_Bm=21.6; m_Br=21.6; m_ESC=0.12; m_TS=0.9; m_BS=0.9
C_B = 1100; C_ESC = 100; C_TS = 1040; C_BS = 1040
Q_B_front=77.6; Q_B_middle=77.6; Q_B_rear=77.6; Q_ESC = 100
C_cond=8.92; C_ESC_Bf_cond=3
A_Bf_conv=0.26617; A_Bm_conv=0.22736; A_Br_conv=0.26617; A_ESC_conv=0.01348
C_Bf_TS_int_rad=rad_coeff(0.9,0.75,0.0427,0.54168,1); C_TS_BS_rad=rad_coeff(0.75,0.75,0.54168,0.54168,0.5)
C_ESC_TS_rad=rad_coeff(0.8,0.75,0.0013959,0.54168,1); C_Bf_ESC_rad=rad_coeff(0.9,0.8,0.038808,0.00206415,1)


# ------------------------------------------------------------------
# 5. Textbook-Validated Convection Model Functions
# ------------------------------------------------------------------

def get_forced_convection_h(p_film, L_char, velocity):
    """Calculates h for FORCED convection using cited flat plate correlations."""
    k, Pr, rho, mu = p_film[2], p_film[5], p_film[0], p_film[3]
    Re_L = rho * velocity * L_char / max(mu, 1e-12)
    if Re_L < 100: return 0.0 # No meaningful forced flow

    # Source: Incropera & DeWitt, 6th Ed., Chapter 7
    Re_crit = 5e5
    if Re_L <= Re_crit:
        # Laminar Flow Correlation: Eq. 7.30
        Nu = 0.664 * (Re_L**0.5) * (Pr**(1/3))
    else:
        # Mixed Flow Correlation: Eq. 7.41 (constant 871 is for Re_crit=5e5)
        Nu = (0.037 * (Re_L**0.8) - 871) * (Pr**(1/3))
    return Nu * k / L_char

def get_natural_convection_h(p_film, T_surface, T_fluid, L_char, is_vertical):
    """Calculates h for NATURAL convection using cited correlations from Incropera 6th Ed."""
    k, Pr, nu_val = p_film[2], p_film[5], p_film[4]
    if abs(T_surface - T_fluid) < 1e-4: return 0.0
    beta = 1.0 / ((T_surface + T_fluid) / 2)
    Gr = (g * beta * abs(T_surface - T_fluid) * L_char**3) / (nu_val**2)
    Ra = Gr * Pr
    Nu = 1.0 # Default fallback

    if is_vertical: # Source: Incropera, 6th Ed., Eq. 9.26 (Churchill and Chu)
        Nu = (0.825 + (0.387 * Ra**(1/6)) / (1 + (0.492 / Pr)**(9/16))**(8/27))**2
    else: # Horizontal
        if T_surface > T_fluid: # Hot Surface Facing Up - Source: Eqs. 9.30 & 9.31
            if 1e4 <= Ra <= 1e7: Nu = 0.54 * Ra**(1/4)
            elif Ra > 1e7: Nu = 0.15 * Ra**(1/3)
        else: # Hot Surface Facing Down - Source: Eq. 9.32
            if 1e5 <= Ra <= 1e10: Nu = 0.27 * Ra**(1/4)
    return Nu * k / L_char

def get_external_convection_h(p_film, T_surface, T_fluid, L_forced, L_natural, velocity, is_vertical):
    """Manager function to determine the dominant external convection coefficient."""
    h_forced = get_forced_convection_h(p_film, L_forced, velocity)
    h_natural = get_natural_convection_h(p_film, T_surface, T_fluid, L_natural, is_vertical)
    return max(h_forced, h_natural)

# -------------------------
# 6. Define the System of ODEs
# -------------------------
def f(t, x):
    # --- 6a. Unpack Temperature Array ---
    T_Bf, T_Bm, T_Br, T_ESC, T_TS_int, T_TS_ext, T_BS_int, T_BS_ext, T_air = x

    # --- 6b. Calculate Film Properties for All Surfaces ---
    p_air = prop(T_air)
    p_ts_ext_film = prop((T_TS_ext + T_E) / 2)
    p_bs_ext_film = prop((T_BS_ext + T_E) / 2)
    p_batt_film = prop((T_Bf + T_air) / 2)
    p_esc_film = prop((T_ESC + T_air) / 2)
    p_ts_int_film = prop((T_TS_int + T_air) / 2)
    p_bs_int_film = prop((T_BS_int + T_air) / 2)

    # --- 6c. External Heat Loads ---
    Q_S, Q_A, Q_P = 0.0, 0.0, 0.0 # Default to zero (no external radiation)
    if ENABLE_DAY_NIGHT_CYCLE:
        time_in_cycle = t % DAY_DURATION_S
        solar_factor = np.clip(np.sin(2 * np.pi * time_in_cycle / DAY_DURATION_S), 0, 1)
        Q_S = SOLAR_FLUX_PEAK * A_TS * SOLAR_ABSORPTIVITY_TS * solar_factor
        Q_A = ALBEDO_FLUX_PEAK * A_BS * SOLAR_ABSORPTIVITY_BS * solar_factor
        Q_P = PLANETARY_IR_FLUX * A_BS * IR_EMISSIVITY_BS

    # --- 6d. Calculate Heat Flows (Q) for all Transfer Modes ---
    # -- External Convection --
    h_TS_ext = get_external_convection_h(p_ts_ext_film, T_TS_ext, T_E, L_NACELLE_FORCED, H_NACELLE_NATURAL, velocity, is_vertical=False)
    Q_conv_TS_ext = h_TS_ext * A_TS * (T_E - T_TS_ext)

    h_BS_ext = get_external_convection_h(p_bs_ext_film, T_BS_ext, T_E, L_NACELLE_FORCED, H_NACELLE_NATURAL, velocity, is_vertical=False)
    Q_conv_BS_ext = h_BS_ext * A_BS * (T_E - T_BS_ext)

    # -- Internal Natural Convection --
    h_batt_top = get_natural_convection_h(p_batt_film, T_Bf, T_air, L_char=0.277, is_vertical=False)
    h_batt_vert = get_natural_convection_h(p_batt_film, T_Bf, T_air, L_char=0.252, is_vertical=True)
    h_batt_avg = (2*h_batt_top + 2*h_batt_vert)/4.0 # Simple average over 4 faces
    Q_conv_Bf = h_batt_avg * A_Bf_conv * (T_air - T_Bf)
    Q_conv_Bm = h_batt_avg * A_Bm_conv * (T_air - T_Bm)
    Q_conv_Br = h_batt_avg * A_Br_conv * (T_air - T_Br)

    h_ESC = get_natural_convection_h(p_esc_film, T_ESC, T_air, L_char=0.0695, is_vertical=False)
    Q_conv_ESC = h_ESC * A_ESC_conv * (T_air - T_ESC)
    
    h_TS_in = get_natural_convection_h(p_ts_int_film, T_TS_int, T_air, L_char=L_NACELLE_FORCED, is_vertical=False)
    Q_conv_TS_in = h_TS_in * A_TS * (T_air - T_TS_int)

    h_BS_in = get_natural_convection_h(p_bs_int_film, T_BS_int, T_air, L_char=L_NACELLE_FORCED, is_vertical=False)
    Q_conv_BS_in = h_BS_in * A_BS * (T_air - T_BS_int)

    # -- Radiation --
    T4_Bf,T4_Bm,T4_Br,T4_ESC,T4_TS_int,T4_BS_int=T_power4(np.array([T_Bf,T_Bm,T_Br,T_ESC,T_TS_int,T_BS_int]))
    Q_rad_Bf = (C_Bf_TS_int_rad*(T4_TS_int-T4_Bf)+C_Bf_ESC_rad*(T4_ESC-T4_Bf)+C_Bf_TS_int_rad*(T4_BS_int-T4_Bf))
    Q_rad_Bm = (C_Bf_TS_int_rad*(T4_TS_int-T4_Bm)+C_Bf_TS_int_rad*(T4_BS_int-T4_Bm))
    Q_rad_Br = (C_Bf_TS_int_rad*(T4_TS_int-T4_Br)+C_Bf_TS_int_rad*(T4_BS_int-T4_Br))
    Q_rad_ESC = (C_Bf_ESC_rad*(T4_Bf-T4_ESC)+C_ESC_TS_rad*(T4_TS_int-T4_ESC)+C_ESC_TS_rad*(T4_BS_int-T4_ESC))
    Q_rad_TS = (C_Bf_TS_int_rad*((T4_Bf-T4_TS_int)+(T4_Bm-T4_TS_int)+(T4_Br-T4_TS_int))+C_TS_BS_rad*(T4_BS_int-T4_TS_int))
    Q_rad_BS = (C_Bf_TS_int_rad*((T4_Bf-T4_BS_int)+(T4_Bm-T4_BS_int)+(T4_Br-T4_BS_int))+C_TS_BS_rad*(T4_TS_int-T4_BS_int))

    # -- Conduction --
    Q_cond_Bf = C_cond*(T_Bm-T_Bf); Q_cond_Bm = C_cond*((T_Bf-T_Bm)+(T_Br-T_Bm)); Q_cond_Br = C_cond*(T_Bm-T_Br)
    Q_cond_ESC_Bf = C_ESC_Bf_cond*(T_ESC-T_Bf)
    Q_cond_TS_ext = C_cond_cfrp*(T_TS_int-T_TS_ext); Q_cond_BS_ext = C_cond_cfrp*(T_BS_int-T_BS_ext)
    
    # -- Convection sum for air node --
    Q_conv_air = -(Q_conv_Bf + Q_conv_Bm + Q_conv_Br + Q_conv_ESC + Q_conv_TS_in + Q_conv_BS_in)
    m_a = p_air[0] * 0.11

    # --- 6e. Calculate Temperature Derivatives (dT/dt) ---
    dT_Bf_dt = (Q_B_front + Q_cond_Bf + Q_conv_Bf + Q_rad_Bf + Q_cond_ESC_Bf) / (m_Bf * C_B)
    dT_Bm_dt = (Q_B_middle + Q_cond_Bm + Q_conv_Bm + Q_rad_Bm) / (m_Bm * C_B)
    dT_Br_dt = (Q_B_rear + Q_cond_Br + Q_conv_Br + Q_rad_Br) / (m_Br * C_B)
    dT_ESC_dt = (Q_ESC + Q_conv_ESC + Q_rad_ESC - Q_cond_ESC_Bf) / (m_ESC * C_ESC)
    dT_TS_int_dt = (-Q_cond_TS_ext + Q_conv_TS_in + Q_rad_TS) / (m_TS * C_TS)
    dT_TS_ext_dt = (Q_cond_TS_ext + Q_conv_TS_ext + Q_S) / (m_TS * C_TS)
    dT_BS_int_dt = (-Q_cond_BS_ext + Q_conv_BS_in + Q_rad_BS) / (m_BS * C_BS)
    dT_BS_ext_dt = (Q_cond_BS_ext + Q_conv_BS_ext + Q_A + Q_P) / (m_BS * C_BS)
    dT_air_dt = Q_conv_air / (m_a * p_air[1])
    
    return np.array([dT_Bf_dt, dT_Bm_dt, dT_Br_dt, dT_ESC_dt, dT_TS_int_dt, dT_TS_ext_dt, dT_BS_int_dt, dT_BS_ext_dt, dT_air_dt])


# -------------------------
# 7. Set Up and Run the Simulation
# -------------------------
print(f"Conditions: Ambient Temp = {T_E}K, HAP Velocity = {velocity}m/s")
if ENABLE_DAY_NIGHT_CYCLE:
    print("Day/Night Cycle: ENABLED")
    T_total = 2 * DAY_DURATION_S
else:
    print("Day/Night Cycle: DISABLED (No external radiation)")
    T_total = 3600 * 2 # Simulate for 2 hours to ensure steady state

x0 = np.array([T_E] * 9) # Start all components at the ambient temperature
sol = solve_ivp(fun=f, t_span=[0, T_total], y0=x0, method='BDF', dense_output=True)

# -------------------------
# 8. Process and Display Results
# -------------------------
print(f"\nSolver finished with status: {sol.status} ({sol.message})")
final_temps = sol.y[:, -1]
labels = ['Battery Front', 'Battery Middle', 'Battery Rear', 'ESC', 'Top Shell Internal',
          'Top Shell External', 'Bottom Shell Internal', 'Bottom Shell External', 'Internal Air']
print(f"\nFinal Temperatures at t = {T_total/3600:.1f} hours (K):")
for lab, temp in zip(labels, final_temps):
    print(f"{lab:<25}: {temp:.2f} K")
print(f"\nTotal execution time: {time.time() - start_time:.2f} seconds")

# -------------------------
# 9. Plot the Results
# -------------------------
plt.figure(figsize=(14, 9))
plot_time_unit = 3600 # Plot time axis in hours
time_label = 'Time (hours)'
t_plot_points = min(int(T_total) * 2, 4000)
t_plot = np.linspace(0, T_total, t_plot_points)
x_plot = sol.sol(t_plot).T
for i, lab in enumerate(labels):
    plt.plot(t_plot / plot_time_unit, x_plot[:, i], label=lab)

plt.xlabel(time_label, fontsize=12)
plt.ylabel('Temperature (K)', fontsize=12)
plt.title(f'Nacelle Temperature Evolution (T_amb={T_E}K, v={velocity}m/s)', fontsize=16)
plt.legend(loc='best'); plt.grid(True, which='both', linestyle='--'); plt.tight_layout(); plt.show()