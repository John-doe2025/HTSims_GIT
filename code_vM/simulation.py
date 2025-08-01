# 6_main_simulation.py

import time
import numpy as np
import pandas as pd
from scipy.integrate import solve_ivp
import config, physics_models, environment_model, post_processing

# --- Initialization ---
start_time = time.time()
environment_model.init()
print("Loading altitude properties...")
try:
    df_alt = pd.read_excel("altitude_data.xlsx")
    row = df_alt.iloc[(df_alt['Altitude'] - config.TARGET_ALTITUDE_KM).abs().idxmin()]
    T_E = row['Temperature']; P_amb = row['Pressure']; rho_amb = row['Density']; mu_amb = row['Viscosity']
    _, cp_amb, k_amb, _, nu_amb, Pr_amb = physics_models.prop_internal_air(T_E, P_amb)
    p_ambient = (rho_amb, cp_amb, k_amb, mu_amb, nu_amb, Pr_amb)
    print(f"Set environment for {row['Altitude']} km altitude.")
except Exception as e:
    raise SystemExit(f"FATAL ERROR loading altitude_data.xlsx: {e}")

# --- Pre-Solver Calculations ---
C_cond_batt = config.k_eff_batt * config.A_cross_batt / config.L_node_batt
C_cond_cfrp = config.k_cfrp * config.A_TS / config.t_cfrp
C_cond_ESC_to_Mount = config.k_al_mount * config.A_contact_ESC_Mount / config.L_path_ESC_Mount
C_cond_Mount_to_Bf = config.k_al_mount * config.A_contact_Mount_Bf / config.L_path_Mount_Bf

# Radiation coefficients based on the stable T4-T4 model
C_Bf_TS_rad = physics_models.rad_coeff(config.emis_batt, config.emis_shell_int, config.A_Bf_conv, config.A_TS)
C_Bf_BS_rad = physics_models.rad_coeff(config.emis_batt, config.emis_shell_int, config.A_Bf_conv, config.A_BS)
C_ESC_TS_rad = physics_models.rad_coeff(config.emis_esc, config.emis_shell_int, config.A_ESC_conv, config.A_TS)
C_ESC_BS_rad = physics_models.rad_coeff(config.emis_esc, config.emis_shell_int, config.A_ESC_conv, config.A_BS)
C_Mount_TS_rad = physics_models.rad_coeff(config.emis_mount, config.emis_shell_int, config.A_mount_conv, config.A_TS)
C_Mount_BS_rad = physics_models.rad_coeff(config.emis_mount, config.emis_shell_int, config.A_mount_conv, config.A_BS)
C_TS_BS_rad = physics_models.rad_coeff(config.emis_shell_int, config.emis_shell_int, config.A_TS, config.A_BS, vf=0.5)

# --- The ODE System Definition (Stable Merged Logic) ---
def f(t, x):
    temps = {label: x[i] for i, label in enumerate(config.labels)}
    T4s = {k: physics_models.T_power4(v) for k, v in temps.items()}
    p_air = physics_models.prop_internal_air(temps['Internal Air'], P_amb)
    
    # --- External Heat Loads (from environment_model) ---
    ext_loads = environment_model.calculate_external_heat_loads(t, temps['Top Shell External'], temps['Bottom Shell External'], T_E)
    
    # --- Heat Flow Calculations (Consistent Source-to-Sink Convention) ---
    # Conduction
    Q_cond_Bf_Bm = C_cond_batt * (temps['Battery Front'] - temps['Battery Middle'])
    Q_cond_Bm_Br = C_cond_batt * (temps['Battery Middle'] - temps['Battery Rear'])
    Q_cond_ESC_to_Mount = C_cond_ESC_to_Mount * (temps['ESC'] - temps['ESC Mount'])
    Q_cond_Mount_to_Bf = C_cond_Mount_to_Bf * (temps['ESC Mount'] - temps['Battery Front'])
    Q_cond_TS_int_ext = C_cond_cfrp * (temps['Top Shell Internal'] - temps['Top Shell External'])
    Q_cond_BS_int_ext = C_cond_cfrp * (temps['Bottom Shell Internal'] - temps['Bottom Shell External'])
    
    # Internal Convection (Component -> Air)
    p_batt_film = physics_models.prop_internal_air((temps['Battery Front'] + temps['Internal Air'])/2, P_amb)
    h_batt_avg = (physics_models.natural_convection_h(p_batt_film, temps['Battery Front'], temps['Internal Air'], config.LC_B_horiz, False) * 2 \
                + physics_models.natural_convection_h(p_batt_film, temps['Battery Front'], temps['Internal Air'], config.LC_B_vert, True) * 2) / 4.0
    Q_conv_Bf_air = h_batt_avg * config.A_Bf_conv * (temps['Battery Front'] - temps['Internal Air'])
    Q_conv_Bm_air = h_batt_avg * config.A_Bm_conv * (temps['Battery Middle'] - temps['Internal Air'])
    Q_conv_Br_air = h_batt_avg * config.A_Br_conv * (temps['Battery Rear'] - temps['Internal Air'])
    
    p_esc_film = physics_models.prop_internal_air((temps['ESC'] + temps['Internal Air'])/2, P_amb)
    Q_conv_ESC_air = physics_models.natural_convection_h(p_esc_film, temps['ESC'], temps['Internal Air'], config.LC_ESC, False) * config.A_ESC_conv * (temps['ESC'] - temps['Internal Air'])
    
    p_mount_film = physics_models.prop_internal_air((temps['ESC Mount'] + temps['Internal Air'])/2, P_amb)
    Q_conv_Mount_air = physics_models.natural_convection_h(p_mount_film, temps['ESC Mount'], temps['Internal Air'], config.LC_mount, False) * config.A_mount_conv * (temps['ESC Mount'] - temps['Internal Air'])

    p_ts_int_film = physics_models.prop_internal_air((temps['Top Shell Internal'] + temps['Internal Air'])/2, P_amb)
    Q_conv_TSin_air = physics_models.natural_convection_h(p_ts_int_film, temps['Top Shell Internal'], temps['Internal Air'], config.LC_TS_int, False) * config.A_TS * (temps['Top Shell Internal'] - temps['Internal Air'])
    p_bs_int_film = physics_models.prop_internal_air((temps['Bottom Shell Internal'] + temps['Internal Air'])/2, P_amb)
    Q_conv_BSin_air = physics_models.natural_convection_h(p_bs_int_film, temps['Bottom Shell Internal'], temps['Internal Air'], config.LC_BS_int, False) * config.A_BS * (temps['Bottom Shell Internal'] - temps['Internal Air'])
    
    Q_conv_air_total = Q_conv_Bf_air + Q_conv_Bm_air + Q_conv_Br_air + Q_conv_ESC_air + Q_conv_Mount_air + Q_conv_TSin_air + Q_conv_BSin_air

    # External Convection (Shell -> Ambient)
    Q_conv_TS_ext = physics_models.get_external_convection_h(p_ambient, temps['Top Shell External'], T_E, config.LC_TS_ext, config.LC_TS_int, config.velocity) * config.A_TS * (temps['Top Shell External'] - T_E)
    Q_conv_BS_ext = physics_models.get_external_convection_h(p_ambient, temps['Bottom Shell External'], T_E, config.LC_BS_ext, config.LC_BS_int, config.velocity) * config.A_BS * (temps['Bottom Shell External'] - T_E)

    # Internal Radiation (Net heat from each component)
    Q_rad_Bf = C_Bf_TS_rad * (T4s['Battery Front'] - T4s['Top Shell Internal']) + C_Bf_BS_rad * (T4s['Battery Front'] - T4s['Bottom Shell Internal'])
    Q_rad_ESC = C_ESC_TS_rad * (T4s['ESC'] - T4s['Top Shell Internal']) + C_ESC_BS_rad * (T4s['ESC'] - T4s['Bottom Shell Internal'])
    Q_rad_Mount = C_Mount_TS_rad * (T4s['ESC Mount'] - T4s['Top Shell Internal']) + C_Mount_BS_rad * (T4s['ESC Mount'] - T4s['Bottom Shell Internal'])
    Q_rad_TS_BS = C_TS_BS_rad * (T4s['Top Shell Internal'] - T4s['Bottom Shell Internal'])
    
    # --- Temperature Derivatives (dT/dt = (Heat In - Heat Out) / (m*C)) ---
    dT_Bf_dt = (config.Q_B_front - Q_conv_Bf_air - Q_cond_Bf_Bm + Q_cond_Mount_to_Bf - Q_rad_Bf) / (config.m_Bf * config.C_B)
    dT_Bm_dt = (config.Q_B_middle - Q_conv_Bm_air + Q_cond_Bf_Bm - Q_cond_Bm_Br) / (config.m_Bm * config.C_B)
    dT_Br_dt = (config.Q_B_rear - Q_conv_Br_air + Q_cond_Bm_Br) / (config.m_Br * config.C_B)
    dT_ESC_dt = (config.Q_ESC - Q_conv_ESC_air - Q_cond_ESC_to_Mount - Q_rad_ESC) / (config.m_ESC * config.C_ESC)
    dT_mount_dt = (- Q_conv_Mount_air + Q_cond_ESC_to_Mount - Q_cond_Mount_to_Bf - Q_rad_Mount) / (config.m_mount * config.C_mount)
    dT_TS_int_dt = (-Q_conv_TSin_air + Q_rad_Bf + Q_rad_ESC + Q_rad_Mount - Q_rad_TS_BS - Q_cond_TS_int_ext) / (config.m_TS * config.C_TS)
    dT_TS_ext_dt = (Q_cond_TS_int_ext - Q_conv_TS_ext + ext_loads['Q_ext_top']) / (config.m_TS * config.C_TS)
    dT_BS_int_dt = (-Q_conv_BSin_air + Q_rad_TS_BS - Q_cond_BS_int_ext) / (config.m_BS * config.C_BS)
    dT_BS_ext_dt = (Q_cond_BS_int_ext - Q_conv_BS_ext + ext_loads['Q_ext_bottom']) / (config.m_BS * config.C_BS)
    dT_air_dt = Q_conv_air_total / (p_air[0] * config.V_internal_air * p_air[1])
    
    return [dT_Bf_dt, dT_Bm_dt, dT_Br_dt, dT_ESC_dt, dT_mount_dt, dT_TS_int_dt, dT_TS_ext_dt, dT_BS_int_dt, dT_BS_ext_dt, dT_air_dt]

# --- Main Execution Block ---
if __name__ == "__main__":
    print("\n--- Simulation Initial Conditions ---")
    print(f"Target Altitude: {config.TARGET_ALTITUDE_KM} km")
    print(f"Ambient Temp: {T_E:.2f} K | Ambient Pressure: {P_amb:.2f} Pa")
    print(f"Aircraft Velocity: {config.velocity} m/s")
    print("-------------------------------------\n")
    x0 = np.array([config.initial_temp_K] * len(config.labels))
    print(f"Starting solver for {config.T_total / 86400:.1f} days...")
    sol = solve_ivp(fun=f, t_span=[0, config.T_total], y0=x0, method='BDF', dense_output=True)
    print("... Solver finished.")
    print(f"\nTotal execution time: {time.time() - start_time:.2f} seconds")
    post_processing.print_final_temps(sol)
    post_processing.analyze_peaks(sol)
    post_processing.plot_results(sol)