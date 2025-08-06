# main_simulation.py

import time
import numpy as np
import pandas as pd
from scipy.integrate import solve_ivp
import config
import physics_models
import environment_model  
import post_processing   

# --- 1. Initialization and Pre-Calculations ---
start_time = time.time()
# environment_model.init() # Disabled for now to focus on core physics
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

# Pre-calculate constant conduction coefficients
C_cond_ESC_to_Mount = config.k_mount * config.A_contact_ESC_Mount / config.L_path_ESC_Mount
C_cond_Mount_to_BH1 = config.k_mount * config.A_contact_Mount_BH1 / config.t_bulkhead
C_cond_Batt_Top_Bot = config.k_eff_batt * (config.L_batt_zone * config.H_batt_zone) / config.W_batt_zone
C_cond_Batt_to_BH = config.k_eff_batt * config.A_contact_Batt_BH / (config.L_batt_zone / 2)
C_cond_BH_to_Shell = config.k_bulkhead * config.A_contact_BH_Shell / 0.001
C_cond_cfrp = config.k_cfrp * config.A_TS / config.t_cfrp

# Pre-calculate constant radiation coefficients
C_ESC_TS_rad = physics_models.rad_coeff(config.emis_esc, config.emis_shell_int, config.A_ESC_conv, config.A_TS)
C_ESC_BS_rad = physics_models.rad_coeff(config.emis_esc, config.emis_shell_int, config.A_ESC_conv, config.A_BS)
C_Mount_TS_rad = physics_models.rad_coeff(config.emis_mount, config.emis_shell_int, config.A_mount_conv, config.A_TS)
C_Mount_BS_rad = physics_models.rad_coeff(config.emis_mount, config.emis_shell_int, config.A_mount_conv, config.A_BS)
C_TS_BS_rad = physics_models.rad_coeff(config.emis_shell_int, config.emis_shell_int, config.A_TS, config.A_BS, 0.5)
# --- 2. THE ODE SYSTEM (To be built piece-by-piece) ---

def f(t, x):
    # Unpack all 17 temperatures
    temps = {label: x[i] for i, label in enumerate(config.labels)}
    T4s = {k: physics_models.T_power4(v) for k, v in temps.items()}
    p_air = physics_models.prop_internal_air(temps['Internal_Air'], P_amb)
    
    # Initialize all heat flows to zero
    Q_net = {label: 0.0 for label in config.labels}
    
    # ========================================================================
    # WE WILL FILL THIS SECTION TOGETHER, COMPONENT BY COMPONENT
    # ========================================================================

        # --- A. ESC Calculations ---
    # Heat is generated within the ESC and flows OUT via three paths:
    # 1. Conduction to the Mount.
    # 2. Convection to the Internal Air.
    # 3. Radiation to the internal faces of the Top and Bottom Shells.

    # 1. Conduction (ESC -> Mount)
    Q_cond_ESC_to_Mount = C_cond_ESC_to_Mount * (temps['ESC'] - temps['ESC_Mount'])
    
    # 2. Convection (ESC -> Air)
    p_esc_film = physics_models.prop_internal_air((temps['ESC'] + temps['Internal_Air'])/2, P_amb)
    h_conv_esc = physics_models.natural_convection_h(p_esc_film, temps['ESC'], temps['Internal_Air'], config.LC_ESC, False)
    Q_conv_ESC_air = h_conv_esc * config.A_ESC_conv * (temps['ESC'] - temps['Internal_Air'])
    
    # 3. Radiation (ESC -> Shells)
    Q_rad_ESC_to_TS = C_ESC_TS_rad * (T4s['ESC'] - T4s['Top_Shell_Int'])
    Q_rad_ESC_to_BS = C_ESC_BS_rad * (T4s['ESC'] - T4s['Bot_Shell_Int'])
    Q_rad_ESC_net = Q_rad_ESC_to_TS + Q_rad_ESC_to_BS

    # Apply energy balance to the affected nodes
    # Heat IN to ESC:
    Q_net['ESC'] += config.Q_ESC
    # Heat OUT from ESC:
    Q_net['ESC'] -= Q_cond_ESC_to_Mount
    Q_net['ESC'] -= Q_conv_ESC_air
    Q_net['ESC'] -= Q_rad_ESC_net
    # Corresponding Heat IN to neighbors:
    Q_net['ESC_Mount'] += Q_cond_ESC_to_Mount
    Q_net['Internal_Air'] += Q_conv_ESC_air
    Q_net['Top_Shell_Int'] += Q_rad_ESC_to_TS
    Q_net['Bot_Shell_Int'] += Q_rad_ESC_to_BS

        # --- B. Mount Calculations ---
    # Heat flows INTO the Mount from the ESC, and flows OUT via three paths:
    # 1. Conduction to the Front Bulkhead (BH_1).
    # 2. Convection to the Internal Air.
    # 3. Radiation to the internal faces of the Top and Bottom Shells.

    # 1. Conduction (Mount -> BH_1)
    Q_cond_Mount_to_BH1 = C_cond_Mount_to_BH1 * (temps['ESC_Mount'] - temps['BH_1'])
    
    # 2. Convection (Mount -> Air)
    p_mount_film = physics_models.prop_internal_air((temps['ESC_Mount'] + temps['Internal_Air'])/2, P_amb)
    h_conv_mount = physics_models.natural_convection_h(p_mount_film, temps['ESC_Mount'], temps['Internal_Air'], config.LC_mount, False)
    Q_conv_Mount_air = h_conv_mount * config.A_mount_conv * (temps['ESC_Mount'] - temps['Internal_Air'])
    
    # 3. Radiation (Mount -> Shells)
    Q_rad_Mount_to_TS = C_Mount_TS_rad * (T4s['ESC_Mount'] - T4s['Top_Shell_Int'])
    Q_rad_Mount_to_BS = C_Mount_BS_rad * (T4s['ESC_Mount'] - T4s['Bot_Shell_Int'])
    Q_rad_Mount_net = Q_rad_Mount_to_TS + Q_rad_Mount_to_BS

    # Apply energy balance to the affected nodes
    # Heat OUT from Mount:
    Q_net['ESC_Mount'] -= Q_cond_Mount_to_BH1
    Q_net['ESC_Mount'] -= Q_conv_Mount_air
    Q_net['ESC_Mount'] -= Q_rad_Mount_net
    # Corresponding Heat IN to neighbors:
    Q_net['BH_1'] += Q_cond_Mount_to_BH1
    Q_net['Internal_Air'] += Q_conv_Mount_air
    Q_net['Top_Shell_Int'] += Q_rad_Mount_to_TS
    Q_net['Bot_Shell_Int'] += Q_rad_Mount_to_BS
    
        # --- C. Bulkhead Calculations ---
    # The 4 bulkheads exchange heat with their neighboring battery nodes,
    # convect/radiate to the internal air, and conduct heat out to the shells.
    
    # Pre-calculate a single h_conv and h_rad for all bulkheads for simplicity and stability
    p_bh_film = physics_models.prop_internal_air((temps['BH_2'] + temps['Internal_Air'])/2, P_amb)
    h_conv_bh = physics_models.natural_convection_h(p_bh_film, temps['BH_2'], temps['Internal_Air'], config.LC_bulkhead, True)
    
    # Loop through each of the 4 bulkheads
    for i in range(1, 5):
        bh_label = f'BH_{i}'
        T_bh = temps[bh_label]
        
        # --- Convection and Radiation to Air ---
        Q_conv_BH_air = h_conv_bh * (2 * config.A_bulkhead_face) * (T_bh - temps['Internal_Air'])
        Q_net[bh_label] -= Q_conv_BH_air
        Q_net['Internal_Air'] += Q_conv_BH_air

        # --- Conduction to Shells ---
        # Assume each bulkhead connects to both top and bottom shells equally
        Q_cond_BH_to_TS = (C_cond_BH_to_Shell / 2) * (T_bh - temps['Top_Shell_Int'])
        Q_cond_BH_to_BS = (C_cond_BH_to_Shell / 2) * (T_bh - temps['Bot_Shell_Int'])
        Q_net[bh_label] -= (Q_cond_BH_to_TS + Q_cond_BH_to_BS)
        Q_net['Top_Shell_Int'] += Q_cond_BH_to_TS
        Q_net['Bot_Shell_Int'] += Q_cond_BH_to_BS
        
    # --- Conduction between Bulkheads and Battery Nodes ---
    # BH_1 is a special case as it only has batteries on one side
    Q_cond_BH1_to_BFT = C_cond_Batt_to_BH * (temps['BH_1'] - temps['Batt_BF_Top'])
    Q_cond_BH1_to_BFB = C_cond_Batt_to_BH * (temps['BH_1'] - temps['Batt_BF_Bot'])
    Q_net['BH_1'] -= (Q_cond_BH1_to_BFT + Q_cond_BH1_to_BFB)
    Q_net['Batt_BF_Top'] += Q_cond_BH1_to_BFT
    Q_net['Batt_BF_Bot'] += Q_cond_BH1_to_BFB
    
    # BH_2 is between BF and BM batteries
    Q_cond_BH2_from_BFT = C_cond_Batt_to_BH * (temps['Batt_BF_Top'] - temps['BH_2'])
    Q_cond_BH2_from_BFB = C_cond_Batt_to_BH * (temps['Batt_BF_Bot'] - temps['BH_2'])
    Q_cond_BH2_to_BMT = C_cond_Batt_to_BH * (temps['BH_2'] - temps['Batt_BM_Top'])
    Q_cond_BH2_to_BMB = C_cond_Batt_to_BH * (temps['BH_2'] - temps['Batt_BM_Bot'])
    Q_net['BH_2'] += (Q_cond_BH2_from_BFT + Q_cond_BH2_from_BFB - Q_cond_BH2_to_BMT - Q_cond_BH2_to_BMB)
    Q_net['Batt_BF_Top'] -= Q_cond_BH2_from_BFT
    Q_net['Batt_BF_Bot'] -= Q_cond_BH2_from_BFB
    Q_net['Batt_BM_Top'] -= Q_cond_BH2_to_BMT
    Q_net['Batt_BM_Bot'] -= Q_cond_BH2_to_BMB

    # BH_3 is between BM and BR batteries
    Q_cond_BH3_from_BMT = C_cond_Batt_to_BH * (temps['Batt_BM_Top'] - temps['BH_3'])
    Q_cond_BH3_from_BMB = C_cond_Batt_to_BH * (temps['Batt_BM_Bot'] - temps['BH_3'])
    Q_cond_BH3_to_BRT = C_cond_Batt_to_BH * (temps['BH_3'] - temps['Batt_BR_Top'])
    Q_cond_BH3_to_BRB = C_cond_Batt_to_BH * (temps['BH_3'] - temps['Batt_BR_Bot'])
    Q_net['BH_3'] += (Q_cond_BH3_from_BMT + Q_cond_BH3_from_BMB - Q_cond_BH3_to_BRT - Q_cond_BH3_to_BRB)
    Q_net['Batt_BM_Top'] -= Q_cond_BH3_from_BMT
    Q_net['Batt_BM_Bot'] -= Q_cond_BH3_from_BMB
    Q_net['Batt_BR_Top'] += Q_cond_BH3_to_BRT
    Q_net['Batt_BR_Bot'] += Q_cond_BH3_to_BRB

    # BH_4 is a special case at the rear
    Q_cond_BH4_from_BRT = C_cond_Batt_to_BH * (temps['Batt_BR_Top'] - temps['BH_4'])
    Q_cond_BH4_from_BRB = C_cond_Batt_to_BH * (temps['Batt_BR_Bot'] - temps['BH_4'])
    Q_net['BH_4'] += (Q_cond_BH4_from_BRT + Q_cond_BH4_from_BRB)
    Q_net['Batt_BR_Top'] -= Q_cond_BH4_from_BRT
    Q_net['Batt_BR_Bot'] -= Q_cond_BH4_from_BRB

        # --- D. Battery Node Calculations ---
    # The 6 battery zones generate heat, convect/radiate to the environment,
    # and conduct heat to each other.
    
    # --- Conduction Between Battery Nodes (Internal Pack Gradients) ---
    # 1. Front-to-Middle Conduction
    Q_cond_BFT_to_BMT = C_cond_Batt_to_BH * (temps['Batt_BF_Top'] - temps['Batt_BM_Top'])
    Q_cond_BFB_to_BMB = C_cond_Batt_to_BH * (temps['Batt_BF_Bot'] - temps['Batt_BM_Bot'])
    Q_net['Batt_BF_Top'] -= Q_cond_BFT_to_BMT; Q_net['Batt_BM_Top'] += Q_cond_BFT_to_BMT
    Q_net['Batt_BF_Bot'] -= Q_cond_BFB_to_BMB; Q_net['Batt_BM_Bot'] += Q_cond_BFB_to_BMB
    
    # 2. Middle-to-Rear Conduction
    Q_cond_BMT_to_BRT = C_cond_Batt_to_BH * (temps['Batt_BM_Top'] - temps['Batt_BR_Top'])
    Q_cond_BMB_to_BRB = C_cond_Batt_to_BH * (temps['Batt_BM_Bot'] - temps['Batt_BR_Bot'])
    Q_net['Batt_BM_Top'] -= Q_cond_BMT_to_BRT; Q_net['Batt_BR_Top'] += Q_cond_BMT_to_BRT
    Q_net['Batt_BM_Bot'] -= Q_cond_BMB_to_BRB; Q_net['Batt_BR_Bot'] += Q_cond_BMB_to_BRB

    # 3. Top-to-Bottom Conduction
    Q_cond_BFT_to_BFB = C_cond_Batt_Top_Bot * (temps['Batt_BF_Top'] - temps['Batt_BF_Bot'])
    Q_cond_BMT_to_BMB = C_cond_Batt_Top_Bot * (temps['Batt_BM_Top'] - temps['Batt_BM_Bot'])
    Q_cond_BRT_to_BRB = C_cond_Batt_Top_Bot * (temps['Batt_BR_Top'] - temps['Batt_BR_Bot'])
    Q_net['Batt_BF_Top'] -= Q_cond_BFT_to_BFB; Q_net['Batt_BF_Bot'] += Q_cond_BFT_to_BFB
    Q_net['Batt_BM_Top'] -= Q_cond_BMT_to_BMB; Q_net['Batt_BM_Bot'] += Q_cond_BMT_to_BMB
    Q_net['Batt_BR_Top'] -= Q_cond_BRT_to_BRB; Q_net['Batt_BR_Bot'] += Q_cond_BRT_to_BRB

    # --- Convection and Radiation from Battery Nodes to Environment ---
    batt_zones = [
        ('Batt_BF_Top', config.A_conv_batt_end), ('Batt_BF_Bot', config.A_conv_batt_end),
        ('Batt_BM_Top', config.A_conv_batt_middle), ('Batt_BM_Bot', config.A_conv_batt_middle),
        ('Batt_BR_Top', config.A_conv_batt_end), ('Batt_BR_Bot', config.A_conv_batt_end)
    ]

    for label, area in batt_zones:
        T_batt = temps[label]
        
        # Calculate a weighted average h_conv for this zone
        p_batt_film = physics_models.prop_internal_air((T_batt + temps['Internal_Air'])/2, P_amb)
        h_horiz = physics_models.natural_convection_h(p_batt_film, T_batt, temps['Internal_Air'], config.LC_batt_horiz, False)
        h_vert = physics_models.natural_convection_h(p_batt_film, T_batt, temps['Internal_Air'], config.LC_batt_vert, True)
        
        # Simplified weighted average based on area types
        A_horiz = 2 * config.L_batt_zone * config.W_batt_zone
        A_vert = area - A_horiz
        h_avg = (h_horiz * A_horiz + h_vert * A_vert) / area if area > 0 else 0
        
        # Convection to Air
        Q_conv_batt_air = h_avg * area * (T_batt - temps['Internal_Air'])
        Q_net[label] -= Q_conv_batt_air
        Q_net['Internal_Air'] += Q_conv_batt_air

        # Radiation to Shells
        C_rad_batt_TS = physics_models.rad_coeff(config.emis_batt, config.emis_shell_int, area, config.A_TS)
        C_rad_batt_BS = physics_models.rad_coeff(config.emis_batt, config.emis_shell_int, area, config.A_BS)
        Q_rad_batt_net = C_rad_batt_TS * (T4s[label] - T4s['Top_Shell_Int']) + \
                         C_rad_batt_BS * (T4s[label] - T4s['Bot_Shell_Int'])
        
        Q_net[label] -= Q_rad_batt_net
        Q_net['Top_Shell_Int'] += C_rad_batt_TS * (T4s[label] - T4s['Top_Shell_Int'])
        Q_net['Bot_Shell_Int'] += C_rad_batt_BS * (T4s[label] - T4s['Bot_Shell_Int'])


        # --- E. Shell and Air Calculations ---
    # This section handles the energy balance for the shell walls and the
    # internal air volume, which receives heat from all other components.

    # --- Shell Conduction (Internal <-> External) ---
    Q_cond_TS_int_ext = C_cond_cfrp * (temps['Top_Shell_Int'] - temps['Top_Shell_Ext'])
    Q_cond_BS_int_ext = C_cond_cfrp * (temps['Bot_Shell_Int'] - temps['Bot_Shell_Ext'])
    
    Q_net['Top_Shell_Int'] -= Q_cond_TS_int_ext
    Q_net['Top_Shell_Ext'] += Q_cond_TS_int_ext
    Q_net['Bot_Shell_Int'] -= Q_cond_BS_int_ext
    Q_net['Bot_Shell_Ext'] += Q_cond_BS_int_ext
    
    # --- Internal Convection (Shells -> Air) ---
    p_ts_int_film = physics_models.prop_internal_air((temps['Top_Shell_Int'] + temps['Internal_Air'])/2, P_amb)
    h_conv_TSin = physics_models.natural_convection_h(p_ts_int_film, temps['Top_Shell_Int'], temps['Internal_Air'], config.LC_TS_int, False)
    Q_conv_TSin_air = h_conv_TSin * config.A_TS * (temps['Top_Shell_Int'] - temps['Internal_Air'])
    
    p_bs_int_film = physics_models.prop_internal_air((temps['Bot_Shell_Int'] + temps['Internal_Air'])/2, P_amb)
    h_conv_BSin = physics_models.natural_convection_h(p_bs_int_film, temps['Bot_Shell_Int'], temps['Internal_Air'], config.LC_BS_int, False)
    Q_conv_BSin_air = h_conv_BSin * config.A_BS * (temps['Bot_Shell_Int'] - temps['Internal_Air'])
    
    Q_net['Top_Shell_Int'] -= Q_conv_TSin_air
    Q_net['Bot_Shell_Int'] -= Q_conv_BSin_air
    Q_net['Internal_Air'] += Q_conv_TSin_air + Q_conv_BSin_air
    
    # --- External Convection (Shells -> Ambient) ---
    p_ts_ext_film = physics_models.prop_internal_air((temps['Top_Shell_Ext'] + T_E)/2, P_amb)
    h_conv_TSext = physics_models.get_external_convection_h(p_ts_ext_film, temps['Top_Shell_Ext'], T_E, config.LC_TS_ext, config.LC_TS_int, config.velocity)
    Q_conv_TSext_amb = h_conv_TSext * config.A_TS * (temps['Top_Shell_Ext'] - T_E)
    
    p_bs_ext_film = physics_models.prop_internal_air((temps['Bot_Shell_Ext'] + T_E)/2, P_amb)
    h_conv_BSext = physics_models.get_external_convection_h(p_bs_ext_film, temps['Bot_Shell_Ext'], T_E, config.LC_BS_ext, config.LC_BS_int, config.velocity)
    Q_conv_BSext_amb = h_conv_BSext * config.A_BS * (temps['Bot_Shell_Ext'] - T_E)
    
    Q_net['Top_Shell_Ext'] -= Q_conv_TSext_amb
    Q_net['Bot_Shell_Ext'] -= Q_conv_BSext_amb
    # --- Internal Radiation between Shells ---
    Q_rad_TS_to_BS = C_TS_BS_rad * (T4s['Top_Shell_Int'] - T4s['Bot_Shell_Int'])
    Q_net['Top_Shell_Int'] -= Q_rad_TS_to_BS
    Q_net['Bot_Shell_Int'] += Q_rad_TS_to_BS

    # --- External Radiation (from environment_model) ---
    ext_loads = environment_model.calculate_external_heat_loads(t, temps['Top_Shell_Ext'], temps['Bot_Shell_Ext'], T_E)
    Q_net['Top_Shell_Ext'] += ext_loads['Q_ext_top']
    Q_net['Bot_Shell_Ext'] += ext_loads['Q_ext_bottom']
    # ========================================================================
        # --- Final Derivative Calculation (dT/dt = Q_net / (m*C)) ---
    m_batt = config.m_batt_zone
    dTdt = [
        # Battery Nodes (6)
        (Q_net['Batt_BF_Top'] + config.Q_batt_zone) / (m_batt * config.C_B),
        (Q_net['Batt_BF_Bot'] + config.Q_batt_zone) / (m_batt * config.C_B),
        (Q_net['Batt_BM_Top'] + config.Q_batt_zone) / (m_batt * config.C_B),
        (Q_net['Batt_BM_Bot'] + config.Q_batt_zone) / (m_batt * config.C_B),
        (Q_net['Batt_BR_Top'] + config.Q_batt_zone) / (m_batt * config.C_B),
        (Q_net['Batt_BR_Bot'] + config.Q_batt_zone) / (m_batt * config.C_B),
        
        # Avionics Nodes (2)
        (Q_net['ESC'] + config.Q_ESC) / (config.m_ESC * config.C_ESC),
        (Q_net['ESC_Mount']) / (config.m_mount * config.C_mount),
        
        # Bulkhead Nodes (4)
        (Q_net['BH_1']) / (config.m_bulkhead * config.C_bulkhead),
        (Q_net['BH_2']) / (config.m_bulkhead * config.C_bulkhead),
        (Q_net['BH_3']) / (config.m_bulkhead * config.C_bulkhead),
        (Q_net['BH_4']) / (config.m_bulkhead * config.C_bulkhead),

        # Shell Nodes (4)
        (Q_net['Top_Shell_Int']) / (config.m_TS * config.C_TS),
        (Q_net['Top_Shell_Ext']) / (config.m_TS * config.C_TS),
        (Q_net['Bot_Shell_Int']) / (config.m_BS * config.C_BS),
        (Q_net['Bot_Shell_Ext']) / (config.m_BS * config.C_BS),
        
        # Air Node (1)
        (Q_net['Internal_Air']) / (p_air[0] * config.V_internal_air * p_air[1])
    ]
    
    return dTdt

# --- 3. Main Execution Block ---
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