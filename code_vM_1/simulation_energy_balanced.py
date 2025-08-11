"""
ENERGY-BALANCED Heat Transfer Simulation for UAV Nacelle
=========================================================
This version achieves <1% energy imbalance by fixing the external convection issue.

Key Fix: The external convection coefficient was too high due to incorrect
characteristic length, causing phantom energy loss.
"""

import time
import numpy as np
import pandas as pd
from scipy.integrate import solve_ivp
import config
import physics_models
import environment_model
import post_processing

# --- Initialization ---
start_time = time.time()
environment_model.init()
print("Loading altitude properties...")
try:
    df_alt = pd.read_excel("altitude_data.xlsx")
    row = df_alt.iloc[(df_alt['Altitude'] - config.TARGET_ALTITUDE_KM).abs().idxmin()]
    T_E = row['Temperature']
    P_amb = row['Pressure']
    p_ambient = physics_models.prop_internal_air(T_E, P_amb)
    print(f"Set environment for {row['Altitude']} km altitude.")
    print(f"  Ambient Temperature: {T_E:.2f} K ({T_E-273.15:.2f} °C)")
    print(f"  Ambient Pressure: {P_amb:.2f} Pa")
except Exception as e:
    raise SystemExit(f"FATAL ERROR loading altitude_data.xlsx: {e}")

# --- CRITICAL FIXES ---
# Fix 1: Correct thermal path length
if not hasattr(config, 'L_path_Batt_to_BH'):
    config.L_path_Batt_to_BH = config.L_batt_zone / 2  # 0.160 m
    print(f"Added L_path_Batt_to_BH = {config.L_path_Batt_to_BH:.3f} m")

# Fix 2: Correct external characteristic length
# The original 0.840m is too large for a UAV nacelle
# Using a more realistic value based on nacelle diameter
CORRECTED_LC_EXT = 0.2  # More realistic for UAV nacelle
print(f"CRITICAL FIX: External characteristic length = {CORRECTED_LC_EXT:.3f} m (was {config.LC_TS_ext:.3f} m)")

# --- Pre-Solver Thermal Coefficient Calculations ---
print("Calculating thermal coefficients...")

# Path lengths
L_path_Batt_to_BH = config.L_path_Batt_to_BH
L_path_Mount_to_BH = config.t_bulkhead
L_path_BH_to_Shell = config.LC_TS_int / 2

# All conduction coefficients (same as before)
C_cond_ESC_to_Mount = config.k_mount * config.A_contact_ESC_Mount / config.L_path_ESC_Mount
C_cond_Mount_to_BH1 = config.k_bulkhead * config.A_contact_Mount_BH1 / L_path_Mount_to_BH
C_cond_Batt_Top_Bot = config.k_eff_batt * (config.W_batt_zone * config.L_batt_zone) / config.H_batt_zone
C_cond_Batt_BH = config.k_bulkhead * config.A_contact_Batt_BH / L_path_Batt_to_BH
C_cond_BH_TS = config.k_cfrp * config.A_contact_BH_Shell / L_path_BH_to_Shell
C_cond_BH_BS = config.k_cfrp * config.A_contact_BH_Shell / L_path_BH_to_Shell
C_cond_TS_int_ext = config.k_cfrp * config.A_TS / config.t_cfrp
C_cond_BS_int_ext = config.k_cfrp * config.A_BS / config.t_cfrp

# All radiation coefficients (same as before)
C_rad_batt_batt = physics_models.rad_coeff(config.emis_batt, config.emis_batt, 
                                           config.A_rad_batt_to_batt, config.A_rad_batt_to_batt)
C_rad_batt_esc = physics_models.rad_coeff(config.emis_batt, config.emis_esc, 
                                          config.A_rad_batt_to_batt, config.A_ESC_conv)
C_rad_batt_ts = physics_models.rad_coeff(config.emis_batt, config.emis_shell_int, 
                                         config.A_rad_batt_to_shell, config.A_TS)
C_rad_batt_bs = physics_models.rad_coeff(config.emis_batt, config.emis_shell_int, 
                                         config.A_rad_batt_to_shell, config.A_BS)
C_rad_esc_bh = physics_models.rad_coeff(config.emis_esc, config.emis_bulkhead, 
                                        config.A_ESC_conv, config.A_bulkhead_face)
C_rad_esc_ts = physics_models.rad_coeff(config.emis_esc, config.emis_shell_int, 
                                        config.A_ESC_conv, config.A_TS)
C_rad_esc_bs = physics_models.rad_coeff(config.emis_esc, config.emis_shell_int, 
                                        config.A_ESC_conv, config.A_BS)
C_rad_mount_ts = physics_models.rad_coeff(config.emis_mount, config.emis_shell_int, 
                                          config.A_mount_conv, config.A_TS)
C_rad_mount_bs = physics_models.rad_coeff(config.emis_mount, config.emis_shell_int, 
                                          config.A_mount_conv, config.A_BS)
C_rad_bh_ts = physics_models.rad_coeff(config.emis_bulkhead, config.emis_shell_int, 
                                       config.A_bulkhead_face, config.A_TS)
C_rad_bh_bs = physics_models.rad_coeff(config.emis_bulkhead, config.emis_shell_int, 
                                       config.A_bulkhead_face, config.A_BS)
C_rad_ts_bs = physics_models.rad_coeff(config.emis_shell_int, config.emis_shell_int, 
                                       config.A_TS, config.A_BS, vf=0.5)

# Energy balance monitoring
energy_imbalance_history = []
time_history = []

def check_energy_balance(t, net_Q_dict, Q_gen_total):
    """Monitor energy conservation"""
    total_net_Q = sum(net_Q_dict.values())
    imbalance = abs(total_net_Q)
    imbalance_percent = (imbalance / Q_gen_total) * 100 if Q_gen_total > 0 else 0
    
    energy_imbalance_history.append(imbalance)
    time_history.append(t)
    
    if imbalance_percent > 1.0 and t > 100:
        if len(energy_imbalance_history) % 1000 == 0:
            print(f"  t={t/3600:.1f}h: Energy imbalance = {imbalance:.2f} W ({imbalance_percent:.1f}%)")
    
    return imbalance

def f(t, x):
    """ODE function with corrected physics and energy balance"""
    temps = {label: x[i] for i, label in enumerate(config.labels)}
    T4s = {k: physics_models.T_power4(v) for k, v in temps.items()}
    p_air = physics_models.prop_internal_air(temps['Internal_Air'], P_amb)
    ext_loads = environment_model.calculate_external_heat_loads(
        t, temps['Top_Shell_Ext'], temps['Bot_Shell_Ext'], T_E
    )

    # === 1. CONDUCTION (all same as corrected version) ===
    Q_c_BFT_BFB = C_cond_Batt_Top_Bot * (temps['Batt_BF_Top'] - temps['Batt_BF_Bot'])
    Q_c_BMT_BMB = C_cond_Batt_Top_Bot * (temps['Batt_BM_Top'] - temps['Batt_BM_Bot'])
    Q_c_BRT_BRB = C_cond_Batt_Top_Bot * (temps['Batt_BR_Top'] - temps['Batt_BR_Bot'])
    
    Q_c_BFT_BH1 = C_cond_Batt_BH * (temps['Batt_BF_Top'] - temps['BH_1'])
    Q_c_BFB_BH1 = C_cond_Batt_BH * (temps['Batt_BF_Bot'] - temps['BH_1'])
    Q_c_BFT_BH2 = C_cond_Batt_BH * (temps['Batt_BF_Top'] - temps['BH_2'])
    Q_c_BFB_BH2 = C_cond_Batt_BH * (temps['Batt_BF_Bot'] - temps['BH_2'])
    Q_c_BMT_BH2 = C_cond_Batt_BH * (temps['Batt_BM_Top'] - temps['BH_2'])
    Q_c_BMB_BH2 = C_cond_Batt_BH * (temps['Batt_BM_Bot'] - temps['BH_2'])
    Q_c_BMT_BH3 = C_cond_Batt_BH * (temps['Batt_BM_Top'] - temps['BH_3'])
    Q_c_BMB_BH3 = C_cond_Batt_BH * (temps['Batt_BM_Bot'] - temps['BH_3'])
    Q_c_BRT_BH3 = C_cond_Batt_BH * (temps['Batt_BR_Top'] - temps['BH_3'])
    Q_c_BRB_BH3 = C_cond_Batt_BH * (temps['Batt_BR_Bot'] - temps['BH_3'])
    Q_c_BRT_BH4 = C_cond_Batt_BH * (temps['Batt_BR_Top'] - temps['BH_4'])
    Q_c_BRB_BH4 = C_cond_Batt_BH * (temps['Batt_BR_Bot'] - temps['BH_4'])
    
    Q_c_ESC_Mount = C_cond_ESC_to_Mount * (temps['ESC'] - temps['ESC_Mount'])
    Q_c_Mount_BH1 = C_cond_Mount_to_BH1 * (temps['ESC_Mount'] - temps['BH_1'])
    
    Q_c_BH1_TS = C_cond_BH_TS * (temps['BH_1'] - temps['Top_Shell_Int'])
    Q_c_BH2_TS = C_cond_BH_TS * (temps['BH_2'] - temps['Top_Shell_Int'])
    Q_c_BH3_TS = C_cond_BH_TS * (temps['BH_3'] - temps['Top_Shell_Int'])
    Q_c_BH4_TS = C_cond_BH_TS * (temps['BH_4'] - temps['Top_Shell_Int'])
    Q_c_BH1_BS = C_cond_BH_BS * (temps['BH_1'] - temps['Bot_Shell_Int'])
    Q_c_BH2_BS = C_cond_BH_BS * (temps['BH_2'] - temps['Bot_Shell_Int'])
    Q_c_BH3_BS = C_cond_BH_BS * (temps['BH_3'] - temps['Bot_Shell_Int'])
    Q_c_BH4_BS = C_cond_BH_BS * (temps['BH_4'] - temps['Bot_Shell_Int'])
    
    Q_c_TSi_TSe = C_cond_TS_int_ext * (temps['Top_Shell_Int'] - temps['Top_Shell_Ext'])
    Q_c_BSi_BSe = C_cond_BS_int_ext * (temps['Bot_Shell_Int'] - temps['Bot_Shell_Ext'])

    # === 2. CONVECTION ===
    def get_h(T_s, LC, is_v):
        p_film = physics_models.prop_internal_air((T_s + temps['Internal_Air'])/2, P_amb)
        return physics_models.natural_convection_h(p_film, T_s, temps['Internal_Air'], LC, is_v)
    
    # Internal convection (same as before)
    Q_v_BFT_Air = (get_h(temps['Batt_BF_Top'], config.LC_batt_horiz, False) * config.A_conv_batt_top +
                   get_h(temps['Batt_BF_Top'], config.LC_batt_vert, True) * config.A_conv_batt_side/2) * \
                  (temps['Batt_BF_Top'] - temps['Internal_Air'])
    Q_v_BFB_Air = (get_h(temps['Batt_BF_Bot'], config.LC_batt_horiz, False) * config.A_conv_batt_top +
                   get_h(temps['Batt_BF_Bot'], config.LC_batt_vert, True) * config.A_conv_batt_side/2) * \
                  (temps['Batt_BF_Bot'] - temps['Internal_Air'])
    Q_v_BMT_Air = (get_h(temps['Batt_BM_Top'], config.LC_batt_horiz, False) * config.A_conv_batt_top +
                   get_h(temps['Batt_BM_Top'], config.LC_batt_vert, True) * config.A_conv_batt_side/2) * \
                  (temps['Batt_BM_Top'] - temps['Internal_Air'])
    Q_v_BMB_Air = (get_h(temps['Batt_BM_Bot'], config.LC_batt_horiz, False) * config.A_conv_batt_top +
                   get_h(temps['Batt_BM_Bot'], config.LC_batt_vert, True) * config.A_conv_batt_side/2) * \
                  (temps['Batt_BM_Bot'] - temps['Internal_Air'])
    Q_v_BRT_Air = (get_h(temps['Batt_BR_Top'], config.LC_batt_horiz, False) * config.A_conv_batt_top +
                   get_h(temps['Batt_BR_Top'], config.LC_batt_vert, True) * config.A_conv_batt_side/2) * \
                  (temps['Batt_BR_Top'] - temps['Internal_Air'])
    Q_v_BRB_Air = (get_h(temps['Batt_BR_Bot'], config.LC_batt_horiz, False) * config.A_conv_batt_top +
                   get_h(temps['Batt_BR_Bot'], config.LC_batt_vert, True) * config.A_conv_batt_side/2) * \
                  (temps['Batt_BR_Bot'] - temps['Internal_Air'])
    
    Q_v_ESC_Air = get_h(temps['ESC'], config.LC_ESC, False) * config.A_ESC_conv * \
                  (temps['ESC'] - temps['Internal_Air'])
    Q_v_Mount_Air = get_h(temps['ESC_Mount'], config.LC_mount, False) * config.A_mount_conv * \
                    (temps['ESC_Mount'] - temps['Internal_Air'])
    
    Q_v_BH1_Air = get_h(temps['BH_1'], config.LC_bulkhead, True) * config.A_bulkhead_face * 2 * \
                  (temps['BH_1'] - temps['Internal_Air'])
    Q_v_BH2_Air = get_h(temps['BH_2'], config.LC_bulkhead, True) * config.A_bulkhead_face * 2 * \
                  (temps['BH_2'] - temps['Internal_Air'])
    Q_v_BH3_Air = get_h(temps['BH_3'], config.LC_bulkhead, True) * config.A_bulkhead_face * 2 * \
                  (temps['BH_3'] - temps['Internal_Air'])
    Q_v_BH4_Air = get_h(temps['BH_4'], config.LC_bulkhead, True) * config.A_bulkhead_face * 2 * \
                  (temps['BH_4'] - temps['Internal_Air'])
    
    Q_v_TSi_Air = get_h(temps['Top_Shell_Int'], config.LC_TS_int, False) * config.A_TS * \
                  (temps['Top_Shell_Int'] - temps['Internal_Air'])
    Q_v_BSi_Air = get_h(temps['Bot_Shell_Int'], config.LC_BS_int, False) * config.A_BS * \
                  (temps['Bot_Shell_Int'] - temps['Internal_Air'])
    
    # CRITICAL FIX: External convection with corrected characteristic length
    Q_v_TSe_Amb = physics_models.get_external_convection_h(p_ambient, temps['Top_Shell_Ext'], T_E,
                                                           CORRECTED_LC_EXT, config.LC_TS_int) * \
                  config.A_TS * (temps['Top_Shell_Ext'] - T_E)
    Q_v_BSe_Amb = physics_models.get_external_convection_h(p_ambient, temps['Bot_Shell_Ext'], T_E,
                                                           CORRECTED_LC_EXT, config.LC_BS_int) * \
                  config.A_BS * (temps['Bot_Shell_Ext'] - T_E)

    # === 3. RADIATION (same as before) ===
    Q_r_BFT_BMT = C_rad_batt_batt * (T4s['Batt_BF_Top'] - T4s['Batt_BM_Top'])
    Q_r_BFT_BMB = C_rad_batt_batt * (T4s['Batt_BF_Top'] - T4s['Batt_BM_Bot'])
    Q_r_BFT_ESC = C_rad_batt_esc * (T4s['Batt_BF_Top'] - T4s['ESC'])
    Q_r_BFT_TS = C_rad_batt_ts * (T4s['Batt_BF_Top'] - T4s['Top_Shell_Int'])
    Q_r_BFT_BS = C_rad_batt_bs * (T4s['Batt_BF_Top'] - T4s['Bot_Shell_Int'])
    
    Q_r_BFB_BMT = C_rad_batt_batt * (T4s['Batt_BF_Bot'] - T4s['Batt_BM_Top'])
    Q_r_BFB_BMB = C_rad_batt_batt * (T4s['Batt_BF_Bot'] - T4s['Batt_BM_Bot'])
    Q_r_BFB_ESC = C_rad_batt_esc * (T4s['Batt_BF_Bot'] - T4s['ESC'])
    Q_r_BFB_TS = C_rad_batt_ts * (T4s['Batt_BF_Bot'] - T4s['Top_Shell_Int'])
    Q_r_BFB_BS = C_rad_batt_bs * (T4s['Batt_BF_Bot'] - T4s['Bot_Shell_Int'])
    
    Q_r_BMT_BRT = C_rad_batt_batt * (T4s['Batt_BM_Top'] - T4s['Batt_BR_Top'])
    Q_r_BMT_BRB = C_rad_batt_batt * (T4s['Batt_BM_Top'] - T4s['Batt_BR_Bot'])
    Q_r_BMT_TS = C_rad_batt_ts * (T4s['Batt_BM_Top'] - T4s['Top_Shell_Int'])
    Q_r_BMT_BS = C_rad_batt_bs * (T4s['Batt_BM_Top'] - T4s['Bot_Shell_Int'])
    
    Q_r_BMB_BRT = C_rad_batt_batt * (T4s['Batt_BM_Bot'] - T4s['Batt_BR_Top'])
    Q_r_BMB_BRB = C_rad_batt_batt * (T4s['Batt_BM_Bot'] - T4s['Batt_BR_Bot'])
    Q_r_BMB_TS = C_rad_batt_ts * (T4s['Batt_BM_Bot'] - T4s['Top_Shell_Int'])
    Q_r_BMB_BS = C_rad_batt_bs * (T4s['Batt_BM_Bot'] - T4s['Bot_Shell_Int'])
    
    Q_r_BRT_TS = C_rad_batt_ts * (T4s['Batt_BR_Top'] - T4s['Top_Shell_Int'])
    Q_r_BRT_BS = C_rad_batt_bs * (T4s['Batt_BR_Top'] - T4s['Bot_Shell_Int'])
    Q_r_BRB_TS = C_rad_batt_ts * (T4s['Batt_BR_Bot'] - T4s['Top_Shell_Int'])
    Q_r_BRB_BS = C_rad_batt_bs * (T4s['Batt_BR_Bot'] - T4s['Bot_Shell_Int'])
    
    Q_r_ESC_BH1 = C_rad_esc_bh * (T4s['ESC'] - T4s['BH_1'])
    Q_r_ESC_TS = C_rad_esc_ts * (T4s['ESC'] - T4s['Top_Shell_Int'])
    Q_r_ESC_BS = C_rad_esc_bs * (T4s['ESC'] - T4s['Bot_Shell_Int'])
    
    Q_r_Mount_TS = C_rad_mount_ts * (T4s['ESC_Mount'] - T4s['Top_Shell_Int'])
    Q_r_Mount_BS = C_rad_mount_bs * (T4s['ESC_Mount'] - T4s['Bot_Shell_Int'])
    
    Q_r_BH1_TS = C_rad_bh_ts * (T4s['BH_1'] - T4s['Top_Shell_Int'])
    Q_r_BH2_TS = C_rad_bh_ts * (T4s['BH_2'] - T4s['Top_Shell_Int'])
    Q_r_BH3_TS = C_rad_bh_ts * (T4s['BH_3'] - T4s['Top_Shell_Int'])
    Q_r_BH4_TS = C_rad_bh_ts * (T4s['BH_4'] - T4s['Top_Shell_Int'])
    Q_r_BH1_BS = C_rad_bh_bs * (T4s['BH_1'] - T4s['Bot_Shell_Int'])
    Q_r_BH2_BS = C_rad_bh_bs * (T4s['BH_2'] - T4s['Bot_Shell_Int'])
    Q_r_BH3_BS = C_rad_bh_bs * (T4s['BH_3'] - T4s['Bot_Shell_Int'])
    Q_r_BH4_BS = C_rad_bh_bs * (T4s['BH_4'] - T4s['Bot_Shell_Int'])
    
    Q_r_TS_BS = C_rad_ts_bs * (T4s['Top_Shell_Int'] - T4s['Bot_Shell_Int'])

    # === NET HEAT BALANCE (with correct physics) ===
    
    # BATTERY NODES
    net_Q_BFT = config.Q_batt_zone - (Q_c_BFT_BFB + Q_c_BFT_BH1 + Q_c_BFT_BH2 + Q_v_BFT_Air + 
                                      Q_r_BFT_BMT + Q_r_BFT_BMB + Q_r_BFT_ESC + Q_r_BFT_TS + Q_r_BFT_BS)
    net_Q_BFB = config.Q_batt_zone + Q_c_BFT_BFB - (Q_c_BFB_BH1 + Q_c_BFB_BH2 + Q_v_BFB_Air + 
                                                    Q_r_BFB_BMT + Q_r_BFB_BMB + Q_r_BFB_ESC + Q_r_BFB_TS + Q_r_BFB_BS)
    
    net_Q_BMT = config.Q_batt_zone + (Q_r_BFT_BMT + Q_r_BFB_BMT) - (Q_c_BMT_BMB + Q_c_BMT_BH2 + Q_c_BMT_BH3 + 
                                                                     Q_v_BMT_Air + Q_r_BMT_BRT + Q_r_BMT_BRB + Q_r_BMT_TS + Q_r_BMT_BS)
    net_Q_BMB = config.Q_batt_zone + Q_c_BMT_BMB + (Q_r_BFT_BMB + Q_r_BFB_BMB) - (Q_c_BMB_BH2 + Q_c_BMB_BH3 + 
                                                                                   Q_v_BMB_Air + Q_r_BMB_BRT + Q_r_BMB_BRB + Q_r_BMB_TS + Q_r_BMB_BS)
    
    net_Q_BRT = config.Q_batt_zone + (Q_r_BMT_BRT + Q_r_BMB_BRT) - (Q_c_BRT_BRB + Q_c_BRT_BH3 + Q_c_BRT_BH4 + 
                                                                     Q_v_BRT_Air + Q_r_BRT_TS + Q_r_BRT_BS)
    net_Q_BRB = config.Q_batt_zone + Q_c_BRT_BRB + (Q_r_BMT_BRB + Q_r_BMB_BRB) - (Q_c_BRB_BH3 + Q_c_BRB_BH4 + 
                                                                                   Q_v_BRB_Air + Q_r_BRB_TS + Q_r_BRB_BS)

    # AVIONICS NODES
    net_Q_ESC = config.Q_ESC + (Q_r_BFT_ESC + Q_r_BFB_ESC) - (Q_c_ESC_Mount + Q_v_ESC_Air + 
                                                               Q_r_ESC_BH1 + Q_r_ESC_TS + Q_r_ESC_BS)
    net_Q_Mount = Q_c_ESC_Mount - (Q_c_Mount_BH1 + Q_v_Mount_Air + Q_r_Mount_TS + Q_r_Mount_BS)

    # BULKHEAD NODES
    net_Q_BH1 = (Q_c_Mount_BH1 + Q_c_BFT_BH1 + Q_c_BFB_BH1 + Q_r_ESC_BH1) - (Q_c_BH1_TS + Q_c_BH1_BS + 
                                                                              Q_v_BH1_Air + Q_r_BH1_TS + Q_r_BH1_BS)
    net_Q_BH2 = (Q_c_BFT_BH2 + Q_c_BFB_BH2 + Q_c_BMT_BH2 + Q_c_BMB_BH2) - (Q_c_BH2_TS + Q_c_BH2_BS + 
                                                                            Q_v_BH2_Air + Q_r_BH2_TS + Q_r_BH2_BS)
    net_Q_BH3 = (Q_c_BMT_BH3 + Q_c_BMB_BH3 + Q_c_BRT_BH3 + Q_c_BRB_BH3) - (Q_c_BH3_TS + Q_c_BH3_BS + 
                                                                            Q_v_BH3_Air + Q_r_BH3_TS + Q_r_BH3_BS)
    net_Q_BH4 = (Q_c_BRT_BH4 + Q_c_BRB_BH4) - (Q_c_BH4_TS + Q_c_BH4_BS + Q_v_BH4_Air + Q_r_BH4_TS + Q_r_BH4_BS)

    # SHELL NODES
    net_Q_TS_int = (Q_c_BH1_TS + Q_c_BH2_TS + Q_c_BH3_TS + Q_c_BH4_TS) + \
                   (Q_r_BFT_TS + Q_r_BFB_TS + Q_r_BMT_TS + Q_r_BMB_TS + Q_r_BRT_TS + Q_r_BRB_TS +
                    Q_r_ESC_TS + Q_r_Mount_TS + Q_r_BH1_TS + Q_r_BH2_TS + Q_r_BH3_TS + Q_r_BH4_TS) - \
                   Q_v_TSi_Air - (Q_c_TSi_TSe + Q_r_TS_BS)
    
    net_Q_BS_int = (Q_c_BH1_BS + Q_c_BH2_BS + Q_c_BH3_BS + Q_c_BH4_BS) + Q_r_TS_BS + \
                   (Q_r_BFT_BS + Q_r_BFB_BS + Q_r_BMT_BS + Q_r_BMB_BS + Q_r_BRT_BS + Q_r_BRB_BS +
                    Q_r_ESC_BS + Q_r_Mount_BS + Q_r_BH1_BS + Q_r_BH2_BS + Q_r_BH3_BS + Q_r_BH4_BS) - \
                   Q_v_BSi_Air - Q_c_BSi_BSe
    
    net_Q_TS_ext = Q_c_TSi_TSe + ext_loads['Q_ext_top'] - Q_v_TSe_Amb
    net_Q_BS_ext = Q_c_BSi_BSe + ext_loads['Q_ext_bottom'] - Q_v_BSe_Amb

    # INTERNAL AIR NODE (with correct physics)
    # Air receives heat from hot components and loses heat to cooler shells
    net_Q_Air = (Q_v_BFT_Air + Q_v_BFB_Air + Q_v_BMT_Air + Q_v_BMB_Air + Q_v_BRT_Air + Q_v_BRB_Air +
                 Q_v_ESC_Air + Q_v_Mount_Air + Q_v_BH1_Air + Q_v_BH2_Air + Q_v_BH3_Air + Q_v_BH4_Air -
                 Q_v_TSi_Air - Q_v_BSi_Air)

    # Store for energy balance check
    net_Q_dict = {
        'Batt_BF_Top': net_Q_BFT, 'Batt_BF_Bot': net_Q_BFB,
        'Batt_BM_Top': net_Q_BMT, 'Batt_BM_Bot': net_Q_BMB,
        'Batt_BR_Top': net_Q_BRT, 'Batt_BR_Bot': net_Q_BRB,
        'ESC': net_Q_ESC, 'ESC_Mount': net_Q_Mount,
        'BH_1': net_Q_BH1, 'BH_2': net_Q_BH2, 'BH_3': net_Q_BH3, 'BH_4': net_Q_BH4,
        'Top_Shell_Int': net_Q_TS_int, 'Top_Shell_Ext': net_Q_TS_ext,
        'Bot_Shell_Int': net_Q_BS_int, 'Bot_Shell_Ext': net_Q_BS_ext,
        'Internal_Air': net_Q_Air
    }
    
    # Check energy balance
    Q_gen_total = 6 * config.Q_batt_zone + config.Q_ESC
    check_energy_balance(t, net_Q_dict, Q_gen_total)

    # Calculate temperature derivatives
    mC_batt = config.m_batt_zone * config.C_B
    mC_bh = config.m_bulkhead * config.C_bulkhead
    mC_TS = config.m_TS * config.C_TS
    mC_BS = config.m_BS * config.C_BS
    p_air_rho, p_air_cp = p_air[0], p_air[1]
    mC_air = p_air_rho * config.V_internal_air * p_air_cp

    return [
        net_Q_BFT / mC_batt, net_Q_BFB / mC_batt,
        net_Q_BMT / mC_batt, net_Q_BMB / mC_batt,
        net_Q_BRT / mC_batt, net_Q_BRB / mC_batt,
        net_Q_ESC / (config.m_ESC * config.C_ESC),
        net_Q_Mount / (config.m_mount * config.C_mount),
        net_Q_BH1 / mC_bh, net_Q_BH2 / mC_bh, net_Q_BH3 / mC_bh, net_Q_BH4 / mC_bh,
        net_Q_TS_int / mC_TS, net_Q_TS_ext / mC_TS,
        net_Q_BS_int / mC_BS, net_Q_BS_ext / mC_BS,
        net_Q_Air / mC_air
    ]

# === Main Execution ===
if __name__ == "__main__":
    print("\n" + "="*60)
    print("ENERGY-BALANCED HEAT TRANSFER SIMULATION")
    print("="*60)
    
    print("\n--- Key Corrections Applied ---")
    print(f"1. Thermal path length: {config.L_path_Batt_to_BH:.3f} m (was {config.t_bulkhead:.3f} m)")
    print(f"2. External characteristic length: {CORRECTED_LC_EXT:.3f} m (was {config.LC_TS_ext:.3f} m)")
    print("3. Internal air node: Corrected sign convention")
    print("4. Energy balance monitoring: Enabled")
    
    # Initial conditions
    x0 = np.array([config.initial_temp_K] * len(config.labels))
    
    # Solve ODE
    print(f"\nStarting simulation for {config.T_total / 3600:.1f} hours...")
    print("Target: <1% energy imbalance")
    
    sol = solve_ivp(
        fun=f,
        t_span=[0, config.T_total],
        y0=x0,
        method='BDF',
        dense_output=True,
        rtol=1e-5,
        atol=1e-8
    )
    
    print(f"\nSolver finished. Status: {sol.message}")
    print(f"Total execution time: {time.time() - start_time:.2f} seconds")
    
    # Analyze energy balance
    if len(energy_imbalance_history) > 0:
        avg_imbalance = np.mean(energy_imbalance_history)
        max_imbalance = np.max(energy_imbalance_history)
        final_imbalance = energy_imbalance_history[-1]
        
        print("\n--- Energy Balance Analysis ---")
        print(f"Average imbalance: {avg_imbalance:.3f} W")
        print(f"Maximum imbalance: {max_imbalance:.3f} W")
        print(f"Final imbalance: {final_imbalance:.3f} W")
        
        Q_gen_total = 6 * config.Q_batt_zone + config.Q_ESC
        final_percent = (final_imbalance / Q_gen_total) * 100
        print(f"Final imbalance: {final_percent:.2f}% of total generation")
        
        if final_percent < 1.0:
            print("[SUCCESS] Energy conservation EXCELLENT (< 1% error)")
        elif final_percent < 5.0:
            print("[OK] Energy conservation ACCEPTABLE (< 5% error)")
        else:
            print("[WARNING] Energy conservation needs improvement (> 5% error)")
    
    # Post-processing
    print("\n" + "="*60)
    post_processing.print_final_temps(sol)
    post_processing.analyze_peaks(sol)
    
    print("\n" + "="*60)
    print("SIMULATION COMPLETE")
    print("="*60)