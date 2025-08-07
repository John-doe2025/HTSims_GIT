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
#All the files are imported now we will work on all predefined variables refer to the excel for amy confusion in heat transfer paths
#Conduction and radiation variables stay constant throughout but they change for convection so that will be placed inside the f(t,x)

# ---------------------ESC---------------------------
# Radiation 
C_ESC_TS_rad = physics_models.rad_coeff(config.emis_esc, config.emis_shell_int, config.A_ESC_rad, config.A_TS, vf = 1)
C_ESC_BS_rad = physics_models.rad_coeff(config.emis_esc, config.emis_shell_int, config.A_ESC_rad, config.A_BS, vf = 1)
C_ESC_BF_Top_rad = physics_models.rad_coeff(config.emis_esc, config.emis_batt, config.A_ESC_rad, config.A_conv_batt_end, vf=1) 
C_ESC_BF_Bot_rad = physics_models.rad_coeff(config.emis_esc, config.emis_batt, config.A_ESC_rad, config.A_conv_batt_end, vf=1)
C_ESC_BH1_rad = physics_models.rad_coeff(config.emis_esc, config.emis_bulkhead, config.A_ESC_rad,config.A_bulkhead_face, vf=1)

#Conduction
C_cond_ESC_to_Mount = config.k_mount * config.A_contact_ESC_Mount / config.L_path_ESC_Mount

#------------------ESC Mount----------------------------
#Radiation
C_Mount_TS_rad = physics_models.rad_coeff(config.emis_mount, config.emis_shell_int, config.A_mount_rad, config.A_TS)
C_Mount_BS_rad = physics_models.rad_coeff(config.emis_mount, config.emis_shell_int, config.A_mount_rad, config.A_BS)

#Conduction
C_cond_Mount_to_BH1 = config.k_mount * config.A_contact_Mount_BH1 / config.L_path_ESC_Mount

#------------------- Battery Front Top-------------------
#Radiation
C_BFT_BMT_rad =  physics_models.rad_coeff(config.emis_batt, config.emis_batt, config.A_rad_batt_to_batt, config.A_rad_batt_to_batt, vf = 1)
C_BFT_BMB_rad =  physics_models.rad_coeff(config.emis_batt, config.emis_batt, config.A_rad_batt_to_batt, config.A_rad_batt_to_batt, vf = 1)
C_BFT_TS_rad = physics_models.rad_coeff(config.emis_batt, config.emis_shell_int, config.A_rad_batt_to_shell, config.A_TS, vf = 1)
C_BFT_BS_rad = physics_models.rad_coeff(config.emis_batt, config.emis_shell_int, config.A_rad_batt_to_shell, config.A_BS, vf = 1)

#Conduction
C_cond_BFT_BH2 = config.k_bulkhead * config.A_contact_Batt_BH / config.L_path_Batt_BH
C_cond_BFT_BFB = config.k_BattPack * config.A_contact_batt_batt / config.H_batt_zone





def f(t, x):
    temps = {label: x[i] for i, label in enumerate(config.labels)}
    T4s = {k: physics_models.T_power4(v) for k, v in temps.items()}
    p_air = physics_models.prop_internal_air(temps['Internal Air'], P_amb)
    
    # --- External Heat Loads (from environment_model) ---
    ext_loads = environment_model.calculate_external_heat_loads(t, temps['Top Shell External'], temps['Bottom Shell External'], T_E)

    # --- A. ESC Calculations ---
    # 1. Conduction from ESC to Mount
    Q_cond_ESC_to_Mount = C_cond_ESC_to_Mount * (temps['ESC'] - temps['ESC_Mount'])
    
    # 2. Convection from ESC to Internal Air
    p_esc_film = physics_models.prop_internal_air((temps['ESC'] + temps['Internal_Air'])/2, P_amb)
    h_conv_esc = physics_models.natural_convection_h(p_esc_film, temps['ESC'], temps['Internal_Air'], config.LC_ESC, False)
    Q_conv_ESC_air = h_conv_esc * config.A_ESC_conv * (temps['ESC'] - temps['Internal_Air'])
    
    # 3. Radiation from ESC to other surfaces (Shells AND front Batteries)
    Q_rad_ESC_to_TS = C_ESC_TS_rad * (T4s['ESC'] - T4s['Top_Shell_Int'])
    Q_rad_ESC_to_BS = C_ESC_BS_rad * (T4s['ESC'] - T4s['Bot_Shell_Int'])
    Q_rad_ESC_to_BFT = C_ESC_BF_Top_rad * (T4s['ESC'] - T4s['Batt_BF_Top'])
    Q_rad_ESC_to_BFB = C_ESC_BF_Bot_rad * (T4s['ESC'] - T4s['Batt_BF_Bot'])
    Q_rad_ESC_to_BH1 = C_ESC_BH1_rad * (T4s['ESC'] - T4s['BH_1'])
    # --- Heat Balance Definition for Final Derivative Calculation ---
    
    # Total heat flowing IN to the ESC
    Q_in_ESC = config.Q_ESC

    # Total heat flowing OUT of the ESC
    Q_out_ESC = Q_cond_ESC_to_Mount + Q_conv_ESC_air + Q_rad_ESC_to_TS + Q_rad_ESC_to_BS + Q_rad_ESC_to_BFT + Q_rad_ESC_to_BFB + Q_rad_ESC_to_BH1

    #------ B. Mount Calculations ----------
    # Conduction from mount to Bulkhead 1
    Q_cond_Mount_to_BH1 = C_cond_Mount_to_BH1 * (temps['ESC_Mount'] - temps['BH_1'])

    # Convection from Esc Mount to Internal Air
    p_esc_film = physics_models.prop_internal_air((temps['ESC_Mount'] + temps['Internal_Air'])/2, P_amb)
    h_conv_mount = physics_models.natural_convection_h(p_esc_film, temps['ESC_Mount'], temps['Internal_Air'], config.LC_mount, False)
    Q_conv_Mount_air = h_conv_mount * config.A_mount_conv * (temps['ESC_Mount'] - temps['Internal_Air'])

    #Radiation from ESC Mount to Shell 
    Q_rad_Mount_to_TS = C_Mount_TS_rad * (T4s['ESC_Mount'] - T4s['Top_Shell_Int'])
    Q_rad_Mount_to_BS = C_Mount_BS_rad * (T4s['ESC_Mount'] - T4s['Bot_Shell_Int'])

    #Total heat flowing in ESC_Mount
    Q_in_Mount = Q_cond_ESC_to_Mount

    #Total heat flowing out of mount
    Q_out_Mount = Q_cond_Mount_to_BH1 + Q_conv_Mount_air + Q_rad_Mount_to_BS + Q_rad_Mount_to_TS

    #------ C. BFT Calculations ----------

    # Convection from BFT to internal air
    p_bft_film = physics_models.prop_internal_air((temps['Batt_BF_Top'] + temps['Internal_Air'])/2, P_amb)
    h_bft_top = physics_models.natural_convection_h(p_bft_film, temps['Batt_BF_Top'], temps['Internal_Air'], config.LC_batt_horiz, False)
    h_bft_ends = physics_models.natural_convection_h(p_bft_film, temps['Batt_BF_Top'], temps['Internal_Air'], config.LC_batt_vert, True)

    h_avg_bft = ((h_bft_top * config.A_conv_batt_top) + (h_bft_ends * config.A_conv_batt_side))/(config.A_conv_batt_total)
    Q_conv_bft_air = h_avg_bft * config.A_conv_batt_total * (temps['Batt_BF_Top'] - temps['Internal Air'])
    
    # Conduction from BFT to BFB and Bulkhead 2

    Q_cond_bft_bfb = C_cond_BFT_BFB * (temps['Batt_BF_Top'] - temps['Batt_BF_Bot'])
    Q_cond_bft_BH2 = C_cond_BFT_BH2 * (temps['Batt_BF_Top'] - temps['BH_2'])

    # Radiation to bmt bmb and shell

    Q_bft_bmt_rad = C_BFT_BMT_rad * (T4s['Batt_BF_Top'] - T4s['Batt_BM_Top'])
    Q_bft_bmb_rad = C_BFT_BMB_rad * (T4s['Batt_BF_Top'] - T4s['Batt_BM_Bot'])
    Q_bft_ts_rad = C_BFT_TS_rad * (T4s['Batt_BF_Top'] - T4s['Top_Shell_Int'])
    Q_bft_bs_rad = C_BFT_BS_rad * (T4s['Batt_BF_Top'] - T4s['Bot_Shell_Int'])

    # Total Heat flowing in BFT
    Q_in_BFT = config.Q_batt_zone + Q_rad_ESC_to_BFT #+ Q_cond_BH1_BFT

    #Total heat flowing out of BFT
    Q_out_BFT = Q_conv_bft_air + Q_cond_bft_bfb + Q_cond_bft_BH2 + Q_bft_bs_rad + Q_bft_ts_rad + Q_bft_bmt_rad + Q_bft_bmb_rad