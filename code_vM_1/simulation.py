# 6_main_simulation.py (Final, Rigorously Checked Version)

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
    T_E = row['Temperature']; P_amb = row['Pressure']
    p_ambient = physics_models.prop_internal_air(T_E, P_amb)
    print(f"Set environment for {row['Altitude']} km altitude.")
except Exception as e:
    raise SystemExit(f"FATAL ERROR loading altitude_data.xlsx: {e}")

# --- Pre-Solver Thermal Coefficient Calculations ---
print("Calculating thermal coefficients...")
L_path_Batt_to_BH = config.L_batt_zone/2
L_path_Mount_to_BH = config.LC_ESC/2
L_path_BH_to_Shell = config.LC_TS_int/2

C_cond_ESC_to_Mount = config.k_mount*config.A_contact_ESC_Mount/config.L_path_ESC_Mount
C_cond_Mount_to_BH1 = config.k_bulkhead*config.A_contact_Mount_BH1/L_path_Mount_to_BH
C_cond_Batt_Top_Bot = config.k_eff_batt*(config.W_batt_zone*config.L_batt_zone)/config.H_batt_zone

#if the area turns out to be same merge into one
C_cond_BFT_BH1 =  config.k_bulkhead*config.A_contact_Batt_BH/L_path_Batt_to_BH
C_cond_BFB_BH1 =  config.k_bulkhead*config.A_contact_Batt_BH/L_path_Batt_to_BH
C_cond_BFT_BH2 =  config.k_bulkhead*config.A_contact_Batt_BH/L_path_Batt_to_BH
C_cond_BFB_BH2 =  config.k_bulkhead*config.A_contact_Batt_BH/L_path_Batt_to_BH
C_cond_BMT_BH2 =  config.k_bulkhead*config.A_contact_Batt_BH/L_path_Batt_to_BH
C_cond_BMB_BH2 =  config.k_bulkhead*config.A_contact_Batt_BH/L_path_Batt_to_BH
C_cond_BMT_BH3 =  config.k_bulkhead*config.A_contact_Batt_BH/L_path_Batt_to_BH
C_cond_BMB_BH3 =  config.k_bulkhead*config.A_contact_Batt_BH/L_path_Batt_to_BH
C_cond_BRT_BH3 =  config.k_bulkhead*config.A_contact_Batt_BH/L_path_Batt_to_BH
C_cond_BRB_BH3 =  config.k_bulkhead*config.A_contact_Batt_BH/L_path_Batt_to_BH
C_cond_BRT_BH4 =  config.k_bulkhead*config.A_contact_Batt_BH/L_path_Batt_to_BH
C_cond_BRB_BH4 =  config.k_bulkhead*config.A_contact_Batt_BH/L_path_Batt_to_BH

C_cond_BH_TS = config.k_cfrp*config.A_contact_BH_Shell/L_path_BH_to_Shell
C_cond_BH_BS = config.k_cfrp*config.A_contact_BH_Shell/L_path_BH_to_Shell

C_cond_TS_int_ext = config.k_cfrp*config.A_TS/config.t_cfrp
C_cond_BS_int_ext = config.k_cfrp*config.A_BS/config.t_cfrp


C_rad_batt_batt = physics_models.rad_coeff(config.emis_batt,config.emis_batt,config.A_rad_batt_to_batt,config.A_rad_batt_to_batt)
C_rad_batt_esc = physics_models.rad_coeff(config.emis_esc,config.emis_batt,config.A_ESC_conv,config.A_rad_batt_to_batt)
C_rad_batt_ts = physics_models.rad_coeff(config.emis_batt,config.emis_shell_int,config.A_rad_batt_to_shell,config.A_TS)
C_rad_batt_bs = physics_models.rad_coeff(config.emis_batt,config.emis_shell_int,config.A_rad_batt_to_shell,config.A_BS)

C_rad_esc_bh = physics_models.rad_coeff(config.emis_esc,config.emis_bulkhead,config.A_ESC_conv,config.A_bulkhead_face)
C_rad_esc_ts = physics_models.rad_coeff(config.emis_esc,config.emis_shell_int,config.A_ESC_conv,config.A_TS)
C_rad_esc_bs = physics_models.rad_coeff(config.emis_esc,config.emis_shell_int,config.A_ESC_conv,config.A_BS,vf=0.3)

C_rad_mount_ts = physics_models.rad_coeff(config.emis_mount,config.emis_shell_int,config.A_mount_conv,config.A_TS,vf=0.3)
C_rad_mount_bs = physics_models.rad_coeff(config.emis_mount,config.emis_shell_int,config.A_mount_conv,config.A_BS)

C_rad_bh_ts = physics_models.rad_coeff(config.emis_bulkhead,config.emis_shell_int,config.A_bulkhead_face,config.A_TS)
C_rad_bh_bs = physics_models.rad_coeff(config.emis_bulkhead,config.emis_shell_int,config.A_bulkhead_face,config.A_BS)
C_rad_bh_bh = physics_models.rad_coeff(config.emis_bulkhead,config.emis_bulkhead,config.A_bulkhead_face,config.A_bulkhead_face)

C_rad_ts_bs = physics_models.rad_coeff(config.emis_shell_int,config.emis_shell_int,config.A_TS,config.A_BS,vf=0.5)

def f(t, x):
    temps={label:x[i] for i, label in enumerate(config.labels)}
    T4s={k:physics_models.T_power4(v) for k,v in temps.items()}
    p_air = physics_models.prop_internal_air(temps['Internal_Air'], P_amb)
    ext_loads=environment_model.calculate_external_heat_loads(t,temps['Top_Shell_Ext'],temps['Bot_Shell_Ext'], T_E)

    # 1. CONDUCTION
    Q_c_BFT_BFB = C_cond_Batt_Top_Bot*(temps['Batt_BF_Top']-temps['Batt_BF_Bot'])
    Q_c_BMT_BMB = C_cond_Batt_Top_Bot*(temps['Batt_BM_Top']-temps['Batt_BM_Bot'])
    Q_c_BRT_BRB = C_cond_Batt_Top_Bot*(temps['Batt_BR_Top']-temps['Batt_BR_Bot'])

    Q_c_BFT_BH1 = C_cond_BFT_BH1*(temps['BH_1']-temps['Batt_BF_Top'])
    Q_c_BFB_BH1 = C_cond_BFB_BH1*(temps['BH_1']-temps['Batt_BF_Bot'])
    Q_c_BFT_BH2 = C_cond_BFT_BH2*(temps['Batt_BF_Top']-temps['BH_2'])
    Q_c_BFB_BH2 = C_cond_BFB_BH2*(temps['Batt_BF_Bot']-temps['BH_2'])
    Q_c_BMT_BH2 = C_cond_BMT_BH2*(temps['Batt_BM_Top']-temps['BH_2'])
    Q_c_BMB_BH2 = C_cond_BMB_BH2*(temps['Batt_BM_Bot']-temps['BH_2'])
    Q_c_BMT_BH3 = C_cond_BMT_BH3*(temps['Batt_BM_Top']-temps['BH_3'])
    Q_c_BMB_BH3 = C_cond_BMB_BH3*(temps['Batt_BM_Bot']-temps['BH_3'])
    Q_c_BRT_BH3 = C_cond_BRT_BH3*(temps['Batt_BR_Top']-temps['BH_3'])
    Q_c_BRB_BH3 = C_cond_BRB_BH3*(temps['Batt_BR_Bot']-temps['BH_3'])
    Q_c_BRT_BH4 = C_cond_BRT_BH4*(temps['Batt_BR_Top']-temps['BH_4'])
    Q_c_BRB_BH4 = C_cond_BRB_BH4*(temps['Batt_BR_Bot']-temps['BH_4'])

    Q_c_ESC_Mount = C_cond_ESC_to_Mount*(temps['ESC']-temps['ESC_Mount'])
    Q_c_Mount_BH1 = C_cond_Mount_to_BH1*(temps['ESC_Mount']-temps['BH_1'])

    Q_c_BH1_TS = C_cond_BH_TS*(temps['BH_1']-temps['Top_Shell_Int'])
    Q_c_BH2_TS = C_cond_BH_TS*(temps['BH_2']-temps['Top_Shell_Int'])
    Q_c_BH3_TS = C_cond_BH_TS*(temps['BH_3']-temps['Top_Shell_Int'])
    Q_c_BH4_TS = C_cond_BH_TS*(temps['BH_4']-temps['Top_Shell_Int'])
    Q_c_BH1_BS = C_cond_BH_BS*(temps['BH_1']-temps['Bot_Shell_Int'])
    Q_c_BH2_BS = C_cond_BH_BS*(temps['BH_2']-temps['Bot_Shell_Int'])
    Q_c_BH3_BS = C_cond_BH_BS*(temps['BH_3']-temps['Bot_Shell_Int'])
    Q_c_BH4_BS = C_cond_BH_BS*(temps['BH_4']-temps['Bot_Shell_Int'])

    Q_c_TSi_TSe = C_cond_TS_int_ext*(temps['Top_Shell_Int']-temps['Top_Shell_Ext'])
    Q_c_BSi_BSe = C_cond_BS_int_ext*(temps['Bot_Shell_Int']-temps['Bot_Shell_Ext'])

    # 2. CONVECTION
    def get_h(T_s,LC,is_v): return physics_models.natural_convection_h(physics_models.prop_internal_air((T_s+temps['Internal_Air'])/2, P_amb), T_s, temps['Internal_Air'], LC, is_v)
    Q_v_BFT_Air = (get_h(temps['Batt_BF_Top'],config.LC_batt_horiz,False)*config.A_conv_batt_top+get_h(temps['Batt_BF_Top'],config.LC_batt_vert,True)*config.A_conv_batt_side)*(temps['Batt_BF_Top']-temps['Internal_Air'])
    Q_v_BFB_Air = (get_h(temps['Batt_BF_Bot'],config.LC_batt_horiz,False)*config.A_conv_batt_top+get_h(temps['Batt_BF_Bot'],config.LC_batt_vert,True)*config.A_conv_batt_side)*(temps['Batt_BF_Bot']-temps['Internal_Air'])
    Q_v_BMT_Air = (get_h(temps['Batt_BM_Top'],config.LC_batt_horiz,False)*config.A_conv_batt_top+get_h(temps['Batt_BM_Top'],config.LC_batt_vert,True)*config.A_conv_batt_side)*(temps['Batt_BM_Top']-temps['Internal_Air'])
    Q_v_BMB_Air = (get_h(temps['Batt_BM_Bot'],config.LC_batt_horiz,False)*config.A_conv_batt_top+get_h(temps['Batt_BM_Bot'],config.LC_batt_vert,True)*config.A_conv_batt_side)*(temps['Batt_BM_Bot']-temps['Internal_Air'])
    Q_v_BRT_Air = (get_h(temps['Batt_BR_Top'],config.LC_batt_horiz,False)*config.A_conv_batt_top+get_h(temps['Batt_BR_Top'],config.LC_batt_vert,True)*config.A_conv_batt_side)*(temps['Batt_BR_Top']-temps['Internal_Air'])
    Q_v_BRB_Air = (get_h(temps['Batt_BR_Bot'],config.LC_batt_horiz,False)*config.A_conv_batt_top+get_h(temps['Batt_BR_Bot'],config.LC_batt_vert,True)*config.A_conv_batt_side)*(temps['Batt_BR_Bot']-temps['Internal_Air'])
    
    Q_v_ESC_Air = get_h(temps['ESC'],config.LC_ESC,False)*config.A_ESC_conv*(temps['ESC']-temps['Internal_Air'])
    Q_v_Mount_Air = get_h(temps['ESC_Mount'],config.LC_mount,False)*config.A_mount_conv*(temps['ESC_Mount']-temps['Internal_Air'])
    
    Q_v_BH1_Air = get_h(temps['BH_1'],config.LC_bulkhead,True)*config.A_bulkhead_face*2*(temps['BH_1']-temps['Internal_Air'])
    Q_v_BH2_Air = get_h(temps['BH_2'],config.LC_bulkhead,True)*config.A_bulkhead_face*2*(temps['BH_2']-temps['Internal_Air'])
    Q_v_BH3_Air = get_h(temps['BH_3'],config.LC_bulkhead,True)*config.A_bulkhead_face*2*(temps['BH_3']-temps['Internal_Air'])
    Q_v_BH4_Air = get_h(temps['BH_4'],config.LC_bulkhead,True)*config.A_bulkhead_face*2*(temps['BH_4']-temps['Internal_Air'])
    
    Q_v_TSi_Air = get_h(temps['Top_Shell_Int'],config.LC_TS_int,False)*config.A_TS*(temps['Internal_Air']-temps['Top_Shell_Int'])
    Q_v_BSi_Air = get_h(temps['Bot_Shell_Int'],config.LC_BS_int,False)*config.A_BS*(temps['Internal_Air']-temps['Bot_Shell_Int'])
    Q_v_TSe_Amb = physics_models.get_external_convection_h(p_ambient,temps['Top_Shell_Ext'],T_E,config.LC_TS_ext)*config.A_TS*(temps['Top_Shell_Ext']-T_E)
    Q_v_BSe_Amb = physics_models.get_external_convection_h(p_ambient,temps['Bot_Shell_Ext'],T_E,config.LC_BS_ext)*config.A_BS*(temps['Bot_Shell_Ext']-T_E)

    # 3. RADIATION
    Q_r_BFT_BMT = C_rad_batt_batt*(T4s['Batt_BF_Top']-T4s['Batt_BM_Top'])
    Q_r_BFT_BMB = C_rad_batt_batt*(T4s['Batt_BF_Top']-T4s['Batt_BM_Bot'])
    Q_r_BFT_ESC = C_rad_batt_esc*(T4s['ESC']-T4s['Batt_BF_Top'])
    Q_r_BFT_TS = C_rad_batt_ts*(T4s['Batt_BF_Top']-T4s['Top_Shell_Int'])
    Q_r_BFT_BS = C_rad_batt_bs*(T4s['Batt_BF_Top']-T4s['Bot_Shell_Int'])

    Q_r_BFB_BMT = C_rad_batt_batt*(T4s['Batt_BF_Bot']-T4s['Batt_BM_Top'])
    Q_r_BFB_BMB = C_rad_batt_batt*(T4s['Batt_BF_Bot']-T4s['Batt_BM_Bot'])
    Q_r_BFB_ESC = C_rad_batt_esc*(T4s['ESC']-T4s['Batt_BF_Bot'])
    Q_r_BFB_TS = C_rad_batt_ts*(T4s['Batt_BF_Bot']-T4s['Top_Shell_Int'])
    Q_r_BFB_BS = C_rad_batt_bs*(T4s['Batt_BF_Bot']-T4s['Bot_Shell_Int'])

    Q_r_BMT_BRT = C_rad_batt_batt*(T4s['Batt_BM_Top']-T4s['Batt_BR_Top'])
    Q_r_BMT_BRB = C_rad_batt_batt*(T4s['Batt_BM_Top']-T4s['Batt_BR_Bot'])
    Q_r_BMT_TS = C_rad_batt_ts*(T4s['Batt_BM_Top']-T4s['Top_Shell_Int'])
    Q_r_BMT_BS = C_rad_batt_bs*(T4s['Batt_BM_Top']-T4s['Bot_Shell_Int'])

    Q_r_BMB_BRT = C_rad_batt_batt*(T4s['Batt_BM_Bot']-T4s['Batt_BR_Top'])
    Q_r_BMB_BRB = C_rad_batt_batt*(T4s['Batt_BM_Bot']-T4s['Batt_BR_Bot'])
    Q_r_BMB_TS = C_rad_batt_ts*(T4s['Batt_BM_Bot']-T4s['Top_Shell_Int'])
    Q_r_BMB_BS = C_rad_batt_bs*(T4s['Batt_BM_Bot']-T4s['Bot_Shell_Int'])

    Q_r_BRT_TS = C_rad_batt_ts*(T4s['Batt_BR_Top']-T4s['Top_Shell_Int'])
    Q_r_BRT_BS = C_rad_batt_bs*(T4s['Batt_BR_Top']-T4s['Bot_Shell_Int'])

    Q_r_BRB_TS = C_rad_batt_ts*(T4s['Batt_BR_Bot']-T4s['Top_Shell_Int'])
    Q_r_BRB_BS = C_rad_batt_bs*(T4s['Batt_BR_Bot']-T4s['Bot_Shell_Int'])

    Q_r_ESC_BH1 = C_rad_esc_bh*(T4s['ESC']-T4s['BH_1'])
    Q_r_ESC_TS = C_rad_esc_ts*(T4s['ESC']-T4s['Top_Shell_Int'])
    Q_r_ESC_BS = C_rad_esc_bs*(T4s['ESC']-T4s['Bot_Shell_Int'])

    Q_r_Mount_TS = C_rad_mount_ts*(T4s['ESC_Mount']-T4s['Top_Shell_Int'])
    Q_r_Mount_BS = C_rad_mount_bs*(T4s['ESC_Mount']-T4s['Bot_Shell_Int'])

    Q_r_BH1_BH2 = C_rad_bh_bh*(T4s['BH_1']-T4s['BH_2'])
    Q_r_BH2_BH3 = C_rad_bh_bh*(T4s['BH_2']-T4s['BH_3'])
    Q_r_BH3_BH4 = C_rad_bh_bh*(T4s['BH_3']-T4s['BH_4'])

    Q_r_TS_BS = C_rad_ts_bs*(T4s['Top_Shell_Int']-T4s['Bot_Shell_Int'])

    # --- NET HEAT BALANCE: (All Heat IN) - (All Heat OUT) ---
    # Heat flows are defined as positive FROM first node TO second.
    # Therefore, Q_A_B is an outflow from A and an inflow to B.
    # net_Q = Generation + Inflow - Outflow
    net_Q_BFT = config.Q_batt_zone + (Q_c_BFT_BH1 + Q_r_BFT_ESC) - (Q_c_BFT_BFB + Q_c_BFT_BH2 + Q_v_BFT_Air + Q_r_BFT_BMT + Q_r_BFT_BMB + Q_r_BFT_TS + Q_r_BFT_BS)
    net_Q_BFB = config.Q_batt_zone + (Q_r_BFB_ESC + Q_c_BFT_BFB + Q_c_BFB_BH1) - (Q_c_BFB_BH2 + Q_v_BFB_Air + Q_r_BFB_BMT + Q_r_BFB_BMB + Q_r_BFB_TS + Q_r_BFB_BS)
    net_Q_BMT = config.Q_batt_zone + (Q_r_BFT_BMT + Q_r_BFB_BMT) - (Q_c_BMT_BMB + Q_c_BMT_BH2 + Q_c_BMT_BH3 + Q_v_BMT_Air + Q_r_BMT_BRT + Q_r_BMT_BRB + Q_r_BMT_TS + Q_r_BMT_BS)
    net_Q_BMB = config.Q_batt_zone + (Q_r_BFT_BMB + Q_r_BFB_BMB + Q_c_BMT_BMB) - (Q_c_BMB_BH3 + Q_c_BMB_BH2 + Q_v_BMB_Air + Q_r_BMB_BRT + Q_r_BMB_BRB + Q_r_BMB_TS + Q_r_BMB_BS)
    net_Q_BRT = config.Q_batt_zone + (Q_r_BMT_BRT + Q_r_BMB_BRT) - (Q_c_BRT_BRB + Q_c_BRT_BH3 + Q_c_BRT_BH4 + Q_v_BRT_Air + Q_r_BRT_TS + Q_r_BRT_BS)
    net_Q_BRB = config.Q_batt_zone + (Q_r_BMT_BRB + Q_r_BMB_BRB + Q_c_BRT_BRB) - (Q_c_BRB_BH3 + Q_c_BRB_BH4 + Q_v_BRB_Air + Q_r_BRB_TS + Q_r_BRB_BS)

    # AVIONICS NODES
    net_Q_ESC = config.Q_ESC - (Q_r_BFT_ESC + Q_r_BFB_ESC + Q_c_ESC_Mount + Q_v_ESC_Air + Q_r_ESC_BH1 + Q_r_ESC_TS + Q_r_ESC_BS)
    net_Q_Mount = Q_c_ESC_Mount - (Q_c_Mount_BH1 + Q_v_Mount_Air + Q_r_Mount_TS + Q_r_Mount_BS)

    # BULKHEAD NODES
    net_Q_BH1 = (Q_c_Mount_BH1 + Q_r_ESC_BH1) - (Q_c_BH1_TS + Q_c_BH1_BS + Q_v_BH1_Air + Q_c_BFT_BH1 + Q_c_BFB_BH1 + Q_r_BH1_BH2)
    net_Q_BH2 = (Q_c_BFT_BH2 + Q_c_BFB_BH2 + Q_r_BH1_BH2 + Q_c_BMT_BH2 + Q_c_BMB_BH2 ) - (Q_c_BH2_TS + Q_c_BH2_BS + Q_v_BH2_Air + Q_r_BH2_BH3)
    net_Q_BH3 = (Q_c_BMT_BH3 + Q_c_BMB_BH3 + Q_c_BRT_BH3 + Q_c_BRB_BH3 + Q_r_BH2_BH3) - (Q_c_BH3_TS + Q_c_BH3_BS + Q_v_BH3_Air + Q_r_BH3_BH4)
    net_Q_BH4 = (Q_c_BRT_BH4 + Q_c_BRB_BH4 + Q_r_BH3_BH4) - (Q_c_BH4_TS + Q_c_BH4_BS + Q_v_BH4_Air)

    # SHELL NODES
    net_Q_TS_int = (Q_c_BH1_TS + Q_c_BH2_TS + Q_c_BH3_TS + Q_c_BH4_TS + Q_r_BFT_TS + Q_r_BFB_TS  + Q_r_BMT_TS + Q_r_BMB_TS + Q_r_BRT_TS + Q_r_BRB_TS + Q_r_ESC_TS + Q_r_Mount_TS + Q_v_TSi_Air) - (Q_c_TSi_TSe + Q_r_TS_BS)
    net_Q_BS_int = (Q_c_BH1_BS + Q_c_BH2_BS + Q_c_BH3_BS + Q_c_BH4_BS + Q_r_TS_BS + Q_r_BFT_BS + Q_r_BFB_BS + Q_r_BMT_BS + Q_r_BMB_BS + Q_r_BRT_BS + Q_r_BRB_BS + Q_r_ESC_BS + Q_r_Mount_BS + Q_v_BSi_Air) - (Q_c_BSi_BSe)
    net_Q_TS_ext = Q_c_TSi_TSe + ext_loads['Q_ext_top'] - Q_v_TSe_Amb
    net_Q_BS_ext = Q_c_BSi_BSe + ext_loads['Q_ext_bottom'] - Q_v_BSe_Amb

    # INTERNAL AIR NODE
    net_Q_Air = (Q_v_BFT_Air + Q_v_BFB_Air + Q_v_BMT_Air + Q_v_BMB_Air + Q_v_BRT_Air + Q_v_BRB_Air + Q_v_ESC_Air + Q_v_Mount_Air + Q_v_BH1_Air + Q_v_BH2_Air + Q_v_BH3_Air + Q_v_BH4_Air) - Q_v_TSi_Air - Q_v_BSi_Air
    
    
    '''# BATTERY NODES
    net_Q_BFT = config.Q_batt_zone - (Q_c_BFT_BFB + Q_c_BFT_BH2 + Q_c_BFT_BH1 + Q_r_BFT_ESC + Q_v_BFT_Air + Q_r_BFT_BMT + Q_r_BFT_BMB + Q_r_BFT_TS + Q_r_BFT_BS)
    net_Q_BFB = config.Q_batt_zone + Q_c_BFT_BFB - (Q_c_BFB_BH1 + Q_c_BFB_BH2 + Q_v_BFB_Air + Q_r_BFB_BMT + Q_r_BFB_BMB + Q_r_BFB_ESC + Q_r_BFB_TS + Q_r_BFB_BS)
    net_Q_BMT = config.Q_batt_zone + (Q_r_BFT_BMT + Q_r_BFB_BMT) - (Q_c_BMT_BMB + Q_c_BMT_BH2 + Q_c_BMT_BH3 + Q_v_BMT_Air + Q_r_BMT_BRT + Q_r_BMT_BRB + Q_r_BMT_TS + Q_r_BMT_BS)
    net_Q_BMB = config.Q_batt_zone + Q_c_BMT_BMB + (Q_r_BFT_BMB + Q_r_BFB_BMB) - (Q_c_BMB_BH2 + Q_c_BMB_BH3 + Q_v_BMB_Air + Q_r_BMB_BRT + Q_r_BMB_BRB + Q_r_BMB_TS + Q_r_BMB_BS)
    net_Q_BRT = config.Q_batt_zone + (Q_r_BMT_BRT + Q_r_BMB_BRT) - (Q_c_BRT_BRB + Q_c_BRT_BH3 + Q_c_BRT_BH4 + Q_v_BRT_Air + Q_r_BRT_TS + Q_r_BRT_BS)
    net_Q_BRB = config.Q_batt_zone + Q_c_BRT_BRB + (Q_r_BMT_BRB + Q_r_BMB_BRB) - (Q_c_BRB_BH3 + Q_c_BRB_BH4 + Q_v_BRB_Air + Q_r_BRB_TS + Q_r_BRB_BS)

    # AVIONICS NODES
    net_Q_ESC = config.Q_ESC + (Q_r_BFT_ESC + Q_r_BFB_ESC) - (Q_c_ESC_Mount + Q_v_ESC_Air + Q_r_ESC_BH1 + Q_r_ESC_TS + Q_r_ESC_BS)
    net_Q_Mount = Q_c_ESC_Mount - (Q_c_Mount_BH1 + Q_v_Mount_Air + Q_r_Mount_TS + Q_r_Mount_BS)

    # BULKHEAD NODES
    net_Q_BH1 = (Q_c_Mount_BH1 + Q_c_BFT_BH1 + Q_c_BFB_BH1 + Q_r_ESC_BH1) - (Q_c_BH1_TS + Q_c_BH1_BS + Q_v_BH1_Air + Q_r_BH1_TS + Q_r_BH1_BS)
    net_Q_BH2 = (Q_c_BFT_BH2 + Q_c_BFB_BH2 + Q_c_BMT_BH2 + Q_c_BMB_BH2) - (Q_c_BH2_TS + Q_c_BH2_BS + Q_v_BH2_Air + Q_r_BH2_TS + Q_r_BH2_BS)
    net_Q_BH3 = (Q_c_BMT_BH3 + Q_c_BMB_BH3 + Q_c_BRT_BH3 + Q_c_BRB_BH3) - (Q_c_BH3_TS + Q_c_BH3_BS + Q_v_BH3_Air + Q_r_BH3_TS + Q_r_BH3_BS)
    net_Q_BH4 = (Q_c_BRT_BH4 + Q_c_BRB_BH4) - (Q_c_BH4_TS + Q_c_BH4_BS + Q_v_BH4_Air + Q_r_BH4_TS + Q_r_BH4_BS)

    # SHELL NODES
    net_Q_TS_int = (Q_c_BH1_TS+Q_c_BH2_TS+Q_c_BH3_TS+Q_c_BH4_TS) + (Q_r_BFT_TS+Q_r_BFB_TS+Q_r_BMT_TS+Q_r_BMB_TS+Q_r_BRT_TS+Q_r_BRB_TS+Q_r_ESC_TS+Q_r_Mount_TS+Q_r_BH1_TS+Q_r_BH2_TS+Q_r_BH3_TS+Q_r_BH4_TS) - Q_v_TSi_Air - (Q_c_TSi_TSe + Q_r_TS_BS)
    net_Q_BS_int = (Q_c_BH1_BS+Q_c_BH2_BS+Q_c_BH3_BS+Q_c_BH4_BS) + Q_r_TS_BS + (Q_r_BFT_BS+Q_r_BFB_BS+Q_r_BMT_BS+Q_r_BMB_BS+Q_r_BRT_BS+Q_r_BRB_BS+Q_r_ESC_BS+Q_r_Mount_BS+Q_r_BH1_BS+Q_r_BH2_BS+Q_r_BH3_BS+Q_r_BH4_BS) - Q_v_BSi_Air - (Q_c_BSi_BSe)
    net_Q_TS_ext = Q_c_TSi_TSe + ext_loads['Q_ext_top'] - Q_v_TSe_Amb
    net_Q_BS_ext = Q_c_BSi_BSe + ext_loads['Q_ext_bottom'] - Q_v_BSe_Amb

    # INTERNAL AIR NODE
    net_Q_Air = (Q_v_BFT_Air + Q_v_BFB_Air + Q_v_BMT_Air + Q_v_BMB_Air + Q_v_BRT_Air + Q_v_BRB_Air + Q_v_ESC_Air + Q_v_Mount_Air + Q_v_BH1_Air + Q_v_BH2_Air + Q_v_BH3_Air + Q_v_BH4_Air) - Q_v_TSi_Air - Q_v_BSi_Air
    '''
    # dT/dt
    mC_batt = config.m_batt_zone*config.C_B 
    mC_bh = config.m_bulkhead*config.C_bulkhead
    mC_TS = config.m_TS*config.C_TS
    mC_BS = config.m_BS*config.C_BS
    p_air_rho, p_air_cp = p_air[0], p_air[1]

    return [
        net_Q_BFT/mC_batt,net_Q_BFB/mC_batt,net_Q_BMT/mC_batt,net_Q_BMB/mC_batt,net_Q_BRT/mC_batt,net_Q_BRB/mC_batt,net_Q_ESC/(config.m_ESC*config.C_ESC),net_Q_Mount/(config.m_mount*config.C_mount),net_Q_BH1/mC_bh,net_Q_BH2/mC_bh,net_Q_BH3/mC_bh,net_Q_BH4/mC_bh,net_Q_TS_int/mC_TS,net_Q_TS_ext/mC_TS,net_Q_BS_int/mC_BS,net_Q_BS_ext/mC_BS,net_Q_Air/(p_air_rho*config.V_internal_air*p_air_cp)
    ]

# --- Main Execution Block ---
if __name__ == "__main__":
    print("\n--- Simulation Initial Conditions ---")
    print(f"Target Altitude: {config.TARGET_ALTITUDE_KM} km")
    print(f"Ambient Temp: {T_E:.2f} K | Ambient Pressure: {P_amb:.2f} Pa")
    print(f"Aircraft Velocity: {config.velocity} m/s")
    print(f"Node Count: {len(config.labels)}")
    print("-------------------------------------\n")
    x0 = np.array([config.initial_temp_K] * len(config.labels))
    print(f"Starting BDF solver for a simulation time of {config.T_total / 3600:.2f} hours...")
    sol = solve_ivp(fun=f, t_span=[0, config.T_total], y0=x0, method='BDF', dense_output=True, rtol=1e-5, atol=1e-8)
    print(f"... Solver finished. Status: {sol.message}")
    print(f"\nTotal execution time: {time.time() - start_time:.2f} seconds")
    post_processing.print_final_temps(sol)
    post_processing.analyze_peaks(sol)
    post_processing.plot_grouped_results(sol)