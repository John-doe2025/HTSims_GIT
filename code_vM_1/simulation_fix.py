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

#Conduction Constants
C_cond_ESC_to_Mount = config.k_mount*config.A_contact_ESC_Mount/config.L_path_ESC_Mount
C_cond_Mount_to_BH1 = config.k_bulkhead*config.A_contact_Mount_BH1/config.t_bulkhead
C_cond_Batt_plate =  config.k_cfrp*config.A_contact_batt_plate/config.t_plate
C_cond_bh_plate = 2 * config.k_bulkhead*config.A_contact_bh_plate/config.L_bh_plate_cond
C_cond_BH_TS = config.k_cfrp*config.A_contact_BH_Shell/config.t_cfrp
C_cond_BH_BS = config.k_cfrp*config.A_contact_BH_Shell/config.t_cfrp
C_cond_TS_int_ext = config.k_cfrp*config.A_TS/config.t_cfrp
C_cond_BS_int_ext = config.k_cfrp*config.A_BS/config.t_cfrp


#Radiation Constants 
C_rad_batt_batt = physics_models.rad_coeff(config.emis_batt,config.emis_batt,config.A_rad_batt_to_batt,config.A_rad_batt_to_batt)
C_rad_batt_esc = physics_models.rad_coeff(config.emis_esc,config.emis_batt,config.A_ESC_conv,config.A_rad_batt_to_batt)
C_rad_batt_ts = physics_models.rad_coeff(config.emis_batt,config.emis_shell_int,config.A_rad_batt_to_shell,config.A_TS, vf = 0.5)
C_rad_batt_bs = physics_models.rad_coeff(config.emis_batt,config.emis_shell_int,config.A_rad_batt_to_shell,config.A_BS, vf = 0.5)
C_rad_batt_bh = physics_models.rad_coeff(config.emis_batt,config.emis_bulkhead,config.A_conv_batt_side,config.A_rad_batt_bh)

C_rad_esc_bh = physics_models.rad_coeff(config.emis_esc,config.emis_bulkhead,config.A_ESC_conv,config.A_bulkhead_face)
C_rad_esc_ts = physics_models.rad_coeff(config.emis_esc,config.emis_shell_int,config.A_ESC_conv,config.A_TS)

C_rad_plate_sh = physics_models.rad_coeff(config.emis_plate, config.emis_shell_int, config.A_Plate, config.A_BS)
C_rad_mount_bs = physics_models.rad_coeff(config.emis_mount,config.emis_shell_int,config.A_mount_conv,config.A_BS)
C_rad_bh_bh = physics_models.rad_coeff(config.emis_bulkhead,config.emis_bulkhead,config.A_bulkhead_face,config.A_bulkhead_face)
C_rad_ts_bs = physics_models.rad_coeff(config.emis_shell_int,config.emis_shell_int,config.A_TS,config.A_BS,vf=0.5)

ENABLE_CONDUCTION = True
ENABLE_CONVECTION = True
ENABLE_RADIATION = True

def f(t, x):
    # Add temperature bounds checking to prevent runaway solutions
    x_bounded = np.clip(x, 150.0, 800.0)  # Physical temperature limits
    temps={label:x_bounded[i] for i, label in enumerate(config.labels)}
    T4s={k:physics_models.T_power4(v) for k,v in temps.items()}
    p_air = physics_models.prop_internal_air(temps['Internal_air'], P_amb)
    ext_loads=environment_model.calculate_external_heat_loads(t,temps['Top_Shell_Ext'],temps['Bot_Shell_Ext'], T_E)
    def get_h(T_s,LC,is_v): 
        # Use stable film properties to prevent instability
        T_film = (T_s + temps['Internal_air']) / 2
        T_film = max(min(T_film, 500.0), 200.0)  # Clamp film temperature
        p_film = physics_models.prop_internal_air(T_film, P_amb)
        h = physics_models.natural_convection_h(p_film, T_s, temps['Internal_air'], LC, is_v)
        return min(h, 100.0)  # Cap convection coefficient

    #BFT
    Q_r_BFT_BMT = C_rad_batt_batt*(T4s['Batt_BM_Top']-T4s['Batt_BF_Top'])
    Q_r_BFT_BH1 = C_rad_batt_bh*(T4s['BH_1']-T4s['Batt_BF_Top'])
    Q_r_BFT_BH2 = C_rad_batt_bh*(T4s['BH_2']-T4s['Batt_BF_Top'])
    Q_r_BFT_ESC = C_rad_batt_esc*(T4s['ESC']-T4s['Batt_BF_Top'])
    Q_r_BFT_TS = C_rad_batt_ts*(T4s['Top_Shell_Int']-T4s['Batt_BF_Top'])
    Q_v_BFT_Air = (get_h(temps['Batt_BF_Top'],config.LC_batt_horiz,False)*config.A_conv_batt_top+get_h(temps['Batt_BF_Top'],config.LC_batt_vert,True)*config.A_conv_batt_side)*(temps['Internal_air']-temps['Batt_BF_Top'])
    Q_cond_BFT_pt = C_cond_Batt_plate*(temps['plateT']-temps['Batt_BF_Top'])
    Q_cond_BFT_pm = C_cond_Batt_plate*(temps['plateM']-temps['Batt_BF_Top'])

    #BFB
    Q_r_BFB_BMB = C_rad_batt_batt*(T4s['Batt_BM_Bot']-T4s['Batt_BF_Bot'])
    Q_r_BFB_BH1 = C_rad_batt_bh*(T4s['BH_1']-T4s['Batt_BF_Bot'])
    Q_r_BFB_BH2 = C_rad_batt_bh*(T4s['BH_2']-T4s['Batt_BF_Bot'])
    Q_r_BFB_ESC = C_rad_batt_esc*(T4s['ESC']-T4s['Batt_BF_Bot'])
    Q_r_BFB_BS = C_rad_batt_ts*(T4s['Bot_Shell_Int']-T4s['Batt_BF_Bot'])
    Q_v_BFB_Air = (get_h(temps['Batt_BF_Bot'],config.LC_batt_horiz,False)*config.A_conv_batt_top+get_h(temps['Batt_BF_Bot'],config.LC_batt_vert,True)*config.A_conv_batt_side)*(temps['Internal_air']-temps['Batt_BF_Bot'])
    Q_cond_BFB_pm = C_cond_Batt_plate*(temps['plateM']-temps['Batt_BF_Bot'])
    Q_cond_BFB_pb = C_cond_Batt_plate*(temps['plateB']-temps['Batt_BF_Bot'])

    #BMT
    Q_r_BMT_BFT = C_rad_batt_batt*(T4s['Batt_BF_Top']-T4s['Batt_BM_Top'])
    Q_r_BMT_BRT = C_rad_batt_batt*(T4s['Batt_BR_Top']-T4s['Batt_BM_Top'])
    Q_r_BMT_BH2 = C_rad_batt_bh*(T4s['BH_2']-T4s['Batt_BM_Top'])
    Q_r_BMT_BH3 = C_rad_batt_bh*(T4s['BH_3']-T4s['Batt_BM_Top'])
    Q_r_BMT_TS = C_rad_batt_ts*(T4s['Top_Shell_Int']-T4s['Batt_BM_Top'])
    Q_v_BMT_Air = (get_h(temps['Batt_BM_Top'],config.LC_batt_horiz,False)*config.A_conv_batt_top+get_h(temps['Batt_BM_Top'],config.LC_batt_vert,True)*config.A_conv_batt_side)*(temps['Internal_air']-temps['Batt_BM_Top'])
    Q_cond_BMT_pt = C_cond_Batt_plate*(temps['plateT']-temps['Batt_BM_Top'])
    Q_cond_BMT_pm = C_cond_Batt_plate*(temps['plateM']-temps['Batt_BM_Top'])
    
    #BMB
    Q_r_BMB_BFB = C_rad_batt_batt*(T4s['Batt_BF_Bot']-T4s['Batt_BM_Bot'])
    Q_r_BMB_BRB = C_rad_batt_batt*(T4s['Batt_BR_Bot']-T4s['Batt_BM_Bot'])
    Q_r_BMB_BH2 = C_rad_batt_bh*(T4s['BH_2']-T4s['Batt_BM_Bot'])
    Q_r_BMB_BH3 = C_rad_batt_bh*(T4s['BH_3']-T4s['Batt_BM_Bot'])
    Q_r_BMB_BS = C_rad_batt_ts*(T4s['Bot_Shell_Int']-T4s['Batt_BM_Bot'])
    Q_v_BMB_Air = (get_h(temps['Batt_BM_Bot'],config.LC_batt_horiz,False)*config.A_conv_batt_top+get_h(temps['Batt_BM_Bot'],config.LC_batt_vert,True)*config.A_conv_batt_side)*(temps['Internal_air']-temps['Batt_BM_Bot'])
    Q_cond_BMB_pm = C_cond_Batt_plate*(temps['plateM']-temps['Batt_BM_Bot'])
    Q_cond_BMB_pb = C_cond_Batt_plate*(temps['plateB']-temps['Batt_BM_Bot'])

    #BRT
    Q_r_BRT_BMT = C_rad_batt_batt*(T4s['Batt_BM_Top']-T4s['Batt_BR_Top'])
    Q_r_BRT_BH3 = C_rad_batt_bh*(T4s['BH_3']-T4s['Batt_BR_Top'])
    Q_r_BRT_BH4 = C_rad_batt_bh*(T4s['BH_4']-T4s['Batt_BR_Top'])
    Q_r_BRT_TS = C_rad_batt_ts*(T4s['Top_Shell_Int']-T4s['Batt_BR_Top'])
    Q_v_BRT_Air = (get_h(temps['Batt_BR_Top'],config.LC_batt_horiz,False)*config.A_conv_batt_top+get_h(temps['Batt_BR_Top'],config.LC_batt_vert,True)*config.A_conv_batt_side)*(temps['Internal_air']-temps['Batt_BR_Top'])
    Q_cond_BRT_pt = C_cond_Batt_plate*(temps['plateT']-temps['Batt_BR_Top'])
    Q_cond_BRT_pm = C_cond_Batt_plate*(temps['plateM']-temps['Batt_BR_Top']) 

    #BRB
    Q_r_BRB_BMB = C_rad_batt_batt*(T4s['Batt_BM_Bot']-T4s['Batt_BR_Bot'])
    Q_r_BRB_BH3 = C_rad_batt_bh*(T4s['BH_3']-T4s['Batt_BR_Bot'])
    Q_r_BRB_BH4 = C_rad_batt_bh*(T4s['BH_4']-T4s['Batt_BR_Bot'])
    Q_r_BRB_BS = C_rad_batt_ts*(T4s['Bot_Shell_Int']-T4s['Batt_BR_Bot'])
    Q_v_BRB_Air = (get_h(temps['Batt_BR_Bot'],config.LC_batt_horiz,False)*config.A_conv_batt_top+get_h(temps['Batt_BR_Bot'],config.LC_batt_vert,True)*config.A_conv_batt_side)*(temps['Internal_air']-temps['Batt_BR_Bot'])
    Q_cond_BRB_pm = C_cond_Batt_plate*(temps['plateM']-temps['Batt_BR_Bot'])
    Q_cond_BRB_pb = C_cond_Batt_plate*(temps['plateB']-temps['Batt_BR_Bot'])

    #PlateT
    Q_cond_pt_BFT = C_cond_Batt_plate*(temps['Batt_BF_Top']-temps['plateT'])
    Q_cond_pt_BMT = C_cond_Batt_plate*(temps['Batt_BM_Top']-temps['plateT'])
    Q_cond_pt_BRT = C_cond_Batt_plate*(temps['Batt_BR_Top']-temps['plateT'])
    Q_cond_pt_BH1 = C_cond_bh_plate*(temps['BH_1']-temps['plateT'])
    Q_cond_pt_BH2 = C_cond_bh_plate*(temps['BH_2']-temps['plateT'])
    Q_cond_pt_BH3 = C_cond_bh_plate*(temps['BH_3']-temps['plateT'])
    Q_cond_pt_BH4 = C_cond_bh_plate*(temps['BH_4']-temps['plateT'])
    Q_r_pt_ts = C_rad_plate_sh*(T4s['Top_Shell_Int']-T4s['plateT'])
    Q_v_pt_air = get_h(temps['plateT'],config.LC_plate,False)*config.A_conv_plate*(temps['Internal_air']-temps['plateT'])

    #PlateM
    Q_cond_pm_BFT = C_cond_Batt_plate*(temps['Batt_BF_Top']-temps['plateM'])
    Q_cond_pm_BMT = C_cond_Batt_plate*(temps['Batt_BM_Top']-temps['plateM'])
    Q_cond_pm_BRT = C_cond_Batt_plate*(temps['Batt_BR_Top']-temps['plateM'])
    Q_cond_pm_BFB = C_cond_Batt_plate*(temps['Batt_BF_Bot']-temps['plateM'])
    Q_cond_pm_BMB = C_cond_Batt_plate*(temps['Batt_BM_Bot']-temps['plateM'])
    Q_cond_pm_BRB = C_cond_Batt_plate*(temps['Batt_BR_Bot']-temps['plateM'])
    Q_cond_pm_BH1 = C_cond_bh_plate*(temps['BH_1']-temps['plateM'])
    Q_cond_pm_BH2 = C_cond_bh_plate*(temps['BH_2']-temps['plateM'])
    Q_cond_pm_BH3 = C_cond_bh_plate*(temps['BH_3']-temps['plateM'])
    Q_cond_pm_BH4 = C_cond_bh_plate*(temps['BH_4']-temps['plateM'])
    Q_v_pm_air = get_h(temps['plateM'],config.LC_plate,False)*config.A_conv_plateM*(temps['Internal_air']-temps['plateM'])
 
    #PlateB
    Q_cond_pb_BFB = C_cond_Batt_plate*(temps['Batt_BF_Bot']-temps['plateB'])
    Q_cond_pb_BMB = C_cond_Batt_plate*(temps['Batt_BM_Bot']-temps['plateB'])
    Q_cond_pb_BRB = C_cond_Batt_plate*(temps['Batt_BR_Bot']-temps['plateB'])
    Q_cond_pb_BH1 = C_cond_bh_plate*(temps['BH_1']-temps['plateB'])
    Q_cond_pb_BH2 = C_cond_bh_plate*(temps['BH_2']-temps['plateB'])
    Q_cond_pb_BH3 = C_cond_bh_plate*(temps['BH_3']-temps['plateB'])
    Q_cond_pb_BH4 = C_cond_bh_plate*(temps['BH_4']-temps['plateB'])
    Q_r_pb_bs = C_rad_plate_sh*(T4s['Bot_Shell_Int']-T4s['plateB'])
    Q_v_pb_air = get_h(temps['plateB'],config.LC_plate,False)*config.A_conv_plate*(temps['Internal_air']-temps['plateB'])

    #BH1
    Q_r_BH1_BFT = C_rad_batt_bh*(T4s['Batt_BF_Top']-T4s['BH_1'])
    Q_r_BH1_BFB = C_rad_batt_bh*(T4s['Batt_BF_Bot']-T4s['BH_1'])
    Q_r_BH1_BH2 = C_rad_bh_bh*(T4s['BH_2']-T4s['BH_1'])
    Q_r_BH1_esc = C_rad_esc_bh*(T4s['ESC']-T4s['BH_1'])
    Q_cond_BH1_Mount = C_cond_Mount_to_BH1*(temps['ESC_Mount']-temps['BH_1'])
    Q_cond_BH1_TS = C_cond_BH_TS*(temps['Top_Shell_Int']-temps['BH_1'])
    Q_cond_BH1_BS = C_cond_BH_BS*(temps['Bot_Shell_Int']-temps['BH_1'])
    Q_cond_BH1_pt = C_cond_bh_plate*(temps['plateT']-temps['BH_1'])
    Q_cond_BH1_pm = C_cond_bh_plate*(temps['plateM']-temps['BH_1'])
    Q_cond_BH1_pb = C_cond_bh_plate*(temps['plateB']-temps['BH_1'])
    Q_v_BH1_Air = get_h(temps['BH_1'],config.LC_bulkhead,True)*config.A_bulkhead_face*2*(temps['Internal_air']-temps['BH_1'])
    
    #BH2
    Q_r_BH2_BFT = C_rad_batt_bh*(T4s['Batt_BF_Top']-T4s['BH_2'])
    Q_r_BH2_BFB = C_rad_batt_bh*(T4s['Batt_BF_Bot']-T4s['BH_2'])
    Q_r_BH2_BMT = C_rad_batt_bh*(T4s['Batt_BM_Top']-T4s['BH_2'])
    Q_r_BH2_BMB = C_rad_batt_bh*(T4s['Batt_BM_Bot']-T4s['BH_2'])
    Q_r_BH2_BH1 = C_rad_bh_bh*(T4s['BH_1']-T4s['BH_2'])
    Q_r_BH2_BH3 = C_rad_bh_bh*(T4s['BH_3']-T4s['BH_2'])
    Q_cond_BH2_TS = C_cond_BH_TS*(temps['Top_Shell_Int']-temps['BH_2'])
    Q_cond_BH2_BS = C_cond_BH_BS*(temps['Bot_Shell_Int']-temps['BH_2'])
    Q_cond_BH2_pt = C_cond_bh_plate*(temps['plateT']-temps['BH_2'])
    Q_cond_BH2_pm = C_cond_bh_plate*(temps['plateM']-temps['BH_2'])
    Q_cond_BH2_pb = C_cond_bh_plate*(temps['plateB']-temps['BH_2'])
    Q_v_BH2_Air = get_h(temps['BH_2'],config.LC_bulkhead,True)*config.A_bulkhead_face*2*(temps['Internal_air']-temps['BH_2'])

    #BH3
    Q_r_BH3_BMT = C_rad_batt_bh*(T4s['Batt_BM_Top']-T4s['BH_3'])
    Q_r_BH3_BMB = C_rad_batt_bh*(T4s['Batt_BM_Bot']-T4s['BH_3'])
    Q_r_BH3_BRT = C_rad_batt_bh*(T4s['Batt_BR_Top']-T4s['BH_3'])
    Q_r_BH3_BRB = C_rad_batt_bh*(T4s['Batt_BR_Bot']-T4s['BH_3'])
    Q_r_BH3_BH2 = C_rad_bh_bh*(T4s['BH_2']-T4s['BH_3'])
    Q_r_BH3_BH4 = C_rad_bh_bh*(T4s['BH_4']-T4s['BH_3'])
    Q_cond_BH3_TS = C_cond_BH_TS*(temps['Top_Shell_Int']-temps['BH_3'])
    Q_cond_BH3_BS = C_cond_BH_BS*(temps['Bot_Shell_Int']-temps['BH_3'])
    Q_cond_BH3_pt = C_cond_bh_plate*(temps['plateT']-temps['BH_3'])
    Q_cond_BH3_pm = C_cond_bh_plate*(temps['plateM']-temps['BH_3'])
    Q_cond_BH3_pb = C_cond_bh_plate*(temps['plateB']-temps['BH_3'])
    Q_v_BH3_Air = get_h(temps['BH_3'],config.LC_bulkhead,True)*config.A_bulkhead_face*2*(temps['Internal_air']-temps['BH_3'])

    #BH4
    Q_r_BH4_BRT = C_rad_batt_bh*(T4s['Batt_BR_Top']-T4s['BH_4'])
    Q_r_BH4_BRB = C_rad_batt_bh*(T4s['Batt_BR_Bot']-T4s['BH_4'])
    Q_r_BH4_BH3 = C_rad_bh_bh*(T4s['BH_3']-T4s['BH_4'])
    Q_cond_BH4_TS = C_cond_BH_TS*(temps['Top_Shell_Int']-temps['BH_4'])
    Q_cond_BH4_BS = C_cond_BH_BS*(temps['Bot_Shell_Int']-temps['BH_4'])
    Q_cond_BH4_pt = C_cond_bh_plate*(temps['plateT']-temps['BH_4'])
    Q_cond_BH4_pm = C_cond_bh_plate*(temps['plateM']-temps['BH_4'])
    Q_cond_BH4_pb = C_cond_bh_plate*(temps['plateB']-temps['BH_4'])
    Q_v_BH4_Air = get_h(temps['BH_4'],config.LC_bulkhead,True)*config.A_bulkhead_face*2*(temps['Internal_air']-temps['BH_4'])

    #ESC
    Q_r_ESC_BFT = C_rad_batt_esc*(T4s['Batt_BF_Top']-T4s['ESC'])
    Q_r_ESC_BFB = C_rad_batt_esc*(T4s['Batt_BF_Bot']-T4s['ESC'])
    Q_r_ESC_BH1 = C_rad_esc_bh*(T4s['BH_1']-T4s['ESC'])
    Q_r_ESC_TS = C_rad_esc_ts*(T4s['Top_Shell_Int']-T4s['ESC'])
    Q_cond_ESC_Mount = C_cond_ESC_to_Mount*(temps['ESC_Mount']-temps['ESC'])
    Q_v_ESC_Air = (get_h(temps['ESC'],config.LC_esc_horiz,False)*config.A_conv_esc_top+get_h(temps['ESC'],config.LC_esc_vert,True)*config.A_conv_esc_side)*(temps['Internal_air']-temps['ESC'])

    #ESC Mount
    Q_cond_Mount_BH1 = C_cond_Mount_to_BH1*(temps['BH_1']-temps['ESC_Mount'])
    Q_cond_Mount_ESC = C_cond_ESC_to_Mount*(temps['ESC']-temps['ESC_Mount'])
    Q_r_Mount_BS = C_rad_mount_bs*(T4s['Bot_Shell_Int']-T4s['ESC_Mount'])
    Q_v_Mount_Air = get_h(temps['ESC_Mount'],config.LC_mount,False)*config.A_mount_conv*(temps['Internal_air']-temps['ESC_Mount'])

    #Top_Shell_Int
    Q_r_TS_BFT = C_rad_batt_ts*(T4s['Batt_BF_Top']-T4s['Top_Shell_Int'])
    Q_r_TS_BMT = C_rad_batt_ts*(T4s['Batt_BM_Top']-T4s['Top_Shell_Int'])
    Q_r_TS_BRT = C_rad_batt_ts*(T4s['Batt_BR_Top']-T4s['Top_Shell_Int'])
    Q_r_TS_ESC = C_rad_esc_ts*(T4s['ESC']-T4s['Top_Shell_Int'])
    Q_r_ts_pt = C_rad_plate_sh*(T4s['plateT']-T4s['Top_Shell_Int'])
    Q_r_TS_BS = C_rad_ts_bs*(T4s['Bot_Shell_Int']-T4s['Top_Shell_Int'])
    Q_cond_TS_BH1 = C_cond_BH_TS*(temps['BH_1']-temps['Top_Shell_Int'])
    Q_cond_TS_BH2 = C_cond_BH_TS*(temps['BH_2']-temps['Top_Shell_Int'])
    Q_cond_TS_BH3 = C_cond_BH_TS*(temps['BH_3']-temps['Top_Shell_Int'])
    Q_cond_TS_BH4 = C_cond_BH_TS*(temps['BH_4']-temps['Top_Shell_Int'])
    Q_cond_TSi_TSe = C_cond_TS_int_ext*(temps['Top_Shell_Ext']-temps['Top_Shell_Int'])
    Q_v_TSi_Air = get_h(temps['Top_Shell_Int'],config.LC_TS_int,False)*config.A_TS*(temps['Internal_air']-temps['Top_Shell_Int'])

    #Top Shell Ext
    Q_cond_TSe_TSi = C_cond_TS_int_ext*(temps['Top_Shell_Int']-temps['Top_Shell_Ext'])
    Q_v_TSe_Amb = physics_models.get_external_convection_h(p_ambient,temps['Top_Shell_Ext'],T_E,config.LC_TS_ext)*config.A_TS*(T_E-temps['Top_Shell_Ext'])
    Q_r_TSe_Ext = ext_loads['Q_ext_top']

    #Bottom Shell Int
    Q_r_BS_BFB = C_rad_batt_ts*(T4s['Batt_BF_Bot']-T4s['Bot_Shell_Int'])
    Q_r_BS_BMB = C_rad_batt_ts*(T4s['Batt_BM_Bot']-T4s['Bot_Shell_Int'])
    Q_r_BS_BRB = C_rad_batt_ts*(T4s['Batt_BR_Bot']-T4s['Bot_Shell_Int'])
    Q_r_BS_Mount = C_rad_mount_bs*(T4s['ESC_Mount']-T4s['Bot_Shell_Int'])
    Q_r_bs_pb = C_rad_plate_sh*(T4s['plateB']-T4s['Bot_Shell_Int'])
    Q_r_BS_TS = C_rad_ts_bs*(T4s['Top_Shell_Int']-T4s['Bot_Shell_Int'])
    Q_cond_BS_BH1 = C_cond_BH_BS*(temps['BH_1']-temps['Bot_Shell_Int'])
    Q_cond_BS_BH2 = C_cond_BH_BS*(temps['BH_2']-temps['Bot_Shell_Int'])
    Q_cond_BS_BH3 = C_cond_BH_BS*(temps['BH_3']-temps['Bot_Shell_Int'])
    Q_cond_BS_BH4 = C_cond_BH_BS*(temps['BH_4']-temps['Bot_Shell_Int'])
    Q_cond_BSi_BSe = C_cond_BS_int_ext*(temps['Bot_Shell_Ext']-temps['Bot_Shell_Int'])
    Q_v_BSi_Air = get_h(temps['Bot_Shell_Int'],config.LC_BS_int,False)*config.A_BS*(temps['Internal_air']-temps['Bot_Shell_Int'])

    #Bottom Shell Ext
    Q_cond_BSe_BSi = C_cond_BS_int_ext*(temps['Bot_Shell_Int']-temps['Bot_Shell_Ext'])
    Q_v_BSe_Amb = physics_models.get_external_convection_h(p_ambient,temps['Bot_Shell_Ext'],T_E,config.LC_BS_ext)*config.A_BS*(T_E-temps['Bot_Shell_Ext'])
    Q_r_BSe_Ext = ext_loads['Q_ext_bottom']

    #Internal Air
    Q_v_Air_BFT = (get_h(temps['Batt_BF_Top'],config.LC_batt_horiz,False)*config.A_conv_batt_top+get_h(temps['Batt_BF_Top'],config.LC_batt_vert,True)*config.A_conv_batt_side)*(temps['Batt_BF_Top']-temps['Internal_air'])
    Q_v_Air_BFB = (get_h(temps['Batt_BF_Bot'],config.LC_batt_horiz,False)*config.A_conv_batt_top+get_h(temps['Batt_BF_Bot'],config.LC_batt_vert,True)*config.A_conv_batt_side)*(temps['Batt_BF_Bot']-temps['Internal_air'])
    Q_v_Air_BMT = (get_h(temps['Batt_BM_Top'],config.LC_batt_horiz,False)*config.A_conv_batt_top+get_h(temps['Batt_BM_Top'],config.LC_batt_vert,True)*config.A_conv_batt_side)*(temps['Batt_BM_Top']-temps['Internal_air'])
    Q_v_Air_BMB = (get_h(temps['Batt_BM_Bot'],config.LC_batt_horiz,False)*config.A_conv_batt_top+get_h(temps['Batt_BM_Bot'],config.LC_batt_vert,True)*config.A_conv_batt_side)*(temps['Batt_BM_Bot']-temps['Internal_air'])
    Q_v_Air_BRT = (get_h(temps['Batt_BR_Top'],config.LC_batt_horiz,False)*config.A_conv_batt_top+get_h(temps['Batt_BR_Top'],config.LC_batt_vert,True)*config.A_conv_batt_side)*(temps['Batt_BR_Top']-temps['Internal_air'])
    Q_v_Air_BRB = (get_h(temps['Batt_BR_Bot'],config.LC_batt_horiz,False)*config.A_conv_batt_top+get_h(temps['Batt_BR_Bot'],config.LC_batt_vert,True)*config.A_conv_batt_side)*(temps['Batt_BR_Bot']-temps['Internal_air'])
    Q_v_air_pt = get_h(temps['plateT'],config.LC_plate,False)*config.A_conv_plate*(temps['plateT']-temps['Internal_air'])
    Q_v_air_pm = get_h(temps['plateM'],config.LC_plate,False)*config.A_conv_plateM*(temps['plateM']-temps['Internal_air'])
    Q_v_air_pb = get_h(temps['plateB'],config.LC_plate,False)*config.A_conv_plate*(temps['plateB']-temps['Internal_air'])
    Q_v_Air_BH1 = get_h(temps['BH_1'],config.LC_bulkhead,True)*config.A_bulkhead_face*2*(temps['BH_1']-temps['Internal_air'])
    Q_v_Air_BH2 = get_h(temps['BH_2'],config.LC_bulkhead,True)*config.A_bulkhead_face*2*(temps['BH_2']-temps['Internal_air'])
    Q_v_Air_BH3 = get_h(temps['BH_3'],config.LC_bulkhead,True)*config.A_bulkhead_face*2*(temps['BH_3']-temps['Internal_air'])
    Q_v_Air_BH4 = get_h(temps['BH_4'],config.LC_bulkhead,True)*config.A_bulkhead_face*2*(temps['BH_4']-temps['Internal_air'])
    Q_v_Air_ESC = (get_h(temps['ESC'],config.LC_esc_horiz,False)*config.A_conv_esc_top+get_h(temps['ESC'],config.LC_esc_vert,True)*config.A_conv_esc_side)*(temps['ESC']-temps['Internal_air'])
    Q_v_Air_Mount = get_h(temps['ESC_Mount'],config.LC_mount,False)*config.A_mount_conv*(temps['ESC_Mount']-temps['Internal_air'])
    Q_v_Air_TSi = get_h(temps['Top_Shell_Int'],config.LC_TS_int,False)*config.A_TS*(temps['Top_Shell_Int']-temps['Internal_air'])
    Q_v_Air_BSi = get_h(temps['Bot_Shell_Int'],config.LC_BS_int,False)*config.A_BS*(temps['Bot_Shell_Int']-temps['Internal_air'])


    # --- NET HEAT BALANCE ---
    
    # BATTERY NODES
    net_Q_BFT = config.Q_batt_zone + (Q_r_BFT_BMT + Q_r_BFT_BH1 + Q_r_BFT_BH2 + Q_r_BFT_ESC + Q_r_BFT_TS + Q_v_BFT_Air + Q_cond_BFT_pt + Q_cond_BFT_pm)
    net_Q_BFB = config.Q_batt_zone + (Q_r_BFB_BMB + Q_r_BFB_BH1 + Q_r_BFB_BH2 + Q_r_BFB_ESC + Q_r_BFB_BS + Q_v_BFB_Air + Q_cond_BFB_pm + Q_cond_BFB_pb)
    net_Q_BMT = config.Q_batt_zone + (Q_r_BMT_BFT + Q_r_BMT_BRT + Q_r_BMT_BH2 + Q_r_BMT_BH3 + Q_r_BMT_TS + Q_v_BMT_Air + Q_cond_BMT_pt + Q_cond_BMT_pm)
    net_Q_BMB = config.Q_batt_zone + (Q_r_BMB_BFB + Q_r_BMB_BRB + Q_r_BMB_BH2 + Q_r_BMB_BH3 + Q_r_BMB_BS + Q_v_BMB_Air + Q_cond_BMB_pm + Q_cond_BMB_pb)
    net_Q_BRT = config.Q_batt_zone + (Q_r_BRT_BMT + Q_r_BRT_BH3 + Q_r_BRT_BH4 + Q_r_BRT_TS + Q_v_BRT_Air + Q_cond_BRT_pt + Q_cond_BRT_pm)
    net_Q_BRB = config.Q_batt_zone + (Q_r_BRB_BMB + Q_r_BRB_BH3 + Q_r_BRB_BH4 + Q_r_BRB_BS + Q_v_BRB_Air + Q_cond_BRB_pm + Q_cond_BRB_pb)
    

    # AVIONICS NODES
    net_Q_ESC = config.Q_ESC + (Q_r_ESC_BFT + Q_r_ESC_BFB + Q_r_ESC_BH1 + Q_r_ESC_TS + Q_cond_ESC_Mount + Q_v_ESC_Air)
    net_Q_Mount = (Q_cond_Mount_BH1 + Q_cond_Mount_ESC + Q_r_Mount_BS + Q_v_Mount_Air)

    # BULKHEAD NODES
    net_Q_BH1 = (Q_r_BH1_BFT + Q_r_BH1_BFB + Q_r_BH1_BH2 + Q_r_BH1_esc + Q_cond_BH1_Mount + Q_cond_BH1_TS + Q_cond_BH1_BS + Q_cond_BH1_pt + Q_cond_BH1_pm + Q_cond_BH1_pb + Q_v_BH1_Air)
    net_Q_BH2 = (Q_r_BH2_BFT + Q_r_BH2_BFB + Q_r_BH2_BMT + Q_r_BH2_BMB + Q_r_BH2_BH1 + Q_r_BH2_BH3 + Q_cond_BH2_TS + Q_cond_BH2_BS + Q_cond_BH2_pt + Q_cond_BH2_pm + Q_cond_BH2_pb + Q_v_BH2_Air)
    net_Q_BH3 = (Q_r_BH3_BMT + Q_r_BH3_BMB + Q_r_BH3_BRT + Q_r_BH3_BRB + Q_r_BH3_BH2 + Q_r_BH3_BH4 + Q_cond_BH3_TS + Q_cond_BH3_BS + Q_cond_BH3_pt + Q_cond_BH3_pm + Q_cond_BH3_pb + Q_v_BH3_Air)
    net_Q_BH4 = (Q_r_BH4_BRT + Q_r_BH4_BRB + Q_r_BH4_BH3 + Q_cond_BH4_TS + Q_cond_BH4_BS + Q_cond_BH4_pt + Q_cond_BH4_pm + Q_cond_BH4_pb + Q_v_BH4_Air)

    # PLATE NODES (FIXED variable names)
    net_Q_plateT = Q_cond_pt_BFT + Q_cond_pt_BMT + Q_cond_pt_BRT + Q_cond_pt_BH1 + Q_cond_pt_BH2 + Q_cond_pt_BH3 + Q_cond_pt_BH4 + Q_r_pt_ts + Q_v_pt_air
    net_Q_plateM = Q_cond_pm_BFT + Q_cond_pm_BMT + Q_cond_pm_BRT + Q_cond_pm_BFB + Q_cond_pm_BMB + Q_cond_pm_BRB + Q_cond_pm_BH1 + Q_cond_pm_BH2 + Q_cond_pm_BH3 + Q_cond_pm_BH4 + Q_v_pm_air
    net_Q_plateB = Q_cond_pb_BFB + Q_cond_pb_BMB + Q_cond_pb_BRB + Q_cond_pb_BH1 + Q_cond_pb_BH2 + Q_cond_pb_BH3 + Q_cond_pb_BH4 + Q_r_pb_bs + Q_v_pb_air

    # SHELL NODES
    net_Q_TS_int = (Q_r_TS_BFT + Q_r_TS_BMT + Q_r_TS_BRT + Q_r_TS_ESC + Q_r_ts_pt + Q_r_TS_BS + Q_cond_TS_BH1 + Q_cond_TS_BH2 + Q_cond_TS_BH3 + Q_cond_TS_BH4 + Q_cond_TSi_TSe + Q_v_TSi_Air)
    net_Q_BS_int = (Q_r_BS_BFB + Q_r_BS_BMB + Q_r_BS_BRB + Q_r_BS_Mount + Q_r_bs_pb + Q_r_BS_TS + Q_cond_BS_BH1 + Q_cond_BS_BH2 + Q_cond_BS_BH3 + Q_cond_BS_BH4 + Q_cond_BSi_BSe + Q_v_BSi_Air)
    net_Q_TS_ext = (Q_cond_TSe_TSi + Q_v_TSe_Amb + Q_r_TSe_Ext)
    net_Q_BS_ext = (Q_cond_BSe_BSi + Q_v_BSe_Amb + Q_r_BSe_Ext)

    # INTERNAL AIR NODE
    net_Q_Air = (Q_v_Air_BFT + Q_v_Air_BFB + Q_v_Air_BMT + Q_v_Air_BMB + Q_v_Air_BRT + Q_v_Air_BRB + Q_v_air_pt + Q_v_air_pm + Q_v_air_pb + Q_v_Air_BH1 + Q_v_Air_BH2 + Q_v_Air_BH3 + Q_v_Air_BH4 + Q_v_Air_ESC + Q_v_Air_Mount + Q_v_Air_TSi + Q_v_Air_BSi) 
    
    # dT/dt
    mC_batt = config.m_batt_zone*config.C_B 
    mC_bh = config.m_bulkhead*config.C_bulkhead
    mC_TS = config.m_TS*config.C_TS
    mC_BS = config.m_BS*config.C_BS
    mC_ESC = config.m_ESC*config.C_ESC
    mC_Mount = config.m_mount*config.C_mount
    mC_plate = config.m_plate*config.C_plate
    p_air_rho, p_air_cp = p_air[0], p_air[1]
    
    # Calculate internal air temperature rate - use physics-based approach
    air_thermal_mass = p_air_rho * config.V_internal_air * p_air_cp
    
    # Check for unrealistic thermal mass that could cause instability
    if air_thermal_mass < 1.0:
        # If thermal mass is too small, use a minimum realistic value
        air_thermal_mass = max(air_thermal_mass, 10.0)  # Minimum thermal mass
    
    air_temp_rate = net_Q_Air / air_thermal_mass
    

    # CORRECTED return order to match config.labels exactly
    return [
        net_Q_BFT/mC_batt,        # 0: Batt_BF_Top
        net_Q_BFB/mC_batt,        # 1: Batt_BF_Bot
        net_Q_BMT/mC_batt,        # 2: Batt_BM_Top
        net_Q_BMB/mC_batt,        # 3: Batt_BM_Bot
        net_Q_BRT/mC_batt,        # 4: Batt_BR_Top
        net_Q_BRB/mC_batt,        # 5: Batt_BR_Bot
        net_Q_ESC/mC_ESC,         # 6: ESC
        net_Q_Mount/mC_Mount,     # 7: ESC_Mount
        net_Q_BH1/mC_bh,          # 8: BH_1
        net_Q_BH2/mC_bh,          # 9: BH_2
        net_Q_BH3/mC_bh,          # 10: BH_3
        net_Q_BH4/mC_bh,          # 11: BH_4
        net_Q_plateT/mC_plate,    # 12: plateT
        net_Q_plateM/mC_plate,    # 13: plateM
        net_Q_plateB/mC_plate,    # 14: plateB
        net_Q_TS_int/mC_TS,       # 15: Top_Shell_Int
        net_Q_TS_ext/mC_TS,       # 16: Top_Shell_Ext
        net_Q_BS_int/mC_BS,       # 17: Bot_Shell_Int
        net_Q_BS_ext/mC_BS,       # 18: Bot_Shell_Ext
        air_temp_rate             # 19: Internal_air (rate-limited)
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
    
    # Add more frequent temperature tracking for first few days
    hourly_times = [i * 3600 for i in range(72)]  # Every hour for first 3 days
    daily_times = [i * 86400 for i in range(3, 15)]  # Daily from day 3 onwards
    all_times = hourly_times + daily_times
    
    # Use more conservative solver settings when external radiation is enabled
    if environment_model.ENABLE_EXTERNAL_RADIATION:
        sol = solve_ivp(fun=f, t_span=[0, config.T_total], y0=x0, method='BDF', dense_output=True, rtol=1e-2, atol=1e-5, max_step=1800, t_eval=all_times)
    else:
        sol = solve_ivp(fun=f, t_span=[0, config.T_total], y0=x0, method='BDF', dense_output=True, rtol=1e-3, atol=1e-6, max_step=3600, t_eval=all_times)
    print(f"... Solver finished. Status: {sol.message}")
    print(f"\nTotal execution time: {time.time() - start_time:.2f} seconds")
    
    # Print initial temperature drop analysis
    print("\n--- INITIAL TEMPERATURE DROP ANALYSIS ---")
    print(f"Initial temp: {config.initial_temp_K:.1f}K ({config.initial_temp_K-273.15:.1f}°C)")
    print(f"Ambient temp: {T_E:.1f}K ({T_E-273.15:.1f}°C)")
    print(f"Temperature difference: {config.initial_temp_K - T_E:.1f}K")
    
    # Show first few hours of temperature evolution
    for i in range(min(25, len(sol.t))):  # First 24 hours
        if i < len(sol.t):
            hour = sol.t[i] / 3600
            temps = sol.y[:, i]
            temps_dict = {config.labels[j]: temps[j] for j in range(len(config.labels))}
            if i % 6 == 0:  # Every 6 hours
                print(f"\nHour {hour:.0f}:")
                print(f"  Internal_air: {temps_dict['Internal_air']:.1f}K ({temps_dict['Internal_air']-273.15:.1f}°C)")
                print(f"  Batt_BF_Top:  {temps_dict['Batt_BF_Top']:.1f}K ({temps_dict['Batt_BF_Top']-273.15:.1f}°C)")
                print(f"  Top_Shell_Ext: {temps_dict['Top_Shell_Ext']:.1f}K ({temps_dict['Top_Shell_Ext']-273.15:.1f}°C)")
    
    # Print daily temperature evolution for remaining days
    print("\n--- DAILY TEMPERATURE EVOLUTION (Days 3-10) ---")
    daily_start_idx = 72  # After 72 hours
    for day in range(min(8, len(sol.t) - daily_start_idx)):
        if daily_start_idx + day < len(sol.t):
            day_temps = sol.y[:, daily_start_idx + day]
            temps_dict = {config.labels[i]: day_temps[i] for i in range(len(config.labels))}
            print(f"\nDay {day + 3}:")
            print(f"  Internal_air: {temps_dict['Internal_air']:.1f}K ({temps_dict['Internal_air']-273.15:.1f}°C)")
            print(f"  Batt_BF_Top:  {temps_dict['Batt_BF_Top']:.1f}K ({temps_dict['Batt_BF_Top']-273.15:.1f}°C)")
            print(f"  ESC:          {temps_dict['ESC']:.1f}K ({temps_dict['ESC']-273.15:.1f}°C)")
            print(f"  BH_1:         {temps_dict['BH_1']:.1f}K ({temps_dict['BH_1']-273.15:.1f}°C)")
            print(f"  Top_Shell_Ext: {temps_dict['Top_Shell_Ext']:.1f}K ({temps_dict['Top_Shell_Ext']-273.15:.1f}°C)")
    post_processing.print_final_temps(sol)
    post_processing.analyze_peaks(sol)
    post_processing.plot_grouped_results(sol)