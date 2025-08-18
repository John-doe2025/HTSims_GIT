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
C_cond_BH_TS = config.k_cfrp*config.A_contact_BH_Shell/config.L_bh_plate_cond
C_cond_BH_BS = config.k_cfrp*config.A_contact_BH_Shell/config.L_bh_plate_cond
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
    temps={label:x[i] for i, label in enumerate(config.labels)}
    T4s={k:physics_models.T_power4(v) for k,v in temps.items()}
    p_air = physics_models.prop_internal_air(temps['Internal_air'], P_amb)
    ext_loads=environment_model.calculate_external_heat_loads(t,temps['Top_Shell_Ext'],temps['Bot_Shell_Ext'], T_E)
    def get_h(T_s,LC,is_v): return physics_models.natural_convection_h(physics_models.prop_internal_air((T_s+temps['Internal_air'])/2, P_amb), T_s, temps['Internal_air'], LC, is_v)

    #BFT
    Q_r_BFT_BMT = ENABLE_RADIATION * C_rad_batt_batt*(T4s['Batt_BM_Top']-T4s['Batt_BF_Top'])
    Q_r_BFT_BH1 = ENABLE_RADIATION * C_rad_batt_bh*(T4s['BH_1']-T4s['Batt_BF_Top'])
    Q_r_BFT_BH2 = ENABLE_RADIATION * C_rad_batt_bh*(T4s['BH_2']-T4s['Batt_BF_Top'])
    Q_r_BFT_ESC = ENABLE_RADIATION * C_rad_batt_esc*(T4s['ESC']-T4s['Batt_BF_Top'])
    Q_r_BFT_TS = ENABLE_RADIATION * C_rad_batt_ts*(T4s['Top_Shell_Int']-T4s['Batt_BF_Top'])
    Q_v_BFT_air = (ENABLE_CONVECTION * get_h(temps['Batt_BF_Top'],config.LC_batt_horiz,False)*config.A_conv_batt_top+ENABLE_CONVECTION * get_h(temps['Batt_BF_Top'],config.LC_batt_vert,True)*config.A_conv_batt_side)*(temps['Internal_air']-temps['Batt_BF_Top'])
    Q_cond_BFT_pt = ENABLE_CONDUCTION * C_cond_Batt_plate*(temps['plateT']-temps['Batt_BF_Top'])
    Q_cond_BFT_pm = ENABLE_CONDUCTION * C_cond_Batt_plate*(temps['plateM']-temps['Batt_BF_Top'])

    #BFB
    Q_r_BFB_BMB = ENABLE_RADIATION * C_rad_batt_batt*(T4s['Batt_BM_Bot']-T4s['Batt_BF_Bot'])
    Q_r_BFB_BH1 = ENABLE_RADIATION * C_rad_batt_bh*(T4s['BH_1']-T4s['Batt_BF_Bot'])
    Q_r_BFB_BH2 = ENABLE_RADIATION * C_rad_batt_bh*(T4s['BH_2']-T4s['Batt_BF_Bot'])
    Q_r_BFB_ESC = ENABLE_RADIATION * C_rad_batt_esc*(T4s['ESC']-T4s['Batt_BF_Bot'])
    Q_r_BFB_BS = ENABLE_RADIATION * C_rad_batt_ts*(T4s['Bot_Shell_Int']-T4s['Batt_BF_Bot'])
    Q_v_BFB_air = (ENABLE_CONVECTION * get_h(temps['Batt_BF_Bot'],config.LC_batt_horiz,False)*config.A_conv_batt_top+ENABLE_CONVECTION * get_h(temps['Batt_BF_Bot'],config.LC_batt_vert,True)*config.A_conv_batt_side)*(temps['Internal_air']-temps['Batt_BF_Bot'])
    Q_cond_BFB_pm = ENABLE_CONDUCTION * C_cond_Batt_plate*(temps['plateM']-temps['Batt_BF_Bot'])
    Q_cond_BFB_pb = ENABLE_CONDUCTION * C_cond_Batt_plate*(temps['plateB']-temps['Batt_BF_Bot'])

    #BMT
    Q_r_BMT_BFT = ENABLE_RADIATION * C_rad_batt_batt*(T4s['Batt_BF_Top']-T4s['Batt_BM_Top'])
    Q_r_BMT_BRT = ENABLE_RADIATION * C_rad_batt_batt*(T4s['Batt_BR_Top']-T4s['Batt_BM_Top'])
    Q_r_BMT_BH2 = ENABLE_RADIATION * C_rad_batt_bh*(T4s['BH_2']-T4s['Batt_BM_Top'])
    Q_r_BMT_BH3 = ENABLE_RADIATION * C_rad_batt_bh*(T4s['BH_3']-T4s['Batt_BM_Top'])
    Q_r_BMT_TS = ENABLE_RADIATION * C_rad_batt_ts*(T4s['Top_Shell_Int']-T4s['Batt_BM_Top'])
    Q_v_BMT_air = (ENABLE_CONVECTION * get_h(temps['Batt_BM_Top'],config.LC_batt_horiz,False)*config.A_conv_batt_top+ENABLE_CONVECTION * get_h(temps['Batt_BM_Top'],config.LC_batt_vert,True)*config.A_conv_batt_side)*(temps['Internal_air']-temps['Batt_BM_Top'])
    Q_cond_BMT_pt = ENABLE_CONDUCTION * C_cond_Batt_plate*(temps['plateT']-temps['Batt_BM_Top'])
    Q_cond_BMT_pm = ENABLE_CONDUCTION * C_cond_Batt_plate*(temps['plateM']-temps['Batt_BM_Top'])
    
    #BMB
    Q_r_BMB_BFB = ENABLE_RADIATION * C_rad_batt_batt*(T4s['Batt_BF_Bot']-T4s['Batt_BM_Bot'])
    Q_r_BMB_BRB = ENABLE_RADIATION * C_rad_batt_batt*(T4s['Batt_BR_Bot']-T4s['Batt_BM_Bot'])
    Q_r_BMB_BH2 = ENABLE_RADIATION * C_rad_batt_bh*(T4s['BH_2']-T4s['Batt_BM_Bot'])
    Q_r_BMB_BH3 = ENABLE_RADIATION * C_rad_batt_bh*(T4s['BH_3']-T4s['Batt_BM_Bot'])
    Q_r_BMB_BS = ENABLE_RADIATION * C_rad_batt_ts*(T4s['Bot_Shell_Int']-T4s['Batt_BM_Bot'])
    Q_v_BMB_air = (ENABLE_CONVECTION * get_h(temps['Batt_BM_Bot'],config.LC_batt_horiz,False)*config.A_conv_batt_top+ENABLE_CONVECTION * get_h(temps['Batt_BM_Bot'],config.LC_batt_vert,True)*config.A_conv_batt_side)*(temps['Internal_air']-temps['Batt_BM_Bot'])
    Q_cond_BMB_pm = ENABLE_CONDUCTION * C_cond_Batt_plate*(temps['plateM']-temps['Batt_BM_Bot'])
    Q_cond_BMB_pb = ENABLE_CONDUCTION * C_cond_Batt_plate*(temps['plateB']-temps['Batt_BM_Bot'])

    #BRT
    Q_r_BRT_BMT = ENABLE_RADIATION * C_rad_batt_batt*(T4s['Batt_BM_Top']-T4s['Batt_BR_Top'])
    Q_r_BRT_BH3 = ENABLE_RADIATION * C_rad_batt_bh*(T4s['BH_3']-T4s['Batt_BR_Top'])
    Q_r_BRT_BH4 = ENABLE_RADIATION * C_rad_batt_bh*(T4s['BH_4']-T4s['Batt_BR_Top'])
    Q_r_BRT_TS = ENABLE_RADIATION * C_rad_batt_ts*(T4s['Top_Shell_Int']-T4s['Batt_BR_Top'])
    Q_v_BRT_air = (ENABLE_CONVECTION * get_h(temps['Batt_BR_Top'],config.LC_batt_horiz,False)*config.A_conv_batt_top+ENABLE_CONVECTION * get_h(temps['Batt_BR_Top'],config.LC_batt_vert,True)*config.A_conv_batt_side)*(temps['Internal_air']-temps['Batt_BR_Top'])
    Q_cond_BRT_pt = ENABLE_CONDUCTION * C_cond_Batt_plate*(temps['plateT']-temps['Batt_BR_Top'])
    Q_cond_BRT_pm = ENABLE_CONDUCTION * C_cond_Batt_plate*(temps['plateM']-temps['Batt_BR_Top']) 

    #BRB
    Q_r_BRB_BMB = ENABLE_RADIATION * C_rad_batt_batt*(T4s['Batt_BM_Bot']-T4s['Batt_BR_Bot'])
    Q_r_BRB_BH3 = ENABLE_RADIATION * C_rad_batt_bh*(T4s['BH_3']-T4s['Batt_BR_Bot'])
    Q_r_BRB_BH4 = ENABLE_RADIATION * C_rad_batt_bh*(T4s['BH_4']-T4s['Batt_BR_Bot'])
    Q_r_BRB_BS = ENABLE_RADIATION * C_rad_batt_ts*(T4s['Bot_Shell_Int']-T4s['Batt_BR_Bot'])
    Q_v_BRB_air = (ENABLE_CONVECTION * get_h(temps['Batt_BR_Bot'],config.LC_batt_horiz,False)*config.A_conv_batt_top+ENABLE_CONVECTION * get_h(temps['Batt_BR_Bot'],config.LC_batt_vert,True)*config.A_conv_batt_side)*(temps['Internal_air']-temps['Batt_BR_Bot'])
    Q_cond_BRB_pm = ENABLE_CONDUCTION * C_cond_Batt_plate*(temps['plateM']-temps['Batt_BR_Bot'])
    Q_cond_BRB_pb = ENABLE_CONDUCTION * C_cond_Batt_plate*(temps['plateB']-temps['Batt_BR_Bot'])

    #PlateT
    Q_cond_pt_BFT = ENABLE_CONDUCTION * C_cond_Batt_plate*(temps['Batt_BF_Top']-temps['plateT'])
    Q_cond_pt_BMT = ENABLE_CONDUCTION * C_cond_Batt_plate*(temps['Batt_BM_Top']-temps['plateT'])
    Q_cond_pt_BRT = ENABLE_CONDUCTION * C_cond_Batt_plate*(temps['Batt_BR_Top']-temps['plateT'])
    Q_cond_pt_BH1 = ENABLE_CONDUCTION * C_cond_bh_plate*(temps['BH_1']-temps['plateT'])
    Q_cond_pt_BH2 = ENABLE_CONDUCTION * C_cond_bh_plate*(temps['BH_2']-temps['plateT'])
    Q_cond_pt_BH3 = ENABLE_CONDUCTION * C_cond_bh_plate*(temps['BH_3']-temps['plateT'])
    Q_cond_pt_BH4 = ENABLE_CONDUCTION * C_cond_bh_plate*(temps['BH_4']-temps['plateT'])
    Q_r_pt_ts = ENABLE_RADIATION * C_rad_plate_sh*(T4s['Top_Shell_Int']-T4s['plateT'])
    Q_v_pt_air = ENABLE_CONVECTION * get_h(temps['plateT'],config.LC_plate,False)*config.A_conv_plate*(temps['Internal_air']-temps['plateT'])

    #PlateM
    Q_cond_pm_BFT = ENABLE_CONDUCTION * C_cond_Batt_plate*(temps['Batt_BF_Top']-temps['plateM'])
    Q_cond_pm_BMT = ENABLE_CONDUCTION * C_cond_Batt_plate*(temps['Batt_BM_Top']-temps['plateM'])
    Q_cond_pm_BRT = ENABLE_CONDUCTION * C_cond_Batt_plate*(temps['Batt_BR_Top']-temps['plateM'])
    Q_cond_pm_BFB = ENABLE_CONDUCTION * C_cond_Batt_plate*(temps['Batt_BF_Bot']-temps['plateM'])
    Q_cond_pm_BMB = ENABLE_CONDUCTION * C_cond_Batt_plate*(temps['Batt_BM_Bot']-temps['plateM'])
    Q_cond_pm_BRB = ENABLE_CONDUCTION * C_cond_Batt_plate*(temps['Batt_BR_Bot']-temps['plateM'])
    Q_cond_pm_BH1 = ENABLE_CONDUCTION * C_cond_bh_plate*(temps['BH_1']-temps['plateM'])
    Q_cond_pm_BH2 = ENABLE_CONDUCTION * C_cond_bh_plate*(temps['BH_2']-temps['plateM'])
    Q_cond_pm_BH3 = ENABLE_CONDUCTION * C_cond_bh_plate*(temps['BH_3']-temps['plateM'])
    Q_cond_pm_BH4 = ENABLE_CONDUCTION * C_cond_bh_plate*(temps['BH_4']-temps['plateM'])
    Q_v_pm_air = ENABLE_CONVECTION * get_h(temps['plateM'],config.LC_plate,False)*config.A_conv_plateM*(temps['Internal_air']-temps['plateM'])
 
    #PlateB
    Q_cond_pb_BFB = ENABLE_CONDUCTION * C_cond_Batt_plate*(temps['Batt_BF_Bot']-temps['plateB'])
    Q_cond_pb_BMB = ENABLE_CONDUCTION * C_cond_Batt_plate*(temps['Batt_BM_Bot']-temps['plateB'])
    Q_cond_pb_BRB = ENABLE_CONDUCTION * C_cond_Batt_plate*(temps['Batt_BR_Bot']-temps['plateB'])
    Q_cond_pb_BH1 = ENABLE_CONDUCTION * C_cond_bh_plate*(temps['BH_1']-temps['plateB'])
    Q_cond_pb_BH2 = ENABLE_CONDUCTION * C_cond_bh_plate*(temps['BH_2']-temps['plateB'])
    Q_cond_pb_BH3 = ENABLE_CONDUCTION * C_cond_bh_plate*(temps['BH_3']-temps['plateB'])
    Q_cond_pb_BH4 = ENABLE_CONDUCTION * C_cond_bh_plate*(temps['BH_4']-temps['plateB'])
    Q_r_pb_bs = ENABLE_RADIATION * C_rad_plate_sh*(T4s['Bot_Shell_Int']-T4s['plateB'])
    Q_v_pb_air = ENABLE_CONVECTION * get_h(temps['plateB'],config.LC_plate,False)*config.A_conv_plate*(temps['Internal_air']-temps['plateB'])

    #BH1
    Q_r_BH1_BFT = ENABLE_RADIATION * C_rad_batt_bh*(T4s['Batt_BF_Top']-T4s['BH_1'])
    Q_r_BH1_BFB = ENABLE_RADIATION * C_rad_batt_bh*(T4s['Batt_BF_Bot']-T4s['BH_1'])
    Q_r_BH1_BH2 = ENABLE_RADIATION * C_rad_bh_bh*(T4s['BH_2']-T4s['BH_1'])
    Q_r_BH1_esc = ENABLE_RADIATION * C_rad_esc_bh*(T4s['ESC']-T4s['BH_1'])
    Q_cond_BH1_Mount = ENABLE_CONDUCTION * C_cond_Mount_to_BH1*(temps['ESC_Mount']-temps['BH_1'])
    Q_cond_BH1_TS = ENABLE_CONDUCTION * C_cond_BH_TS*(temps['Top_Shell_Int']-temps['BH_1'])
    Q_cond_BH1_BS = ENABLE_CONDUCTION * C_cond_BH_BS*(temps['Bot_Shell_Int']-temps['BH_1'])
    Q_cond_BH1_pt = ENABLE_CONDUCTION * C_cond_bh_plate*(temps['plateT']-temps['BH_1'])
    Q_cond_BH1_pm = ENABLE_CONDUCTION * C_cond_bh_plate*(temps['plateM']-temps['BH_1'])
    Q_cond_BH1_pb = ENABLE_CONDUCTION * C_cond_bh_plate*(temps['plateB']-temps['BH_1'])
    Q_v_BH1_air = ENABLE_CONVECTION * get_h(temps['BH_1'],config.LC_bulkhead,True)*config.A_bulkhead_face*2*(temps['Internal_air']-temps['BH_1'])
    
    #BH2
    Q_r_BH2_BFT = ENABLE_RADIATION * C_rad_batt_bh*(T4s['Batt_BF_Top']-T4s['BH_2'])
    Q_r_BH2_BFB = ENABLE_RADIATION * C_rad_batt_bh*(T4s['Batt_BF_Bot']-T4s['BH_2'])
    Q_r_BH2_BMT = ENABLE_RADIATION * C_rad_batt_bh*(T4s['Batt_BM_Top']-T4s['BH_2'])
    Q_r_BH2_BMB = ENABLE_RADIATION * C_rad_batt_bh*(T4s['Batt_BM_Bot']-T4s['BH_2'])
    Q_r_BH2_BH1 = ENABLE_RADIATION * C_rad_bh_bh*(T4s['BH_1']-T4s['BH_2'])
    Q_r_BH2_BH3 = ENABLE_RADIATION * C_rad_bh_bh*(T4s['BH_3']-T4s['BH_2'])
    Q_cond_BH2_TS = ENABLE_CONDUCTION * C_cond_BH_TS*(temps['Top_Shell_Int']-temps['BH_2'])
    Q_cond_BH2_BS = ENABLE_CONDUCTION * C_cond_BH_BS*(temps['Bot_Shell_Int']-temps['BH_2'])
    Q_cond_BH2_pt = ENABLE_CONDUCTION * C_cond_bh_plate*(temps['plateT']-temps['BH_2'])
    Q_cond_BH2_pm = ENABLE_CONDUCTION * C_cond_bh_plate*(temps['plateM']-temps['BH_2'])
    Q_cond_BH2_pb = ENABLE_CONDUCTION * C_cond_bh_plate*(temps['plateB']-temps['BH_2'])
    #Q_v_BH2_air = ENABLE_CONVECTION * get_h(temps['BH_2'],config.LC_bulkhead,True)*config.A_bulkhead_face*2*(temps['Internal_air']-temps['BH_2'])

    #BH3
    Q_r_BH3_BMT = ENABLE_RADIATION * C_rad_batt_bh*(T4s['Batt_BM_Top']-T4s['BH_3'])
    Q_r_BH3_BMB = ENABLE_RADIATION * C_rad_batt_bh*(T4s['Batt_BM_Bot']-T4s['BH_3'])
    Q_r_BH3_BRT = ENABLE_RADIATION * C_rad_batt_bh*(T4s['Batt_BR_Top']-T4s['BH_3'])
    Q_r_BH3_BRB = ENABLE_RADIATION * C_rad_batt_bh*(T4s['Batt_BR_Bot']-T4s['BH_3'])
    Q_r_BH3_BH2 = ENABLE_RADIATION * C_rad_bh_bh*(T4s['BH_2']-T4s['BH_3'])
    Q_r_BH3_BH4 = ENABLE_RADIATION * C_rad_bh_bh*(T4s['BH_4']-T4s['BH_3'])
    Q_cond_BH3_TS = ENABLE_CONDUCTION * C_cond_BH_TS*(temps['Top_Shell_Int']-temps['BH_3'])
    Q_cond_BH3_BS = ENABLE_CONDUCTION * C_cond_BH_BS*(temps['Bot_Shell_Int']-temps['BH_3'])
    Q_cond_BH3_pt = ENABLE_CONDUCTION * C_cond_bh_plate*(temps['plateT']-temps['BH_3'])
    Q_cond_BH3_pm = ENABLE_CONDUCTION * C_cond_bh_plate*(temps['plateM']-temps['BH_3'])
    Q_cond_BH3_pb = ENABLE_CONDUCTION * C_cond_bh_plate*(temps['plateB']-temps['BH_3'])
    Q_v_BH3_air = ENABLE_CONVECTION * get_h(temps['BH_3'],config.LC_bulkhead,True)*config.A_bulkhead_face*2*(temps['Internal_air']-temps['BH_3'])

    #BH4
    Q_r_BH4_BRT = ENABLE_RADIATION * C_rad_batt_bh*(T4s['Batt_BR_Top']-T4s['BH_4'])
    Q_r_BH4_BRB = ENABLE_RADIATION * C_rad_batt_bh*(T4s['Batt_BR_Bot']-T4s['BH_4'])
    Q_r_BH4_BH3 = ENABLE_RADIATION * C_rad_bh_bh*(T4s['BH_3']-T4s['BH_4'])
    Q_cond_BH4_TS = ENABLE_CONDUCTION * C_cond_BH_TS*(temps['Top_Shell_Int']-temps['BH_4'])
    Q_cond_BH4_BS = ENABLE_CONDUCTION * C_cond_BH_BS*(temps['Bot_Shell_Int']-temps['BH_4'])
    Q_cond_BH4_pt = ENABLE_CONDUCTION * C_cond_bh_plate*(temps['plateT']-temps['BH_4'])
    Q_cond_BH4_pm = ENABLE_CONDUCTION * C_cond_bh_plate*(temps['plateM']-temps['BH_4'])
    Q_cond_BH4_pb = ENABLE_CONDUCTION * C_cond_bh_plate*(temps['plateB']-temps['BH_4'])
    Q_v_BH4_air = ENABLE_CONVECTION * get_h(temps['BH_4'],config.LC_bulkhead,True)*config.A_bulkhead_face*2*(temps['Internal_air']-temps['BH_4'])

    #ESC
    Q_r_ESC_BFT = ENABLE_RADIATION * C_rad_batt_esc*(T4s['Batt_BF_Top']-T4s['ESC'])
    Q_r_ESC_BFB = ENABLE_RADIATION * C_rad_batt_esc*(T4s['Batt_BF_Bot']-T4s['ESC'])
    Q_r_ESC_BH1 = ENABLE_RADIATION * C_rad_esc_bh*(T4s['BH_1']-T4s['ESC'])
    Q_r_ESC_TS = ENABLE_RADIATION * C_rad_esc_ts*(T4s['Top_Shell_Int']-T4s['ESC'])
    Q_cond_ESC_Mount = ENABLE_CONDUCTION * C_cond_ESC_to_Mount*(temps['ESC_Mount']-temps['ESC'])
    Q_v_ESC_air = (ENABLE_CONVECTION * get_h(temps['ESC'],config.LC_esc_horiz,False)*config.A_conv_esc_top+ENABLE_CONVECTION * get_h(temps['ESC'],config.LC_esc_vert,True)*config.A_conv_esc_side)*(temps['Internal_air']-temps['ESC'])

    #ESC Mount
    Q_cond_Mount_BH1 = ENABLE_CONDUCTION * C_cond_Mount_to_BH1*(temps['BH_1']-temps['ESC_Mount'])
    Q_cond_Mount_ESC = ENABLE_CONDUCTION * C_cond_ESC_to_Mount*(temps['ESC']-temps['ESC_Mount'])
    Q_r_Mount_BS = ENABLE_RADIATION * C_rad_mount_bs*(T4s['Bot_Shell_Int']-T4s['ESC_Mount'])
    Q_v_Mount_air = ENABLE_CONVECTION * get_h(temps['ESC_Mount'],config.LC_mount,False)*config.A_mount_conv*(temps['Internal_air']-temps['ESC_Mount'])

    #Top_Shell_Int
    Q_r_TS_BFT = ENABLE_RADIATION * C_rad_batt_ts*(T4s['Batt_BF_Top']-T4s['Top_Shell_Int'])
    Q_r_TS_BMT = ENABLE_RADIATION * C_rad_batt_ts*(T4s['Batt_BM_Top']-T4s['Top_Shell_Int'])
    Q_r_TS_BRT = ENABLE_RADIATION * C_rad_batt_ts*(T4s['Batt_BR_Top']-T4s['Top_Shell_Int'])
    Q_r_TS_ESC = ENABLE_RADIATION * C_rad_esc_ts*(T4s['ESC']-T4s['Top_Shell_Int'])
    Q_r_ts_pt = ENABLE_RADIATION * C_rad_plate_sh*(T4s['plateT']-T4s['Top_Shell_Int'])
    Q_r_TS_BS = ENABLE_RADIATION * C_rad_ts_bs*(T4s['Bot_Shell_Int']-T4s['Top_Shell_Int'])
    Q_cond_TS_BH1 = ENABLE_CONDUCTION * C_cond_BH_TS*(temps['BH_1']-temps['Top_Shell_Int'])
    Q_cond_TS_BH2 = ENABLE_CONDUCTION * C_cond_BH_TS*(temps['BH_2']-temps['Top_Shell_Int'])
    Q_cond_TS_BH3 = ENABLE_CONDUCTION * C_cond_BH_TS*(temps['BH_3']-temps['Top_Shell_Int'])
    Q_cond_TS_BH4 = ENABLE_CONDUCTION * C_cond_BH_TS*(temps['BH_4']-temps['Top_Shell_Int'])
    Q_cond_TSi_TSe = ENABLE_CONDUCTION * C_cond_TS_int_ext*(temps['Top_Shell_Ext']-temps['Top_Shell_Int'])
    Q_v_TSi_air = ENABLE_CONVECTION * get_h(temps['Top_Shell_Int'],config.LC_TS_int,False)*config.A_TS*(temps['Internal_air']-temps['Top_Shell_Int'])

    #Top Shell Ext
    Q_cond_TSe_TSi = ENABLE_CONDUCTION * C_cond_TS_int_ext*(temps['Top_Shell_Int']-temps['Top_Shell_Ext'])
    Q_v_TSe_Amb = ENABLE_RADIATION * physics_models.get_external_convection_h(p_ambient,temps['Top_Shell_Ext'],T_E,config.LC_TS_ext)*config.A_TS*(T_E-temps['Top_Shell_Ext'])
    Q_r_TSe_Ext = ext_loads['Q_ext_top']

    #Bottom Shell Int
    Q_r_BS_BFB = ENABLE_RADIATION * C_rad_batt_ts*(T4s['Batt_BF_Bot']-T4s['Bot_Shell_Int'])
    Q_r_BS_BMB = ENABLE_RADIATION * C_rad_batt_ts*(T4s['Batt_BM_Bot']-T4s['Bot_Shell_Int'])
    Q_r_BS_BRB = ENABLE_RADIATION * C_rad_batt_ts*(T4s['Batt_BR_Bot']-T4s['Bot_Shell_Int'])
    Q_r_BS_Mount = ENABLE_RADIATION * C_rad_mount_bs*(T4s['ESC_Mount']-T4s['Bot_Shell_Int'])
    Q_r_bs_pb = ENABLE_RADIATION * C_rad_plate_sh*(T4s['plateB']-T4s['Bot_Shell_Int'])
    Q_r_BS_TS = ENABLE_RADIATION * C_rad_ts_bs*(T4s['Top_Shell_Int']-T4s['Bot_Shell_Int'])
    Q_cond_BS_BH1 = ENABLE_CONDUCTION * C_cond_BH_BS*(temps['BH_1']-temps['Bot_Shell_Int'])
    Q_cond_BS_BH2 = ENABLE_CONDUCTION * C_cond_BH_BS*(temps['BH_2']-temps['Bot_Shell_Int'])
    Q_cond_BS_BH3 = ENABLE_CONDUCTION * C_cond_BH_BS*(temps['BH_3']-temps['Bot_Shell_Int'])
    Q_cond_BS_BH4 = ENABLE_CONDUCTION * C_cond_BH_BS*(temps['BH_4']-temps['Bot_Shell_Int'])
    Q_cond_BSi_BSe = ENABLE_CONDUCTION * C_cond_BS_int_ext*(temps['Bot_Shell_Ext']-temps['Bot_Shell_Int'])
    Q_v_BSi_air = ENABLE_CONVECTION * get_h(temps['Bot_Shell_Int'],config.LC_BS_int,False)*config.A_BS*(temps['Internal_air']-temps['Bot_Shell_Int'])

    #Bottom Shell Ext
    Q_cond_BSe_BSi = ENABLE_CONDUCTION * C_cond_BS_int_ext*(temps['Bot_Shell_Int']-temps['Bot_Shell_Ext'])
    Q_v_BSe_Amb = ENABLE_RADIATION * physics_models.get_external_convection_h(p_ambient,temps['Bot_Shell_Ext'],T_E,config.LC_BS_ext)*config.A_BS*(T_E-temps['Bot_Shell_Ext'])
    Q_r_BSe_Ext = ext_loads['Q_ext_bottom']

    #Internal air
    Q_v_air_BFT = (ENABLE_CONVECTION * get_h(temps['Batt_BF_Top'],config.LC_batt_horiz,False)*config.A_conv_batt_top+ENABLE_CONVECTION * get_h(temps['Batt_BF_Top'],config.LC_batt_vert,True)*config.A_conv_batt_side)*(temps['Batt_BF_Top']-temps['Internal_air'])
    Q_v_air_BFB = (ENABLE_CONVECTION * get_h(temps['Batt_BF_Bot'],config.LC_batt_horiz,False)*config.A_conv_batt_top+ENABLE_CONVECTION * get_h(temps['Batt_BF_Bot'],config.LC_batt_vert,True)*config.A_conv_batt_side)*(temps['Batt_BF_Bot']-temps['Internal_air'])
    Q_v_air_BMT = (ENABLE_CONVECTION * get_h(temps['Batt_BM_Top'],config.LC_batt_horiz,False)*config.A_conv_batt_top+ENABLE_CONVECTION * get_h(temps['Batt_BM_Top'],config.LC_batt_vert,True)*config.A_conv_batt_side)*(temps['Batt_BM_Top']-temps['Internal_air'])
    Q_v_air_BMB = (ENABLE_CONVECTION * get_h(temps['Batt_BM_Bot'],config.LC_batt_horiz,False)*config.A_conv_batt_top+ENABLE_CONVECTION * get_h(temps['Batt_BM_Bot'],config.LC_batt_vert,True)*config.A_conv_batt_side)*(temps['Batt_BM_Bot']-temps['Internal_air'])
    Q_v_air_BRT = (ENABLE_CONVECTION * get_h(temps['Batt_BR_Top'],config.LC_batt_horiz,False)*config.A_conv_batt_top+ENABLE_CONVECTION * get_h(temps['Batt_BR_Top'],config.LC_batt_vert,True)*config.A_conv_batt_side)*(temps['Batt_BR_Top']-temps['Internal_air'])
    Q_v_air_BRB = (ENABLE_CONVECTION * get_h(temps['Batt_BR_Bot'],config.LC_batt_horiz,False)*config.A_conv_batt_top+ENABLE_CONVECTION * get_h(temps['Batt_BR_Bot'],config.LC_batt_vert,True)*config.A_conv_batt_side)*(temps['Batt_BR_Bot']-temps['Internal_air'])
    Q_v_air_pt = ENABLE_CONVECTION * get_h(temps['plateT'],config.LC_plate,False)*config.A_conv_plate*(temps['plateT']-temps['Internal_air'])
    Q_v_air_pm = ENABLE_CONVECTION * get_h(temps['plateM'],config.LC_plate,False)*config.A_conv_plateM*(temps['plateM']-temps['Internal_air'])
    Q_v_air_pb = ENABLE_CONVECTION * get_h(temps['plateB'],config.LC_plate,False)*config.A_conv_plate*(temps['plateB']-temps['Internal_air'])
    Q_v_air_BH1 = ENABLE_CONVECTION * get_h(temps['BH_1'],config.LC_bulkhead,True)*config.A_bulkhead_face*2*(temps['BH_1']-temps['Internal_air'])
    #Q_v_air_BH2 = ENABLE_CONVECTION * get_h(temps['BH_2'],config.LC_bulkhead,True)*config.A_bulkhead_face*2*(temps['BH_2']-temps['Internal_air'])
    Q_v_air_BH3 = ENABLE_CONVECTION * get_h(temps['BH_3'],config.LC_bulkhead,True)*config.A_bulkhead_face*2*(temps['BH_3']-temps['Internal_air'])
    Q_v_air_BH4 = ENABLE_CONVECTION * get_h(temps['BH_4'],config.LC_bulkhead,True)*config.A_bulkhead_face*2*(temps['BH_4']-temps['Internal_air'])
    Q_v_air_ESC = (ENABLE_CONVECTION * get_h(temps['ESC'],config.LC_esc_horiz,False)*config.A_conv_esc_top+ENABLE_CONVECTION * get_h(temps['ESC'],config.LC_esc_vert,True)*config.A_conv_esc_side)*(temps['ESC']-temps['Internal_air'])
    Q_v_air_Mount = ENABLE_CONVECTION * get_h(temps['ESC_Mount'],config.LC_mount,False)*config.A_mount_conv*(temps['ESC_Mount']-temps['Internal_air'])
    Q_v_air_TSi = ENABLE_CONVECTION * get_h(temps['Top_Shell_Int'],config.LC_TS_int,False)*config.A_TS*(temps['Top_Shell_Int']-temps['Internal_air'])
    Q_v_air_BSi = ENABLE_CONVECTION * get_h(temps['Bot_Shell_Int'],config.LC_BS_int,False)*config.A_BS*(temps['Bot_Shell_Int']-temps['Internal_air'])
    Q_conv_from_air_to_BH2 = get_h(temps['BH_2'],config.LC_bulkhead,True)*config.A_bulkhead_face*2*(temps['Internal_air']-temps['BH_2'])


    # --- NET HEAT BALANCE ---
    
    # BATTERY NODES
    net_Q_BFT = config.Q_batt_zone + (Q_r_BFT_BMT + Q_r_BFT_BH1 + Q_r_BFT_BH2 + Q_r_BFT_ESC + Q_r_BFT_TS + Q_v_BFT_air + Q_cond_BFT_pt + Q_cond_BFT_pm)
    net_Q_BFB = config.Q_batt_zone + (Q_r_BFB_BMB + Q_r_BFB_BH1 + Q_r_BFB_BH2 + Q_r_BFB_ESC + Q_r_BFB_BS + Q_v_BFB_air + Q_cond_BFB_pm + Q_cond_BFB_pb)
    net_Q_BMT = config.Q_batt_zone + (Q_r_BMT_BFT + Q_r_BMT_BRT + Q_r_BMT_BH2 + Q_r_BMT_BH3 + Q_r_BMT_TS + Q_v_BMT_air + Q_cond_BMT_pt + Q_cond_BMT_pm)
    net_Q_BMB = config.Q_batt_zone + (Q_r_BMB_BFB + Q_r_BMB_BRB + Q_r_BMB_BH2 + Q_r_BMB_BH3 + Q_r_BMB_BS + Q_v_BMB_air + Q_cond_BMB_pm + Q_cond_BMB_pb)
    net_Q_BRT = config.Q_batt_zone + (Q_r_BRT_BMT + Q_r_BRT_BH3 + Q_r_BRT_BH4 + Q_r_BRT_TS + Q_v_BRT_air + Q_cond_BRT_pt + Q_cond_BRT_pm)
    net_Q_BRB = config.Q_batt_zone + (Q_r_BRB_BMB + Q_r_BRB_BH3 + Q_r_BRB_BH4 + Q_r_BRB_BS + Q_v_BRB_air + Q_cond_BRB_pm + Q_cond_BRB_pb)

    # PLATE NODES
    net_Q_plateT = Q_cond_pt_BFT + Q_cond_pt_BMT + Q_cond_pt_BRT + Q_cond_pt_BH1 + Q_cond_pt_BH2 + Q_cond_pt_BH3 + Q_cond_pt_BH4 + Q_r_pt_ts + Q_v_pt_air
    net_Q_plateM = Q_cond_pm_BFT + Q_cond_pm_BMT + Q_cond_pm_BRT + Q_cond_pm_BFB + Q_cond_pm_BMB + Q_cond_pm_BRB + Q_cond_pm_BH1 + Q_cond_pm_BH2 + Q_cond_pm_BH3 + Q_cond_pm_BH4 + Q_v_pm_air
    net_Q_plateB = Q_cond_pb_BFB + Q_cond_pb_BMB + Q_cond_pb_BRB + Q_cond_pb_BH1 + Q_cond_pb_BH2 + Q_cond_pb_BH3 + Q_cond_pb_BH4 + Q_r_pb_bs + Q_v_pb_air

    # AVIONICS NODES
    net_Q_ESC = config.Q_ESC + (Q_r_ESC_BFT + Q_r_ESC_BFB + Q_r_ESC_BH1 + Q_r_ESC_TS + Q_cond_ESC_Mount + Q_v_ESC_air)
    net_Q_Mount = (Q_cond_Mount_BH1 + Q_cond_Mount_ESC + Q_r_Mount_BS + Q_v_Mount_air)

    # BULKHEAD NODES
    net_Q_BH1 = (Q_r_BH1_BFT + Q_r_BH1_BFB + Q_r_BH1_BH2 + Q_r_BH1_esc + Q_cond_BH1_Mount + Q_cond_BH1_TS + Q_cond_BH1_BS + Q_cond_BH1_pt + Q_cond_BH1_pm + Q_cond_BH1_pb + Q_v_BH1_air)
    net_Q_BH2 = (Q_r_BH2_BFT + Q_r_BH2_BFB + Q_r_BH2_BMT + Q_r_BH2_BMB + Q_r_BH2_BH1 + Q_r_BH2_BH3 + Q_cond_BH2_TS + Q_cond_BH2_BS + Q_cond_BH2_pt + Q_cond_BH2_pm + Q_cond_BH2_pb + Q_conv_from_air_to_BH2)
    net_Q_BH3 = (Q_r_BH3_BMT + Q_r_BH3_BMB + Q_r_BH3_BRT + Q_r_BH3_BRB + Q_r_BH3_BH2 + Q_r_BH3_BH4 + Q_cond_BH3_TS + Q_cond_BH3_BS + Q_cond_BH3_pt + Q_cond_BH3_pm + Q_cond_BH3_pb + Q_v_BH3_air)
    net_Q_BH4 = (Q_r_BH4_BRT + Q_r_BH4_BRB + Q_r_BH4_BH3 + Q_cond_BH4_TS + Q_cond_BH4_BS + Q_cond_BH4_pt + Q_cond_BH4_pm + Q_cond_BH4_pb + Q_v_BH4_air)

    # SHELL NODES
    net_Q_TS_int = (Q_r_TS_BFT + Q_r_TS_BMT + Q_r_TS_BRT + Q_r_TS_ESC + Q_r_ts_pt + Q_r_TS_BS + Q_cond_TS_BH1 + Q_cond_TS_BH2 + Q_cond_TS_BH3 + Q_cond_TS_BH4 + Q_cond_TSi_TSe + Q_v_TSi_air)
    net_Q_BS_int = (Q_r_BS_BFB + Q_r_BS_BMB + Q_r_BS_BRB + Q_r_BS_Mount + Q_r_bs_pb + Q_r_BS_TS + Q_cond_BS_BH1 + Q_cond_BS_BH2 + Q_cond_BS_BH3 + Q_cond_BS_BH4 + Q_cond_BSi_BSe + Q_v_BSi_air)
    net_Q_TS_ext = (Q_cond_TSe_TSi + Q_v_TSe_Amb + Q_r_TSe_Ext)
    net_Q_BS_ext = (Q_cond_BSe_BSi + Q_v_BSe_Amb + Q_r_BSe_Ext)

    # INTERNAL air NODE
    #net_Q_air = (Q_v_air_BFT + Q_v_air_BFB + Q_v_air_BMT + Q_v_air_BMB + Q_v_air_BRT + Q_v_air_BRB + Q_v_air_pt + Q_v_air_pm + Q_v_air_pb + Q_v_air_BH1 - Q_conv_from_air_to_BH2 + Q_v_air_BH3 + Q_v_air_BH4 + Q_v_air_ESC + Q_v_air_Mount + Q_v_air_TSi + Q_v_air_BSi) 
    net_Q_air = -(
        Q_v_BFT_air + Q_v_BFB_air + Q_v_BMT_air + Q_v_BMB_air + Q_v_BRT_air + Q_v_BRB_air +
        Q_v_pt_air + Q_v_pm_air + Q_v_pb_air +
        Q_v_BH1_air + Q_conv_from_air_to_BH2 + Q_v_BH3_air + Q_v_BH4_air +
        Q_v_ESC_air + Q_v_Mount_air +
        Q_v_TSi_air + Q_v_BSi_air
    )
    # dT/dt
    mC_batt = config.m_batt_zone*config.C_B 
    mC_bh = config.m_bulkhead*config.C_bulkhead
    mC_TS = config.m_TS*config.C_TS
    mC_BS = config.m_BS*config.C_BS
    mC_ESC = config.m_ESC*config.C_ESC
    mC_Mount = config.m_mount*config.C_mount
    mC_plate = config.m_plate*config.C_plate
    p_air_rho, p_air_cp = p_air[0], p_air[1]

    # --- PROGRAMMATIC ENERGY BALANCE AUDIT (Temporary Debugging Code) ---
    if t == 0: # Only run this check once at the very beginning
        # 1. Sum of all net heat changes calculated for each node
        total_net_Q = (
            net_Q_BFT + net_Q_BFB + net_Q_BMT + net_Q_BMB + net_Q_BRT + net_Q_BRB +
            net_Q_plateT + net_Q_plateM + net_Q_plateB +
            net_Q_ESC + net_Q_Mount +
            net_Q_BH1 + net_Q_BH2 + net_Q_BH3 + net_Q_BH4 +
            net_Q_TS_int + net_Q_TS_ext + net_Q_BS_int + net_Q_BS_ext +
            net_Q_air
        )

        # 2. Sum of all heat sources/sinks entering or leaving the entire system
        # (Internal heat generation + all external loads)
        total_source_Q = (
            (config.Q_batt_zone * 6) + config.Q_ESC +
            Q_r_TSe_Ext + Q_r_BSe_Ext +
            Q_v_TSe_Amb + Q_v_BSe_Amb
        )

        # 3. Calculate the energy residual. In a perfect system, this should be zero.
        residual = total_net_Q - total_source_Q

        print("\n--- ENERGY BALANCE AUDIT (t=0) ---")
        print(f"Sum of all Net Q's: {total_net_Q:.6f} W")
        print(f"Sum of all Source Q's: {total_source_Q:.6f} W")
        print(f"Energy Residual (Net - Source): {residual:.6f} W")
        if abs(residual) > 1e-9:
            print("!!! WARNING: Energy is NOT being conserved. Check summation logic. !!!")
        else:
            print("--- Energy balance appears correct. ---")

    return [
        net_Q_BFT/mC_batt,
        net_Q_BFB/mC_batt,
        net_Q_BMT/mC_batt,
        net_Q_BMB/mC_batt,
        net_Q_BRT/mC_batt,
        net_Q_BRB/mC_batt,
        net_Q_plateT/mC_plate,
        net_Q_plateM/mC_plate,
        net_Q_plateB/mC_plate,
        net_Q_ESC/mC_ESC,
        net_Q_Mount/mC_Mount,
        net_Q_BH1/mC_bh,
        net_Q_BH2/mC_bh,
        net_Q_BH3/mC_bh,
        net_Q_BH4/mC_bh,
        net_Q_TS_int/mC_TS,
        net_Q_TS_ext/mC_TS,
        net_Q_BS_int/mC_BS,
        net_Q_BS_ext/mC_BS,
        net_Q_air/(p_air_rho*config.V_internal_air*p_air_cp)
    ]

# --- Main Execution Block ---
if __name__ == "__main__":
    print("\n--- Simulation Initial Conditions ---")
    print(f"Target Altitude: {config.TARGET_ALTITUDE_KM} km")
    print(f"Ambient Temp: {T_E:.2f} K | Ambient Pressure: {P_amb:.2f} Pa")
    print(f"aircraft Velocity: {config.velocity} m/s")
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