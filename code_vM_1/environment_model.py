# 4_environment_model.py
import numpy as np
import pandas as pd
import config

ENABLE_EXTERNAL_RADIATION = False
SIMULATION_START_HOUR = 8.0
FLUX_ALBEDO_FRACTION = 0.3
FLUX_PLANETARY = 237.0
solar_time_data, solar_flux_data, start_offset_seconds = [], [], 0

def init():
    global solar_time_data, solar_flux_data, start_offset_seconds
    print("Initializing External Environment Model...")
    start_offset_seconds = SIMULATION_START_HOUR * 3600.0
    try:
        df_solar = pd.read_csv("solar_flux_data.csv")
        solar_time_data, solar_flux_data = df_solar['time_seconds'].tolist(), df_solar['flux_w_m2'].tolist()
        print("Solar data loaded successfully.")
    except FileNotFoundError:
        print("!!! WARNING (Environment Model): solar_flux_data.csv not found."); solar_time_data, solar_flux_data = [0, 86400], [0, 0]

def calculate_external_heat_loads(t, T_TS_ext, T_BS_ext, T_E):
    if not ENABLE_EXTERNAL_RADIATION:
        return {'Q_ext_top': 0.0, 'Q_ext_bottom': 0.0}
    
    data_period_seconds = solar_time_data[-1] if solar_time_data else 0
    current_solar_flux = 0
    if data_period_seconds > 0:
        effective_time = t + start_offset_seconds
        lookup_time = effective_time % data_period_seconds
        current_solar_flux = np.interp(lookup_time, solar_time_data, solar_flux_data)
    
    # Calculate INFLOWS only. Outflows are handled in the main solver.
    Q_solar_in_TS = current_solar_flux * config.A_TS * config.alpha_solar_shell
    Q_albedo_in_BS = current_solar_flux * FLUX_ALBEDO_FRACTION * config.A_BS * config.alpha_solar_shell

    # The ground IR radiation is an inflow. We can add it back if needed, but for stability, start with this.
    #sigma = 5.67e-8
    #Q_ground_rad_in = sigma * config.A_BS * config.emis_shell_ext * (T_E**4)
    
    total_Q_top = Q_solar_in_TS
    total_Q_bottom = Q_albedo_in_BS  #+ Q_ground_rad_in
    
    return {'Q_ext_top': total_Q_top, 'Q_ext_bottom': total_Q_bottom}