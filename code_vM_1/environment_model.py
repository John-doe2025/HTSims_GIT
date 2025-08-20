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
    
    # Use time-averaged approach to avoid sharp transitions
    # Average solar flux over a day cycle (simplified model)
    avg_solar_flux = 400.0  # W/m² - typical average for UAV altitude
    
    # Apply gentle sinusoidal variation instead of sharp day/night cycles
    day_seconds = 24 * 3600
    time_of_day = (t + start_offset_seconds) % day_seconds
    solar_factor = 0.5 * (1 + 0.8 * np.sin(2 * np.pi * time_of_day / day_seconds - np.pi/2))
    
    # Smooth solar flux without sharp transitions
    current_solar_flux = avg_solar_flux * solar_factor
    
    # Calculate heat loads with conservative scaling
    Q_solar_in_TS = current_solar_flux * config.A_TS * config.alpha_solar_shell * 0.3  # 30% of theoretical
    Q_albedo_in_BS = current_solar_flux * FLUX_ALBEDO_FRACTION * config.A_BS * config.alpha_solar_shell * 0.3
    
    # Apply strict physical bounds
    Q_solar_in_TS = min(Q_solar_in_TS, 50.0)  # Conservative upper bound
    Q_albedo_in_BS = min(Q_albedo_in_BS, 15.0)  # Conservative upper bound
    
    return {'Q_ext_top': Q_solar_in_TS, 'Q_ext_bottom': Q_albedo_in_BS}