# 3_physics_models.py
import numpy as np
import pandas as pd
import config

# --- Load temperature-dependent data ONCE when the module is imported ---
try:
    df_air_temp = pd.read_excel("298.xlsx", sheet_name=1)
    temperatures_table = df_air_temp['Temperature (K)'].tolist()
    cp_table = df_air_temp['Specific Heat (cp) J/kg.K'].tolist()
    k_table = df_air_temp['Thermal Conductivity (k) W/m.K'].tolist()
    mu_table = df_air_temp['Dynamic Viscosity (m)  kg/m.s'].tolist()
except FileNotFoundError:
    raise SystemExit("FATAL ERROR: 298.xlsx not found. This file is required for the hybrid physics model.")

def prop_internal_air(T_internal, P_amb):
    target_temperature = max(T_internal, 1.0)
    R_specific_air = 287.058
    rho = P_amb / (R_specific_air * target_temperature)
    cp = np.interp(target_temperature, temperatures_table, cp_table)
    k = np.interp(target_temperature, temperatures_table, k_table)
    mu = np.interp(target_temperature, temperatures_table, mu_table)
    nu = mu / rho if rho > 1e-9 else 0
    Pr = (mu * cp) / k if k > 1e-9 else 0
    return rho, cp, k, mu, nu, Pr

def get_external_convection_h(p_film, T_surface, T_fluid, L_char_forced, L_char_natural, velocity):
    if velocity > 0.1:
        return forced_convection_h(p_film, L_char_forced, velocity)
    else:
        return natural_convection_h(p_film, T_surface, T_fluid, L_char_natural, is_vertical=False)

def natural_convection_h(p_film, T_surface, T_fluid, L_char, is_vertical):
    k, Pr, nu_val = p_film[2], p_film[5], p_film[4]
    if abs(T_surface - T_fluid) < 1e-4 or Pr is None or nu_val is None or k is None:
        return 0.0
    
    T_film = (T_surface + T_fluid) / 2
    beta = 1.0 / T_film if T_film > 1e-6 else 0

    Gr = (config.g * beta * abs(T_surface - T_fluid) * L_char**3) / (nu_val**2 + 1e-12) # Safety denominator
    Ra = Gr * Pr

    if Ra < 0: return 0.0 # Prevent complex numbers
    
    Nu = 1.0
    try:
        if is_vertical:
            Nu = (0.825 + (0.387 * Ra**(1/6)) / (1 + (0.492 / Pr)**(9/16))**(8/27))**2
        else: # Horizontal plate
            if T_surface > T_fluid: # Hot surface facing up
                if 1e4 <= Ra <= 1e7: Nu = 0.54 * Ra**(1/4)
                elif Ra > 1e7: Nu = 0.15 * Ra**(1/3)
            else: # Hot surface facing down
                if 1e5 <= Ra <= 1e10: Nu = 0.27 * Ra**(1/4)
    except (ValueError, OverflowError): # Catch math errors from extreme Ra/Pr
        Nu = 1.0 # Fallback
        
    return Nu * k / L_char

def forced_convection_h(p_film, L_char, velocity):
    rho, mu, k, Pr = p_film[0], p_film[3], p_film[2], p_film[5]
    if Pr is None or k is None: return 0.0
    Re_L = rho * velocity * L_char / max(mu, 1e-12)
    if Re_L < 100: return 0.0
    Re_crit = 5e5
    try:
        if Re_L <= Re_crit: # Laminar
            Nu = 0.664 * (Re_L**0.5) * (Pr**(1/3))
        else: # Mixed
            Nu = (0.037 * (Re_L**0.8) - 871) * (Pr**(1/3))
    except (ValueError, OverflowError):
        Nu = 1.0 # Fallback
    return Nu * k / L_char

def T_power4(T): return np.clip(T, 1e-6, 1e6)**4

def rad_coeff(e1, e2, a1, a2, vf=1.0):
    sigma = 5.67e-8
    if e1 == 0 or e2 == 0 or a1 == 0 or a2 == 0: return 0
    return sigma / ((1 - e1) / (e1 * a1) + 1 / (a1 * vf) + (1 - e2) / (e2 * a2))