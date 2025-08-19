# physics_models.py

import numpy as np
import pandas as pd
import config

# --- Load temperature-dependent data ONCE when the module is imported ---
try:
    # MODIFIED: Removed 'sheet_name=1' to make it more robust, defaults to the first sheet.
    df_air_temp = pd.read_excel("298.xlsx", sheet_name = 1)
    temperatures_table = df_air_temp['Temperature (K)'].tolist()
    cp_table = df_air_temp['Specific Heat (cp) J/kg.K'].tolist()
    k_table = df_air_temp['Thermal Conductivity (k) W/m.K'].tolist()
    # MODIFIED: Corrected the column name to include the double space, matching the file.
    mu_table = df_air_temp['Dynamic Viscosity (m)  kg/m.s'].tolist()
except FileNotFoundError:
    raise SystemExit("FATAL ERROR: 298.xlsx not found. Please place it in the same folder.")
except KeyError as e:
    raise SystemExit(f"FATAL ERROR: Column {e} not found in 298.xlsx. Please check the file's headers.")

def prop_internal_air(T_internal, P_amb):
    """
    HYBRID MODEL:
    - Calculates density using the Ideal Gas Law for altitude accuracy.
    - Interpolates other properties from the Excel table for stability.
    """
    target_temperature = max(T_internal, 1.0)
    R_specific_air = 287.058
    
    # 1. Calculate density from physics
    rho = P_amb / (R_specific_air * target_temperature)
    
    # 2. Interpolate other properties from the table for stability
    cp = np.interp(target_temperature, temperatures_table, cp_table)
    k = np.interp(target_temperature, temperatures_table, k_table)
    mu = np.interp(target_temperature, temperatures_table, mu_table)
    
    # 3. Calculate derived properties
    nu = mu / rho if rho > 1e-9 else 0
    Pr = (mu * cp) / k if k > 1e-9 else 0
    
    return rho, cp, k, mu, nu, Pr

def get_external_convection_h(p_film, T_surface, T_fluid, L_char):
    """
    Determines the appropriate external convection coefficient.
    - Uses forced convection if velocity is significant.
    - Falls back to natural convection if velocity is zero or negligible.
    """
    if config.velocity > 0.1:
        return forced_convection_h(p_film, L_char)
    else:
        return natural_convection_h(p_film, T_surface, T_fluid, L_char, is_vertical=False)

def natural_convection_h(p_film, T_surface, T_fluid, L_char, is_vertical):
    # Unpack the fluid properties from the tuple
    k, Pr, nu_val = p_film[2], p_film[5], p_film[4]

    # --- SAFEGUARDS ---
    # If the temperature difference is negligible or fluid properties are invalid,
    # there is no natural convection. Return h = 0.
    if abs(T_surface - T_fluid) < 1e-6 or not all([Pr, nu_val, k]) or Pr <= 0:
        return 0.0

    # --- CALCULATIONS ---
    # Calculate properties at the film temperature
    T_film = (T_surface + T_fluid) / 2
    beta = 1.0 / T_film if T_film > 1e-6 else 0

    # Calculate Grashof and Rayleigh numbers
    # Added a small epsilon to the denominator to prevent division by zero if nu_val is exactly 0
    Gr = (config.g * beta * abs(T_surface - T_fluid) * L_char**3) / (nu_val**2 + 1e-12)
    Ra = Gr * Pr

    # Safeguard against negative Rayleigh number from potential floating point errors
    if Ra < 0:
        return 0.0

    # Set a safe, physically reasonable default for the Nusselt number.
    # Nu=1 implies pure conduction through the fluid layer.
    Nu = 1.0

    # --- NUSSELT NUMBER CORRELATIONS ---
    try:
        if is_vertical:
            # Churchill and Chu correlation for vertical plates, valid for all Ra
            Nu = (0.825 + (0.387 * Ra**(1/6)) / (1 + (0.492 / Pr)**(9/16))**(8/27))**2
        else:  # Horizontal plate
            if T_surface > T_fluid:  # Hot surface facing up (cooling from below)
                if 1e4 <= Ra <= 1e7:
                    Nu = 0.54 * Ra**(1/4)  # Laminar
                elif Ra > 1e7:
                    Nu = 0.15 * Ra**(1/3)  # Turbulent
                # If Ra < 1e4, Nu remains the default value of 1.0
            else:  # Cold surface facing up (cooling from above)
                if 1e5 <= Ra <= 1e10:
                    Nu = 0.27 * Ra**(1/4)
                # If Ra < 1e5, Nu remains the default value of 1.0

    except (ValueError, OverflowError):
        # If any math error occurs (e.g., power of a negative number),
        # fall back to the safe default Nusselt number.
        Nu = 1.0

    # --- FINAL CALCULATION ---
    # Calculate h AFTER the Nusselt number has been determined. This line will now always execute.
    h = Nu * k / L_char

    # The result h should always be positive, so abs() is no longer needed.
    return h

def forced_convection_h(p_film, L_char):
    rho, mu, k, Pr = p_film[0], p_film[3], p_film[2], p_film[5]
    if not all([Pr, k]) or Pr <= 0: return 0.0
    
    # Enhanced safety check for extremely low density (high altitude)
    if rho < 1e-6:  # Very thin air
        return 0.0
    
    # Additional safety checks for extreme conditions
    if mu < 1e-12 or k < 1e-12:
        return 0.0
    
    # Calculate Reynolds number with enhanced stability
    Re_L = rho * config.velocity * L_char / mu
    
    # More conservative Reynolds number limits for high altitude
    if Re_L < 100: 
        return 0.0
    
    # Cap Reynolds number more aggressively to prevent extreme values
    Re_L = min(Re_L, 1e6)  # Reduced from 1e8
    
    Re_crit = 5e5
    try:
        if Re_L <= Re_crit: # Laminar
            Nu = 0.664 * (Re_L**0.5) * (Pr**(1/3))
        else: # Mixed
            Nu = (0.037 * (Re_L**0.8) - 871) * (Pr**(1/3))
            # Ensure Nu is positive for mixed flow
            Nu = max(Nu, 1.0)
    except (ValueError, OverflowError):
        Nu = 1.0 # Fallback
    
    # Calculate h with enhanced bounds checking
    h = Nu * k / L_char
    
    # More conservative cap for high altitude conditions
    max_h = 50.0 if rho < 0.1 else 200.0  # Lower cap for thin air
    return min(h, max_h)

# --- ADDED: Helper function for radiation calculations ---
def T_power4(T):
    """Safely computes T^4 for radiation calculations."""
    return np.clip(T, 1.0, 1e7)**4

def rad_coeff(e1, e2, a1, a2, vf=1.0):
    """Calculates the radiation heat transfer coefficient between two surfaces."""
    sigma = 5.67e-8
    if e1 * a1 == 0 or e2 * a2 == 0 or vf == 0:
        return 0.0
    return sigma / ((1 - e1) / (e1 * a1) + 1 / (a1 * vf) + (1 - e2) / (e2 * a2))