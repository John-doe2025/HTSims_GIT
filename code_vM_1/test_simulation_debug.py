"""
Debug version of the simulation to identify the root cause of instability
"""

import numpy as np
import pandas as pd
import config
import physics_models
import environment_model

# Initialize
environment_model.init()
df_alt = pd.read_excel("altitude_data.xlsx")
row = df_alt.iloc[(df_alt['Altitude'] - config.TARGET_ALTITUDE_KM).abs().idxmin()]
T_E = row['Temperature']
P_amb = row['Pressure']

print("="*60)
print("DEBUGGING HEAT TRANSFER EQUATIONS")
print("="*60)

# Set initial temperatures (all at ambient + small delta)
temps = {label: T_E + 10.0 for label in config.labels}
temps['Internal_Air'] = T_E + 8.0
temps['Top_Shell_Ext'] = T_E + 5.0
temps['Bot_Shell_Ext'] = T_E + 5.0

print(f"\nInitial Temperatures:")
for label, T in temps.items():
    print(f"  {label:20s}: {T:.2f} K ({T-273.15:.2f} °C)")

# Add missing config parameter
if not hasattr(config, 'L_path_Batt_to_BH'):
    config.L_path_Batt_to_BH = config.L_batt_zone / 2
    print(f"\nAdded L_path_Batt_to_BH = {config.L_path_Batt_to_BH:.3f} m")

# Calculate heat generation
Q_gen_batteries = 6 * config.Q_batt_zone
Q_gen_ESC = config.Q_ESC
Q_gen_total = Q_gen_batteries + Q_gen_ESC

print(f"\nHeat Generation:")
print(f"  Batteries: {Q_gen_batteries:.2f} W")
print(f"  ESC: {Q_gen_ESC:.2f} W")
print(f"  Total: {Q_gen_total:.2f} W")

# Test 1: Check conduction coefficients
print(f"\n--- Conduction Coefficients Check ---")
L_path_Batt_to_BH_OLD = config.t_bulkhead  # Old incorrect value
L_path_Batt_to_BH_NEW = config.L_path_Batt_to_BH  # Corrected value

C_cond_OLD = config.k_bulkhead * config.A_contact_Batt_BH / L_path_Batt_to_BH_OLD
C_cond_NEW = config.k_bulkhead * config.A_contact_Batt_BH / L_path_Batt_to_BH_NEW

print(f"Battery to Bulkhead conduction coefficient:")
print(f"  OLD (wrong): {C_cond_OLD:.3f} W/K (L={L_path_Batt_to_BH_OLD:.4f} m)")
print(f"  NEW (correct): {C_cond_NEW:.3f} W/K (L={L_path_Batt_to_BH_NEW:.4f} m)")
print(f"  Ratio: {C_cond_OLD/C_cond_NEW:.1f}x too high!")

# Test 2: Check a simple heat balance for one battery node
print(f"\n--- Simple Heat Balance Test (Battery Front Top) ---")

# Heat generation
Q_gen = config.Q_batt_zone
print(f"Heat generation: {Q_gen:.2f} W")

# Conduction to bulkhead (using corrected coefficient)
dT_batt_bh = 10.0  # Assume 10K temperature difference
Q_cond_to_bh = C_cond_NEW * dT_batt_bh
print(f"Conduction to bulkhead (dT={dT_batt_bh}K): {Q_cond_to_bh:.2f} W")

# Convection to air
h_conv = 5.0  # Typical natural convection coefficient
A_conv = config.A_conv_batt_total
dT_batt_air = 5.0  # Assume 5K difference to air
Q_conv_to_air = h_conv * A_conv * dT_batt_air
print(f"Convection to air (h={h_conv}, dT={dT_batt_air}K): {Q_conv_to_air:.2f} W")

# Radiation (simplified)
sigma = 5.67e-8
epsilon = config.emis_batt
A_rad = config.A_rad_batt_to_shell
T_batt = 300.0
T_shell = 295.0
Q_rad = sigma * epsilon * A_rad * (T_batt**4 - T_shell**4)
print(f"Radiation to shell (dT={T_batt-T_shell}K): {Q_rad:.2f} W")

# Net heat
Q_net = Q_gen - (Q_cond_to_bh + Q_conv_to_air + Q_rad)
print(f"\nNet heat: {Q_net:.2f} W")

# Temperature rise rate
m_batt = config.m_batt_zone
Cp_batt = config.C_B
dT_dt = Q_net / (m_batt * Cp_batt)
print(f"Temperature rise rate: {dT_dt:.4f} K/s = {dT_dt*3600:.2f} K/hr")

# Test 3: Check if the problem is with the internal air node
print(f"\n--- Internal Air Node Balance Test ---")

# Sum of convection from all components (simplified)
n_components = 12  # 6 batteries + ESC + Mount + 4 bulkheads
h_avg = 5.0
A_avg = 0.05  # Average area per component
dT_avg = 5.0  # Average temperature difference
Q_from_components = n_components * h_avg * A_avg * dT_avg
print(f"Heat from components: {Q_from_components:.2f} W")

# Convection to shells
A_shell = config.A_TS + config.A_BS
dT_air_shell = 2.0  # Air to shell temperature difference
Q_to_shells = h_avg * A_shell * dT_air_shell
print(f"Heat to shells: {Q_to_shells:.2f} W")

# OLD equation (wrong)
Q_net_air_OLD = Q_from_components - Q_to_shells
print(f"\nOLD air node net heat: {Q_net_air_OLD:.2f} W")

# NEW equation (corrected)
Q_net_air_NEW = Q_from_components + Q_to_shells  # Both should add if shells are cooler
print(f"NEW air node net heat: {Q_net_air_NEW:.2f} W")

# Air temperature rise
rho_air = 1.2  # kg/m³
cp_air = 1005  # J/kg·K
V_air = config.V_internal_air
m_air = rho_air * V_air
dT_dt_air_OLD = Q_net_air_OLD / (m_air * cp_air)
dT_dt_air_NEW = Q_net_air_NEW / (m_air * cp_air)

print(f"\nAir temperature rise rate:")
print(f"  OLD: {dT_dt_air_OLD*3600:.2f} K/hr")
print(f"  NEW: {dT_dt_air_NEW*3600:.2f} K/hr")

# Test 4: Energy conservation check
print(f"\n--- Energy Conservation Check ---")
print("For a closed system at steady state:")
print(f"  Heat generated = {Q_gen_total:.2f} W")
print("  Heat lost to environment = ? (through external shells only)")

# Estimate steady-state shell temperature
# At steady state, all generated heat must be lost to environment
h_ext = 10.0  # External convection coefficient
A_ext = config.A_TS + config.A_BS
dT_required = Q_gen_total / (h_ext * A_ext)
T_shell_steady = T_E + dT_required

print(f"\nRequired shell temperature for steady state:")
print(f"  T_shell = {T_shell_steady:.2f} K ({T_shell_steady-273.15:.2f} °C)")
print(f"  dT above ambient = {dT_required:.2f} K")

# Check if this is reasonable
if dT_required > 100:
    print(f"  WARNING: This seems too high! Check heat transfer coefficients.")
elif dT_required < 1:
    print(f"  WARNING: This seems too low! Check heat generation values.")
else:
    print(f"  This seems reasonable.")

print("\n" + "="*60)
print("DIAGNOSIS SUMMARY")
print("="*60)

issues_found = []

# Check 1: Conduction coefficient
if C_cond_OLD/C_cond_NEW > 10:
    issues_found.append(f"1. Conduction coefficient was {C_cond_OLD/C_cond_NEW:.0f}x too high")

# Check 2: Air node equation
if abs(Q_net_air_OLD - Q_net_air_NEW) > 1:
    issues_found.append("2. Internal air node had wrong sign convention")

# Check 3: Steady state temperature
if dT_required > 100 or dT_required < 1:
    issues_found.append("3. Steady-state temperature seems unrealistic")

if issues_found:
    print("Issues identified:")
    for issue in issues_found:
        print(f"  {issue}")
else:
    print("No obvious issues found in basic checks.")

print("\nRecommendations:")
print("1. The corrected thermal path length reduces conduction significantly")
print("2. This means less heat transfer between components")
print("3. Components may reach higher temperatures than before")
print("4. Consider adding more heat transfer paths or increasing areas")
print("5. Check if all heat generation values are correct")

# Additional check: Time constant
print(f"\n--- System Time Constants ---")
tau_batt = m_batt * Cp_batt / (h_conv * A_conv)
tau_air = m_air * cp_air / (h_avg * A_shell)
print(f"Battery thermal time constant: {tau_batt:.1f} s = {tau_batt/60:.1f} min")
print(f"Air thermal time constant: {tau_air:.1f} s = {tau_air/60:.1f} min")

if tau_batt < 10 or tau_air < 10:
    print("WARNING: Very short time constants may cause numerical instability!")