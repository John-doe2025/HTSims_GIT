"""
Test the external convection coefficient calculation
"""

import physics_models
import config
import pandas as pd

# Load environment
df_alt = pd.read_excel("altitude_data.xlsx")
row = df_alt.iloc[(df_alt['Altitude'] - config.TARGET_ALTITUDE_KM).abs().idxmin()]
T_E = row['Temperature']
P_amb = row['Pressure']

print("="*60)
print("EXTERNAL CONVECTION COEFFICIENT TEST")
print("="*60)

print(f"\nEnvironment:")
print(f"  Altitude: {config.TARGET_ALTITUDE_KM} km")
print(f"  Ambient T: {T_E:.2f} K")
print(f"  Ambient P: {P_amb:.2f} Pa")
print(f"  Velocity: {config.velocity} m/s")

# Test conditions
T_surface = 645.0  # Shell temperature from simulation
T_fluid = T_E

print(f"\nTest conditions:")
print(f"  Surface T: {T_surface:.2f} K")
print(f"  Fluid T: {T_fluid:.2f} K")
print(f"  Delta T: {T_surface - T_fluid:.2f} K")

# Get air properties
p_ambient = physics_models.prop_internal_air(T_E, P_amb)

# Calculate external convection
h_ext = physics_models.get_external_convection_h(
    p_ambient, T_surface, T_fluid,
    config.LC_TS_ext, config.LC_TS_int
)

print(f"\nExternal convection coefficient:")
print(f"  h = {h_ext:.3f} W/m²K")

# Check if this is reasonable
if config.velocity == 0:
    print(f"\nFor natural convection (v=0):")
    print(f"  Expected range: 0.5-5 W/m²K")
    if h_ext > 5:
        print(f"  WARNING: h is too high!")
    elif h_ext < 0.5:
        print(f"  WARNING: h is too low!")
    else:
        print(f"  OK: h is in reasonable range")

# Calculate heat loss with this h
A_ext = config.A_TS + config.A_BS
Q_loss = h_ext * A_ext * (T_surface - T_fluid)
print(f"\nHeat loss with this h:")
print(f"  Q = {Q_loss:.2f} W")
print(f"  Generated: {112.6} W")
print(f"  Ratio: {Q_loss/112.6:.1f}x")

# What h do we need for balance?
h_needed = 112.6 / (A_ext * (T_surface - T_fluid))
print(f"\nh needed for energy balance: {h_needed:.3f} W/m²K")

# Let's also check the natural convection calculation directly
print(f"\n--- Testing Natural Convection Function ---")
L_char = config.LC_TS_ext
print(f"Characteristic length: {L_char:.3f} m")

# Film properties
T_film = (T_surface + T_fluid) / 2
p_film = physics_models.prop_internal_air(T_film, P_amb)
print(f"Film temperature: {T_film:.2f} K")

h_nat = physics_models.natural_convection_h(
    p_film, T_surface, T_fluid, L_char, is_vertical=False
)
print(f"Natural convection h: {h_nat:.3f} W/m²K")

# Check Grashof and Rayleigh numbers
g = 9.81
beta = 1/T_film
nu = p_film[4]
k = p_film[2]
Pr = p_film[5]

Gr = g * beta * abs(T_surface - T_fluid) * L_char**3 / nu**2
Ra = Gr * Pr

print(f"\nDimensionless numbers:")
print(f"  Gr = {Gr:.2e}")
print(f"  Ra = {Ra:.2e}")
print(f"  Pr = {Pr:.3f}")

if Ra > 1e9:
    print(f"  Flow regime: Turbulent natural convection")
elif Ra > 1e4:
    print(f"  Flow regime: Laminar natural convection")
else:
    print(f"  Flow regime: Conduction dominant")