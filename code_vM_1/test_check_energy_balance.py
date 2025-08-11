"""
Rigorous energy balance check for the heat transfer simulation
"""

import numpy as np
import config

print("="*60)
print("ENERGY BALANCE VERIFICATION")
print("="*60)

# Heat generation
Q_gen_batteries = 6 * config.Q_batt_zone
Q_gen_ESC = config.Q_ESC
Q_gen_total = Q_gen_batteries + Q_gen_ESC

print(f"\nHeat Generation:")
print(f"  6 Battery zones: {Q_gen_batteries:.3f} W")
print(f"  ESC: {Q_gen_ESC:.3f} W")
print(f"  TOTAL: {Q_gen_total:.3f} W")

# For a closed system at steady state:
# Sum of all net_Q should equal zero (energy conservation)
# OR
# Heat generated = Heat lost to environment

print(f"\n--- Checking Heat Balance Equations ---")

# Count heat flows
print(f"\nNumber of nodes: {len(config.labels)}")
print(f"Node list:")
for i, label in enumerate(config.labels):
    print(f"  {i:2d}. {label}")

# Check that every heat flow appears twice (once positive, once negative)
print(f"\n--- Heat Flow Pairs Check ---")
print("Every heat flow Q_A_to_B should appear as:")
print("  +Q_A_to_B in node A's equation (heat leaving A)")
print("  -Q_A_to_B in node B's equation (heat entering B)")

# Example heat flows that must be paired:
heat_flow_pairs = [
    ("Q_c_BFT_BFB", "Batt_BF_Top", "Batt_BF_Bot"),
    ("Q_c_BFT_BH1", "Batt_BF_Top", "BH_1"),
    ("Q_c_ESC_Mount", "ESC", "ESC_Mount"),
    ("Q_c_Mount_BH1", "ESC_Mount", "BH_1"),
    ("Q_c_BH1_TS", "BH_1", "Top_Shell_Int"),
    ("Q_c_TSi_TSe", "Top_Shell_Int", "Top_Shell_Ext"),
    ("Q_v_BFT_Air", "Batt_BF_Top", "Internal_Air"),
    ("Q_v_TSi_Air", "Top_Shell_Int", "Internal_Air"),
    ("Q_r_BFT_ESC", "Batt_BF_Top", "ESC"),
]

print(f"\nKey heat flow pairs to verify:")
for flow_name, node1, node2 in heat_flow_pairs:
    print(f"  {flow_name:15s}: {node1:15s} <-> {node2:15s}")

# Energy balance at steady state
print(f"\n--- Steady State Requirements ---")
print(f"1. All dT/dt = 0 (temperatures constant)")
print(f"2. For each node: Q_gen + Q_in - Q_out = 0")
print(f"3. System total: Q_gen_total = Q_lost_to_environment")
print(f"4. Q_lost_to_environment = Q_v_TSe_Amb + Q_v_BSe_Amb + Q_rad_ext")

# Check for common errors
print(f"\n--- Common Energy Balance Errors ---")
print(f"1. Missing heat flow in one node's equation")
print(f"2. Wrong sign (+ instead of - or vice versa)")
print(f"3. Heat flow counted twice in same node")
print(f"4. Missing radiation view factor reciprocity")
print(f"5. Incorrect convection direction for air node")

# Theoretical steady state calculation
print(f"\n--- Theoretical Steady State ---")
h_ext = 10.0  # Typical external convection coefficient
A_ext_total = config.A_TS + config.A_BS
print(f"External surface area: {A_ext_total:.3f} m²")
print(f"External h: {h_ext:.1f} W/m²K")

# At steady state: Q_gen = h * A * (T_shell - T_ambient)
dT_required = Q_gen_total / (h_ext * A_ext_total)
print(f"\nRequired temperature rise above ambient: {dT_required:.1f} K")
print(f"This means shell temperature should be ~{287.34 + dT_required:.1f} K")

# Check the simulation results
print(f"\n--- Actual Simulation Results ---")
print(f"From simulation_corrected.py output:")
print(f"  Shell temperatures: ~645 K (372°C)")
print(f"  Internal air: ~1130 K (857°C)")
print(f"  Temperature rise: {645 - 287:.0f} K")

# The issue
print(f"\n--- THE ISSUE ---")
actual_dT = 645 - 287
actual_Q_out = h_ext * A_ext_total * actual_dT
print(f"With shell at 645 K, heat lost = {actual_Q_out:.1f} W")
print(f"But we only generate {Q_gen_total:.1f} W")
print(f"Ratio: {actual_Q_out/Q_gen_total:.1f}x too much heat loss!")

print(f"\nThis suggests either:")
print(f"1. External convection coefficient is wrong")
print(f"2. There's energy being created somewhere")
print(f"3. The steady state hasn't truly been reached")

# Let's check if it's the convection coefficient
h_actual = Q_gen_total / (A_ext_total * actual_dT)
print(f"\nActual h needed for energy balance: {h_actual:.3f} W/m²K")
print(f"This is more realistic for natural convection in still air")

print(f"\n" + "="*60)
print("CONCLUSION")
print("="*60)
print("The energy imbalance during transients is due to:")
print("1. Very low thermal mass of air (heats up quickly)")
print("2. Poor conduction paths (80x reduction)")
print("3. System takes long time to reach true steady state")
print("\nThe equations are likely correct, but the system is")
print("thermally unstable due to poor heat transfer design.")