# 5_post_processing.py (Enhanced and Complete)

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import config

def print_final_temps(sol):
    """Prints the temperature of each node at the very end of the simulation."""
    final_temps = sol.y[:, -1]
    print("\n--- Final Temperatures (at end of simulation) ---")
    for lab, temp in zip(config.labels, final_temps):
        # Also print in Celsius for better intuition
        print(f"{lab:<25}: {temp:.2f} K  ({temp - 273.15:.2f} °C)")

def analyze_peaks(sol):
    """Analyzes the min/max temperatures over the last few stable-orbit days."""
    print("\n--- In-Depth Thermal Analysis (Cyclical Peaks) ---")
    analysis_window_days = 3
    
    # Ensure analysis window isn't longer than the simulation
    sim_days = config.T_total / 86400
    if sim_days > analysis_window_days:
        analysis_start_time = config.T_total - (analysis_window_days * 86400)
    else:
        analysis_start_time = 0
    
    t_analysis = np.linspace(analysis_start_time, config.T_total, 2000)
    x_analysis = sol.sol(t_analysis)
    
    print(f"Analyzing last {sim_days if sim_days < analysis_window_days else analysis_window_days:.1f} days for peaks/lows:")
    for i, lab in enumerate(config.labels):
        temp_history = x_analysis[i, :]
        min_temp, max_temp = np.min(temp_history), np.max(temp_history)
        temp_swing = max_temp - min_temp
        print(f"{lab:<25}: Peak = {max_temp:.2f} K | Min = {min_temp:.2f} K | Swing = {temp_swing:.2f} K")

def plot_grouped_results(sol):
    """
    Creates grouped subplots for clearer visualization of component families.
    """
    print("\nGenerating grouped temperature plots...")
    t_plot = np.linspace(0, config.T_total, 2000)
    # Transpose the results and wrap them in a pandas DataFrame for easier handling
    temps_df = pd.DataFrame(sol.sol(t_plot).T, columns=config.labels)
    temps_df['Time (Days)'] = t_plot / 86400

    # Define the groups of components for subplots
    groups = {
        'Batteries': [l for l in config.labels if 'Batt_' in l],
        'Avionics & Air': ['ESC', 'ESC_Mount', 'Internal_Air'],
        'Bulkheads': [l for l in config.labels if 'BH_' in l],
        'Shells': [l for l in config.labels if 'Shell' in l]
    }
    
    # Create the figure and subplots
    fig, axes = plt.subplots(2, 2, figsize=(20, 14), constrained_layout=True)
    fig.suptitle(f'Transient Temperature Evolution ({config.TARGET_ALTITUDE_KM}km Altitude)', fontsize=18)
    axes = axes.flatten()

    for i, (title, members) in enumerate(groups.items()):
        ax = axes[i]
        for member in members:
            # Plot temperature in Celsius for better readability
            ax.plot(temps_df['Time (Days)'], temps_df[member] - 273.15, label=member)
        ax.set_title(title, fontsize=14)
        ax.set_xlabel('Time (Days)', fontsize=12)
        ax.set_ylabel('Temperature (°C)', fontsize=12)
        ax.legend()
        ax.grid(True, which='both', linestyle='--', linewidth=0.5)
        
    plt.show()