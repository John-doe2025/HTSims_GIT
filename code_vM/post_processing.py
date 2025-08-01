# 5_post_processing.py
import numpy as np
import matplotlib.pyplot as plt
import config

def print_final_temps(sol):
    final_temps = sol.y[:, -1]
    print("\n--- Final Temperatures (at end of simulation) ---")
    for lab, temp in zip(config.labels, final_temps):
        print(f"{lab:<25}: {temp:.2f} K")

def analyze_peaks(sol):
    print("\n--- In-Depth Thermal Analysis ---")
    analysis_window_days = 3
    if config.T_total / 86400 > analysis_window_days:
        analysis_start_time = config.T_total - (analysis_window_days * 86400)
    else: analysis_start_time = 0
    t_analysis = np.linspace(analysis_start_time, config.T_total, 2000)
    x_analysis = sol.sol(t_analysis)
    print(f"Analyzing last {analysis_window_days} days for cyclical peaks/lows:")
    for i, lab in enumerate(config.labels):
        temp_history = x_analysis[i, :]
        min_temp, max_temp = np.min(temp_history), np.max(temp_history)
        temp_swing = max_temp - min_temp
        print(f"{lab:<25}: Peak = {max_temp:.2f} K | Min = {min_temp:.2f} K | Swing = {temp_swing:.2f} K")

def plot_results(sol):
    plt.figure(figsize=(14, 9))
    t_plot = np.linspace(0, config.T_total, 2000)
    x_plot = sol.sol(t_plot).T
    for i, lab in enumerate(config.labels): plt.plot(t_plot / 86400, x_plot[:, i], label=lab)
    plt.xlabel('Time (Days)', fontsize=12); plt.ylabel('Temperature (K)', fontsize=12)
    plt.title(f'Transient Temperature Evolution ({config.TARGET_ALTITUDE_KM}km Altitude)', fontsize=16)
    plt.legend(loc='best', fontsize=10); plt.grid(True, which='both', linestyle='--', linewidth=0.5)
    plt.tight_layout(); plt.show()
