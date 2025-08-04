# run_complete_analysis.py

import time
import copy
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

# --- 1. Physics and Simulation Function ---

# --- Define your Excel column names here ---
COLUMN_NAMES = {
    "temp": "Temperature (K)", "rho": "Density (rho) kg/m3",
    "cp": "Specific Heat (cp) J/kg.K", "k": "Thermal Conductivity (k) W/m.K",
    "mu": "Dynamic Viscosity (m) kg/m.s"
}

# Load air properties ONCE for efficiency
try:
    path = "./298.xlsx"
    df_air = pd.read_excel(path)
    temperatures = df_air[COLUMN_NAMES["temp"]].tolist()
    rho_values   = df_air[COLUMN_NAMES["rho"]].tolist()
    cp_values    = df_air[COLUMN_NAMES["cp"]].tolist()
    k_values     = df_air[COLUMN_NAMES["k"]].tolist()
    mu_values    = df_air[COLUMN_NAMES["mu"]].tolist()
except FileNotFoundError:
    raise SystemExit("FATAL ERROR: 298.xlsx not found. Please place it in the same folder as this script.")
except KeyError as e:
    raise SystemExit(f"FATAL ERROR: Column {e} not found in 298.xlsx. Please check the COLUMN_NAMES dictionary.")

def prop(T):
    r  = np.interp(T, temperatures, rho_values)
    cp = np.interp(T, temperatures, cp_values)
    k  = np.interp(T, temperatures, k_values)
    mu = np.interp(T, temperatures, mu_values)
    nu = mu / r if r > 1e-6 else 0
    Pr = (mu * cp) / k if k > 1e-6 else 0
    return r, cp, k, mu, nu, Pr

def T_power4(T): return np.clip(T, 1e-6, 1e6)**4

def natural_convection_h(p_film, T_surface, T_fluid, L_char, is_vertical, g):
    k, Pr, nu_val = p_film[2], p_film[5], p_film[4]
    if abs(T_surface - T_fluid) < 1e-4 or Pr is None or nu_val is None or k is None: return 0.0
    T_film = (T_surface + T_fluid) / 2
    beta = 1.0 / T_film if T_film > 1e-6 else 0
    Gr = (g * beta * abs(T_surface - T_fluid) * L_char**3) / (nu_val**2 + 1e-12)
    Ra = Gr * Pr
    if Ra < 0: return 0.0
    Nu = 1.0
    try:
        if is_vertical: Nu = (0.825 + (0.387 * Ra**(1/6)) / (1 + (0.492 / Pr)**(9/16))**(8/27))**2
        else:
            if T_surface > T_fluid:
                if 1e4 <= Ra <= 1e7: Nu = 0.54 * Ra**(1/4)
                elif Ra > 1e7: Nu = 0.15 * Ra**(1/3)
            else:
                if 1e5 <= Ra <= 1e10: Nu = 0.27 * Ra**(1/4)
    except (ValueError, OverflowError): Nu = 1.0
    return Nu * k / L_char

def get_external_surface_h(p_film, T_surface, T_fluid, L_char, velocity, g):
    if velocity > 0.1:
        k, Pr, rho, mu = p_film[2], p_film[5], p_film[0], p_film[3]
        Re_L = rho * velocity * L_char / max(mu, 1e-12)
        if Re_L < 100: return 0.0
        Re_crit = 5e5
        if Re_L <= Re_crit: Nu = 0.664 * (Re_L**0.5) * (Pr**(1/3))
        else: Nu = (0.037 * (Re_L**0.8) - 871) * (Pr**(1/3))
        return Nu * k / L_char
    else:
        return natural_convection_h(p_film, T_surface, T_fluid, L_char, is_vertical=False, g=g)

def run_simulation(params):
    def f(t, x, params, return_flows=False):
        for key, value in params.items(): globals()[key] = value
        m_Bf = m_Bm = m_Br = m_B_total / 3
        Q_B_front = Q_B_middle = Q_B_rear = Q_B_total / 3
        C_cond = k_eff * A_cross / 0.280
        C_cond_ESC_to_Mount = k_mount * A_contact_ESC_Mount / L_path_ESC_Mount
        C_cond_Mount_to_Bf = k_mount * A_contact_Mount_Bf / L_path_Mount_Bf
        C_cond_cfrp = k_cfrp * A_cfrp / t_cfrp
        
        sigma = 5.67e-8
        def rad_coeff(e1, e2, a1, a2, vf=1.0):
            if e1*a1 == 0 or e2*a2 == 0 or vf == 0: return 0
            return sigma / ((1 - e1) / (e1 * a1) + 1 / (a1 * vf) + (1 - e2) / (e2 * a2))

        C_Bf_TS_int_rad = rad_coeff(0.9, 0.2, 0.0427, A_TS, 1)
        C_TS_BS_rad = rad_coeff(0.2, 0.2, A_TS, A_BS, 0.5)
        C_ESC_TS_rad = rad_coeff(0.8, 0.2, 0.0013959, A_TS, 1)
        C_Bf_ESC_rad = rad_coeff(0.2, 0.8, 0.038808, 0.00206415, 1)
        C_Mount_rad = rad_coeff(0.7, 0.2, A_mount_conv, A_TS, 1)
        
        T_Bf, T_Bm, T_Br, T_ESC, T_mount, T_TS_int, T_TS_ext, T_BS_int, T_BS_ext, T_air = x
        p_air = prop(T_air)
        p_batt_film = prop((T_Bf + T_air)/2); p_esc_film = prop((T_ESC + T_air)/2)
        p_mount_film = prop((T_mount + T_air)/2); p_ts_int_film = prop((T_TS_int + T_air)/2)
        p_bs_int_film = prop((T_BS_int + T_air)/2); p_ts_ext_film = prop((T_TS_ext + T_E)/2)
        p_bs_ext_film = prop((T_BS_ext + T_E)/2)
        T4s = {k: T_power4(v) for k, v in zip(['Bf','Bm','Br','ESC','mount','TS_int','BS_int'], [T_Bf,T_Bm,T_Br,T_ESC,T_mount,T_TS_int,T_BS_int])}

        h_batt_avg = (natural_convection_h(p_batt_film, T_Bf, T_air, LC_B_horiz, False, g) * 2 + natural_convection_h(p_batt_film, T_Bf, T_air, LC_B_vert, True, g) * 2) / 4.0
        Q_conv_Bf = h_batt_avg * A_Bf_conv * (T_air - T_Bf)
        Q_conv_Bm = h_batt_avg * A_Bm_conv * (T_air - T_Bm)
        Q_conv_Br = h_batt_avg * A_Br_conv * (T_air - T_Br)
        Q_cond_Bf = C_cond * (T_Bm - T_Bf); Q_cond_Bm = C_cond * ((T_Bf - T_Bm) + (T_Br - T_Bm)); Q_cond_Br = C_cond * (T_Bm - T_Br)
        Q_rad_Bf = (C_Bf_TS_int_rad * (T4s['TS_int'] - T4s['Bf']) + C_Bf_ESC_rad * (T4s['ESC'] - T4s['Bf']) + C_Bf_TS_int_rad * (T4s['BS_int'] - T4s['Bf']))
        Q_rad_Bm = (C_Bf_TS_int_rad * (T4s['TS_int'] - T4s['Bm']) + C_Bf_TS_int_rad * (T4s['BS_int'] - T4s['Bm']))
        Q_rad_Br = (C_Bf_TS_int_rad * (T4s['TS_int'] - T4s['Br']) + C_Bf_TS_int_rad * (T4s['BS_int'] - T4s['Br']))
        Q_cond_Mount_to_Bf = C_cond_Mount_to_Bf * (T_Bf - T_mount)
        Q_conv_ESC = natural_convection_h(p_esc_film, T_ESC, T_air, LC_ESC, False, g) * A_ESC_conv * (T_air - T_ESC)
        Q_rad_ESC = (C_Bf_ESC_rad * (T4s['Bf'] - T4s['ESC']) + C_ESC_TS_rad * (T4s['TS_int'] - T4s['ESC']) + C_ESC_TS_rad * (T4s['BS_int'] - T4s['ESC']))
        Q_cond_ESC_to_Mount = C_cond_ESC_to_Mount * (T_mount - T_ESC)
        Q_conv_Mount = natural_convection_h(p_mount_film, T_mount, T_air, LC_mount, False, g) * A_mount_conv * (T_air - T_mount)
        Q_rad_Mount = C_Mount_rad * (T4s['TS_int'] - T4s['mount'])
        Q_cond_TS_ext = C_cond_cfrp * (T_TS_int - T_TS_ext); Q_cond_BS_ext = C_cond_cfrp * (T_BS_int - T_BS_ext)
        Q_conv_TS_in = natural_convection_h(p_ts_int_film, T_TS_int, T_air, LC_TS, False, g) * A_TS * (T_air - T_TS_int)
        Q_conv_TS_ext = get_external_surface_h(p_ts_ext_film, T_TS_ext, T_E, LC_TS, velocity, g) * A_TS * (T_E - T_TS_ext)
        Q_conv_BS_in = natural_convection_h(p_bs_int_film, T_BS_int, T_air, LC_BS, False, g) * A_BS * (T_air - T_BS_int)
        Q_conv_BS_ext = get_external_surface_h(p_bs_ext_film, T_BS_ext, T_E, LC_BS, velocity, g) * A_BS * (T_E - T_BS_ext)
        Q_rad_TS = (C_Bf_TS_int_rad * ((T4s['Bf'] - T4s['TS_int']) + (T4s['Bm'] - T4s['TS_int']) + (T4s['Br'] - T4s['TS_int'])) + C_TS_BS_rad * (T4s['BS_int'] - T4s['TS_int'])) + Q_rad_Mount
        Q_rad_BS = (C_Bf_TS_int_rad * ((T4s['Bf'] - T4s['BS_int']) + (T4s['Bm'] - T4s['BS_int']) + (T4s['Br'] - T4s['BS_int'])) + C_TS_BS_rad * (T4s['TS_int'] - T4s['BS_int']))
        Q_conv_air = -(Q_conv_Bf + Q_conv_Bm + Q_conv_Br + Q_conv_ESC + Q_conv_Mount + Q_conv_TS_in + Q_conv_BS_in)
        m_a = prop(T_air)[0] * 0.11

        if return_flows:
            return {
                'Q_gen_ESC': Q_ESC, 'Q_out_ESC_conv': -Q_conv_ESC, 'Q_out_ESC_rad': -Q_rad_ESC, 'Q_out_ESC_cond': -Q_cond_ESC_to_Mount,
                'Q_total_shell_rejection': -Q_conv_TS_ext - Q_conv_BS_ext }

        dT_Bf_dt = (Q_B_front + Q_cond_Bf + Q_conv_Bf + Q_rad_Bf - Q_cond_Mount_to_Bf) / (m_Bf * C_B)
        dT_Bm_dt = (Q_B_middle + Q_cond_Bm + Q_conv_Bm + Q_rad_Bm) / (m_Bm * C_B)
        dT_Br_dt = (Q_B_rear + Q_cond_Br + Q_conv_Br + Q_rad_Br) / (m_Br * C_B)
        dT_ESC_dt = (Q_ESC + Q_conv_ESC + Q_rad_ESC + Q_cond_ESC_to_Mount) / (m_ESC * C_ESC)
        dT_mount_dt = (-Q_cond_ESC_to_Mount + Q_cond_Mount_to_Bf + Q_conv_Mount + Q_rad_Mount) / (m_mount * C_mount)
        dT_TS_int_dt = (-Q_cond_TS_ext + Q_conv_TS_in + Q_rad_TS) / (m_TS * C_TS)
        dT_TS_ext_dt = (Q_cond_TS_ext + Q_conv_TS_ext) / (m_TS * C_TS)
        dT_BS_int_dt = (-Q_cond_BS_ext + Q_conv_BS_in + Q_rad_BS) / (m_BS * C_BS)
        dT_BS_ext_dt = (Q_cond_BS_ext + Q_conv_BS_ext) / (m_BS * C_BS)
        dT_air_dt = Q_conv_air / (m_a * prop(T_air)[1])
        
        return np.array([dT_Bf_dt, dT_Bm_dt, dT_Br_dt, dT_ESC_dt, dT_mount_dt, dT_TS_int_dt, dT_TS_ext_dt, dT_BS_int_dt, dT_BS_ext_dt, dT_air_dt])

    x0 = np.array([params.get('initial_temp', 298.15)] * 10)
    t_span = [0, params.get('T_total', 300000)]
    solver_func = lambda t, x: f(t, x, params, return_flows=False)
    sol = solve_ivp(fun=solver_func, t_span=t_span, y0=x0, method='BDF', dense_output=True)
    
    final_temps = sol.y[:, -1]
    heat_flows = f(sol.t[-1], final_temps, params, return_flows=True)
    
    analysis_start_time = t_span[1] * 0.8
    t_analysis = np.linspace(analysis_start_time, t_span[1], 500)
    temps = sol.sol(t_analysis)
    peak_temp_batt_front = np.max(temps[0, :])
    
    return peak_temp_batt_front, heat_flows

# --- 2. Main Analysis Execution Block ---
if __name__ == "__main__":
    start_analysis_time = time.time()
    
    baseline_params = {
        'm_B_total': 64.8, 'm_ESC': 0.12, 'm_TS': 0.9, 'm_BS': 0.9, 'm_mount': 0.023,
        'C_mount': 800, 'C_B': 1100, 'C_ESC': 100, 'C_TS': 1040, 'C_BS': 1040,
        'Q_B_total': 232.8, 'Q_ESC': 100,
        'A_cross': 0.0388, 'A_Bf_conv': 0.266, 'A_Bm_conv': 0.227, 'A_Br_conv': 0.266,
        'A_ESC_conv': 0.0135, 'A_mount_conv': 0.0067, 'A_TS': 0.542, 'A_BS': 0.542, 'A_cfrp': 0.542,
        'LC_mount': 0.0087, 'LC_B_horiz': 0.277, 'LC_B_vert': 0.252, 'LC_ESC': 0.0695,
        'LC_TS': 0.84, 'LC_BS': 0.84, 't_cfrp': 0.0005,
        'k_eff': 2.5, 'k_mount': 0.3, 'k_cfrp': 1.0,
        'A_contact_ESC_Mount': 0.0012, 'L_path_ESC_Mount': 0.005,
        'A_contact_Mount_Bf': 0.0026, 'L_path_Mount_Bf': 0.005,
        'g': 9.81, 'velocity': 0.0, 'T_E': 298.15,
        'initial_temp': 298.15, 'T_total': 300000
    }
    
    variables_to_test = [
        'Q_ESC', 'Q_B_total', 'k_mount', 'k_eff', 'k_cfrp',
        'A_contact_ESC_Mount', 'A_contact_Mount_Bf', 'L_path_ESC_Mount', 'L_path_Mount_Bf',
        'A_mount_conv', 'A_ESC_conv', 'A_Bf_conv', 'T_E'
    ]
    
    PERTURBATION = 0.10
    print("--- Starting Sensitivity Analysis ---")
    print(f"Figure of Merit: Peak Temperature of Battery Front Node")
    print(f"Perturbation: +/- {PERTURBATION:.0%}\n")
    print("Running baseline simulation...")
    
    baseline_fom, baseline_flows = run_simulation(baseline_params)
    print(f"Baseline Peak Battery Temp: {baseline_fom:.2f} K\n")
    
    print("--- Baseline Heat Flow Report ---")
    Q_gen = baseline_flows.get('Q_gen_ESC', 0)
    Q_conv = baseline_flows.get('Q_out_ESC_conv', 0)
    Q_cond = baseline_flows.get('Q_out_ESC_cond', 0)
    Q_rad = baseline_flows.get('Q_out_ESC_rad', 0)
    Q_total_out = Q_conv + Q_cond + Q_rad
    print(f"ESC Heat Generation: {Q_gen:.2f} W")
    print(f"Heat Rejection Paths for ESC (Steady State):")
    if Q_total_out > 0:
        print(f"  - Conduction to Mount: {Q_cond:.2f} W ({Q_cond/Q_total_out:.1%})")
        print(f"  - Convection to Air:   {Q_conv:.2f} W ({Q_conv/Q_total_out:.1%})")
        print(f"  - Radiation to Walls:  {Q_rad:.2f} W ({Q_rad/Q_total_out:.1%})")
    print("-" * 35)
    print(f"Total Nacelle Heat Rejection to Environment: {baseline_flows.get('Q_total_shell_rejection', 0):.2f} W\n")

    results = []
    
    for var in variables_to_test:
        print(f"Testing variable: {var}...")
        params_plus = copy.deepcopy(baseline_params)
        params_plus[var] *= (1 + PERTURBATION)
        fom_plus, _ = run_simulation(params_plus)
        
        params_minus = copy.deepcopy(baseline_params)
        params_minus[var] *= (1 - PERTURBATION)
        fom_minus, _ = run_simulation(params_minus)
        
        sensitivity = ((fom_plus - fom_minus) / baseline_fom) / (2 * PERTURBATION) if baseline_fom != 0 else 0
        absolute_change = fom_plus - baseline_fom 
        
        results.append({
            'Parameter': var, 'Sensitivity (%)': sensitivity * 100,
            'Abs_Change_K_for_+10%': absolute_change,
            'T_peak_+10%': fom_plus, 'T_peak_-10%': fom_minus
        })

    df_results = pd.DataFrame(results)
    df_results['Abs_Sensitivity'] = df_results['Sensitivity (%)'].abs()
    df_results = df_results.sort_values(by='Abs_Sensitivity', ascending=False).drop(columns='Abs_Sensitivity')
    
    print("\n--- Sensitivity Analysis Report ---")
    print(df_results.to_string(index=False))
    
    # --- Corrected File Saving ---
    script_dir = os.path.dirname(os.path.abspath(__file__))
    output_filename = os.path.join(script_dir, "sensitivity_analysis_report.xlsx")
    df_results.to_excel(output_filename, index=False, engine='openpyxl')
    print(f"\nFull report saved to '{output_filename}'")
    
    print(f"\nTotal analysis execution time: {time.time() - start_analysis_time:.2f} seconds")

    # --- Plotting ---
    df_plot = df_results[['Parameter', 'Sensitivity (%)']].set_index('Parameter')
    plt.figure(figsize=(12, 10))
    bars = plt.barh(df_plot.index, df_plot['Sensitivity (%)'], 
                    color=(df_plot['Sensitivity (%)'] > 0).map({True: '#d62728', False: '#1f77b4'}))
    plt.title('Sensitivity Analysis of Nacelle Thermal Model', fontsize=16)
    plt.xlabel('Sensitivity (% Change in Peak Temp for a +1% Parameter Change)', fontsize=12)
    plt.ylabel('Parameter', fontsize=12)
    plt.axvline(0, color='black', linewidth=0.8)
    plt.grid(axis='x', linestyle='--', alpha=0.7)
    plt.gca().invert_yaxis()
    plt.tight_layout()
    plt.show()