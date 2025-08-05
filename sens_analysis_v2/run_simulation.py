# run_full_analysis.py

import time
import copy
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

# --- 1. SETUP AND PARAMETERS ---

# All Simulation Parameters in one place
baseline_params = {
    'm_B_total': 64.8, 'm_ESC': 0.12, 'm_TS': 0.9, 'm_BS': 0.9, 'm_mount': 0.023,
    'C_mount': 800, 'C_B': 1100, 'C_ESC': 100, 'C_TS': 1040, 'C_BS': 1040,
    'Q_B_total': 232.8, 'Q_ESC': 100, 'Q_S': 0, 'Q_A': 0, 'Q_P': 0,
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

# --- 2. DATA LOADING ---
COLUMN_NAMES = {
    "temp": "Temperature (K)", "rho": "Density (rho) kg/m3", "cp": "Specific Heat (cp) J/kg.K",
    "k": "Thermal Conductivity (k) W/m.K", "mu": "Dynamic Viscosity (m)  kg/m.s"
}
try:
    path = "./298.xlsx"
    df_air = pd.read_excel(path, sheet_name = 1)
    temperatures = df_air[COLUMN_NAMES["temp"]].tolist()
    rho_values   = df_air[COLUMN_NAMES["rho"]].tolist()
    cp_values    = df_air[COLUMN_NAMES["cp"]].tolist()
    k_values     = df_air[COLUMN_NAMES["k"]].tolist()
    mu_values    = df_air[COLUMN_NAMES["mu"]].tolist()
except FileNotFoundError:
    raise SystemExit("FATAL ERROR: 298.xlsx not found.")
except KeyError as e:
    raise SystemExit(f"FATAL ERROR: Column {e} not found in 298.xlsx.")

# --- 3. PHYSICS MODELS ---

def prop(T):
    r  = np.interp(T, temperatures, rho_values)
    cp = np.interp(T, temperatures, cp_values)
    k  = np.interp(T, temperatures, k_values)
    mu = np.interp(T, temperatures, mu_values)
    nu = mu / r if r > 1e-9 else 0
    Pr = (mu * cp) / k if k > 1e-9 else 0
    return r, cp, k, mu, nu, Pr

def T_power4(T): return np.clip(T, 1.0, 1e7)**4

def natural_convection_h(p_film, T_surface, T_fluid, L_char, is_vertical, g):
    k, Pr, nu_val = p_film[2], p_film[5], p_film[4]
    if abs(T_surface - T_fluid) < 1e-4 or not all([Pr, nu_val, k]) or Pr <= 0: return 0.0
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
    if velocity > 0.1: # Forced convection
        k, Pr, rho, mu = p_film[2], p_film[5], p_film[0], p_film[3]
        if not all([Pr, k]) or Pr <= 0: return 0.0
        Re_L = rho * velocity * L_char / max(mu, 1e-12)
        if Re_L < 100: return 0.0
        Re_crit = 5e5
        try:
            if Re_L <= Re_crit: Nu = 0.664 * (Re_L**0.5) * (Pr**(1/3))
            else: Nu = (0.037 * (Re_L**0.8) - 871) * (Pr**(1/3))
        except (ValueError, OverflowError): Nu = 0.0
        return Nu * k / L_char
    else: # Natural convection
        return natural_convection_h(p_film, T_surface, T_fluid, L_char, is_vertical=False, g=g)

# --- 4. SIMULATION AND ANALYSIS FUNCTIONS ---

def f(t, x, p, return_flows=False):
    # This is your trusted physics logic, meticulously integrated and corrected.
    for key, value in p.items(): globals()[key] = value
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
    C_Bf_TS_int_rad = rad_coeff(0.9, 0.2, A_Bf_conv, A_TS)
    C_Bf_BS_int_rad = rad_coeff(0.9, 0.2, A_Bf_conv, A_BS)
    C_TS_BS_rad = rad_coeff(0.2, 0.2, A_TS, A_BS, 0.5)
    C_ESC_TS_rad = rad_coeff(0.8, 0.2, A_ESC_conv, A_TS)
    C_ESC_BS_rad = rad_coeff(0.8, 0.2, A_ESC_conv, A_BS)
    C_Bf_ESC_rad = rad_coeff(0.9, 0.8, A_Bf_conv, A_ESC_conv) # Corrected areas for Batt-ESC rad
    C_Mount_rad = rad_coeff(0.7, 0.2, A_mount_conv, A_TS)
    
    T_Bf, T_Bm, T_Br, T_ESC, T_mount, T_TS_int, T_TS_ext, T_BS_int, T_BS_ext, T_air = x
    p_air = prop(T_air)
    T4s = {k: T_power4(v) for k, v in zip(['Bf','Bm','Br','ESC','mount','TS_int','BS_int'], [T_Bf,T_Bm,T_Br,T_ESC,T_mount,T_TS_int,T_BS_int])}

    h_batt_avg = (natural_convection_h(prop((T_Bf+T_air)/2), T_Bf, T_air, LC_B_horiz, False, g)*2 + natural_convection_h(prop((T_Bf+T_air)/2), T_Bf, T_air, LC_B_vert, True, g)*2)/4
    Q_conv_Bf = h_batt_avg * A_Bf_conv * (T_air - T_Bf)
    Q_conv_Bm = h_batt_avg * A_Bm_conv * (T_air - T_Bm)
    Q_conv_Br = h_batt_avg * A_Br_conv * (T_air - T_Br)
    Q_cond_Bf_in = C_cond * (T_Bm - T_Bf)
    Q_cond_Bm_net = C_cond * ((T_Bf - T_Bm) + (T_Br - T_Bm))
    Q_cond_Br_in = C_cond * (T_Bm - T_Br)
    Q_rad_Bf = C_Bf_TS_int_rad * (T4s['TS_int'] - T4s['Bf']) + C_Bf_ESC_rad * (T4s['ESC'] - T4s['Bf']) + C_Bf_BS_int_rad * (T4s['BS_int'] - T4s['Bf'])
    Q_rad_Bm = C_Bf_TS_int_rad * (T4s['TS_int'] - T4s['Bm']) + C_Bf_BS_int_rad * (T4s['BS_int'] - T4s['Bm'])
    Q_rad_Br = C_Bf_TS_int_rad * (T4s['TS_int'] - T4s['Br']) + C_Bf_BS_int_rad * (T4s['BS_int'] - T4s['Br'])
    Q_cond_Mount_to_Bf = C_cond_Mount_to_Bf * (T_mount - T_Bf)
    Q_conv_ESC = natural_convection_h(prop((T_ESC+T_air)/2), T_ESC, T_air, LC_ESC, False, g) * A_ESC_conv * (T_air - T_ESC)
    Q_rad_ESC = C_Bf_ESC_rad * (T4s['Bf'] - T4s['ESC']) + C_ESC_TS_rad * (T4s['TS_int'] - T4s['ESC']) + C_ESC_BS_rad * (T4s['BS_int'] - T4s['ESC'])
    Q_cond_ESC_to_Mount = C_cond_ESC_to_Mount * (T_ESC - T_mount)
    Q_conv_Mount = natural_convection_h(prop((T_mount+T_air)/2), T_mount, T_air, LC_mount, False, g) * A_mount_conv * (T_air - T_mount)
    Q_rad_Mount_TS = C_Mount_rad * (T4s['TS_int'] - T4s['mount'])
    Q_cond_TS_ext = C_cond_cfrp * (T_TS_int - T_TS_ext)
    Q_cond_BS_ext = C_cond_cfrp * (T_BS_int - T_BS_ext)
    Q_conv_TS_in = natural_convection_h(prop((T_TS_int+T_air)/2), T_TS_int, T_air, LC_TS, False, g) * A_TS * (T_air - T_TS_int)
    Q_conv_TS_ext = get_external_surface_h(prop((T_TS_ext+T_E)/2), T_TS_ext, T_E, LC_TS, velocity, g) * A_TS * (T_E - T_TS_ext)
    Q_conv_BS_in = natural_convection_h(prop((T_BS_int+T_air)/2), T_BS_int, T_air, LC_BS, False, g) * A_BS * (T_air - T_BS_int)
    Q_conv_BS_ext = get_external_surface_h(prop((T_BS_ext+T_E)/2), T_BS_ext, T_E, LC_BS, velocity, g) * A_BS * (T_E - T_BS_ext)
    Q_rad_TS_in = C_Bf_TS_int_rad * (T4s['Bf'] - T4s['TS_int']) + C_Bf_TS_int_rad * (T4s['Bm'] - T4s['TS_int']) + C_Bf_TS_int_rad * (T4s['Br'] - T4s['TS_int']) + C_TS_BS_rad * (T4s['BS_int'] - T4s['TS_int']) + C_Mount_rad * (T4s['mount'] - T4s['TS_int'])
    Q_rad_BS_in = C_Bf_BS_int_rad * (T4s['Bf'] - T4s['BS_int']) + C_Bf_BS_int_rad * (T4s['Bm'] - T4s['BS_int']) + C_Bf_BS_int_rad * (T4s['Br'] - T4s['BS_int']) + C_TS_BS_rad * (T4s['TS_int'] - T4s['BS_int'])
    Q_conv_air_total = -(Q_conv_Bf + Q_conv_Bm + Q_conv_Br + Q_conv_ESC + Q_conv_Mount + Q_conv_TS_in + Q_conv_BS_in)
    m_a = prop(T_air)[0] * 0.11

    if return_flows:
        # Note: Signs are flipped here so that "out" flows are positive values
        return {
            'Q_gen_ESC': Q_ESC, 'Q_gen_Bf': Q_B_front, 'Q_gen_Bm': Q_B_middle, 'Q_gen_Br': Q_B_rear,
            'Q_ESC_out_conv': -Q_conv_ESC, 'Q_ESC_out_rad': -Q_rad_ESC, 'Q_ESC_out_cond': Q_cond_ESC_to_Mount,
            'Q_Mount_in_cond': Q_cond_ESC_to_Mount, 'Q_Mount_out_cond': -Q_cond_Mount_to_Bf, 'Q_Mount_out_conv': -Q_conv_Mount, 'Q_Mount_out_rad': -Q_rad_Mount_TS,
            'Q_Bf_in_cond_mount': Q_cond_Mount_to_Bf, 'Q_Bf_in_cond_batt': Q_cond_Bf_in, 'Q_Bf_out_conv': -Q_conv_Bf, 'Q_Bf_out_rad': -Q_rad_Bf, 'Q_Bf_out_cond_batt': -Q_cond_Bf_in,
            'Q_Bm_in_cond_Bf': -Q_cond_Bf_in, 'Q_Bm_out_cond_Br': C_cond * (T_Bm - T_Br), 'Q_Bm_out_conv': -Q_conv_Bm,
            'Q_Br_in_cond_Bm': C_cond * (T_Bm - T_Br), 'Q_Br_out_conv': -Q_conv_Br,
            'Q_TSin_in_rad': -Q_rad_TS_in, 'Q_TSin_in_conv': -Q_conv_TS_in, 'Q_TSin_out_cond': Q_cond_TS_ext,
            'Q_BSin_in_rad': -Q_rad_BS_in, 'Q_BSin_in_conv': -Q_conv_BS_in, 'Q_BSin_out_cond': Q_cond_BS_ext,
            'Q_Air_in_total': -Q_conv_air_total,
            'Total_System_Generation': Q_B_total + Q_ESC, 'Total_System_Rejection': -Q_conv_TS_ext - Q_conv_BS_ext
        }

    dT_Bf_dt     = (Q_B_front + Q_cond_Bf_in + Q_conv_Bf + Q_rad_Bf + Q_cond_Mount_to_Bf) / (m_Bf * C_B)
    dT_Bm_dt     = (Q_B_middle + Q_cond_Bm_net + Q_conv_Bm + Q_rad_Bm) / (m_Bm * C_B)
    dT_Br_dt     = (Q_B_rear + Q_cond_Br_in + Q_conv_Br + Q_rad_Br) / (m_Br * C_B)
    dT_ESC_dt    = (Q_ESC + Q_conv_ESC + Q_rad_ESC - Q_cond_ESC_to_Mount) / (m_ESC * C_ESC)
    dT_mount_dt  = (Q_cond_ESC_to_Mount - Q_cond_Mount_to_Bf + Q_conv_Mount + Q_rad_Mount_TS) / (m_mount * C_mount)
    dT_TS_int_dt = (-Q_cond_TS_ext + Q_conv_TS_in + Q_rad_TS_in) / (m_TS * C_TS)
    dT_TS_ext_dt = (Q_cond_TS_ext + Q_conv_TS_ext + Q_S) / (m_TS * C_TS)
    dT_BS_int_dt = (-Q_cond_BS_ext + Q_conv_BS_in + Q_rad_BS_in) / (m_BS * C_BS)
    dT_BS_ext_dt = (Q_cond_BS_ext + Q_conv_BS_ext + Q_A + Q_P) / (m_BS * C_BS)
    dT_air_dt    = Q_conv_air_total / (m_a * p_air[1])
    
    return np.array([dT_Bf_dt, dT_Bm_dt, dT_Br_dt, dT_ESC_dt, dT_mount_dt, dT_TS_int_dt, dT_TS_ext_dt, dT_BS_int_dt, dT_BS_ext_dt, dT_air_dt])

def run_simulation(params):
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
    
    return peak_temp_batt_front, heat_flows, sol

def print_detailed_heat_flow_report(flows):
    print("\n" + "="*50); print(" DETAILED HEAT FLOW PATH ANALYSIS (Steady State)"); print("="*50)
    
    def print_component_balance(name, Q_in, Q_out):
        print(f"\n--- Component: {name} ---")
        total_in = sum(Q_in.values()); total_out = sum(Q_out.values())
        print(f"  {'TOTAL HEAT IN:':<28} {total_in:.2f} W")
        for key, val in Q_in.items(): print(f"  {'  - ' + key:<27} {val:.2f} W")
        print(f"  {'TOTAL HEAT OUT:':<28} {total_out:.2f} W")
        if total_out > 1e-3:
            for key, val in Q_out.items(): print(f"  {'  - ' + key:<27} {val:.2f} W ({abs(val/total_out):.1%})")

    print_component_balance("ESC", {'Generation': flows['Q_gen_ESC']}, {'Conduction to Mount': flows['Q_ESC_out_cond'], 'Convection to Air': flows['Q_ESC_out_conv'], 'Radiation': flows['Q_ESC_out_rad']})
    print_component_balance("ESC Mount", {'Conduction from ESC': flows['Q_Mount_in_cond']}, {'Conduction to Battery': flows['Q_Mount_out_cond'], 'Convection to Air': flows['Q_Mount_out_conv'], 'Radiation': flows['Q_Mount_out_rad']})
    print_component_balance("Battery Front", {'Generation': flows['Q_gen_Bf'], 'Conduction from Mount': flows['Q_Bf_in_cond_mount'], 'Conduction from Batt Mid': flows['Q_Bf_in_cond_batt']}, {'Conduction to Batt Mid': flows['Q_Bf_out_cond_batt'], 'Convection to Air': flows['Q_Bf_out_conv'], 'Radiation': flows['Q_Bf_out_rad']})
    print_component_balance("Battery Middle", {'Generation': flows['Q_gen_Bm'], 'Conduction from Batt Front': -flows['Q_Bf_in_cond_batt']}, {'Conduction to Batt Rear': -flows['Q_Bf_in_cond_batt'] + flows['Q_gen_Bm'], 'Convection to Air': flows['Q_Bm_out_conv']}) # Simplified
    print_component_balance("Top Shell Internal", {'Convection from Air': -flows['Q_TSin_in_conv'], 'Radiation from internals': -flows['Q_TSin_in_rad']}, {'Conduction to Ext Face': flows['Q_TSin_out_cond']})
    print_component_balance("Internal Air", {'Heat from all components': flows['Q_Air_in_total']}, {})

    print("\n" + "="*50)
    total_gen = flows['Total_System_Generation']; total_rej = flows['Total_System_Rejection']
    print(f"{'OVERALL ENERGY BALANCE CHECK':^50}")
    print(f"  {'Total System Heat Generation:':<35} {total_gen:.2f} W")
    print(f"  {'Total Shell Heat Rejection:':<35} {total_rej:.2f} W")
    balance = abs((total_gen - total_rej) / total_gen * 100) if total_gen > 0 else 0
    print(f"  {'Energy Balance Error:':<35} {balance:.2f}%")
    print("="*50)

# --- 5. MAIN EXECUTION BLOCK ---
if __name__ == "__main__":
    start_time = time.time()
    
    print("--- Running Analysis with Detailed Report ---")
    
    # Run the baseline simulation
    peak_temp, heat_flows, solution = run_simulation(baseline_params)
    
    # Print the detailed report for the baseline case
    final_temps = solution.y[:, -1]
    labels = ['Battery Front', 'Battery Middle', 'Battery Rear', 'ESC', 'ESC Mount', 
              'Top Shell Internal', 'Top Shell External', 'Bottom Shell Internal', 
              'Bottom Shell External', 'Internal Air']
    print(f"\nTotal execution time: {time.time() - start_time:.2f} seconds")
    print("\n--- Final Temperatures (K) ---")
    for i, lab in enumerate(labels):
        print(f"{lab:<25}: {final_temps[i]:.2f}")
    print_detailed_heat_flow_report(heat_flows)
    
    # --- Sensitivity Analysis Section ---
    variables_to_test = [
        'Q_ESC', 'Q_B_total', 'k_mount', 'k_eff', 'k_cfrp',
        'A_contact_ESC_Mount', 'A_contact_Mount_Bf', 'L_path_ESC_Mount', 'L_path_Mount_Bf',
        'A_mount_conv', 'A_ESC_conv', 'A_Bf_conv', 'T_E'
    ]
    PERTURBATION = 0.10
    print("\n--- Starting Sensitivity Analysis ---")
    
    results = []
    for var in variables_to_test:
        print(f"Testing variable: {var}...")
        params_plus = copy.deepcopy(baseline_params)
        params_plus[var] *= (1 + PERTURBATION)
        fom_plus, _, _ = run_simulation(params_plus)
        
        params_minus = copy.deepcopy(baseline_params)
        params_minus[var] *= (1 - PERTURBATION)
        fom_minus, _, _ = run_simulation(params_minus)
        
        sensitivity = ((fom_plus - peak_temp) / peak_temp) / PERTURBATION if peak_temp != 0 else 0 # Simplified sensitivity
        
        results.append({'Parameter': var, 'Sensitivity (%)': sensitivity * 100})

    df_results = pd.DataFrame(results).sort_values(by='Sensitivity (%)', key=abs, ascending=False)
    
    print("\n--- Sensitivity Analysis Report ---")
    print(df_results.to_string(index=False))
    
    # Save to Excel
    script_dir = os.path.dirname(os.path.abspath(__file__))
    output_filename = os.path.join(script_dir, "sensitivity_analysis_report.xlsx")
    try:
        df_results.to_excel(output_filename, index=False, engine='openpyxl')
        print(f"\nFull report saved to '{output_filename}'")
    except ImportError:
        print("\nNOTE: 'openpyxl' is not installed. Report not saved to Excel. Run: pip install openpyxl")
    except Exception as e:
        print(f"\nCould not save Excel file. Error: {e}")

    # Plotting
    df_plot = df_results.set_index('Parameter')
    plt.figure(figsize=(12, 10))
    df_plot['Sensitivity (%)'].plot(kind='barh', color=(df_plot['Sensitivity (%)'] > 0).map({True: '#d62728', False: '#1f77b4'}))
    plt.title('Sensitivity Analysis'); plt.xlabel('Sensitivity (%)'); plt.ylabel('Parameter')
    plt.axvline(0, color='black', lw=0.8); plt.grid(axis='x', ls='--'); plt.gca().invert_yaxis(); plt.tight_layout(); plt.show()