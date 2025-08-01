# 2_config.py

# --- Simulation Control ---
TARGET_ALTITUDE_KM = 0.1
T_total = 86400 * 30  # Total simulation time in seconds (e.g., 3 days)
velocity = 0 # Aircraft velocity [m/s]
initial_temp_K = 298.15 # Initial temperature of all components

# --- Node Labels ---
labels = ['Battery Front', 'Battery Middle', 'Battery Rear', 'ESC', 'ESC Mount', 'Top Shell Internal',
          'Top Shell External', 'Bottom Shell Internal', 'Bottom Shell External', 'Internal Air']

# --- Component Masses (kg) ---
m_Bf, m_Bm, m_Br = 64.8 / 3, 64.8 / 3, 64.8 / 3
m_ESC, m_TS, m_BS, m_mount = 0.12, 0.9, 0.9, 0.050

# --- Specific Heat Capacities (J/kg·K) ---
C_B, C_ESC, C_TS, C_BS, C_mount = 1100, 100, 1040, 1040, 900

# --- Internal Heat Generation (W) ---
Q_B_front, Q_B_middle, Q_B_rear = 232.8 / 3, 232.8 / 3, 232.8 / 3
Q_ESC = 100

# --- Geometry: Areas (m^2) and Lengths (m) ---
A_cross_batt = 0.252 * 0.154
A_Bf_conv, A_Bm_conv, A_Br_conv = 0.26617, 0.22736, 0.26617
A_ESC_conv, A_mount_conv = 0.01348, 0.01
A_TS, A_BS = 0.54168, 0.54168
V_internal_air = 0.11
LC_B_horiz, LC_B_vert, LC_ESC, LC_mount = 0.277, 0.252, 0.0695, 0.05
LC_TS_int, LC_BS_int, LC_TS_ext, LC_BS_ext = 0.065, 0.065, 0.840, 0.840

# --- Material Properties: Thermal Conductivity (W/m·K) ---
k_eff_batt, k_cfrp, k_al_mount = 2.5, 1.0, 180

# --- Interface Geometry ---
L_node_batt, t_cfrp = 0.280, 0.0005
A_contact_ESC_Mount, L_path_ESC_Mount = 0.005, 0.005
A_contact_Mount_Bf, L_path_Mount_Bf = 0.005, 0.005

# --- Surface Properties: Emissivity (Dimensionless) ---
emis_batt = 0.9; emis_esc = 0.8; emis_mount = 0.7
emis_shell_int = 0.85 # Using a more realistic value for internal composite
emis_shell_ext = 0.8
alpha_solar_shell = 0.6

# --- Environment ---
g = 9.81