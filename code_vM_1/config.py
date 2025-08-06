# config.py

# --- Simulation Control ---
TARGET_ALTITUDE_KM = 0.1
T_total = 86400 * 0.9  # Total simulation time in seconds (e.g., 3 days)
velocity = 0 # Aircraft velocity [m/s]
initial_temp_K = 287.34 # Initial temperature of all components

# --- Node Labels (17 Total Nodes) ---
labels = [
    # 6 Battery Nodes
    'Batt_BF_Top', 'Batt_BF_Bot', 'Batt_BM_Top', 'Batt_BM_Bot', 'Batt_BR_Top', 'Batt_BR_Bot',
    # 2 Avionics Nodes
    'ESC', 'ESC_Mount',
    # 4 Bulkhead Nodes
    'BH_1', 'BH_2', 'BH_3', 'BH_4',
    # 4 Shell Nodes
    'Top_Shell_Int', 'Top_Shell_Ext', 'Bot_Shell_Int', 'Bot_Shell_Ext',
    # 1 Air Node
    'Internal_Air'
]

# --- BATTERY ZONE PROPERTIES (Based on 6 zones of 4 cells each) ---
m_B_total = 32.4
Q_B_total = 12.6
m_batt_zone = m_B_total / 6  # Mass per 4-cell zone
Q_batt_zone = Q_B_total / 6  # Heat per 4-cell zone
L_batt_zone = 0.320  # Length of a 4-cell zone
W_batt_zone = 0.196  # Width of a 4-cell zone
H_batt_zone = 0.141  # Height of a 4-cell zone

# --- OTHER COMPONENT MASSES (kg) ---
m_ESC = 0.12; m_TS  = 0.9; m_BS  = 0.9
m_mount = 0.044 # Recalculated for Aluminum with new ESC base area
m_bulkhead = 0.035 # Mass of one "hollow frame" bulkhead

# --- SPECIFIC HEAT CAPACITIES (J/kg·K) ---
C_B = 1100; C_ESC = 100; C_TS = 1040; C_BS = 1040
C_mount = 896     # Aluminum 6061
C_bulkhead = 1040 # CFRP

# --- OTHER HEAT GENERATION (W) ---
Q_ESC = 100

# --- GEOMETRY: AREAS (m^2) AND LENGTHS (m) ---
# Convective Areas
A_conv_batt_middle = (2 * L_batt_zone * W_batt_zone) + (L_batt_zone * H_batt_zone) # Top, Bottom, 1 Side
A_conv_batt_end = A_conv_batt_middle + (W_batt_zone * H_batt_zone) # Middle area + 1 End
A_ESC_conv = 2 * (0.047 * 0.0695 + 0.047 * 0.0269 + 0.0695 * 0.0269) # Total surface area of ESC
A_mount_conv = 0.007 # Recalculated for new mount dimensions
A_bulkhead_face = 0.011 # Area of one face of the "hollow frame"
A_TS = 0.542; A_BS = 0.542
V_internal_air = 0.11

# Characteristic Lengths
LC_batt_horiz = (L_batt_zone * W_batt_zone) / (2 * (L_batt_zone + W_batt_zone))
LC_batt_vert = H_batt_zone
LC_ESC = 0.0695
LC_mount = 0.02
LC_bulkhead = 0.282 # Total vertical height of the nacelle interior
LC_TS_int = 0.065; LC_BS_int = 0.065; LC_TS_ext = 0.840; LC_BS_ext = 0.840

# --- MATERIAL PROPERTIES: THERMAL CONDUCTIVITY (W/m·K) ---
k_eff_batt = 0.5   # Effective conductivity within a 4-cell zone (low due to air gaps)
k_cfrp = 1.0     # Shell material
k_mount = 180.0    # Mount material (Aluminum 6061)
k_bulkhead = 1.0   # Bulkhead material (CFRP)

# --- INTERFACE GEOMETRY ---
# Path lengths (thicknesses)
t_cfrp = 0.0005
t_bulkhead = 0.002
L_path_ESC_Mount = 0.005 # Thickness of mount

# Contact Areas
A_contact_ESC_Mount = 0.047 * 0.0695 # Full base of ESC
A_contact_Mount_BH1 = A_contact_ESC_Mount # Mount is flush with bulkhead
A_contact_Batt_BH = W_batt_zone * H_batt_zone # Face of a battery node
A_contact_BH_Shell = 1.066 * t_bulkhead # Perimeter of bulkhead * its thickness

# --- SURFACE PROPERTIES: EMISSIVITY (Dimensionless) ---
emis_batt = 0.9; emis_esc = 0.8; emis_mount = 0.8
emis_shell_int = 0.85; emis_shell_ext = 0.8; emis_bulkhead = 0.85
alpha_solar_shell = 0.6

# --- ENVIRONMENT ---
g = 9.81