# config.py
# --- Simulation Control ---
TARGET_ALTITUDE_KM = 0.1
T_total = 86400 * 14  # Total simulation time in seconds (e.g., 3 days)
velocity = 0 # Aircraft velocity [m/s]
initial_temp_K = 305.0 # Initial temperature closer to ambient to reduce thermal shock

# --- Node Labels (20 Total Nodes) ---
labels = [
    # 6 Battery Nodes
    'Batt_BF_Top', 'Batt_BF_Bot', 'Batt_BM_Top', 'Batt_BM_Bot', 'Batt_BR_Top', 'Batt_BR_Bot',
    # 2 Avionics Nodes
    'ESC', 'ESC_Mount',
    # 4 Bulkhead Nodes
    'BH_1', 'BH_2', 'BH_3', 'BH_4',
    #Plates
    'plateT', 'plateM', 'plateB',
    # 4 Shell Nodes
    'Top_Shell_Int', 'Top_Shell_Ext', 'Bot_Shell_Int', 'Bot_Shell_Ext',
    # 1 Air Node
    'Internal_air'
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
m_ESC = 0.12 
m_TS  = 0.9
m_BS  = 0.9
m_mount = 0.73 # Recalculated for Aluminum with new ESC base area
m_bulkhead = 0.163 # Mass of one bulkhead
m_plate = 1.91 # mass of one plate in kg

# --- SPECIFIC HEAT CAPACITIES (J/kg·K) ---
C_B = 1100 
C_ESC = 100
C_TS = 1040
C_BS = 1040
C_mount = 896     # Aluminum 6061
C_bulkhead = 1040 # CFRP
C_plate = 1040 #CFRP

# --- OTHER HEAT GENERATION (W) ---
Q_ESC = 100
L_esc = 0.047
W_esc = 0.0695
H_esc = 0.0269

# --- GEOMETRY: AREAS (m^2) AND LENGTHS (m) ---
# Radiaiton Areas
A_rad_batt_to_batt = 0.196 * 0.141
A_rad_batt_to_shell = 0.196 * 0.32
A_rad_batt_bh = 0.001008 # i assumed one battery sees only 20% of the internal surface area of the bulkhead so total internal area * 0.2

# Convective Areas
A_conv_batt_side = 2 * (L_batt_zone * H_batt_zone + H_batt_zone * W_batt_zone) # area of all battery sides combined
A_conv_batt_top = (W_batt_zone * L_batt_zone) # Area of top surface of battery
A_conv_batt_total = A_conv_batt_top + A_conv_batt_side
A_conv_plate = 0.6 - (3 * A_conv_batt_top)
A_conv_plateM = 0.6 - (6 * A_conv_batt_top) #area of plate - area of battery top
A_conv_esc_side = 2 * (L_esc * H_esc + H_esc * W_esc) # area of all esc sides combined
A_conv_esc_top = (W_esc * L_esc) # Area of top surface of esc
A_ESC_conv = A_conv_esc_top + A_conv_esc_side # Total surface area of ESC

A_mount_conv = 0.1545 * 0.3492 # Recalculated for new mount dimensions
A_bulkhead_face = 0.034 # Area of one face of the bulkhead
A_Plate =0.3 
A_TS = 0.542; A_BS = 0.542 # to be finalised still
V_internal_air = 0.5 # Increased thermal mass to prevent instability

# Characteristic Lengths
LC_batt_horiz = (L_batt_zone * W_batt_zone) / (2 * (L_batt_zone + W_batt_zone))
LC_batt_vert = H_batt_zone
LC_plate = 0.3/2.6
LC_esc_horiz = (L_esc * W_esc) / (2 * (L_esc + W_esc)) 
LC_esc_vert = H_esc
LC_mount = 0.3492 * 0.1545/(2 * (0.3492 + 0.1545))
LC_bulkhead = 0.380 # Total vertical height of the nacelle interior
LC_TS_int = 0.211; LC_BS_int = 0.211; LC_TS_ext = 1.010; LC_BS_ext = 1.010

# --- MATERIAL PROPERTIES: THERMAL CONDUCTIVITY (W/m·K) ---
k_eff_batt = 0.5   # Effective conductivity within a 4-cell zone (low due to air gaps)
k_cfrp = 1.0     # Shell material
k_mount = 10.0    # Mount material (Aluminum 6061)
k_bulkhead = 1.0   # Bulkhead material (CFRP)

# --- INTERFACE GEOMETRY ---
# Path lengths (thicknesses)
t_cfrp = 0.0005
t_bulkhead = 0.002
t_plate = 0.003
L_path_ESC_Mount = 0.005 # Thickness of mount
L_bh_plate_cond = 0.025
# Contact Areas
A_contact_ESC_Mount = 0.047 * 0.0695 # Full base of ESC
A_contact_Mount_BH1 = 2 * (0.005 * 0.025) # 2 connections mount_t * bh_w
A_contact_BH_Shell = 2 * 0.780 * t_bulkhead # Perimeter of bulkhead * its thickness
A_contact_batt_plate = 0.196 * 0.32
A_contact_bh_plate = 0.002 * 0.003 
# --- SURFACE PROPERTIES: EMISSIVITY (Dimensionless) ---
emis_batt = 0.9
emis_esc = 0.8
emis_mount = 0.8
emis_shell_int = 0.85
emis_shell_ext = 0.8
emis_bulkhead = 0.85
emis_plate = 0.85
alpha_solar_shell = 0.6

# --- ENVIRONMENT ---
g = 9.81