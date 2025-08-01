print('Conditions: Ambient Temperature = 293K, Ground Level (1 atm), HAP velocity = 0 m/s')

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# -------------------------
# Load Air Properties from Excel # -------------------------
path = "./298.xlsx"
df_air = pd.read_excel(path, sheet_name=1)
temperatures = df_air['Temperature (K)'].tolist()
rho_values   = df_air['Density (rho) kg/m3'].tolist()
cp_values    = df_air['Specific Heat (cp) J/kg.K'].tolist()
k_values     = df_air['Thermal Conductivity (k) W/m.K'].tolist()
mu_values    = df_air['Dynamic Viscosity (m)  kg/m.s'].tolist()
nu_values    = df_air['Kinematic Viscosity (n)  m2/s'].tolist()
Pr_values    = df_air['Prandtl Number (Pr)'].tolist()

def prop(T):
    # Interpolate air properties at temperature T.
    r  = np.interp(T, temperatures, rho_values)
    cp = np.interp(T, temperatures, cp_values)
    k  = np.interp(T, temperatures, k_values)
    mu = np.interp(T, temperatures, mu_values)
    nu = np.interp(T, temperatures, nu_values)
    Pr = np.interp(T, temperatures, Pr_values)
    return r, cp, k, mu, nu, Pr

def T_power4(T):
    # Compute T^4 safely.
    return np.clip(T, 1e-6, 1e4)**4

def rad_coeff(e1,e2,a1,a2,vf):
    sigma = 5.67e-8
    C_rad = sigma/((1-e1)/(e1*a1) + 1/(a1*vf) + (1-e2)/(e2*a2))
    return(C_rad)


# -------------------------
# Constants and Given Values
# -------------------------
# Masses (kg)
m_B_total = 64.8         # Total battery pack mass (24 cells, 2.7 kg each)
m_Bf = m_B_total / 3     # Battery Front Node
m_Bm = m_B_total / 3     # Battery Middle Node
m_Br = m_B_total / 3     # Battery Rear Node           # Motor mass (assumed)
m_ESC = 0.12             # ESC mass (given)
m_TS  = 0.9              # Top Shell mass
m_BS  = 0.9              # Bottom Shell mass

# Specific Heat Capacities (J/kg·K) All assumed 
C_B   = 1100   # Battery effective cp
C_M   = 450    # Motor cp
C_ESC = 100    # ESC cp
C_TS  = 1040   # Top Shell cp
C_BS  = 1040   # Bottom Shell cp


# Heat Generation (W)
Q_B_total = 232.8         # Total battery heat generation (W)
Q_B_front  = Q_B_total/3  # Battery generation, distributed equally
Q_B_middle = Q_B_total/3
Q_B_rear   = Q_B_total/3
Q_ESC = 100           # ESC
Q_S = 0 #solar
Q_A = 0 #albedo
Q_P = 0 #planetary
# -------------------------
# Geometry and Calculated Coefficients
# -------------------------
# Battery Pack dimensions: L=0.840 m, H=0.252 m, W=0.154 m; each node length = 0.840/3 = 0.280 m.
A_cross = 0.252 * 0.154         
k_eff = 2.5                    
C_cond = k_eff * A_cross / 0.280   
A_Bf_conv = 0.26617   # Battery Front node exposed area (m^2)
A_Bm_conv = 0.22736   # Battery Middle node
A_Br_conv = 0.26617   # Battery Rear node

# Motor and ESC convection areas
A_ESC_conv = 0.01348  # ESC total area (m^2)

# Nacelle external shells 
A_TS = 0.54168       # Top Shell effective area (m^2)
A_BS = 0.54168       # Bottom Shell effective area (m^2)

#VIEW FACTORS FOR DIFFERENT NODES
Bt_ts = 1
Bb_bs = 1
Bf_M = 1
Ts_Bs = 0.05

#Emissivity Factors
Battery_Pack= 0.90
Motor = 0.80
ESC = 0.80
Top_skin = 0.8
Bottom_skin = 0.8
# -------------------------
# Radiation Coefficients 
C_Bf_TS_int_rad  = rad_coeff(0.9, 0.75, 0.0427, 0.54168, 1)
C_TS_BS_rad  = rad_coeff(0.75, 0.75, 0.54168, 0.54168, 0.5)
C_ESC_TS_rad = rad_coeff(0.8, 0.75, 0.0013959, 0.54168, 1)
C_Bf_ESC_rad = rad_coeff(0.9, 0.8 , 0.038808, 0.00206415,1 )
# -------------------------
# Characteristic Lengths for Convection (m)
# -------------------------
# For battery surfaces:
LC_B_horiz = 0.277     # For horizontal surfaces (top/bottom) of battery nodes
LC_B_vert  = 0.252     # For vertical surfaces (front, sides) of battery nodes
# For ESC: use largest dimension, here 0.0695 m
LC_ESC = 0.0695
# For Top and Bottom Shells
LC_TS = LC_BS = 0.840
k_mount = 3       # W/m·K 
A_contact = 0.005   # m^2 (Assumed)
L_mount   = 0.005   # m (Assumed)

# Conduction coefficients (in W/K)
C_ESC_Bf_cond = k_mount * A_contact / L_mount     
g = 9.81
velocity = 0
T_E = 293.15
#print(T_E, V_a)
k_cfrp = 400
A_cfrp = 0.54168
t_cfrp = 0.0005

C_cond_cfrp = k_cfrp * A_cfrp / t_cfrp


# -------------------------
def convection_h(horizontal=True, T_surface=298.15, T_fluid=298.15, LC=0.28):

    p = prop((T_surface + T_fluid) / 2)
    beta = 2 / ((T_surface + T_fluid) / 2)
    deltaT = T_surface - T_fluid
    # Grashof number
    Gr = abs(LC**3 * 9.81 * beta * deltaT / (p[4] if p[4]>1e-12 else 1e-12)**2)
    Ra = Gr * p[5]
    if horizontal:
        if deltaT > 0:
            Nu = 0.54 * Ra**(1/4) if 1e4 < Ra < 1e7 else 0.15 * Ra**(1/3)
        else:
            Nu = 0.27 * Ra**(1/4) if 1e5 < Ra < 1e10 else 0.15 * Ra**(1/3)
    else:
        # Vertical surface correlation:
        if Ra <= 1e9:
            Nu = 0.68 + (0.67 * Ra**(1/4)) / (1 + (0.492/p[5])**(9/16))**(4/9)
        else:
            Nu = (0.085 + (0.387 * Ra**(1/6)) / (1 + (0.492/p[5])**(9/16))**(8/27))**2
    return Nu * p[2] / LC
def forced_convection_h(
        T_surface=298.15, T_fluid=298.15, L_char=LC_TS, velocity=0):
        """
        External forced convection, fallback to natural if Re ~ 0
        """
        p = prop((T_surface + T_fluid)/2)
        Re = p[0] * velocity * L_char / max(p[3],1e-12)

        if Re < 1e-12:
            # fallback to natural
            beta = 2.0/(T_surface + T_fluid)
            Gr   = abs(L_char**3*g*beta*(T_surface - T_fluid)/max(p[4],1e-12)**2)
            Ra   = Gr*p[5]
            if 1e4 < Ra < 1e7:
                Nu = 0.54 * Ra**0.25
            else:
                Nu = 0.15 * Ra**(1/3)
        else:
            # laminar vs. turbulent
            if Re <= 5e5:
                Nu = 0.664 * (Re**0.5) * (p[5]**(1/3))
            else:
                Nu = 0.037 * (Re**(4/5)) * (p[5]**(1/3))
        return Nu * p[2] / L_char

# -------------------------
# ODE Function: Unsteady State Energy Balance
# -------------------------
def f(t, x):
    T_Bf, T_Bm, T_Br, T_ESC, T_TS_int, T_TS_ext, T_BS_int, T_BS_ext, T_air = x
    p_air = prop(T_air)
    p1 = prop(T_air)
    V_a = 0.11
    m_a = p1[0]*V_a
    k = p1[1]*m_a

    h_top_batt = convection_h(horizontal=True, T_surface=T_Bf, T_fluid=T_air, LC=LC_B_horiz)
    h_bottom_batt = convection_h(horizontal=True, T_surface=T_Bf, T_fluid=T_air, LC=LC_B_horiz)
    h_vert_batt = convection_h(horizontal=False, T_surface=T_Bf, T_fluid=T_air, LC=LC_B_vert)
    h_batt = (h_top_batt + h_bottom_batt + 2*h_vert_batt) / 4.0

    Q_conv_Bf = h_batt * A_Bf_conv * (T_air - T_Bf)
    Q_conv_Bm = h_batt * A_Bm_conv * (T_air - T_Bm)
    Q_conv_Br = h_batt * A_Br_conv * (T_air - T_Br)

    Q_cond_Bf = C_cond * (T_Bm - T_Bf)
    Q_cond_Bm = C_cond * ((T_Bf - T_Bm) + (T_Br - T_Bm))
    Q_cond_Br = C_cond * (T_Bm - T_Br)

    Q_rad_Bf = (C_Bf_TS_int_rad * (T_power4(T_TS_int) - T_power4(T_Bf)) +
                C_Bf_ESC_rad* (T_power4(T_ESC)- T_power4(T_Bf)) + C_Bf_TS_int_rad * (T_power4(T_BS_int) - T_power4(T_Bf)))
    Q_rad_Bm =  (C_Bf_TS_int_rad * (T_power4(T_TS_int) - T_power4(T_Bm)) +
                      C_Bf_TS_int_rad * (T_power4(T_BS_int) - T_power4(T_Bm)))
    Q_rad_Br =  (C_Bf_TS_int_rad * (T_power4(T_TS_int) - T_power4(T_Br)) +
                      C_Bf_TS_int_rad * (T_power4(T_BS_int) - T_power4(T_Br)))

    p_ESC = prop((T_ESC+T_air)/2)
    h_ESC = convection_h(horizontal=True, T_surface=T_ESC, T_fluid=T_air, LC=LC_ESC)
    Q_conv_ESC = h_ESC * A_ESC_conv * (T_air - T_ESC)

    Q_rad_ESC = (C_Bf_ESC_rad * (T_power4(T_Bf) - T_power4(T_ESC)) +
                    C_ESC_TS_rad * (T_power4(T_TS_int) - T_power4(T_ESC)) +
                     C_ESC_TS_rad * (T_power4(T_BS_int) - T_power4(T_ESC)))
    Q_cond_ESC_Bf = C_ESC_Bf_cond * (T_ESC - T_Bf)

    Q_cond_TS_ext = C_cond_cfrp * (T_TS_int - T_TS_ext)
    Q_cond_TS_int = C_cond_cfrp * (T_TS_ext - T_TS_int)
    Q_cond_BS_ext = C_cond_cfrp * (T_BS_int - T_BS_ext)
    Q_cond_BS_int = C_cond_cfrp * (T_BS_ext - T_BS_int)

    h_TS_in = convection_h(horizontal=True, T_surface=T_TS_int, T_fluid=T_air, LC=LC_TS)
    Q_conv_TS_in = h_TS_in * A_TS * (T_air - T_TS_int)
    h_TS_ext      = forced_convection_h(T_TS_ext, T_E, LC_TS, velocity)
    Q_conv_TS_ext = h_TS_ext*A_TS*(T_E - T_TS_ext) 
    h_BS_in = convection_h(horizontal=True, T_surface=T_BS_int, T_fluid=T_air, LC=LC_BS)
    Q_conv_BS_in = h_BS_in * A_BS * (T_air - T_BS_int)
    h_BS_ext      = forced_convection_h(T_BS_ext, T_E, LC_BS, velocity)
    Q_conv_BS_ext = h_BS_ext*A_TS*(T_E - T_BS_ext)  

    Q_rad_TS = (C_Bf_TS_int_rad * ((T_power4(T_Bf) - T_power4(T_TS_int)) + (T_power4(T_Bm) - T_power4(T_TS_int)) + (T_power4(T_Br) - T_power4(T_TS_int))) +
                C_TS_BS_rad * (T_power4(T_BS_int) - T_power4(T_TS_int)))

    Q_rad_BS = (C_Bf_TS_int_rad * ((T_power4(T_Bf) - T_power4(T_BS_int)) + (T_power4(T_Bm) - T_power4(T_BS_int)) + (T_power4(T_Br) - T_power4(T_BS_int))) +
                C_TS_BS_rad * (T_power4(T_TS_int) - T_power4(T_BS_int)))

    Q_conv_air = (h_batt * A_Bf_conv * (T_Bf - T_air) +
                  h_batt * A_Bm_conv * (T_Bm - T_air) +
                  h_batt * A_Br_conv * (T_Br - T_air) +
                  h_ESC * A_ESC_conv * (T_ESC - T_air) +
                  h_TS_in * A_TS * (T_TS_int - T_air) +
                  h_BS_in * A_BS * (T_BS_int - T_air))

    dT_Bf_dt  = (Q_B_front  + Q_cond_Bf  + Q_conv_Bf  + Q_rad_Bf + Q_cond_ESC_Bf  )  / (m_Bf * C_B)
    dT_Bm_dt  = (Q_B_middle + Q_cond_Bm  + Q_conv_Bm  + Q_rad_Bm)  / (m_Bm * C_B)
    dT_Br_dt  = (Q_B_rear   + Q_cond_Br  + Q_conv_Br  + Q_rad_Br)  / (m_Br * C_B)
    dT_ESC_dt = (Q_ESC + Q_conv_ESC + Q_rad_ESC - Q_cond_ESC_Bf ) / (m_ESC * C_ESC)
    dT_TS_int_dt  = (Q_cond_TS_int + Q_conv_TS_in + Q_rad_TS)          / (m_TS * C_TS)
    dT_TS_ext_dt  = (Q_cond_TS_ext + Q_conv_TS_ext + Q_S) / (m_TS * C_TS)
    dT_BS_int_dt  = (Q_cond_BS_int + Q_conv_BS_in + Q_rad_BS )          / (m_BS * C_BS)
    dT_BS_ext_dt  = (Q_cond_BS_ext + Q_conv_BS_ext + Q_A + Q_P ) / (m_BS * C_BS)
    dT_air_dt = Q_conv_air / (m_a*p1[1])

    return np.array([dT_Bf_dt, dT_Bm_dt, dT_Br_dt, dT_ESC_dt, dT_TS_int_dt, dT_TS_ext_dt, dT_BS_int_dt,dT_BS_ext_dt, dT_air_dt])


# -------------------------
# Runge-Kutta 4th Order Solver
# -------------------------
def runge_kutta_4(t0, x0, dt, T_total):
    t_vals = [t0]
    x_vals = [x0.copy()]
    t = t0
    x = x0.copy()
    while t < T_total:
        k1 = dt * f(t, x)
        k2 = dt * f(t + dt/2, x + k1/2)
        k3 = dt * f(t + dt/2, x + k2/2)
        k4 = dt * f(t + dt, x + k3)
        x = x + (k1 + 2*k2 + 2*k3 + k4) / 6.0
        t += dt
        t_vals.append(t)
        x_vals.append(x.copy())
    return np.array(t_vals), np.array(x_vals)

# -------------------------
# Initial Conditions (in Kelvin)
# -------------------------
T_Bf0   = 298.15
T_Bm0   = 298.15
T_Br0   = 298.15
T_ESC0  = 298.15
T_TS_int0   = 298.15
T_TS_ext0   = 298.15
T_BS_int0   = 298.15
T_BS_ext0   = 298.15
T_air0  = 298.15

x0 = np.array([T_Bf0, T_Bm0, T_Br0, T_ESC0, T_TS_int0, T_TS_ext0, T_BS_int0, T_BS_ext0, T_air0])
print("Initial Temperatures (K):", x0)

# -------------------------
# Simulation Parameters
# -------------------------
t0 = 0
dt = 0.001        # time step in seconds
T_total = 1800    # total simulation time in seconds

t_vals, x_vals = runge_kutta_4(t0, x0, dt, T_total)

# -------------------------
# Print Final Temperatures
# -------------------------
final_temps = x_vals[-1]
labels = ['Battery Front', 'Battery Middle', 'Battery Rear','ESC', 'Top Shell Internal','Top shell external','Bottom shell internal', 'Bottom Shell external', 'Internal Air']
print("Final Temperatures (K):")

for lab, temp in zip(labels, final_temps):
    print(f"{lab}: {temp:.2f} K")

# -------------------------
# Plotting Results
# -------------------------
plt.figure(figsize=(12, 8))
for i, lab in enumerate(labels):
    plt.plot(t_vals, x_vals[:, i], label=lab)
plt.xlabel('Time (s)')
plt.ylabel('Temperature (K)')
plt.title('Transient Temperature Evolution in Nacelle Components')
plt.legend(loc='upper right')
plt.grid(True)
plt.show()