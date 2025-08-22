# simple_heat_analysis.py
# Simple Heat Transfer Analysis for 20-Node UAV Thermal Model
# Shows heat gained/lost per node by transfer mode (conduction, convection, radiation)

import sys
import os
import numpy as np
import pandas as pd

# Add the code_vM_1 directory to the path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'code_vM_1'))
import config
import physics_models
import environment_model

class SimpleHeatAnalyzer:
    """
    Simple heat transfer analysis showing heat gained/lost per node by mode
    """
    
    def __init__(self):
        self.node_labels = config.labels
        
        # Initialize environment
        environment_model.init()
        
        # Calculate thermal coefficients
        self.thermal_coeffs = self._get_thermal_coefficients()
    
    def _get_thermal_coefficients(self):
        """Get all thermal coefficients"""
        return {
            # Conduction coefficients
            'C_cond_ESC_to_Mount': config.k_mount * config.A_contact_ESC_Mount / config.L_path_ESC_Mount,
            'C_cond_Mount_to_BH1': config.k_bulkhead * config.A_contact_Mount_BH1 / config.t_bulkhead,
            'C_cond_Batt_plate': config.k_plate * config.A_contact_batt_plate / config.t_plate,
            'C_cond_bh_plate': 2 * config.k_bulkhead * config.A_contact_bh_plate / config.L_bh_plate_cond,
            'C_cond_BH_TS': config.k_cfrp * config.A_contact_BH_Shell / config.t_cfrp,
            'C_cond_BH_BS': config.k_cfrp * config.A_contact_BH_Shell / config.t_cfrp,
            'C_cond_TS_int_ext': config.k_cfrp * config.A_TS / config.t_cfrp,
            'C_cond_BS_int_ext': config.k_cfrp * config.A_BS / config.t_cfrp,
            
            # Radiation coefficients
            'C_rad_batt_batt': physics_models.rad_coeff(config.emis_batt, config.emis_batt, 
                                                       config.A_rad_batt_to_batt, config.A_rad_batt_to_batt),
            'C_rad_batt_esc': physics_models.rad_coeff(config.emis_esc, config.emis_batt, 
                                                      config.A_ESC_conv, config.A_rad_batt_to_batt),
            'C_rad_batt_ts': physics_models.rad_coeff(config.emis_batt, config.emis_shell_int, 
                                                     config.A_rad_batt_to_shell, config.A_TS, vf=0.5),
            'C_rad_batt_bs': physics_models.rad_coeff(config.emis_batt, config.emis_shell_int, 
                                                     config.A_rad_batt_to_shell, config.A_BS, vf=0.5),
            'C_rad_batt_bh': physics_models.rad_coeff(config.emis_batt, config.emis_bulkhead, 
                                                     config.A_conv_batt_side, config.A_rad_batt_bh),
            'C_rad_esc_bh': physics_models.rad_coeff(config.emis_esc, config.emis_bulkhead, 
                                                    config.A_ESC_conv, config.A_bulkhead_face),
            'C_rad_esc_ts': physics_models.rad_coeff(config.emis_esc, config.emis_shell_int, 
                                                    config.A_ESC_conv, config.A_TS),
            'C_rad_plate_sh': physics_models.rad_coeff(config.emis_plate, config.emis_shell_int, 
                                                      config.A_Plate, config.A_BS),
            'C_rad_mount_bs': physics_models.rad_coeff(config.emis_mount, config.emis_shell_int, 
                                                      config.A_mount_conv, config.A_BS),
            'C_rad_bh_bh': physics_models.rad_coeff(config.emis_bulkhead, config.emis_bulkhead, 
                                                   config.A_bulkhead_face, config.A_bulkhead_face),
            'C_rad_ts_bs': physics_models.rad_coeff(config.emis_shell_int, config.emis_shell_int, 
                                                   config.A_TS, config.A_BS, vf=0.5),
        }
    
    def analyze_heat_transfer(self, temperatures, ambient_temp=287.3, ambient_pressure=99832.0):
        """
        Analyze heat transfer for all nodes showing gained/lost heat by mode
        """
        # Create temperature dictionary and T^4 values
        temps = {label: temperatures[i] for i, label in enumerate(self.node_labels)}
        T4s = {k: physics_models.T_power4(v) for k, v in temps.items()}
        
        # Initialize results
        results = {}
        for node in self.node_labels:
            results[node] = {
                'generation': 0.0,
                'conduction_total': 0.0,
                'convection_total': 0.0,
                'radiation_total': 0.0,
                'net_total': 0.0
            }
        
        # HEAT GENERATION
        for node in self.node_labels:
            if 'Batt' in node:
                results[node]['generation'] = config.Q_batt_zone
            elif node == 'ESC':
                results[node]['generation'] = config.Q_ESC
        
        # CONVECTION CALCULATIONS
        def get_h(T_s, LC, is_v):
            T_film = max(min((T_s + temps['Internal_air']) / 2, 500.0), 200.0)
            p_film = physics_models.prop_internal_air(T_film, ambient_pressure)
            
            if config.internal_air_velocity > 0:
                h_forced = physics_models.internal_forced_convection_h(p_film, LC, config.internal_air_velocity)
                h_natural = physics_models.natural_convection_h(p_film, T_s, temps['Internal_air'], LC, is_v)
                h = max(h_forced, h_natural)
            else:
                h = physics_models.natural_convection_h(p_film, T_s, temps['Internal_air'], LC, is_v)
            
            return min(h, 100.0)
        
        # Internal convection for each node
        convection_map = {
            # Battery nodes (top and side surfaces)
            'Batt_BF_Top': (get_h(temps['Batt_BF_Top'], config.LC_batt_horiz, False) * config.A_conv_batt_top + 
                           get_h(temps['Batt_BF_Top'], config.LC_batt_vert, True) * config.A_conv_batt_side),
            'Batt_BF_Bot': (get_h(temps['Batt_BF_Bot'], config.LC_batt_horiz, False) * config.A_conv_batt_top + 
                           get_h(temps['Batt_BF_Bot'], config.LC_batt_vert, True) * config.A_conv_batt_side),
            'Batt_BM_Top': (get_h(temps['Batt_BM_Top'], config.LC_batt_horiz, False) * config.A_conv_batt_top + 
                           get_h(temps['Batt_BM_Top'], config.LC_batt_vert, True) * config.A_conv_batt_side),
            'Batt_BM_Bot': (get_h(temps['Batt_BM_Bot'], config.LC_batt_horiz, False) * config.A_conv_batt_top + 
                           get_h(temps['Batt_BM_Bot'], config.LC_batt_vert, True) * config.A_conv_batt_side),
            'Batt_BR_Top': (get_h(temps['Batt_BR_Top'], config.LC_batt_horiz, False) * config.A_conv_batt_top + 
                           get_h(temps['Batt_BR_Top'], config.LC_batt_vert, True) * config.A_conv_batt_side),
            'Batt_BR_Bot': (get_h(temps['Batt_BR_Bot'], config.LC_batt_horiz, False) * config.A_conv_batt_top + 
                           get_h(temps['Batt_BR_Bot'], config.LC_batt_vert, True) * config.A_conv_batt_side),
            
            # ESC and mount
            'ESC': (get_h(temps['ESC'], config.LC_esc_horiz, False) * config.A_conv_esc_top + 
                   get_h(temps['ESC'], config.LC_esc_vert, True) * config.A_conv_esc_side),
            'ESC_Mount': get_h(temps['ESC_Mount'], config.LC_mount, False) * config.A_mount_conv,
            
            # Bulkheads
            'BH_1': get_h(temps['BH_1'], config.LC_bulkhead, True) * config.A_bulkhead_face * 2,
            'BH_2': get_h(temps['BH_2'], config.LC_bulkhead, True) * config.A_bulkhead_face * 2,
            'BH_3': get_h(temps['BH_3'], config.LC_bulkhead, True) * config.A_bulkhead_face * 2,
            'BH_4': get_h(temps['BH_4'], config.LC_bulkhead, True) * config.A_bulkhead_face * 2,
            
            # Plates
            'plateT': get_h(temps['plateT'], config.LC_plate, False) * config.A_conv_plate,
            'plateM': get_h(temps['plateM'], config.LC_plate, False) * config.A_conv_plateM,
            'plateB': get_h(temps['plateB'], config.LC_plate, False) * config.A_conv_plate,
            
            # Internal shells
            'Top_Shell_Int': get_h(temps['Top_Shell_Int'], config.LC_TS_int, False) * config.A_TS,
            'Bot_Shell_Int': get_h(temps['Bot_Shell_Int'], config.LC_BS_int, False) * config.A_BS,
        }
        
        # Calculate internal convection
        for node in convection_map:
            results[node]['convection_total'] = convection_map[node] * (temps['Internal_air'] - temps[node])
        
        # External convection for shell exteriors
        p_ambient = physics_models.prop_internal_air(ambient_temp, ambient_pressure)
        
        h_ext_top = physics_models.get_external_convection_h(p_ambient, temps['Top_Shell_Ext'], ambient_temp, config.LC_TS_ext)
        results['Top_Shell_Ext']['convection_total'] = h_ext_top * config.A_TS * (ambient_temp - temps['Top_Shell_Ext'])
        
        h_ext_bot = physics_models.get_external_convection_h(p_ambient, temps['Bot_Shell_Ext'], ambient_temp, config.LC_BS_ext)
        results['Bot_Shell_Ext']['convection_total'] = h_ext_bot * config.A_BS * (ambient_temp - temps['Bot_Shell_Ext'])
        
        # CONDUCTION CALCULATIONS
        # ESC to Mount
        q_cond = self.thermal_coeffs['C_cond_ESC_to_Mount'] * (temps['ESC_Mount'] - temps['ESC'])
        results['ESC']['conduction_total'] += q_cond
        results['ESC_Mount']['conduction_total'] += -q_cond
        
        # Mount to BH1
        q_cond = self.thermal_coeffs['C_cond_Mount_to_BH1'] * (temps['BH_1'] - temps['ESC_Mount'])
        results['ESC_Mount']['conduction_total'] += q_cond
        results['BH_1']['conduction_total'] += -q_cond
        
        # Battery to plate connections
        battery_plate_pairs = [
            ('Batt_BF_Top', 'plateT'), ('Batt_BF_Top', 'plateM'),
            ('Batt_BF_Bot', 'plateM'), ('Batt_BF_Bot', 'plateB'),
            ('Batt_BM_Top', 'plateT'), ('Batt_BM_Top', 'plateM'),
            ('Batt_BM_Bot', 'plateM'), ('Batt_BM_Bot', 'plateB'),
            ('Batt_BR_Top', 'plateT'), ('Batt_BR_Top', 'plateM'),
            ('Batt_BR_Bot', 'plateM'), ('Batt_BR_Bot', 'plateB'),
        ]
        
        for batt, plate in battery_plate_pairs:
            q_cond = self.thermal_coeffs['C_cond_Batt_plate'] * (temps[plate] - temps[batt])
            results[batt]['conduction_total'] += q_cond
            results[plate]['conduction_total'] += -q_cond
        
        # Bulkhead to plate connections
        bulkheads = ['BH_1', 'BH_2', 'BH_3', 'BH_4']
        plates = ['plateT', 'plateM', 'plateB']
        
        for bh in bulkheads:
            for plate in plates:
                q_cond = self.thermal_coeffs['C_cond_bh_plate'] * (temps[plate] - temps[bh])
                results[bh]['conduction_total'] += q_cond
                results[plate]['conduction_total'] += -q_cond
        
        # Bulkhead to shell connections
        for bh in bulkheads:
            # To top shell
            q_cond = self.thermal_coeffs['C_cond_BH_TS'] * (temps['Top_Shell_Int'] - temps[bh])
            results[bh]['conduction_total'] += q_cond
            results['Top_Shell_Int']['conduction_total'] += -q_cond
            
            # To bottom shell
            q_cond = self.thermal_coeffs['C_cond_BH_BS'] * (temps['Bot_Shell_Int'] - temps[bh])
            results[bh]['conduction_total'] += q_cond
            results['Bot_Shell_Int']['conduction_total'] += -q_cond
        
        # Shell internal to external
        q_cond = self.thermal_coeffs['C_cond_TS_int_ext'] * (temps['Top_Shell_Ext'] - temps['Top_Shell_Int'])
        results['Top_Shell_Int']['conduction_total'] += q_cond
        results['Top_Shell_Ext']['conduction_total'] += -q_cond
        
        q_cond = self.thermal_coeffs['C_cond_BS_int_ext'] * (temps['Bot_Shell_Ext'] - temps['Bot_Shell_Int'])
        results['Bot_Shell_Int']['conduction_total'] += q_cond
        results['Bot_Shell_Ext']['conduction_total'] += -q_cond
        
        # RADIATION CALCULATIONS
        # Battery to ESC radiation
        batteries = ['Batt_BF_Top', 'Batt_BF_Bot', 'Batt_BM_Top', 'Batt_BM_Bot', 'Batt_BR_Top', 'Batt_BR_Bot']
        for batt in batteries:
            q_rad = self.thermal_coeffs['C_rad_batt_esc'] * (T4s['ESC'] - T4s[batt])
            results[batt]['radiation_total'] += q_rad
            results['ESC']['radiation_total'] += -q_rad
        
        # Battery to shell radiation
        top_batteries = ['Batt_BF_Top', 'Batt_BM_Top', 'Batt_BR_Top']
        bot_batteries = ['Batt_BF_Bot', 'Batt_BM_Bot', 'Batt_BR_Bot']
        
        for batt in top_batteries:
            q_rad = self.thermal_coeffs['C_rad_batt_ts'] * (T4s['Top_Shell_Int'] - T4s[batt])
            results[batt]['radiation_total'] += q_rad
            results['Top_Shell_Int']['radiation_total'] += -q_rad
        
        for batt in bot_batteries:
            q_rad = self.thermal_coeffs['C_rad_batt_bs'] * (T4s['Bot_Shell_Int'] - T4s[batt])
            results[batt]['radiation_total'] += q_rad
            results['Bot_Shell_Int']['radiation_total'] += -q_rad
        
        # ESC to shell radiation
        q_rad = self.thermal_coeffs['C_rad_esc_ts'] * (T4s['Top_Shell_Int'] - T4s['ESC'])
        results['ESC']['radiation_total'] += q_rad
        results['Top_Shell_Int']['radiation_total'] += -q_rad
        
        # Plate to shell radiation
        q_rad = self.thermal_coeffs['C_rad_plate_sh'] * (T4s['Top_Shell_Int'] - T4s['plateT'])
        results['plateT']['radiation_total'] += q_rad
        results['Top_Shell_Int']['radiation_total'] += -q_rad
        
        q_rad = self.thermal_coeffs['C_rad_plate_sh'] * (T4s['Bot_Shell_Int'] - T4s['plateB'])
        results['plateB']['radiation_total'] += q_rad
        results['Bot_Shell_Int']['radiation_total'] += -q_rad
        
        # Mount to shell radiation
        q_rad = self.thermal_coeffs['C_rad_mount_bs'] * (T4s['Bot_Shell_Int'] - T4s['ESC_Mount'])
        results['ESC_Mount']['radiation_total'] += q_rad
        results['Bot_Shell_Int']['radiation_total'] += -q_rad
        
        # Shell to shell radiation
        q_rad = self.thermal_coeffs['C_rad_ts_bs'] * (T4s['Bot_Shell_Int'] - T4s['Top_Shell_Int'])
        results['Top_Shell_Int']['radiation_total'] += q_rad
        results['Bot_Shell_Int']['radiation_total'] += -q_rad
        
        # Calculate net totals
        for node in self.node_labels:
            results[node]['net_total'] = (results[node]['generation'] + 
                                        results[node]['conduction_total'] + 
                                        results[node]['convection_total'] + 
                                        results[node]['radiation_total'])
        
        return results
    
    def create_summary_table(self, results):
        """Create summary table of heat transfer by node and mode"""
        data = []
        
        for node in self.node_labels:
            data.append({
                'Node': node,
                'Generation_W': results[node]['generation'],
                'Conduction_W': results[node]['conduction_total'],
                'Convection_W': results[node]['convection_total'],
                'Radiation_W': results[node]['radiation_total'],
                'Net_Total_W': results[node]['net_total'],
                'Temperature_K': 0  # Will be filled when we have temperatures
            })
        
        return pd.DataFrame(data)

# Example usage
if __name__ == "__main__":
    analyzer = SimpleHeatAnalyzer()
    
    # Example temperatures (would come from simulation)
    example_temps = np.array([315.0] * len(config.labels))  # All nodes at 315K
    example_temps[config.labels.index('ESC')] = 390.0  # Hotter ESC
    example_temps[config.labels.index('Internal_air')] = 318.0  # Warm air
    
    # Analyze heat transfer
    results = analyzer.analyze_heat_transfer(example_temps)
    
    # Create summary table
    summary_df = analyzer.create_summary_table(results)
    summary_df['Temperature_K'] = example_temps
    
    print("=== HEAT TRANSFER ANALYSIS BY NODE ===")
    print("Positive values = Heat GAINED, Negative values = Heat LOST")
    print()
    print(summary_df.round(3))
    
    # Show nodes with highest heat generation/rejection
    print("\n=== HEAT GENERATION NODES ===")
    gen_nodes = summary_df[summary_df['Generation_W'] > 0].sort_values('Generation_W', ascending=False)
    print(gen_nodes[['Node', 'Generation_W', 'Net_Total_W']].round(3))
    
    print("\n=== HIGHEST HEAT REJECTION NODES ===")
    rejection_nodes = summary_df[summary_df['Net_Total_W'] < -1].sort_values('Net_Total_W')
    print(rejection_nodes[['Node', 'Conduction_W', 'Convection_W', 'Radiation_W', 'Net_Total_W']].round(3))
    
    # Save results
    summary_df.to_csv('simple_heat_analysis.csv', index=False)
    print(f"\nResults saved to: simple_heat_analysis.csv")
