# heat_transfer_mapper.py
# Detailed Heat Transfer Mapping for 20-Node UAV Thermal Model
# Shows exact heat flow quantities and directions for each node by transfer mode

import sys
import os
import numpy as np
import pandas as pd
from collections import defaultdict

# Add the code_vM_1 directory to the path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'code_vM_1'))
import config
import physics_models
import environment_model

class HeatTransferMapper:
    """
    Creates detailed heat transfer maps showing exact quantities and directions
    of heat flow for each node across conduction, convection, and radiation modes
    """
    
    def __init__(self):
        self.node_labels = config.labels
        self.thermal_coefficients = {}
        
        # Initialize environment
        environment_model.init()
        self._calculate_thermal_coefficients()
    
    def _calculate_thermal_coefficients(self):
        """Calculate all thermal coefficients used in the simulation"""
        
        # Conduction coefficients
        self.thermal_coefficients.update({
            'C_cond_ESC_to_Mount': config.k_mount * config.A_contact_ESC_Mount / config.L_path_ESC_Mount,
            'C_cond_Mount_to_BH1': config.k_bulkhead * config.A_contact_Mount_BH1 / config.t_bulkhead,
            'C_cond_Batt_plate': config.k_plate * config.A_contact_batt_plate / config.t_plate,
            'C_cond_bh_plate': 2 * config.k_bulkhead * config.A_contact_bh_plate / config.L_bh_plate_cond,
            'C_cond_BH_TS': config.k_cfrp * config.A_contact_BH_Shell / config.t_cfrp,
            'C_cond_BH_BS': config.k_cfrp * config.A_contact_BH_Shell / config.t_cfrp,
            'C_cond_TS_int_ext': config.k_cfrp * config.A_TS / config.t_cfrp,
            'C_cond_BS_int_ext': config.k_cfrp * config.A_BS / config.t_cfrp,
        })
        
        # Radiation coefficients
        self.thermal_coefficients.update({
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
        })
    
    def create_detailed_heat_transfer_map(self, temperatures, ambient_temp, ambient_pressure):
        """
        Create comprehensive heat transfer map showing all heat flows by mode and direction
        """
        # Create temperature dictionary and T^4 values
        temps = {label: temperatures[i] for i, label in enumerate(self.node_labels)}
        T4s = {k: physics_models.T_power4(v) for k, v in temps.items()}
        
        # Air properties
        p_air = physics_models.prop_internal_air(temps['Internal_air'], ambient_pressure)
        
        # Convection heat transfer function
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
        
        # Initialize heat transfer map
        heat_map = {}
        for node in self.node_labels:
            heat_map[node] = {
                'generation': 0.0,
                'conduction_flows': {},
                'convection_flow': 0.0,
                'radiation_flows': {},
                'net_conduction': 0.0,
                'net_radiation': 0.0,
                'total_net': 0.0
            }
        
        # HEAT GENERATION
        for node in self.node_labels:
            if 'Batt' in node:
                heat_map[node]['generation'] = config.Q_batt_zone
            elif node == 'ESC':
                heat_map[node]['generation'] = config.Q_ESC
        
        # CONDUCTION HEAT FLOWS
        conduction_pairs = [
            # ESC connections
            ('ESC', 'ESC_Mount', self.thermal_coefficients['C_cond_ESC_to_Mount']),
            ('ESC_Mount', 'BH_1', self.thermal_coefficients['C_cond_Mount_to_BH1']),
            
            # Battery to plate connections
            ('Batt_BF_Top', 'plateT', self.thermal_coefficients['C_cond_Batt_plate']),
            ('Batt_BF_Top', 'plateM', self.thermal_coefficients['C_cond_Batt_plate']),
            ('Batt_BF_Bot', 'plateM', self.thermal_coefficients['C_cond_Batt_plate']),
            ('Batt_BF_Bot', 'plateB', self.thermal_coefficients['C_cond_Batt_plate']),
            ('Batt_BM_Top', 'plateT', self.thermal_coefficients['C_cond_Batt_plate']),
            ('Batt_BM_Top', 'plateM', self.thermal_coefficients['C_cond_Batt_plate']),
            ('Batt_BM_Bot', 'plateM', self.thermal_coefficients['C_cond_Batt_plate']),
            ('Batt_BM_Bot', 'plateB', self.thermal_coefficients['C_cond_Batt_plate']),
            ('Batt_BR_Top', 'plateT', self.thermal_coefficients['C_cond_Batt_plate']),
            ('Batt_BR_Top', 'plateM', self.thermal_coefficients['C_cond_Batt_plate']),
            ('Batt_BR_Bot', 'plateM', self.thermal_coefficients['C_cond_Batt_plate']),
            ('Batt_BR_Bot', 'plateB', self.thermal_coefficients['C_cond_Batt_plate']),
            
            # Bulkhead to plate connections
            ('BH_1', 'plateT', self.thermal_coefficients['C_cond_bh_plate']),
            ('BH_1', 'plateM', self.thermal_coefficients['C_cond_bh_plate']),
            ('BH_1', 'plateB', self.thermal_coefficients['C_cond_bh_plate']),
            ('BH_2', 'plateT', self.thermal_coefficients['C_cond_bh_plate']),
            ('BH_2', 'plateM', self.thermal_coefficients['C_cond_bh_plate']),
            ('BH_2', 'plateB', self.thermal_coefficients['C_cond_bh_plate']),
            ('BH_3', 'plateT', self.thermal_coefficients['C_cond_bh_plate']),
            ('BH_3', 'plateM', self.thermal_coefficients['C_cond_bh_plate']),
            ('BH_3', 'plateB', self.thermal_coefficients['C_cond_bh_plate']),
            ('BH_4', 'plateT', self.thermal_coefficients['C_cond_bh_plate']),
            ('BH_4', 'plateM', self.thermal_coefficients['C_cond_bh_plate']),
            ('BH_4', 'plateB', self.thermal_coefficients['C_cond_bh_plate']),
            
            # Bulkhead to shell connections
            ('BH_1', 'Top_Shell_Int', self.thermal_coefficients['C_cond_BH_TS']),
            ('BH_1', 'Bot_Shell_Int', self.thermal_coefficients['C_cond_BH_BS']),
            ('BH_2', 'Top_Shell_Int', self.thermal_coefficients['C_cond_BH_TS']),
            ('BH_2', 'Bot_Shell_Int', self.thermal_coefficients['C_cond_BH_BS']),
            ('BH_3', 'Top_Shell_Int', self.thermal_coefficients['C_cond_BH_TS']),
            ('BH_3', 'Bot_Shell_Int', self.thermal_coefficients['C_cond_BH_BS']),
            ('BH_4', 'Top_Shell_Int', self.thermal_coefficients['C_cond_BH_TS']),
            ('BH_4', 'Bot_Shell_Int', self.thermal_coefficients['C_cond_BH_BS']),
            
            # Shell internal to external
            ('Top_Shell_Int', 'Top_Shell_Ext', self.thermal_coefficients['C_cond_TS_int_ext']),
            ('Bot_Shell_Int', 'Bot_Shell_Ext', self.thermal_coefficients['C_cond_BS_int_ext']),
        ]
        
        for node1, node2, coeff in conduction_pairs:
            q_flow = coeff * (temps[node2] - temps[node1])
            heat_map[node1]['conduction_flows'][node2] = q_flow
            heat_map[node2]['conduction_flows'][node1] = -q_flow
        
        # CONVECTION HEAT FLOWS
        convection_specs = [
            # Battery nodes
            ('Batt_BF_Top', config.LC_batt_horiz, False, config.A_conv_batt_top, config.LC_batt_vert, True, config.A_conv_batt_side),
            ('Batt_BF_Bot', config.LC_batt_horiz, False, config.A_conv_batt_top, config.LC_batt_vert, True, config.A_conv_batt_side),
            ('Batt_BM_Top', config.LC_batt_horiz, False, config.A_conv_batt_top, config.LC_batt_vert, True, config.A_conv_batt_side),
            ('Batt_BM_Bot', config.LC_batt_horiz, False, config.A_conv_batt_top, config.LC_batt_vert, True, config.A_conv_batt_side),
            ('Batt_BR_Top', config.LC_batt_horiz, False, config.A_conv_batt_top, config.LC_batt_vert, True, config.A_conv_batt_side),
            ('Batt_BR_Bot', config.LC_batt_horiz, False, config.A_conv_batt_top, config.LC_batt_vert, True, config.A_conv_batt_side),
            
            # ESC and mount
            ('ESC', config.LC_esc_horiz, False, config.A_conv_esc_top, config.LC_esc_vert, True, config.A_conv_esc_side),
            ('ESC_Mount', config.LC_mount, False, config.A_mount_conv, None, None, 0),
            
            # Bulkheads
            ('BH_1', config.LC_bulkhead, True, config.A_bulkhead_face * 2, None, None, 0),
            ('BH_2', config.LC_bulkhead, True, config.A_bulkhead_face * 2, None, None, 0),
            ('BH_3', config.LC_bulkhead, True, config.A_bulkhead_face * 2, None, None, 0),
            ('BH_4', config.LC_bulkhead, True, config.A_bulkhead_face * 2, None, None, 0),
            
            # Plates
            ('plateT', config.LC_plate, False, config.A_conv_plate, None, None, 0),
            ('plateM', config.LC_plate, False, config.A_conv_plateM, None, None, 0),
            ('plateB', config.LC_plate, False, config.A_conv_plate, None, None, 0),
            
            # Internal shells
            ('Top_Shell_Int', config.LC_TS_int, False, config.A_TS, None, None, 0),
            ('Bot_Shell_Int', config.LC_BS_int, False, config.A_BS, None, None, 0),
        ]
        
        for spec in convection_specs:
            node = spec[0]
            if len(spec) == 7 and spec[4] is not None:  # Two surface types
                h_total = (get_h(temps[node], spec[1], spec[2]) * spec[3] + 
                          get_h(temps[node], spec[4], spec[5]) * spec[6])
            else:  # Single surface
                h_total = get_h(temps[node], spec[1], spec[2]) * spec[3]
            
            heat_map[node]['convection_flow'] = h_total * (temps['Internal_air'] - temps[node])
        
        # External convection for shell exteriors
        p_ambient = physics_models.prop_internal_air(ambient_temp, ambient_pressure)
        
        h_ext_top = physics_models.get_external_convection_h(p_ambient, temps['Top_Shell_Ext'], ambient_temp, config.LC_TS_ext)
        heat_map['Top_Shell_Ext']['convection_flow'] = h_ext_top * config.A_TS * (ambient_temp - temps['Top_Shell_Ext'])
        
        h_ext_bot = physics_models.get_external_convection_h(p_ambient, temps['Bot_Shell_Ext'], ambient_temp, config.LC_BS_ext)
        heat_map['Bot_Shell_Ext']['convection_flow'] = h_ext_bot * config.A_BS * (ambient_temp - temps['Bot_Shell_Ext'])
        
        # RADIATION HEAT FLOWS
        radiation_pairs = [
            # Battery to battery radiation
            ('Batt_BF_Top', 'Batt_BM_Top', self.thermal_coefficients['C_rad_batt_batt']),
            ('Batt_BF_Bot', 'Batt_BM_Bot', self.thermal_coefficients['C_rad_batt_batt']),
            ('Batt_BM_Top', 'Batt_BR_Top', self.thermal_coefficients['C_rad_batt_batt']),
            ('Batt_BM_Bot', 'Batt_BR_Bot', self.thermal_coefficients['C_rad_batt_batt']),
            
            # Battery to ESC radiation
            ('Batt_BF_Top', 'ESC', self.thermal_coefficients['C_rad_batt_esc']),
            ('Batt_BF_Bot', 'ESC', self.thermal_coefficients['C_rad_batt_esc']),
            ('Batt_BM_Top', 'ESC', self.thermal_coefficients['C_rad_batt_esc']),
            ('Batt_BM_Bot', 'ESC', self.thermal_coefficients['C_rad_batt_esc']),
            ('Batt_BR_Top', 'ESC', self.thermal_coefficients['C_rad_batt_esc']),
            ('Batt_BR_Bot', 'ESC', self.thermal_coefficients['C_rad_batt_esc']),
            
            # Battery to shell radiation
            ('Batt_BF_Top', 'Top_Shell_Int', self.thermal_coefficients['C_rad_batt_ts']),
            ('Batt_BM_Top', 'Top_Shell_Int', self.thermal_coefficients['C_rad_batt_ts']),
            ('Batt_BR_Top', 'Top_Shell_Int', self.thermal_coefficients['C_rad_batt_ts']),
            ('Batt_BF_Bot', 'Bot_Shell_Int', self.thermal_coefficients['C_rad_batt_bs']),
            ('Batt_BM_Bot', 'Bot_Shell_Int', self.thermal_coefficients['C_rad_batt_bs']),
            ('Batt_BR_Bot', 'Bot_Shell_Int', self.thermal_coefficients['C_rad_batt_bs']),
            
            # ESC radiation
            ('ESC', 'Top_Shell_Int', self.thermal_coefficients['C_rad_esc_ts']),
            
            # Plate radiation
            ('plateT', 'Top_Shell_Int', self.thermal_coefficients['C_rad_plate_sh']),
            ('plateB', 'Bot_Shell_Int', self.thermal_coefficients['C_rad_plate_sh']),
            
            # Mount radiation
            ('ESC_Mount', 'Bot_Shell_Int', self.thermal_coefficients['C_rad_mount_bs']),
            
            # Shell to shell radiation
            ('Top_Shell_Int', 'Bot_Shell_Int', self.thermal_coefficients['C_rad_ts_bs']),
        ]
        
        for node1, node2, coeff in radiation_pairs:
            q_flow = coeff * (T4s[node2] - T4s[node1])
            heat_map[node1]['radiation_flows'][node2] = q_flow
            heat_map[node2]['radiation_flows'][node1] = -q_flow
        
        # Calculate net flows for each node
        for node in self.node_labels:
            # Net conduction
            heat_map[node]['net_conduction'] = sum(heat_map[node]['conduction_flows'].values())
            
            # Net radiation
            heat_map[node]['net_radiation'] = sum(heat_map[node]['radiation_flows'].values())
            
            # Total net flow
            heat_map[node]['total_net'] = (heat_map[node]['generation'] + 
                                         heat_map[node]['net_conduction'] + 
                                         heat_map[node]['convection_flow'] + 
                                         heat_map[node]['net_radiation'])
        
        return heat_map
    
    def create_heat_transfer_summary_table(self, heat_map):
        """
        Create comprehensive summary table of heat transfers
        """
        summary_data = []
        
        for node in self.node_labels:
            data = heat_map[node]
            
            # Find dominant heat transfer modes
            modes = {
                'Generation': abs(data['generation']),
                'Conduction': abs(data['net_conduction']),
                'Convection': abs(data['convection_flow']),
                'Radiation': abs(data['net_radiation'])
            }
            
            dominant_mode = max(modes, key=modes.get) if max(modes.values()) > 0.01 else 'None'
            
            summary_data.append({
                'Node': node,
                'Generation_W': data['generation'],
                'Net_Conduction_W': data['net_conduction'],
                'Convection_W': data['convection_flow'],
                'Net_Radiation_W': data['net_radiation'],
                'Total_Net_W': data['total_net'],
                'Dominant_Mode': dominant_mode,
                'Heat_Rejection_W': abs(data['net_conduction']) + abs(data['convection_flow']) + abs(data['net_radiation'])
            })
        
        return pd.DataFrame(summary_data)
    
    def create_detailed_flow_tables(self, heat_map):
        """
        Create detailed tables showing individual heat flow paths
        """
        # Conduction flows table
        conduction_data = []
        for node in self.node_labels:
            for target, flow in heat_map[node]['conduction_flows'].items():
                if flow != 0:
                    conduction_data.append({
                        'From_Node': node,
                        'To_Node': target,
                        'Heat_Flow_W': flow,
                        'Direction': 'TO' if flow > 0 else 'FROM',
                        'Magnitude_W': abs(flow)
                    })
        
        conduction_df = pd.DataFrame(conduction_data).sort_values('Magnitude_W', ascending=False)
        
        # Radiation flows table
        radiation_data = []
        for node in self.node_labels:
            for target, flow in heat_map[node]['radiation_flows'].items():
                if flow != 0:
                    radiation_data.append({
                        'From_Node': node,
                        'To_Node': target,
                        'Heat_Flow_W': flow,
                        'Direction': 'TO' if flow > 0 else 'FROM',
                        'Magnitude_W': abs(flow)
                    })
        
        radiation_df = pd.DataFrame(radiation_data).sort_values('Magnitude_W', ascending=False)
        
        # Convection flows table
        convection_data = []
        for node in self.node_labels:
            flow = heat_map[node]['convection_flow']
            if abs(flow) > 0.01:
                if node in ['Top_Shell_Ext', 'Bot_Shell_Ext']:
                    target = 'Ambient'
                else:
                    target = 'Internal_Air'
                
                convection_data.append({
                    'Node': node,
                    'Target': target,
                    'Heat_Flow_W': flow,
                    'Direction': 'TO' if flow > 0 else 'FROM',
                    'Magnitude_W': abs(flow)
                })
        
        convection_df = pd.DataFrame(convection_data).sort_values('Magnitude_W', ascending=False)
        
        return conduction_df, radiation_df, convection_df

# Example usage
if __name__ == "__main__":
    mapper = HeatTransferMapper()
    
    # Example temperatures (would come from simulation)
    example_temps = np.array([315.0] * len(config.labels))  # Kelvin
    example_temps[config.labels.index('ESC')] = 390.0  # Hotter ESC
    example_temps[config.labels.index('Internal_air')] = 318.0  # Warm air
    
    # Create heat transfer map
    heat_map = mapper.create_detailed_heat_transfer_map(
        example_temps, 
        ambient_temp=287.3, 
        ambient_pressure=99832.0
    )
    
    # Create summary table
    summary_df = mapper.create_heat_transfer_summary_table(heat_map)
    print("=== HEAT TRANSFER SUMMARY BY NODE ===")
    print(summary_df.round(3))
    
    # Create detailed flow tables
    conduction_df, radiation_df, convection_df = mapper.create_detailed_flow_tables(heat_map)
    
    print("\n=== TOP 10 CONDUCTION HEAT FLOWS ===")
    print(conduction_df.head(10).round(3))
    
    print("\n=== TOP 10 RADIATION HEAT FLOWS ===")
    print(radiation_df.head(10).round(3))
    
    print("\n=== CONVECTION HEAT FLOWS ===")
    print(convection_df.round(3))
    
    # Save detailed results
    summary_df.to_csv('heat_transfer_summary.csv', index=False)
    conduction_df.to_csv('conduction_flows.csv', index=False)
    radiation_df.to_csv('radiation_flows.csv', index=False)
    convection_df.to_csv('convection_flows.csv', index=False)
    
    print(f"\nResults saved to CSV files:")
    print("- heat_transfer_summary.csv")
    print("- conduction_flows.csv") 
    print("- radiation_flows.csv")
    print("- convection_flows.csv")
