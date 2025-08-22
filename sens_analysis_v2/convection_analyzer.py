# convection_analyzer.py
# Detailed Convection Heat Transfer Analysis for 20-Node UAV Thermal Model

import sys
import os
import numpy as np
import pandas as pd

# Add the code_vM_1 directory to the path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'code_vM_1'))
import config
import physics_models
import environment_model

class ConvectionAnalyzer:
    """
    Analyzes all convection heat transfer showing sources and sinks for each node
    """
    
    def __init__(self):
        self.node_labels = config.labels
        environment_model.init()
    
    def analyze_convection_flows(self, temperatures, ambient_temp=287.3, ambient_pressure=99832.0):
        """
        Analyze all convection heat flows and organize by node sources/sinks
        """
        temps = {label: temperatures[i] for i, label in enumerate(self.node_labels)}
        
        # Initialize node convection tracking
        node_convection = {}
        for node in self.node_labels:
            node_convection[node] = {
                'target': None,
                'heat_flow': 0.0,
                'h_coefficient': 0.0,
                'area': 0.0,
                'temp_difference': 0.0,
                'flow_type': 'None'
            }
        
        # Convection heat transfer coefficient function
        def get_h(T_s, LC, is_v):
            T_film = max(min((T_s + temps['Internal_air']) / 2, 500.0), 200.0)
            p_film = physics_models.prop_internal_air(T_film, ambient_pressure)
            
            if config.internal_air_velocity > 0:
                h_forced = physics_models.internal_forced_convection_h(p_film, LC, config.internal_air_velocity)
                h_natural = physics_models.natural_convection_h(p_film, T_s, temps['Internal_air'], LC, is_v)
                h = max(h_forced, h_natural)
                flow_type = 'Forced' if h_forced > h_natural else 'Natural'
            else:
                h = physics_models.natural_convection_h(p_film, T_s, temps['Internal_air'], LC, is_v)
                flow_type = 'Natural'
            
            return min(h, 100.0), flow_type
        
        # Internal convection calculations
        internal_convection_specs = [
            # Battery nodes (combined top and side surfaces)
            ('Batt_BF_Top', 'Internal_air', 
             lambda: self._calculate_battery_convection(temps['Batt_BF_Top'], get_h)),
            ('Batt_BF_Bot', 'Internal_air', 
             lambda: self._calculate_battery_convection(temps['Batt_BF_Bot'], get_h)),
            ('Batt_BM_Top', 'Internal_air', 
             lambda: self._calculate_battery_convection(temps['Batt_BM_Top'], get_h)),
            ('Batt_BM_Bot', 'Internal_air', 
             lambda: self._calculate_battery_convection(temps['Batt_BM_Bot'], get_h)),
            ('Batt_BR_Top', 'Internal_air', 
             lambda: self._calculate_battery_convection(temps['Batt_BR_Top'], get_h)),
            ('Batt_BR_Bot', 'Internal_air', 
             lambda: self._calculate_battery_convection(temps['Batt_BR_Bot'], get_h)),
            
            # ESC (combined top and side surfaces)
            ('ESC', 'Internal_air', 
             lambda: self._calculate_esc_convection(temps['ESC'], get_h)),
            
            # ESC Mount
            ('ESC_Mount', 'Internal_air', 
             lambda: self._calculate_single_surface_convection(temps['ESC_Mount'], config.LC_mount, 
                                                              False, config.A_mount_conv, get_h)),
            
            # Bulkheads (both faces)
            ('BH_1', 'Internal_air', 
             lambda: self._calculate_single_surface_convection(temps['BH_1'], config.LC_bulkhead, 
                                                              True, config.A_bulkhead_face * 2, get_h)),
            ('BH_2', 'Internal_air', 
             lambda: self._calculate_single_surface_convection(temps['BH_2'], config.LC_bulkhead, 
                                                              True, config.A_bulkhead_face * 2, get_h)),
            ('BH_3', 'Internal_air', 
             lambda: self._calculate_single_surface_convection(temps['BH_3'], config.LC_bulkhead, 
                                                              True, config.A_bulkhead_face * 2, get_h)),
            ('BH_4', 'Internal_air', 
             lambda: self._calculate_single_surface_convection(temps['BH_4'], config.LC_bulkhead, 
                                                              True, config.A_bulkhead_face * 2, get_h)),
            
            # Plates
            ('plateT', 'Internal_air', 
             lambda: self._calculate_single_surface_convection(temps['plateT'], config.LC_plate, 
                                                              False, config.A_conv_plate, get_h)),
            ('plateM', 'Internal_air', 
             lambda: self._calculate_single_surface_convection(temps['plateM'], config.LC_plate, 
                                                              False, config.A_conv_plateM, get_h)),
            ('plateB', 'Internal_air', 
             lambda: self._calculate_single_surface_convection(temps['plateB'], config.LC_plate, 
                                                              False, config.A_conv_plate, get_h)),
            
            # Internal shells
            ('Top_Shell_Int', 'Internal_air', 
             lambda: self._calculate_single_surface_convection(temps['Top_Shell_Int'], config.LC_TS_int, 
                                                              False, config.A_TS, get_h)),
            ('Bot_Shell_Int', 'Internal_air', 
             lambda: self._calculate_single_surface_convection(temps['Bot_Shell_Int'], config.LC_BS_int, 
                                                              False, config.A_BS, get_h)),
        ]
        
        # Calculate internal convection
        for node, target, calc_func in internal_convection_specs:
            h_total, area_total, flow_type = calc_func()
            heat_flow = h_total * (temps[target] - temps[node])
            temp_diff = temps[target] - temps[node]
            
            node_convection[node] = {
                'target': target,
                'heat_flow': heat_flow,
                'h_coefficient': h_total / area_total if area_total > 0 else 0,
                'area': area_total,
                'temp_difference': temp_diff,
                'flow_type': flow_type
            }
        
        # External convection for shell exteriors
        p_ambient = physics_models.prop_internal_air(ambient_temp, ambient_pressure)
        
        # Top shell external
        h_ext_top = physics_models.get_external_convection_h(p_ambient, temps['Top_Shell_Ext'], 
                                                           ambient_temp, config.LC_TS_ext)
        heat_flow_top = h_ext_top * config.A_TS * (ambient_temp - temps['Top_Shell_Ext'])
        
        node_convection['Top_Shell_Ext'] = {
            'target': 'Ambient',
            'heat_flow': heat_flow_top,
            'h_coefficient': h_ext_top,
            'area': config.A_TS,
            'temp_difference': ambient_temp - temps['Top_Shell_Ext'],
            'flow_type': 'External'
        }
        
        # Bottom shell external
        h_ext_bot = physics_models.get_external_convection_h(p_ambient, temps['Bot_Shell_Ext'], 
                                                           ambient_temp, config.LC_BS_ext)
        heat_flow_bot = h_ext_bot * config.A_BS * (ambient_temp - temps['Bot_Shell_Ext'])
        
        node_convection['Bot_Shell_Ext'] = {
            'target': 'Ambient',
            'heat_flow': heat_flow_bot,
            'h_coefficient': h_ext_bot,
            'area': config.A_BS,
            'temp_difference': ambient_temp - temps['Bot_Shell_Ext'],
            'flow_type': 'External'
        }
        
        return node_convection
    
    def _calculate_battery_convection(self, T_node, get_h_func):
        """Calculate combined convection for battery (top + side surfaces)"""
        h_top, flow_type_top = get_h_func(T_node, config.LC_batt_horiz, False)
        h_side, flow_type_side = get_h_func(T_node, config.LC_batt_vert, True)
        
        h_total = h_top * config.A_conv_batt_top + h_side * config.A_conv_batt_side
        area_total = config.A_conv_batt_top + config.A_conv_batt_side
        flow_type = flow_type_top  # Use top surface flow type as representative
        
        return h_total, area_total, flow_type
    
    def _calculate_esc_convection(self, T_node, get_h_func):
        """Calculate combined convection for ESC (top + side surfaces)"""
        h_top, flow_type_top = get_h_func(T_node, config.LC_esc_horiz, False)
        h_side, flow_type_side = get_h_func(T_node, config.LC_esc_vert, True)
        
        h_total = h_top * config.A_conv_esc_top + h_side * config.A_conv_esc_side
        area_total = config.A_conv_esc_top + config.A_conv_esc_side
        flow_type = flow_type_top  # Use top surface flow type as representative
        
        return h_total, area_total, flow_type
    
    def _calculate_single_surface_convection(self, T_node, LC, is_vertical, area, get_h_func):
        """Calculate convection for single surface"""
        h, flow_type = get_h_func(T_node, LC, is_vertical)
        h_total = h * area
        
        return h_total, area, flow_type
    
    def create_convection_summary(self, node_convection):
        """Create summary table of convection for all nodes"""
        summary_data = []
        
        for node in self.node_labels:
            data = node_convection[node]
            
            # Determine heat gain/loss
            if data['heat_flow'] > 0:
                gain_loss = 'Gain'
            elif data['heat_flow'] < 0:
                gain_loss = 'Loss'
            else:
                gain_loss = 'None'
            
            summary_data.append({
                'Node': node,
                'Target': data['target'],
                'Heat_Flow_W': data['heat_flow'],
                'Gain_Loss': gain_loss,
                'h_Coefficient': data['h_coefficient'],
                'Area_m2': data['area'],
                'Temp_Difference_K': data['temp_difference'],
                'Flow_Type': data['flow_type']
            })
        
        return pd.DataFrame(summary_data)
    
    def create_detailed_convection_reports(self, node_convection):
        """Create detailed reports for each node showing convection details"""
        reports = {}
        
        for node in self.node_labels:
            data = node_convection[node]
            
            if abs(data['heat_flow']) < 0.001:  # Skip negligible flows
                continue
            
            report = f"\n=== CONVECTION ANALYSIS: {node} ===\n"
            
            if data['heat_flow'] > 0:
                report += f"HEAT GAIN from {data['target']}:\n"
                report += f"  Heat Flow:        +{data['heat_flow']:8.3f} W\n"
            else:
                report += f"HEAT LOSS to {data['target']}:\n"
                report += f"  Heat Flow:        {data['heat_flow']:8.3f} W\n"
            
            report += f"  h Coefficient:     {data['h_coefficient']:8.3f} W/m²K\n"
            report += f"  Surface Area:      {data['area']:8.6f} m²\n"
            report += f"  Temp Difference:   {data['temp_difference']:8.3f} K\n"
            report += f"  Flow Type:         {data['flow_type']}\n"
            
            reports[node] = report
        
        return reports

# Example usage
if __name__ == "__main__":
    analyzer = ConvectionAnalyzer()
    
    # Example temperatures with more realistic differences
    example_temps = np.array([315.0] * len(config.labels))
    example_temps[config.labels.index('ESC')] = 390.0
    example_temps[config.labels.index('Internal_air')] = 318.0
    example_temps[config.labels.index('Top_Shell_Ext')] = 310.0
    example_temps[config.labels.index('Bot_Shell_Ext')] = 310.0
    
    # Analyze convection
    node_convection = analyzer.analyze_convection_flows(example_temps)
    
    # Create summary
    summary_df = analyzer.create_convection_summary(node_convection)
    print("=== CONVECTION SUMMARY BY NODE ===")
    print(summary_df.round(3))
    
    # Show detailed reports for nodes with significant convection
    reports = analyzer.create_detailed_convection_reports(node_convection)
    
    # Show reports for key nodes
    key_nodes = ['ESC', 'Top_Shell_Ext', 'Bot_Shell_Ext', 'Batt_BF_Top']
    for node in key_nodes:
        if node in reports:
            print(reports[node])
    
    # Show nodes with highest convection
    print("\n=== HIGHEST CONVECTION HEAT FLOWS ===")
    significant_flows = summary_df[abs(summary_df['Heat_Flow_W']) > 1.0].sort_values('Heat_Flow_W', key=abs, ascending=False)
    print(significant_flows[['Node', 'Target', 'Heat_Flow_W', 'Flow_Type']].round(3))
    
    # Save results
    summary_df.to_csv('convection_summary.csv', index=False)
    print(f"\nResults saved to: convection_summary.csv")
