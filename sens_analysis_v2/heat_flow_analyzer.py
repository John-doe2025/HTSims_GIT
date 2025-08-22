# heat_flow_analyzer.py
# Detailed Heat Flow Analysis for 20-Node UAV Thermal Model

import sys
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from collections import defaultdict

try:
    import seaborn as sns
    HAS_SEABORN = True
except ImportError:
    HAS_SEABORN = False

# Add the code_vM_1 directory to the path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'code_vM_1'))
import config
import physics_models
import environment_model

class HeatFlowAnalyzer:
    """
    Analyzes detailed heat flow paths and contributions for each node
    in the 20-node thermal model
    """
    
    def __init__(self):
        self.node_labels = config.labels
        self.heat_flows = {}
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
    
    def analyze_heat_flows_at_state(self, temperatures, ambient_temp, ambient_pressure):
        """
        Analyze detailed heat flows for all nodes at a given temperature state
        """
        # Create temperature dictionary and T^4 values
        temps = {label: temperatures[i] for i, label in enumerate(self.node_labels)}
        T4s = {k: physics_models.T_power4(v) for k, v in temps.items()}
        
        # Air properties
        p_air = physics_models.prop_internal_air(temps['Internal_air'], ambient_pressure)
        
        # Initialize heat flow tracking
        node_flows = {node: {
            'generation': 0.0,
            'conduction': {},
            'convection': 0.0,
            'radiation': {},
            'net_flow': 0.0
        } for node in self.node_labels}
        
        # Heat generation
        for node in self.node_labels:
            if 'Batt' in node:
                node_flows[node]['generation'] = config.Q_batt_zone
            elif node == 'ESC':
                node_flows[node]['generation'] = config.Q_ESC
        
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
        
        # Calculate all heat flows systematically
        
        # BATTERY NODES - Conduction to plates
        battery_nodes = ['Batt_BF_Top', 'Batt_BF_Bot', 'Batt_BM_Top', 'Batt_BM_Bot', 'Batt_BR_Top', 'Batt_BR_Bot']
        
        for batt_node in battery_nodes:
            # Convection to internal air
            if 'Top' in batt_node or 'Bot' in batt_node:
                h_total = (get_h(temps[batt_node], config.LC_batt_horiz, False) * config.A_conv_batt_top + 
                          get_h(temps[batt_node], config.LC_batt_vert, True) * config.A_conv_batt_side)
                node_flows[batt_node]['convection'] = h_total * (temps['Internal_air'] - temps[batt_node])
            
            # Conduction to plates
            if 'Top' in batt_node:
                node_flows[batt_node]['conduction']['plateT'] = self.thermal_coefficients['C_cond_Batt_plate'] * (temps['plateT'] - temps[batt_node])
                node_flows[batt_node]['conduction']['plateM'] = self.thermal_coefficients['C_cond_Batt_plate'] * (temps['plateM'] - temps[batt_node])
            elif 'Bot' in batt_node:
                node_flows[batt_node]['conduction']['plateM'] = self.thermal_coefficients['C_cond_Batt_plate'] * (temps['plateM'] - temps[batt_node])
                node_flows[batt_node]['conduction']['plateB'] = self.thermal_coefficients['C_cond_Batt_plate'] * (temps['plateB'] - temps[batt_node])
            
            # Radiation to other components
            node_flows[batt_node]['radiation']['ESC'] = self.thermal_coefficients['C_rad_batt_esc'] * (T4s['ESC'] - T4s[batt_node])
            node_flows[batt_node]['radiation']['Top_Shell_Int'] = self.thermal_coefficients['C_rad_batt_ts'] * (T4s['Top_Shell_Int'] - T4s[batt_node])
            if 'Bot' in batt_node:
                node_flows[batt_node]['radiation']['Bot_Shell_Int'] = self.thermal_coefficients['C_rad_batt_ts'] * (T4s['Bot_Shell_Int'] - T4s[batt_node])
        
        # ESC NODE
        node_flows['ESC']['convection'] = (get_h(temps['ESC'], config.LC_esc_horiz, False) * config.A_conv_esc_top + 
                                          get_h(temps['ESC'], config.LC_esc_vert, True) * config.A_conv_esc_side) * (temps['Internal_air'] - temps['ESC'])
        node_flows['ESC']['conduction']['ESC_Mount'] = self.thermal_coefficients['C_cond_ESC_to_Mount'] * (temps['ESC_Mount'] - temps['ESC'])
        node_flows['ESC']['radiation']['Top_Shell_Int'] = self.thermal_coefficients['C_rad_esc_ts'] * (T4s['Top_Shell_Int'] - T4s['ESC'])
        
        # ESC MOUNT
        node_flows['ESC_Mount']['conduction']['BH_1'] = self.thermal_coefficients['C_cond_Mount_to_BH1'] * (temps['BH_1'] - temps['ESC_Mount'])
        node_flows['ESC_Mount']['conduction']['ESC'] = self.thermal_coefficients['C_cond_ESC_to_Mount'] * (temps['ESC'] - temps['ESC_Mount'])
        node_flows['ESC_Mount']['convection'] = get_h(temps['ESC_Mount'], config.LC_mount, False) * config.A_mount_conv * (temps['Internal_air'] - temps['ESC_Mount'])
        node_flows['ESC_Mount']['radiation']['Bot_Shell_Int'] = self.thermal_coefficients['C_rad_mount_bs'] * (T4s['Bot_Shell_Int'] - T4s['ESC_Mount'])
        
        # BULKHEAD NODES
        bulkhead_nodes = ['BH_1', 'BH_2', 'BH_3', 'BH_4']
        for bh_node in bulkhead_nodes:
            node_flows[bh_node]['convection'] = get_h(temps[bh_node], config.LC_bulkhead, True) * config.A_bulkhead_face * 2 * (temps['Internal_air'] - temps[bh_node])
            node_flows[bh_node]['conduction']['Top_Shell_Int'] = self.thermal_coefficients['C_cond_BH_TS'] * (temps['Top_Shell_Int'] - temps[bh_node])
            node_flows[bh_node]['conduction']['Bot_Shell_Int'] = self.thermal_coefficients['C_cond_BH_BS'] * (temps['Bot_Shell_Int'] - temps[bh_node])
            
            # Plate connections
            node_flows[bh_node]['conduction']['plateT'] = self.thermal_coefficients['C_cond_bh_plate'] * (temps['plateT'] - temps[bh_node])
            node_flows[bh_node]['conduction']['plateM'] = self.thermal_coefficients['C_cond_bh_plate'] * (temps['plateM'] - temps[bh_node])
            node_flows[bh_node]['conduction']['plateB'] = self.thermal_coefficients['C_cond_bh_plate'] * (temps['plateB'] - temps[bh_node])
        
        # Special connections for BH_1
        if 'BH_1' in node_flows:
            node_flows['BH_1']['conduction']['ESC_Mount'] = self.thermal_coefficients['C_cond_Mount_to_BH1'] * (temps['ESC_Mount'] - temps['BH_1'])
        
        # PLATE NODES
        plate_nodes = ['plateT', 'plateM', 'plateB']
        for plate_node in plate_nodes:
            if plate_node == 'plateT':
                area = config.A_conv_plate
            elif plate_node == 'plateM':
                area = config.A_conv_plateM
            else:
                area = config.A_conv_plate
                
            node_flows[plate_node]['convection'] = get_h(temps[plate_node], config.LC_plate, False) * area * (temps['Internal_air'] - temps[plate_node])
            
            # Radiation to shells
            if plate_node == 'plateT':
                node_flows[plate_node]['radiation']['Top_Shell_Int'] = self.thermal_coefficients['C_rad_plate_sh'] * (T4s['Top_Shell_Int'] - T4s[plate_node])
            elif plate_node == 'plateB':
                node_flows[plate_node]['radiation']['Bot_Shell_Int'] = self.thermal_coefficients['C_rad_plate_sh'] * (T4s['Bot_Shell_Int'] - T4s[plate_node])
        
        # SHELL NODES
        shell_nodes = ['Top_Shell_Int', 'Top_Shell_Ext', 'Bot_Shell_Int', 'Bot_Shell_Ext']
        for shell_node in shell_nodes:
            if 'Int' in shell_node:
                if 'Top' in shell_node:
                    area = config.A_TS
                    lc = config.LC_TS_int
                else:
                    area = config.A_BS
                    lc = config.LC_BS_int
                    
                node_flows[shell_node]['convection'] = get_h(temps[shell_node], lc, False) * area * (temps['Internal_air'] - temps[shell_node])
                
                # Conduction to external shell
                if 'Top' in shell_node:
                    node_flows[shell_node]['conduction']['Top_Shell_Ext'] = self.thermal_coefficients['C_cond_TS_int_ext'] * (temps['Top_Shell_Ext'] - temps[shell_node])
                else:
                    node_flows[shell_node]['conduction']['Bot_Shell_Ext'] = self.thermal_coefficients['C_cond_BS_int_ext'] * (temps['Bot_Shell_Ext'] - temps[shell_node])
            
            elif 'Ext' in shell_node:
                # External convection
                if 'Top' in shell_node:
                    area = config.A_TS
                    lc = config.LC_TS_ext
                else:
                    area = config.A_BS
                    lc = config.LC_BS_ext
                
                p_ambient = physics_models.prop_internal_air(ambient_temp, ambient_pressure)
                h_ext = physics_models.get_external_convection_h(p_ambient, temps[shell_node], ambient_temp, lc)
                node_flows[shell_node]['convection'] = h_ext * area * (ambient_temp - temps[shell_node])
                
                # Conduction to internal shell
                if 'Top' in shell_node:
                    node_flows[shell_node]['conduction']['Top_Shell_Int'] = self.thermal_coefficients['C_cond_TS_int_ext'] * (temps['Top_Shell_Int'] - temps[shell_node])
                else:
                    node_flows[shell_node]['conduction']['Bot_Shell_Int'] = self.thermal_coefficients['C_cond_BS_int_ext'] * (temps['Bot_Shell_Int'] - temps[shell_node])
        
        # INTERNAL AIR - collect all convection flows
        air_convection_total = 0.0
        for node in self.node_labels:
            if node != 'Internal_air':
                air_convection_total -= node_flows[node]['convection']  # Opposite sign for air node
        
        node_flows['Internal_air']['convection'] = air_convection_total
        
        # Calculate net flows for each node
        for node in self.node_labels:
            flows = node_flows[node]
            net_flow = flows['generation'] + flows['convection']
            
            # Add conduction contributions
            for target, q_cond in flows['conduction'].items():
                net_flow += q_cond
            
            # Add radiation contributions
            for target, q_rad in flows['radiation'].items():
                net_flow += q_rad
            
            flows['net_flow'] = net_flow
        
        return node_flows
    
    def create_heat_flow_summary(self, node_flows):
        """
        Create a summary DataFrame of heat flows for all nodes
        """
        summary_data = []
        
        for node, flows in node_flows.items():
            # Calculate totals
            total_conduction = sum(flows['conduction'].values()) if flows['conduction'] else 0.0
            total_radiation = sum(flows['radiation'].values()) if flows['radiation'] else 0.0
            
            summary_data.append({
                'Node': node,
                'Generation_W': flows['generation'],
                'Conduction_W': total_conduction,
                'Convection_W': flows['convection'],
                'Radiation_W': total_radiation,
                'Net_Flow_W': flows['net_flow'],
                'Total_Heat_Out_W': abs(total_conduction) + abs(flows['convection']) + abs(total_radiation)
            })
        
        return pd.DataFrame(summary_data)
    
    def create_heat_flow_visualization(self, node_flows, temperatures):
        """
        Create comprehensive heat flow visualizations
        """
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        
        # 1. Heat generation vs heat rejection
        summary_df = self.create_heat_flow_summary(node_flows)
        heat_gen_nodes = summary_df[summary_df['Generation_W'] > 0].sort_values('Generation_W', ascending=False)
        
        if not heat_gen_nodes.empty:
            axes[0,0].bar(heat_gen_nodes['Node'], heat_gen_nodes['Generation_W'], 
                         color='red', alpha=0.7, label='Generation')
            axes[0,0].bar(heat_gen_nodes['Node'], -heat_gen_nodes['Total_Heat_Out_W'], 
                         color='blue', alpha=0.7, label='Heat Rejection')
            axes[0,0].set_title('Heat Generation vs Rejection')
            axes[0,0].set_ylabel('Heat Flow (W)')
            axes[0,0].legend()
            axes[0,0].tick_params(axis='x', rotation=45)
        
        # 2. Heat transfer mode breakdown for top heat generating nodes
        if not heat_gen_nodes.empty:
            top_node = heat_gen_nodes.iloc[0]['Node']
            flows = node_flows[top_node]
            
            modes = ['Conduction', 'Convection', 'Radiation']
            values = [abs(sum(flows['conduction'].values())), 
                     abs(flows['convection']), 
                     abs(sum(flows['radiation'].values()))]
            
            # Filter out zero values
            non_zero_modes = [(mode, val) for mode, val in zip(modes, values) if val > 0.1]
            if non_zero_modes:
                modes, values = zip(*non_zero_modes)
                axes[0,1].pie(values, labels=modes, autopct='%1.1f%%', startangle=90)
                axes[0,1].set_title(f'Heat Transfer Modes\n({top_node})')
        
        # 3. Temperature distribution
        node_temps = [temperatures[i] - 273.15 for i in range(len(temperatures))]
        bars = axes[1,0].bar(range(len(node_temps)), node_temps, 
                            color=plt.cm.coolwarm([t/max(node_temps) for t in node_temps]))
        axes[1,0].set_title('Temperature Distribution')
        axes[1,0].set_xlabel('Node Index')
        axes[1,0].set_ylabel('Temperature (°C)')
        
        # Add node labels for key nodes
        key_indices = [i for i, label in enumerate(self.node_labels) 
                      if any(key in label for key in ['ESC', 'Batt_BF', 'Internal_air'])]
        for i in key_indices:
            axes[1,0].text(i, node_temps[i] + 1, self.node_labels[i][:8], 
                          rotation=45, fontsize=8, ha='left')
        
        # 4. Net heat flow by node
        net_flows = [node_flows[node]['net_flow'] for node in self.node_labels]
        colors = ['red' if flow > 0 else 'blue' for flow in net_flows]
        axes[1,1].bar(range(len(net_flows)), net_flows, color=colors, alpha=0.7)
        axes[1,1].set_title('Net Heat Flow by Node')
        axes[1,1].set_xlabel('Node Index')
        axes[1,1].set_ylabel('Net Heat Flow (W)')
        axes[1,1].axhline(y=0, color='black', linestyle='-', alpha=0.3)
        
        plt.tight_layout()
        return fig

# Example usage
if __name__ == "__main__":
    analyzer = HeatFlowAnalyzer()
    
    # Example temperatures (would come from simulation)
    example_temps = np.array([315.0] * len(config.labels))  # Kelvin
    example_temps[config.labels.index('ESC')] = 390.0  # Hotter ESC
    example_temps[config.labels.index('Internal_air')] = 318.0  # Warm air
    
    # Analyze heat flows
    heat_flows = analyzer.analyze_heat_flows_at_state(
        example_temps, 
        ambient_temp=287.3, 
        ambient_pressure=99832.0
    )
    
    # Create summary
    summary_df = analyzer.create_heat_flow_summary(heat_flows)
    print("Heat Flow Summary:")
    print(summary_df.round(2))
    
    # Create visualization
    fig = analyzer.create_heat_flow_visualization(heat_flows, example_temps)
    plt.show()
