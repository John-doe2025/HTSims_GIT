# radiation_analyzer.py
# Detailed Radiation Heat Transfer Analysis for 20-Node UAV Thermal Model

import sys
import os
import numpy as np
import pandas as pd

# Add the code_vM_1 directory to the path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'code_vM_1'))
import config
import physics_models
import environment_model

class RadiationAnalyzer:
    """
    Analyzes all radiation heat transfer showing sources and sinks for each node
    """
    
    def __init__(self):
        self.node_labels = config.labels
        environment_model.init()
        self.radiation_coeffs = self._calculate_radiation_coefficients()
        self.radiation_paths = self._define_radiation_paths()
    
    def _calculate_radiation_coefficients(self):
        """Calculate all radiation coefficients"""
        return {
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
    
    def _define_radiation_paths(self):
        """Define all radiation exchange paths"""
        paths = []
        
        # Battery to battery radiation
        battery_pairs = [
            ('Batt_BF_Top', 'Batt_BM_Top'), ('Batt_BF_Bot', 'Batt_BM_Bot'),
            ('Batt_BM_Top', 'Batt_BR_Top'), ('Batt_BM_Bot', 'Batt_BR_Bot'),
        ]
        for batt1, batt2 in battery_pairs:
            paths.append((batt1, batt2, 'C_rad_batt_batt'))
        
        # Battery to ESC radiation
        batteries = ['Batt_BF_Top', 'Batt_BF_Bot', 'Batt_BM_Top', 'Batt_BM_Bot', 'Batt_BR_Top', 'Batt_BR_Bot']
        for batt in batteries:
            paths.append((batt, 'ESC', 'C_rad_batt_esc'))
        
        # Battery to shell radiation
        top_batteries = ['Batt_BF_Top', 'Batt_BM_Top', 'Batt_BR_Top']
        bot_batteries = ['Batt_BF_Bot', 'Batt_BM_Bot', 'Batt_BR_Bot']
        
        for batt in top_batteries:
            paths.append((batt, 'Top_Shell_Int', 'C_rad_batt_ts'))
        
        for batt in bot_batteries:
            paths.append((batt, 'Bot_Shell_Int', 'C_rad_batt_bs'))
        
        # ESC radiation
        paths.append(('ESC', 'Top_Shell_Int', 'C_rad_esc_ts'))
        
        # Plate radiation
        paths.append(('plateT', 'Top_Shell_Int', 'C_rad_plate_sh'))
        paths.append(('plateB', 'Bot_Shell_Int', 'C_rad_plate_sh'))
        
        # Mount radiation
        paths.append(('ESC_Mount', 'Bot_Shell_Int', 'C_rad_mount_bs'))
        
        # Shell to shell radiation
        paths.append(('Top_Shell_Int', 'Bot_Shell_Int', 'C_rad_ts_bs'))
        
        return paths
    
    def analyze_radiation_flows(self, temperatures):
        """
        Analyze all radiation heat flows and organize by node sources/sinks
        """
        temps = {label: temperatures[i] for i, label in enumerate(self.node_labels)}
        T4s = {k: physics_models.T_power4(v) for k, v in temps.items()}
        
        # Initialize node radiation tracking
        node_radiation = {}
        for node in self.node_labels:
            node_radiation[node] = {
                'sources': {},  # Heat gained from other nodes
                'sinks': {},    # Heat lost to other nodes
                'net_gain': 0.0,
                'net_loss': 0.0,
                'net_total': 0.0
            }
        
        # Calculate all radiation flows
        radiation_flows = []
        
        for node1, node2, coeff_name in self.radiation_paths:
            coeff = self.radiation_coeffs[coeff_name]
            
            # Radiation flow from node1 to node2 (positive = heat flows from 1 to 2)
            q_flow = coeff * (T4s[node2] - T4s[node1])
            
            # Record the flow
            radiation_flows.append({
                'From_Node': node1,
                'To_Node': node2,
                'Heat_Flow_W': q_flow,
                'Radiation_Coeff': coeff,
                'T1_K': temps[node1],
                'T2_K': temps[node2],
                'T1_4th_Power': T4s[node1],
                'T2_4th_Power': T4s[node2],
                'Direction': 'TO' if q_flow > 0 else 'FROM'
            })
            
            # Update node tracking
            if q_flow > 0:  # Heat flows from node1 to node2
                # node1 loses heat (sink)
                node_radiation[node1]['sinks'][node2] = q_flow
                node_radiation[node1]['net_loss'] += q_flow
                
                # node2 gains heat (source)
                node_radiation[node2]['sources'][node1] = q_flow
                node_radiation[node2]['net_gain'] += q_flow
            else:  # Heat flows from node2 to node1
                # node2 loses heat (sink)
                node_radiation[node2]['sinks'][node1] = abs(q_flow)
                node_radiation[node2]['net_loss'] += abs(q_flow)
                
                # node1 gains heat (source)
                node_radiation[node1]['sources'][node2] = abs(q_flow)
                node_radiation[node1]['net_gain'] += abs(q_flow)
        
        # Calculate net totals
        for node in self.node_labels:
            node_radiation[node]['net_total'] = (node_radiation[node]['net_gain'] - 
                                                node_radiation[node]['net_loss'])
        
        return node_radiation, radiation_flows
    
    def create_node_radiation_summary(self, node_radiation):
        """Create summary table of radiation gains/losses per node"""
        summary_data = []
        
        for node in self.node_labels:
            data = node_radiation[node]
            
            # Count number of connections
            num_sources = len(data['sources'])
            num_sinks = len(data['sinks'])
            
            # Find dominant source and sink
            dominant_source = max(data['sources'], key=data['sources'].get) if data['sources'] else 'None'
            dominant_sink = max(data['sinks'], key=data['sinks'].get) if data['sinks'] else 'None'
            
            summary_data.append({
                'Node': node,
                'Radiation_Sources_Count': num_sources,
                'Radiation_Sinks_Count': num_sinks,
                'Total_Heat_Gained_W': data['net_gain'],
                'Total_Heat_Lost_W': data['net_loss'],
                'Net_Radiation_W': data['net_total'],
                'Dominant_Source': dominant_source,
                'Dominant_Sink': dominant_sink
            })
        
        return pd.DataFrame(summary_data)
    
    def create_detailed_radiation_reports(self, node_radiation):
        """Create detailed reports for each node showing all radiation sources and sinks"""
        reports = {}
        
        for node in self.node_labels:
            data = node_radiation[node]
            
            # Skip nodes with no radiation
            if data['net_gain'] == 0 and data['net_loss'] == 0:
                continue
            
            report = f"\n=== RADIATION ANALYSIS: {node} ===\n"
            
            # Heat sources (gains)
            if data['sources']:
                report += "RADIATION SOURCES (Heat Gains):\n"
                for source_node, heat_flow in sorted(data['sources'].items(), 
                                                   key=lambda x: x[1], reverse=True):
                    report += f"  From {source_node:15}: +{heat_flow:8.3f} W\n"
                report += f"  TOTAL GAINED:           +{data['net_gain']:8.3f} W\n\n"
            else:
                report += "RADIATION SOURCES: None\n\n"
            
            # Heat sinks (losses)
            if data['sinks']:
                report += "RADIATION SINKS (Heat Losses):\n"
                for sink_node, heat_flow in sorted(data['sinks'].items(), 
                                                 key=lambda x: x[1], reverse=True):
                    report += f"  To {sink_node:17}: -{heat_flow:8.3f} W\n"
                report += f"  TOTAL LOST:             -{data['net_loss']:8.3f} W\n\n"
            else:
                report += "RADIATION SINKS: None\n\n"
            
            # Net result
            net_flow = data['net_total']
            if net_flow > 0:
                report += f"NET RADIATION RESULT:     +{net_flow:8.3f} W (Heat Gain)\n"
            elif net_flow < 0:
                report += f"NET RADIATION RESULT:     {net_flow:8.3f} W (Heat Loss)\n"
            else:
                report += f"NET RADIATION RESULT:      {net_flow:8.3f} W (Balanced)\n"
            
            reports[node] = report
        
        return reports
    
    def create_radiation_flows_table(self, radiation_flows):
        """Create table of all radiation flows sorted by magnitude"""
        df = pd.DataFrame(radiation_flows)
        df['Magnitude_W'] = abs(df['Heat_Flow_W'])
        return df.sort_values('Magnitude_W', ascending=False)

# Example usage
if __name__ == "__main__":
    analyzer = RadiationAnalyzer()
    
    # Example temperatures with realistic differences
    example_temps = np.array([315.0] * len(config.labels))
    example_temps[config.labels.index('ESC')] = 390.0
    example_temps[config.labels.index('Internal_air')] = 318.0
    example_temps[config.labels.index('Top_Shell_Int')] = 312.0
    example_temps[config.labels.index('Bot_Shell_Int')] = 312.0
    
    # Analyze radiation
    node_radiation, radiation_flows = analyzer.analyze_radiation_flows(example_temps)
    
    # Create summary
    summary_df = analyzer.create_node_radiation_summary(node_radiation)
    print("=== RADIATION SUMMARY BY NODE ===")
    print(summary_df.round(3))
    
    # Show detailed reports for nodes with significant radiation
    reports = analyzer.create_detailed_radiation_reports(node_radiation)
    
    # Show reports for key nodes
    key_nodes = ['ESC', 'Batt_BF_Top', 'Top_Shell_Int']
    for node in key_nodes:
        if node in reports:
            print(reports[node])
    
    # Show top radiation flows
    flows_df = analyzer.create_radiation_flows_table(radiation_flows)
    print("\n=== TOP 10 RADIATION FLOWS ===")
    print(flows_df.head(10)[['From_Node', 'To_Node', 'Heat_Flow_W', 'T1_K', 'T2_K']].round(3))
    
    # Save results
    summary_df.to_csv('radiation_summary.csv', index=False)
    flows_df.to_csv('radiation_flows_detailed.csv', index=False)
    print(f"\nResults saved to:")
    print("- radiation_summary.csv")
    print("- radiation_flows_detailed.csv")
