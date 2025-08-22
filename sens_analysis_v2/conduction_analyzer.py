# conduction_analyzer.py
# Detailed Conduction Heat Transfer Analysis for 20-Node UAV Thermal Model

import sys
import os
import numpy as np
import pandas as pd

# Add the code_vM_1 directory to the path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'code_vM_1'))
import config
import physics_models
import environment_model

class ConductionAnalyzer:
    """
    Analyzes all conduction heat transfer paths showing sources and sinks for each node
    """
    
    def __init__(self):
        self.node_labels = config.labels
        environment_model.init()
        self.thermal_coeffs = self._calculate_thermal_coefficients()
        self.conduction_paths = self._define_conduction_paths()
    
    def _calculate_thermal_coefficients(self):
        """Calculate all conduction thermal coefficients"""
        return {
            'C_cond_ESC_to_Mount': config.k_mount * config.A_contact_ESC_Mount / config.L_path_ESC_Mount,
            'C_cond_Mount_to_BH1': config.k_bulkhead * config.A_contact_Mount_BH1 / config.t_bulkhead,
            'C_cond_Batt_plate': config.k_plate * config.A_contact_batt_plate / config.t_plate,
            'C_cond_bh_plate': 2 * config.k_bulkhead * config.A_contact_bh_plate / config.L_bh_plate_cond,
            'C_cond_BH_TS': config.k_cfrp * config.A_contact_BH_Shell / config.t_cfrp,
            'C_cond_BH_BS': config.k_cfrp * config.A_contact_BH_Shell / config.t_cfrp,
            'C_cond_TS_int_ext': config.k_cfrp * config.A_TS / config.t_cfrp,
            'C_cond_BS_int_ext': config.k_cfrp * config.A_BS / config.t_cfrp,
        }
    
    def _define_conduction_paths(self):
        """Define all conduction paths in the system"""
        paths = []
        
        # ESC connections
        paths.append(('ESC', 'ESC_Mount', 'C_cond_ESC_to_Mount'))
        paths.append(('ESC_Mount', 'BH_1', 'C_cond_Mount_to_BH1'))
        
        # Battery to plate connections
        battery_plate_connections = [
            ('Batt_BF_Top', 'plateT'), ('Batt_BF_Top', 'plateM'),
            ('Batt_BF_Bot', 'plateM'), ('Batt_BF_Bot', 'plateB'),
            ('Batt_BM_Top', 'plateT'), ('Batt_BM_Top', 'plateM'),
            ('Batt_BM_Bot', 'plateM'), ('Batt_BM_Bot', 'plateB'),
            ('Batt_BR_Top', 'plateT'), ('Batt_BR_Top', 'plateM'),
            ('Batt_BR_Bot', 'plateM'), ('Batt_BR_Bot', 'plateB'),
        ]
        
        for batt, plate in battery_plate_connections:
            paths.append((batt, plate, 'C_cond_Batt_plate'))
        
        # Bulkhead to plate connections
        bulkheads = ['BH_1', 'BH_2', 'BH_3', 'BH_4']
        plates = ['plateT', 'plateM', 'plateB']
        
        for bh in bulkheads:
            for plate in plates:
                paths.append((bh, plate, 'C_cond_bh_plate'))
        
        # Bulkhead to shell connections
        for bh in bulkheads:
            paths.append((bh, 'Top_Shell_Int', 'C_cond_BH_TS'))
            paths.append((bh, 'Bot_Shell_Int', 'C_cond_BH_BS'))
        
        # Shell internal to external
        paths.append(('Top_Shell_Int', 'Top_Shell_Ext', 'C_cond_TS_int_ext'))
        paths.append(('Bot_Shell_Int', 'Bot_Shell_Ext', 'C_cond_BS_int_ext'))
        
        return paths
    
    def analyze_conduction_flows(self, temperatures):
        """
        Analyze all conduction heat flows and organize by node sources/sinks
        """
        temps = {label: temperatures[i] for i, label in enumerate(self.node_labels)}
        
        # Initialize node conduction tracking
        node_conduction = {}
        for node in self.node_labels:
            node_conduction[node] = {
                'sources': {},  # Heat gained from other nodes
                'sinks': {},    # Heat lost to other nodes
                'net_gain': 0.0,
                'net_loss': 0.0,
                'net_total': 0.0
            }
        
        # Calculate all conduction flows
        conduction_flows = []
        
        for node1, node2, coeff_name in self.conduction_paths:
            coeff = self.thermal_coeffs[coeff_name]
            thermal_resistance = 1.0 / coeff if coeff > 0 else float('inf')
            
            # Heat flow from node1 to node2 (positive = heat flows from 1 to 2)
            q_flow = coeff * (temps[node2] - temps[node1])
            
            # Record the flow
            conduction_flows.append({
                'From_Node': node1,
                'To_Node': node2,
                'Heat_Flow_W': q_flow,
                'Thermal_Coeff': coeff,
                'Thermal_Resistance_K_per_W': thermal_resistance,
                'Temp_Difference_K': temps[node2] - temps[node1],
                'Direction': 'TO' if q_flow > 0 else 'FROM'
            })
            
            # Update node tracking
            if q_flow > 0:  # Heat flows from node1 to node2
                # node1 loses heat (sink)
                node_conduction[node1]['sinks'][node2] = q_flow
                node_conduction[node1]['net_loss'] += q_flow
                
                # node2 gains heat (source)
                node_conduction[node2]['sources'][node1] = q_flow
                node_conduction[node2]['net_gain'] += q_flow
            else:  # Heat flows from node2 to node1
                # node2 loses heat (sink)
                node_conduction[node2]['sinks'][node1] = abs(q_flow)
                node_conduction[node2]['net_loss'] += abs(q_flow)
                
                # node1 gains heat (source)
                node_conduction[node1]['sources'][node2] = abs(q_flow)
                node_conduction[node1]['net_gain'] += abs(q_flow)
        
        # Calculate net totals
        for node in self.node_labels:
            node_conduction[node]['net_total'] = (node_conduction[node]['net_gain'] - 
                                                 node_conduction[node]['net_loss'])
        
        return node_conduction, conduction_flows
    
    def create_node_conduction_summary(self, node_conduction):
        """Create summary table of conduction gains/losses per node"""
        summary_data = []
        
        for node in self.node_labels:
            data = node_conduction[node]
            
            # Count number of connections
            num_sources = len(data['sources'])
            num_sinks = len(data['sinks'])
            
            # Find dominant source and sink
            dominant_source = max(data['sources'], key=data['sources'].get) if data['sources'] else 'None'
            dominant_sink = max(data['sinks'], key=data['sinks'].get) if data['sinks'] else 'None'
            
            summary_data.append({
                'Node': node,
                'Heat_Sources_Count': num_sources,
                'Heat_Sinks_Count': num_sinks,
                'Total_Heat_Gained_W': data['net_gain'],
                'Total_Heat_Lost_W': data['net_loss'],
                'Net_Conduction_W': data['net_total'],
                'Dominant_Source': dominant_source,
                'Dominant_Sink': dominant_sink
            })
        
        return pd.DataFrame(summary_data)
    
    def create_detailed_node_reports(self, node_conduction):
        """Create detailed reports for each node showing all sources and sinks"""
        reports = {}
        
        for node in self.node_labels:
            data = node_conduction[node]
            
            report = f"\n=== CONDUCTION ANALYSIS: {node} ===\n"
            
            # Heat sources (gains)
            if data['sources']:
                report += "HEAT SOURCES (Conduction Gains):\n"
                for source_node, heat_flow in sorted(data['sources'].items(), 
                                                   key=lambda x: x[1], reverse=True):
                    report += f"  From {source_node:15}: +{heat_flow:8.3f} W\n"
                report += f"  TOTAL GAINED:           +{data['net_gain']:8.3f} W\n\n"
            else:
                report += "HEAT SOURCES: None\n\n"
            
            # Heat sinks (losses)
            if data['sinks']:
                report += "HEAT SINKS (Conduction Losses):\n"
                for sink_node, heat_flow in sorted(data['sinks'].items(), 
                                                 key=lambda x: x[1], reverse=True):
                    report += f"  To {sink_node:17}: -{heat_flow:8.3f} W\n"
                report += f"  TOTAL LOST:             -{data['net_loss']:8.3f} W\n\n"
            else:
                report += "HEAT SINKS: None\n\n"
            
            # Net result
            net_flow = data['net_total']
            if net_flow > 0:
                report += f"NET CONDUCTION RESULT:    +{net_flow:8.3f} W (Heat Gain)\n"
            elif net_flow < 0:
                report += f"NET CONDUCTION RESULT:    {net_flow:8.3f} W (Heat Loss)\n"
            else:
                report += f"NET CONDUCTION RESULT:     {net_flow:8.3f} W (Balanced)\n"
            
            reports[node] = report
        
        return reports
    
    def create_conduction_flows_table(self, conduction_flows):
        """Create table of all conduction flows sorted by magnitude"""
        df = pd.DataFrame(conduction_flows)
        df['Magnitude_W'] = abs(df['Heat_Flow_W'])
        return df.sort_values('Magnitude_W', ascending=False)

# Example usage
if __name__ == "__main__":
    analyzer = ConductionAnalyzer()
    
    # Example temperatures
    example_temps = np.array([315.0] * len(config.labels))
    example_temps[config.labels.index('ESC')] = 390.0
    example_temps[config.labels.index('Internal_air')] = 318.0
    
    # Analyze conduction
    node_conduction, conduction_flows = analyzer.analyze_conduction_flows(example_temps)
    
    # Create summary
    summary_df = analyzer.create_node_conduction_summary(node_conduction)
    print("=== CONDUCTION SUMMARY BY NODE ===")
    print(summary_df.round(3))
    
    # Show detailed reports for nodes with significant conduction
    reports = analyzer.create_detailed_node_reports(node_conduction)
    
    # Show reports for key nodes
    key_nodes = ['ESC', 'ESC_Mount', 'Batt_BF_Top', 'Top_Shell_Ext']
    for node in key_nodes:
        print(reports[node])
    
    # Show top conduction flows
    flows_df = analyzer.create_conduction_flows_table(conduction_flows)
    print("\n=== TOP 10 CONDUCTION FLOWS ===")
    print(flows_df.head(10)[['From_Node', 'To_Node', 'Heat_Flow_W', 'Thermal_Resistance_K_per_W']].round(3))
    
    # Save results
    summary_df.to_csv('conduction_summary.csv', index=False)
    flows_df.to_csv('conduction_flows_detailed.csv', index=False)
    print(f"\nResults saved to:")
    print("- conduction_summary.csv")
    print("- conduction_flows_detailed.csv")
