# heat_flow_network_mapper.py
# Node-to-Node Heat Flow Network Mapping and Analysis

import sys
import os
import numpy as np
import pandas as pd

# Add the code_vM_1 directory to the path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'code_vM_1'))
import config
import physics_models
import environment_model

from conduction_analyzer import ConductionAnalyzer
from convection_analyzer import ConvectionAnalyzer
from radiation_analyzer import RadiationAnalyzer

class HeatFlowNetworkMapper:
    """
    Creates detailed node-to-node heat flow network maps showing all thermal connections
    """
    
    def __init__(self):
        self.conduction_analyzer = ConductionAnalyzer()
        self.convection_analyzer = ConvectionAnalyzer()
        self.radiation_analyzer = RadiationAnalyzer()
        self.node_labels = config.labels
    
    def create_heat_flow_network(self, temperatures):
        """
        Create complete heat flow network showing all node-to-node connections
        """
        print("Creating heat flow network map...")
        
        # Get all heat transfer data
        node_conduction, conduction_flows = self.conduction_analyzer.analyze_conduction_flows(temperatures)
        node_convection = self.convection_analyzer.analyze_convection_flows(temperatures)
        node_radiation, radiation_flows = self.radiation_analyzer.analyze_radiation_flows(temperatures)
        
        # Create network adjacency matrix
        network_matrix = self._create_network_matrix(conduction_flows, radiation_flows, node_convection)
        
        # Create detailed connection table
        connection_table = self._create_connection_table(conduction_flows, radiation_flows, node_convection)
        
        # Analyze network properties
        network_analysis = self._analyze_network_properties(network_matrix, connection_table)
        
        return {
            'network_matrix': network_matrix,
            'connection_table': connection_table,
            'network_analysis': network_analysis,
            'temperatures': temperatures
        }
    
    def _create_network_matrix(self, conduction_flows, radiation_flows, node_convection):
        """
        Create adjacency matrix showing heat flow magnitudes between all nodes
        """
        n_nodes = len(self.node_labels)
        matrix = np.zeros((n_nodes, n_nodes))
        
        # Add conduction flows
        for flow in conduction_flows:
            from_idx = self.node_labels.index(flow['From_Node'])
            to_idx = self.node_labels.index(flow['To_Node'])
            heat_flow = flow['Heat_Flow_W']
            
            if heat_flow > 0:  # Heat flows from 'from' to 'to'
                matrix[from_idx, to_idx] = heat_flow
            else:  # Heat flows from 'to' to 'from'
                matrix[to_idx, from_idx] = abs(heat_flow)
        
        # Add radiation flows
        for flow in radiation_flows:
            from_idx = self.node_labels.index(flow['From_Node'])
            to_idx = self.node_labels.index(flow['To_Node'])
            heat_flow = flow['Heat_Flow_W']
            
            if heat_flow > 0:  # Heat flows from 'from' to 'to'
                matrix[from_idx, to_idx] += heat_flow
            else:  # Heat flows from 'to' to 'from'
                matrix[to_idx, from_idx] += abs(heat_flow)
        
        return matrix
    
    def _create_connection_table(self, conduction_flows, radiation_flows, node_convection):
        """
        Create detailed table of all heat flow connections
        """
        connections = []
        
        # Add conduction connections
        for flow in conduction_flows:
            if abs(flow['Heat_Flow_W']) > 0.001:  # Skip negligible flows
                connections.append({
                    'From_Node': flow['From_Node'],
                    'To_Node': flow['To_Node'],
                    'Heat_Flow_W': flow['Heat_Flow_W'],
                    'Transfer_Mode': 'Conduction',
                    'Thermal_Resistance': flow['Thermal_Resistance_K_per_W'],
                    'Temp_Difference_K': flow['Temp_Difference_K']
                })
        
        # Add radiation connections
        for flow in radiation_flows:
            if abs(flow['Heat_Flow_W']) > 0.001:  # Skip negligible flows
                connections.append({
                    'From_Node': flow['From_Node'],
                    'To_Node': flow['To_Node'],
                    'Heat_Flow_W': flow['Heat_Flow_W'],
                    'Transfer_Mode': 'Radiation',
                    'Thermal_Resistance': 1.0 / flow['Radiation_Coeff'] if flow['Radiation_Coeff'] > 0 else float('inf'),
                    'Temp_Difference_K': flow['T2_K'] - flow['T1_K']
                })
        
        # Add convection connections (to ambient/internal air)
        for node in self.node_labels:
            conv_data = node_convection[node]
            if abs(conv_data['heat_flow']) > 0.001 and conv_data['target']:
                connections.append({
                    'From_Node': node,
                    'To_Node': conv_data['target'],
                    'Heat_Flow_W': conv_data['heat_flow'],
                    'Transfer_Mode': f"Convection_{conv_data['flow_type']}",
                    'Thermal_Resistance': 1.0 / (conv_data['h_coefficient'] * conv_data['area']) if conv_data['h_coefficient'] * conv_data['area'] > 0 else float('inf'),
                    'Temp_Difference_K': conv_data['temp_difference']
                })
        
        return pd.DataFrame(connections)
    
    def _analyze_network_properties(self, network_matrix, connection_table):
        """
        Analyze thermal network properties
        """
        analysis = {}
        
        # Node connectivity analysis
        node_connections = {}
        for i, node in enumerate(self.node_labels):
            outgoing = np.sum(network_matrix[i, :] > 0.001)
            incoming = np.sum(network_matrix[:, i] > 0.001)
            total_outgoing_heat = np.sum(network_matrix[i, :])
            total_incoming_heat = np.sum(network_matrix[:, i])
            
            node_connections[node] = {
                'outgoing_connections': int(outgoing),
                'incoming_connections': int(incoming),
                'total_connections': int(outgoing + incoming),
                'total_outgoing_heat_W': total_outgoing_heat,
                'total_incoming_heat_W': total_incoming_heat,
                'net_heat_W': total_incoming_heat - total_outgoing_heat
            }
        
        analysis['node_connections'] = node_connections
        
        # Heat flow statistics
        all_flows = connection_table['Heat_Flow_W'].abs()
        analysis['heat_flow_stats'] = {
            'total_heat_flows': len(connection_table),
            'max_heat_flow_W': all_flows.max(),
            'min_heat_flow_W': all_flows.min(),
            'mean_heat_flow_W': all_flows.mean(),
            'std_heat_flow_W': all_flows.std()
        }
        
        # Mode distribution
        mode_counts = connection_table['Transfer_Mode'].value_counts()
        analysis['mode_distribution'] = mode_counts.to_dict()
        
        # Critical heat paths (highest flows)
        critical_paths = connection_table.nlargest(10, 'Heat_Flow_W', keep='all')
        analysis['critical_heat_paths'] = critical_paths
        
        return analysis
    
    def create_network_summary_report(self, network_data):
        """
        Create comprehensive network summary report
        """
        network_analysis = network_data['network_analysis']
        connection_table = network_data['connection_table']
        
        report = "\n" + "="*80 + "\n"
        report += "HEAT FLOW NETWORK ANALYSIS SUMMARY\n"
        report += "="*80 + "\n"
        
        # Network statistics
        stats = network_analysis['heat_flow_stats']
        report += f"\nNETWORK STATISTICS:\n"
        report += f"  Total Heat Flow Connections: {stats['total_heat_flows']}\n"
        report += f"  Maximum Heat Flow:          {stats['max_heat_flow_W']:8.2f} W\n"
        report += f"  Average Heat Flow:          {stats['mean_heat_flow_W']:8.2f} W\n"
        report += f"  Heat Flow Standard Dev:     {stats['std_heat_flow_W']:8.2f} W\n"
        
        # Mode distribution
        report += f"\nHEAT TRANSFER MODE DISTRIBUTION:\n"
        for mode, count in network_analysis['mode_distribution'].items():
            percentage = 100 * count / stats['total_heat_flows']
            report += f"  {mode:20}: {count:3d} connections ({percentage:5.1f}%)\n"
        
        # Node connectivity ranking
        report += f"\nNODE CONNECTIVITY RANKING:\n"
        connections = network_analysis['node_connections']
        sorted_nodes = sorted(connections.items(), key=lambda x: x[1]['total_connections'], reverse=True)
        
        report += f"{'Node':<15} {'Connections':<12} {'Out Heat':<10} {'In Heat':<10} {'Net Heat':<10}\n"
        report += f"{'-'*15} {'-'*12} {'-'*10} {'-'*10} {'-'*10}\n"
        
        for node, data in sorted_nodes[:10]:  # Top 10 most connected
            report += f"{node:<15} {data['total_connections']:<12} "
            report += f"{data['total_outgoing_heat_W']:<10.1f} "
            report += f"{data['total_incoming_heat_W']:<10.1f} "
            report += f"{data['net_heat_W']:<10.1f}\n"
        
        # Critical heat paths
        report += f"\nCRITICAL HEAT PATHS (Top 10):\n"
        report += f"{'From':<15} {'To':<15} {'Heat Flow':<12} {'Mode':<20} {'Temp Diff':<10}\n"
        report += f"{'-'*15} {'-'*15} {'-'*12} {'-'*20} {'-'*10}\n"
        
        critical_paths = network_analysis['critical_heat_paths']
        for _, row in critical_paths.head(10).iterrows():
            report += f"{row['From_Node']:<15} {row['To_Node']:<15} "
            report += f"{row['Heat_Flow_W']:<12.2f} {row['Transfer_Mode']:<20} "
            report += f"{row['Temp_Difference_K']:<10.1f}\n"
        
        return report
    
    def create_thermal_resistance_network(self, network_data):
        """
        Create thermal resistance network analysis
        """
        connection_table = network_data['connection_table']
        
        # Filter out infinite resistances and very small flows
        valid_connections = connection_table[
            (connection_table['Thermal_Resistance'] < 1000) & 
            (abs(connection_table['Heat_Flow_W']) > 0.1)
        ].copy()
        
        # Calculate thermal conductance (1/resistance)
        valid_connections['Thermal_Conductance'] = 1.0 / valid_connections['Thermal_Resistance']
        
        # Analyze resistance distribution by mode
        resistance_analysis = {}
        for mode in valid_connections['Transfer_Mode'].unique():
            mode_data = valid_connections[valid_connections['Transfer_Mode'] == mode]
            resistance_analysis[mode] = {
                'count': len(mode_data),
                'mean_resistance': mode_data['Thermal_Resistance'].mean(),
                'min_resistance': mode_data['Thermal_Resistance'].min(),
                'max_resistance': mode_data['Thermal_Resistance'].max(),
                'mean_conductance': mode_data['Thermal_Conductance'].mean()
            }
        
        return resistance_analysis, valid_connections
    
    def save_network_analysis(self, network_data, base_filename='heat_flow_network'):
        """
        Save all network analysis results to files
        """
        # Save connection table
        network_data['connection_table'].to_csv(f'{base_filename}_connections.csv', index=False)
        
        # Save network matrix
        matrix_df = pd.DataFrame(
            network_data['network_matrix'], 
            index=self.node_labels, 
            columns=self.node_labels
        )
        matrix_df.to_csv(f'{base_filename}_matrix.csv')
        
        # Save node connectivity analysis
        conn_data = []
        for node, data in network_data['network_analysis']['node_connections'].items():
            conn_data.append({'Node': node, **data})
        
        pd.DataFrame(conn_data).to_csv(f'{base_filename}_node_connectivity.csv', index=False)
        
        # Save thermal resistance analysis
        resistance_analysis, valid_connections = self.create_thermal_resistance_network(network_data)
        valid_connections.to_csv(f'{base_filename}_thermal_resistances.csv', index=False)
        
        print(f"\nNetwork analysis saved to:")
        print(f"  - {base_filename}_connections.csv")
        print(f"  - {base_filename}_matrix.csv") 
        print(f"  - {base_filename}_node_connectivity.csv")
        print(f"  - {base_filename}_thermal_resistances.csv")

# Example usage
if __name__ == "__main__":
    mapper = HeatFlowNetworkMapper()
    
    # Example temperatures with realistic differences
    example_temps = np.array([315.0] * len(config.labels))
    example_temps[config.labels.index('ESC')] = 390.0
    example_temps[config.labels.index('Internal_air')] = 318.0
    example_temps[config.labels.index('Top_Shell_Ext')] = 310.0
    example_temps[config.labels.index('Bot_Shell_Ext')] = 310.0
    example_temps[config.labels.index('Top_Shell_Int')] = 312.0
    example_temps[config.labels.index('Bot_Shell_Int')] = 312.0
    
    # Create heat flow network
    network_data = mapper.create_heat_flow_network(example_temps)
    
    # Print summary report
    summary_report = mapper.create_network_summary_report(network_data)
    print(summary_report)
    
    # Analyze thermal resistances
    resistance_analysis, valid_connections = mapper.create_thermal_resistance_network(network_data)
    
    print(f"\n=== THERMAL RESISTANCE ANALYSIS BY MODE ===")
    for mode, data in resistance_analysis.items():
        print(f"\n{mode}:")
        print(f"  Connections:        {data['count']}")
        print(f"  Mean Resistance:    {data['mean_resistance']:.3f} K/W")
        print(f"  Min Resistance:     {data['min_resistance']:.3f} K/W")
        print(f"  Max Resistance:     {data['max_resistance']:.3f} K/W")
        print(f"  Mean Conductance:   {data['mean_conductance']:.3f} W/K")
    
    # Save all results
    mapper.save_network_analysis(network_data)
    
    print(f"\nHeat flow network mapping completed!")
