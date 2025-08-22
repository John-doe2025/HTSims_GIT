# comprehensive_heat_analyzer.py
# Complete Heat Transfer Analysis combining Conduction, Convection, and Radiation

import sys
import os
import numpy as np
import pandas as pd

# Add the code_vM_1 directory to the path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'code_vM_1'))
import config
import physics_models
import environment_model
import simulation_fix

from conduction_analyzer import ConductionAnalyzer
from convection_analyzer import ConvectionAnalyzer
from radiation_analyzer import RadiationAnalyzer

class ComprehensiveHeatAnalyzer:
    """
    Comprehensive heat transfer analysis combining all modes for complete node-by-node breakdown
    """
    
    def __init__(self):
        self.conduction_analyzer = ConductionAnalyzer()
        self.convection_analyzer = ConvectionAnalyzer()
        self.radiation_analyzer = RadiationAnalyzer()
        self.node_labels = config.labels
    
    def run_simulation_and_analyze(self, altitude_km=20.0, velocity_ms=0.0, simulation_time=3600.0):
        """
        Run thermal simulation and analyze all heat transfer modes
        """
        print(f"Running simulation at {altitude_km} km altitude, {velocity_ms} m/s velocity...")
        
        # Set up simulation parameters
        config.altitude = altitude_km * 1000  # Convert to meters
        config.velocity = velocity_ms
        
        # Initialize environment
        environment_model.init()
        
        # Run simulation
        try:
            sol = simulation_fix.run_simulation(T_total=simulation_time)
            final_temps = sol.y[:, -1]  # Final temperatures
            
            print(f"Simulation completed successfully.")
            print(f"Temperature range: {final_temps.min():.1f}K to {final_temps.max():.1f}K")
            
        except Exception as e:
            print(f"Simulation failed: {e}")
            # Use example temperatures for analysis
            final_temps = self._get_example_temperatures()
            print("Using example temperatures for analysis...")
        
        return self.analyze_heat_transfer(final_temps)
    
    def analyze_heat_transfer(self, temperatures):
        """
        Analyze all heat transfer modes for given temperatures
        """
        print("\nAnalyzing heat transfer modes...")
        
        # Analyze each mode
        node_conduction, conduction_flows = self.conduction_analyzer.analyze_conduction_flows(temperatures)
        node_convection = self.convection_analyzer.analyze_convection_flows(temperatures)
        node_radiation, radiation_flows = self.radiation_analyzer.analyze_radiation_flows(temperatures)
        
        # Combine results
        combined_results = self._combine_heat_transfer_modes(node_conduction, node_convection, node_radiation, temperatures)
        
        return {
            'temperatures': temperatures,
            'combined_results': combined_results,
            'conduction_flows': conduction_flows,
            'radiation_flows': radiation_flows,
            'node_convection': node_convection
        }
    
    def _combine_heat_transfer_modes(self, node_conduction, node_convection, node_radiation, temperatures):
        """
        Combine all heat transfer modes into comprehensive node analysis
        """
        combined = {}
        
        for i, node in enumerate(self.node_labels):
            # Conduction data
            cond_gain = node_conduction[node]['net_gain']
            cond_loss = node_conduction[node]['net_loss']
            cond_net = node_conduction[node]['net_total']
            
            # Convection data
            conv_flow = node_convection[node]['heat_flow']
            conv_gain = max(0, conv_flow)
            conv_loss = abs(min(0, conv_flow))
            conv_net = conv_flow
            
            # Radiation data
            rad_gain = node_radiation[node]['net_gain']
            rad_loss = node_radiation[node]['net_loss']
            rad_net = node_radiation[node]['net_total']
            
            # Total heat flows
            total_gain = cond_gain + conv_gain + rad_gain
            total_loss = cond_loss + conv_loss + rad_loss
            total_net = cond_net + conv_net + rad_net
            
            combined[node] = {
                'temperature_K': temperatures[i],
                'temperature_C': temperatures[i] - 273.15,
                
                # By mode
                'conduction_gain_W': cond_gain,
                'conduction_loss_W': cond_loss,
                'conduction_net_W': cond_net,
                
                'convection_gain_W': conv_gain,
                'convection_loss_W': conv_loss,
                'convection_net_W': conv_net,
                'convection_target': node_convection[node]['target'],
                
                'radiation_gain_W': rad_gain,
                'radiation_loss_W': rad_loss,
                'radiation_net_W': rad_net,
                
                # Totals
                'total_heat_gain_W': total_gain,
                'total_heat_loss_W': total_loss,
                'net_heat_flow_W': total_net,
                
                # Dominant modes
                'dominant_gain_mode': self._get_dominant_mode(cond_gain, conv_gain, rad_gain),
                'dominant_loss_mode': self._get_dominant_mode(cond_loss, conv_loss, rad_loss),
                'dominant_net_mode': self._get_dominant_mode(abs(cond_net), abs(conv_net), abs(rad_net))
            }
        
        return combined
    
    def _get_dominant_mode(self, cond, conv, rad):
        """Determine which heat transfer mode is dominant"""
        modes = {'Conduction': cond, 'Convection': conv, 'Radiation': rad}
        return max(modes, key=modes.get) if max(modes.values()) > 0.001 else 'None'
    
    def _get_example_temperatures(self):
        """Generate example temperatures for testing"""
        temps = np.array([315.0] * len(self.node_labels))
        temps[self.node_labels.index('ESC')] = 390.0
        temps[self.node_labels.index('Internal_air')] = 318.0
        temps[self.node_labels.index('Top_Shell_Ext')] = 310.0
        temps[self.node_labels.index('Bot_Shell_Ext')] = 310.0
        return temps
    
    def create_comprehensive_summary(self, combined_results):
        """Create comprehensive summary table"""
        summary_data = []
        
        for node in self.node_labels:
            data = combined_results[node]
            summary_data.append({
                'Node': node,
                'Temp_K': data['temperature_K'],
                'Temp_C': data['temperature_C'],
                'Conduction_Net_W': data['conduction_net_W'],
                'Convection_Net_W': data['convection_net_W'],
                'Radiation_Net_W': data['radiation_net_W'],
                'Total_Net_W': data['net_heat_flow_W'],
                'Total_Gain_W': data['total_heat_gain_W'],
                'Total_Loss_W': data['total_heat_loss_W'],
                'Dominant_Gain_Mode': data['dominant_gain_mode'],
                'Dominant_Loss_Mode': data['dominant_loss_mode']
            })
        
        return pd.DataFrame(summary_data)
    
    def create_detailed_node_reports(self, combined_results):
        """Create detailed reports for each node"""
        reports = {}
        
        for node in self.node_labels:
            data = combined_results[node]
            
            report = f"\n{'='*50}\n"
            report += f"COMPREHENSIVE HEAT ANALYSIS: {node}\n"
            report += f"{'='*50}\n"
            report += f"Temperature: {data['temperature_K']:.1f} K ({data['temperature_C']:.1f}°C)\n\n"
            
            # Heat gains by mode
            report += "HEAT GAINS BY MODE:\n"
            if data['conduction_gain_W'] > 0.001:
                report += f"  Conduction:     +{data['conduction_gain_W']:8.3f} W\n"
            if data['convection_gain_W'] > 0.001:
                report += f"  Convection:     +{data['convection_gain_W']:8.3f} W (from {data['convection_target']})\n"
            if data['radiation_gain_W'] > 0.001:
                report += f"  Radiation:      +{data['radiation_gain_W']:8.3f} W\n"
            
            if data['total_heat_gain_W'] > 0.001:
                report += f"  TOTAL GAIN:     +{data['total_heat_gain_W']:8.3f} W\n"
                report += f"  Dominant Mode:   {data['dominant_gain_mode']}\n\n"
            else:
                report += "  No significant heat gains\n\n"
            
            # Heat losses by mode
            report += "HEAT LOSSES BY MODE:\n"
            if data['conduction_loss_W'] > 0.001:
                report += f"  Conduction:     -{data['conduction_loss_W']:8.3f} W\n"
            if data['convection_loss_W'] > 0.001:
                report += f"  Convection:     -{data['convection_loss_W']:8.3f} W (to {data['convection_target']})\n"
            if data['radiation_loss_W'] > 0.001:
                report += f"  Radiation:      -{data['radiation_loss_W']:8.3f} W\n"
            
            if data['total_heat_loss_W'] > 0.001:
                report += f"  TOTAL LOSS:     -{data['total_heat_loss_W']:8.3f} W\n"
                report += f"  Dominant Mode:   {data['dominant_loss_mode']}\n\n"
            else:
                report += "  No significant heat losses\n\n"
            
            # Net result
            net_flow = data['net_heat_flow_W']
            if abs(net_flow) > 0.001:
                if net_flow > 0:
                    report += f"NET HEAT RESULT:    +{net_flow:8.3f} W (Overall Heat Gain)\n"
                else:
                    report += f"NET HEAT RESULT:    {net_flow:8.3f} W (Overall Heat Loss)\n"
                report += f"Dominant Net Mode:   {data['dominant_net_mode']}\n"
            else:
                report += f"NET HEAT RESULT:     {net_flow:8.3f} W (Thermal Equilibrium)\n"
            
            reports[node] = report
        
        return reports
    
    def print_summary_analysis(self, results):
        """Print key findings and summary"""
        combined = results['combined_results']
        summary_df = self.create_comprehensive_summary(combined)
        
        print("\n" + "="*80)
        print("COMPREHENSIVE HEAT TRANSFER ANALYSIS SUMMARY")
        print("="*80)
        
        # Temperature overview
        temps = [combined[node]['temperature_C'] for node in self.node_labels]
        print(f"\nTemperature Range: {min(temps):.1f}°C to {max(temps):.1f}°C")
        
        # Hottest and coldest nodes
        hottest_node = max(combined.keys(), key=lambda x: combined[x]['temperature_C'])
        coldest_node = min(combined.keys(), key=lambda x: combined[x]['temperature_C'])
        print(f"Hottest Node: {hottest_node} ({combined[hottest_node]['temperature_C']:.1f}°C)")
        print(f"Coldest Node: {coldest_node} ({combined[coldest_node]['temperature_C']:.1f}°C)")
        
        # Heat flow summary
        print(f"\n=== HEAT FLOW SUMMARY ===")
        total_gains = sum(combined[node]['total_heat_gain_W'] for node in self.node_labels)
        total_losses = sum(combined[node]['total_heat_loss_W'] for node in self.node_labels)
        print(f"Total Heat Gains:  {total_gains:8.1f} W")
        print(f"Total Heat Losses: {total_losses:8.1f} W")
        print(f"Heat Balance:      {total_gains - total_losses:8.1f} W")
        
        # Mode breakdown
        print(f"\n=== HEAT TRANSFER MODE BREAKDOWN ===")
        total_cond = sum(abs(combined[node]['conduction_net_W']) for node in self.node_labels)
        total_conv = sum(abs(combined[node]['convection_net_W']) for node in self.node_labels)
        total_rad = sum(abs(combined[node]['radiation_net_W']) for node in self.node_labels)
        total_all = total_cond + total_conv + total_rad
        
        if total_all > 0:
            print(f"Conduction: {total_cond:8.1f} W ({100*total_cond/total_all:.1f}%)")
            print(f"Convection: {total_conv:8.1f} W ({100*total_conv/total_all:.1f}%)")
            print(f"Radiation:  {total_rad:8.1f} W ({100*total_rad/total_all:.1f}%)")
        
        # Show top heat sources and sinks
        print(f"\n=== TOP HEAT SOURCES (Gains) ===")
        heat_sources = [(node, combined[node]['total_heat_gain_W']) for node in self.node_labels]
        heat_sources.sort(key=lambda x: x[1], reverse=True)
        for i, (node, gain) in enumerate(heat_sources[:5]):
            if gain > 0.1:
                print(f"{i+1}. {node:15}: +{gain:7.2f} W")
        
        print(f"\n=== TOP HEAT SINKS (Losses) ===")
        heat_sinks = [(node, combined[node]['total_heat_loss_W']) for node in self.node_labels]
        heat_sinks.sort(key=lambda x: x[1], reverse=True)
        for i, (node, loss) in enumerate(heat_sinks[:5]):
            if loss > 0.1:
                print(f"{i+1}. {node:15}: -{loss:7.2f} W")
        
        return summary_df

# Example usage
if __name__ == "__main__":
    analyzer = ComprehensiveHeatAnalyzer()
    
    # Run analysis
    results = analyzer.run_simulation_and_analyze(altitude_km=20.0, velocity_ms=0.0, simulation_time=1800.0)
    
    # Print summary
    summary_df = analyzer.print_summary_analysis(results)
    
    # Show detailed reports for key nodes
    reports = analyzer.create_detailed_node_reports(results['combined_results'])
    
    key_nodes = ['ESC', 'Top_Shell_Ext', 'Batt_BF_Top', 'Internal_air']
    for node in key_nodes:
        print(reports[node])
    
    # Save comprehensive results
    summary_df.to_csv('comprehensive_heat_analysis.csv', index=False)
    print(f"\nComprehensive analysis saved to: comprehensive_heat_analysis.csv")
