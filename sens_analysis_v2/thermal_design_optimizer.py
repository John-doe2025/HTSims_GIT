# thermal_design_optimizer.py
# Complete Thermal Design Optimization Analysis Tool

import sys
import os
import numpy as np
import pandas as pd

# Add the code_vM_1 directory to the path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'code_vM_1'))
import config
import physics_models
import environment_model

from comprehensive_heat_analyzer import ComprehensiveHeatAnalyzer
from heat_flow_network_mapper import HeatFlowNetworkMapper

class ThermalDesignOptimizer:
    """
    Complete thermal design optimization tool providing actionable insights
    """
    
    def __init__(self):
        self.heat_analyzer = ComprehensiveHeatAnalyzer()
        self.network_mapper = HeatFlowNetworkMapper()
        self.node_labels = config.labels
    
    def run_complete_thermal_analysis(self, temperatures=None):
        """
        Run complete thermal analysis and generate optimization recommendations
        """
        if temperatures is None:
            temperatures = self._get_realistic_temperatures()
        
        print("Running complete thermal design analysis...")
        
        # Run comprehensive heat analysis
        heat_results = self.heat_analyzer.analyze_heat_transfer(temperatures)
        
        # Run network mapping
        network_results = self.network_mapper.create_heat_flow_network(temperatures)
        
        # Generate optimization insights
        optimization_insights = self._generate_optimization_insights(heat_results, network_results)
        
        # Create design recommendations
        design_recommendations = self._create_design_recommendations(heat_results, network_results, optimization_insights)
        
        return {
            'heat_results': heat_results,
            'network_results': network_results,
            'optimization_insights': optimization_insights,
            'design_recommendations': design_recommendations,
            'temperatures': temperatures
        }
    
    def _get_realistic_temperatures(self):
        """Generate realistic temperature distribution for analysis"""
        temps = np.array([315.0] * len(self.node_labels))
        
        # Set realistic temperature differences based on heat sources and thermal paths
        temps[self.node_labels.index('ESC')] = 390.0  # Hot ESC
        temps[self.node_labels.index('ESC_Mount')] = 350.0  # Heated by ESC
        temps[self.node_labels.index('BH_1')] = 325.0  # Near ESC
        temps[self.node_labels.index('Internal_air')] = 318.0  # Slightly warm
        temps[self.node_labels.index('Top_Shell_Int')] = 312.0  # Cooler internal
        temps[self.node_labels.index('Bot_Shell_Int')] = 312.0  # Cooler internal
        temps[self.node_labels.index('Top_Shell_Ext')] = 290.0  # Cold external
        temps[self.node_labels.index('Bot_Shell_Ext')] = 290.0  # Cold external
        
        # Batteries slightly warmer than ambient
        for batt in ['Batt_BF_Top', 'Batt_BF_Bot', 'Batt_BM_Top', 'Batt_BM_Bot', 'Batt_BR_Top', 'Batt_BR_Bot']:
            temps[self.node_labels.index(batt)] = 320.0
        
        return temps
    
    def _generate_optimization_insights(self, heat_results, network_results):
        """
        Generate key thermal optimization insights
        """
        combined = heat_results['combined_results']
        network_analysis = network_results['network_analysis']
        
        insights = {}
        
        # 1. Thermal bottlenecks identification
        thermal_bottlenecks = []
        for node in self.node_labels:
            data = combined[node]
            if data['temperature_C'] > 85:  # High temperature threshold
                thermal_bottlenecks.append({
                    'node': node,
                    'temperature_C': data['temperature_C'],
                    'net_heat_W': data['net_heat_flow_W'],
                    'dominant_mode': data['dominant_net_mode']
                })
        
        insights['thermal_bottlenecks'] = sorted(thermal_bottlenecks, key=lambda x: x['temperature_C'], reverse=True)
        
        # 2. Heat path efficiency analysis
        connection_table = network_results['connection_table']
        
        # Find high resistance paths
        high_resistance_paths = connection_table[
            (connection_table['Thermal_Resistance'] > 10) & 
            (abs(connection_table['Heat_Flow_W']) > 1)
        ].copy()
        
        insights['high_resistance_paths'] = high_resistance_paths.sort_values('Heat_Flow_W', key=abs, ascending=False)
        
        # 3. Heat source/sink imbalances
        heat_sources = []
        heat_sinks = []
        
        for node in self.node_labels:
            data = combined[node]
            if data['net_heat_flow_W'] > 10:  # Significant heat source
                heat_sources.append({'node': node, 'heat_generation_W': data['net_heat_flow_W']})
            elif data['net_heat_flow_W'] < -10:  # Significant heat sink
                heat_sinks.append({'node': node, 'heat_rejection_W': abs(data['net_heat_flow_W'])})
        
        insights['major_heat_sources'] = sorted(heat_sources, key=lambda x: x['heat_generation_W'], reverse=True)
        insights['major_heat_sinks'] = sorted(heat_sinks, key=lambda x: x['heat_rejection_W'], reverse=True)
        
        # 4. Convection effectiveness analysis
        convection_effectiveness = []
        for node in self.node_labels:
            conv_data = heat_results['node_convection'][node]
            if abs(conv_data['heat_flow']) > 1:
                effectiveness = abs(conv_data['heat_flow']) / (conv_data['h_coefficient'] * conv_data['area']) if conv_data['h_coefficient'] * conv_data['area'] > 0 else 0
                convection_effectiveness.append({
                    'node': node,
                    'heat_flow_W': conv_data['heat_flow'],
                    'h_coefficient': conv_data['h_coefficient'],
                    'area_m2': conv_data['area'],
                    'effectiveness': effectiveness
                })
        
        insights['convection_effectiveness'] = sorted(convection_effectiveness, key=lambda x: abs(x['heat_flow_W']), reverse=True)
        
        return insights
    
    def _create_design_recommendations(self, heat_results, network_results, insights):
        """
        Create actionable design recommendations
        """
        recommendations = {
            'critical_actions': [],
            'thermal_management': [],
            'design_modifications': [],
            'material_optimization': []
        }
        
        # Critical temperature issues
        for bottleneck in insights['thermal_bottlenecks']:
            if bottleneck['temperature_C'] > 100:
                recommendations['critical_actions'].append({
                    'priority': 'HIGH',
                    'issue': f"{bottleneck['node']} overheating at {bottleneck['temperature_C']:.1f}°C",
                    'recommendation': f"Add dedicated cooling for {bottleneck['node']} - consider heat sink or forced convection",
                    'expected_benefit': 'Reduce temperature by 20-40°C'
                })
        
        # Thermal path improvements
        for _, path in insights['high_resistance_paths'].head(3).iterrows():
            recommendations['thermal_management'].append({
                'priority': 'MEDIUM',
                'issue': f"High thermal resistance between {path['From_Node']} and {path['To_Node']} ({path['Thermal_Resistance']:.2f} K/W)",
                'recommendation': f"Improve thermal interface - increase contact area or use thermal interface material",
                'expected_benefit': f"Reduce thermal resistance by 50-70%"
            })
        
        # Heat source management
        for source in insights['major_heat_sources'][:2]:
            if source['heat_generation_W'] > 100:
                recommendations['thermal_management'].append({
                    'priority': 'HIGH',
                    'issue': f"{source['node']} generating {source['heat_generation_W']:.1f}W",
                    'recommendation': f"Implement heat spreading from {source['node']} - add thermal spreader or heat pipe",
                    'expected_benefit': 'Distribute heat load and reduce peak temperatures'
                })
        
        # Convection improvements
        for conv in insights['convection_effectiveness'][:3]:
            if conv['h_coefficient'] < 5:  # Low convection coefficient
                recommendations['design_modifications'].append({
                    'priority': 'MEDIUM',
                    'issue': f"Low convection coefficient for {conv['node']} ({conv['h_coefficient']:.2f} W/m²K)",
                    'recommendation': f"Increase surface area or add fins to {conv['node']}",
                    'expected_benefit': 'Improve convection by 2-5x'
                })
        
        # Material optimization
        connection_table = network_results['connection_table']
        conduction_paths = connection_table[connection_table['Transfer_Mode'] == 'Conduction']
        
        for _, path in conduction_paths.head(3).iterrows():
            if path['Thermal_Resistance'] > 1:
                recommendations['material_optimization'].append({
                    'priority': 'LOW',
                    'issue': f"Conduction path {path['From_Node']} → {path['To_Node']} has resistance {path['Thermal_Resistance']:.2f} K/W",
                    'recommendation': f"Consider higher thermal conductivity material or larger cross-section",
                    'expected_benefit': 'Reduce conduction resistance by 30-50%'
                })
        
        return recommendations
    
    def generate_optimization_report(self, analysis_results):
        """
        Generate comprehensive optimization report
        """
        insights = analysis_results['optimization_insights']
        recommendations = analysis_results['design_recommendations']
        temps = analysis_results['temperatures']
        
        report = "\n" + "="*100 + "\n"
        report += "THERMAL DESIGN OPTIMIZATION REPORT\n"
        report += "="*100 + "\n"
        
        # Executive summary
        report += f"\nEXECUTIVE SUMMARY:\n"
        report += f"  Temperature Range: {(temps.min()-273.15):.1f}°C to {(temps.max()-273.15):.1f}°C\n"
        report += f"  Thermal Bottlenecks: {len(insights['thermal_bottlenecks'])} identified\n"
        report += f"  Critical Actions Required: {len(recommendations['critical_actions'])}\n"
        report += f"  Design Improvements: {len(recommendations['thermal_management']) + len(recommendations['design_modifications'])}\n"
        
        # Critical issues
        if insights['thermal_bottlenecks']:
            report += f"\n*** CRITICAL THERMAL ISSUES ***\n"
            for i, bottleneck in enumerate(insights['thermal_bottlenecks'][:5], 1):
                report += f"  {i}. {bottleneck['node']}: {bottleneck['temperature_C']:.1f}°C "
                report += f"(Net: {bottleneck['net_heat_W']:+.1f}W, Mode: {bottleneck['dominant_mode']})\n"
        
        # Major heat sources and sinks
        report += f"\n*** MAJOR HEAT SOURCES ***\n"
        for i, source in enumerate(insights['major_heat_sources'][:3], 1):
            report += f"  {i}. {source['node']}: +{source['heat_generation_W']:.1f}W\n"
        
        report += f"\n*** MAJOR HEAT SINKS ***\n"
        for i, sink in enumerate(insights['major_heat_sinks'][:3], 1):
            report += f"  {i}. {sink['node']}: -{sink['heat_rejection_W']:.1f}W\n"
        
        # Critical actions
        if recommendations['critical_actions']:
            report += f"\n*** CRITICAL ACTIONS REQUIRED ***\n"
            for i, action in enumerate(recommendations['critical_actions'], 1):
                report += f"  {i}. [{action['priority']}] {action['issue']}\n"
                report += f"     -> {action['recommendation']}\n"
                report += f"     -> Expected: {action['expected_benefit']}\n\n"
        
        # Design recommendations
        report += f"\n*** THERMAL MANAGEMENT RECOMMENDATIONS ***\n"
        for i, rec in enumerate(recommendations['thermal_management'][:5], 1):
            report += f"  {i}. [{rec['priority']}] {rec['issue']}\n"
            report += f"     -> {rec['recommendation']}\n"
            report += f"     -> Expected: {rec['expected_benefit']}\n\n"
        
        # Design modifications
        if recommendations['design_modifications']:
            report += f"\n*** DESIGN MODIFICATIONS ***\n"
            for i, mod in enumerate(recommendations['design_modifications'][:3], 1):
                report += f"  {i}. [{mod['priority']}] {mod['issue']}\n"
                report += f"     -> {mod['recommendation']}\n"
                report += f"     -> Expected: {mod['expected_benefit']}\n\n"
        
        # High resistance paths
        if len(insights['high_resistance_paths']) > 0:
            report += f"\n*** HIGH THERMAL RESISTANCE PATHS ***\n"
            for i, (_, path) in enumerate(insights['high_resistance_paths'].head(5).iterrows(), 1):
                report += f"  {i}. {path['From_Node']} -> {path['To_Node']}: "
                report += f"{path['Thermal_Resistance']:.2f} K/W ({path['Heat_Flow_W']:+.1f}W)\n"
        
        return report
    
    def save_optimization_results(self, analysis_results, base_filename='thermal_optimization'):
        """
        Save all optimization results to files
        """
        # Save optimization insights
        insights = analysis_results['optimization_insights']
        
        # Thermal bottlenecks
        if insights['thermal_bottlenecks']:
            pd.DataFrame(insights['thermal_bottlenecks']).to_csv(f'{base_filename}_bottlenecks.csv', index=False)
        
        # Heat sources and sinks
        if insights['major_heat_sources']:
            pd.DataFrame(insights['major_heat_sources']).to_csv(f'{base_filename}_heat_sources.csv', index=False)
        if insights['major_heat_sinks']:
            pd.DataFrame(insights['major_heat_sinks']).to_csv(f'{base_filename}_heat_sinks.csv', index=False)
        
        # High resistance paths
        if len(insights['high_resistance_paths']) > 0:
            insights['high_resistance_paths'].to_csv(f'{base_filename}_high_resistance_paths.csv', index=False)
        
        # Design recommendations
        recommendations = analysis_results['design_recommendations']
        all_recommendations = []
        
        for category, recs in recommendations.items():
            for rec in recs:
                all_recommendations.append({'Category': category, **rec})
        
        if all_recommendations:
            pd.DataFrame(all_recommendations).to_csv(f'{base_filename}_recommendations.csv', index=False)
        
        print(f"\nOptimization results saved to:")
        print(f"  - {base_filename}_bottlenecks.csv")
        print(f"  - {base_filename}_heat_sources.csv") 
        print(f"  - {base_filename}_heat_sinks.csv")
        print(f"  - {base_filename}_high_resistance_paths.csv")
        print(f"  - {base_filename}_recommendations.csv")

# Example usage
if __name__ == "__main__":
    optimizer = ThermalDesignOptimizer()
    
    # Run complete analysis
    analysis_results = optimizer.run_complete_thermal_analysis()
    
    # Generate and print optimization report
    optimization_report = optimizer.generate_optimization_report(analysis_results)
    print(optimization_report)
    
    # Save all results
    optimizer.save_optimization_results(analysis_results)
    
    print(f"\n*** Complete thermal design optimization analysis completed! ***")
    print(f"*** All analysis files have been generated for design review and optimization. ***")
