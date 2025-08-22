# comprehensive_sensitivity_analyzer.py
# Advanced Sensitivity Analysis for 20-Node UAV Thermal Model (code_vM_1)

import time
import copy
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

from scipy.integrate import solve_ivp

# Add the code_vM_1 directory to the path to import modules
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'code_vM_1'))
import config
import physics_models
import environment_model

class ThermalSensitivityAnalyzer:
    """
    Comprehensive sensitivity analysis for the 20-node UAV thermal model.
    Analyzes both parameter sensitivity and heat transfer mode contributions.
    """
    
    def __init__(self):
        self.baseline_params = self._get_baseline_parameters()
        self.node_labels = config.labels
        self.perturbation = 0.10  # 10% perturbation
        self.results = {}
        
        # Initialize environment
        environment_model.init()
        
    def _get_baseline_parameters(self):
        """Extract all relevant parameters from config for sensitivity analysis"""
        return {
            # Heat generation
            'Q_batt_zone': config.Q_batt_zone,
            'Q_ESC': config.Q_ESC,
            
            # Material properties
            'k_eff_batt': config.k_eff_batt,
            'k_cfrp': config.k_cfrp,
            'k_mount': config.k_mount,
            'k_bulkhead': config.k_bulkhead,
            
            # Thermal masses
            'm_batt_zone': config.m_batt_zone,
            'm_ESC': config.m_ESC,
            'm_mount': config.m_mount,
            'm_bulkhead': config.m_bulkhead,
            'm_TS': config.m_TS,
            'm_BS': config.m_BS,
            'm_plate': config.m_plate,
            
            # Heat capacities
            'C_B': config.C_B,
            'C_ESC': config.C_ESC,
            'C_mount': config.C_mount,
            'C_bulkhead': config.C_bulkhead,
            'C_TS': config.C_TS,
            'C_BS': config.C_BS,
            'C_plate': config.C_plate,
            
            # Geometric parameters
            'A_conv_batt_top': config.A_conv_batt_top,
            'A_conv_batt_side': config.A_conv_batt_side,
            'A_conv_esc_top': config.A_conv_esc_top,
            'A_conv_esc_side': config.A_conv_esc_side,
            'A_mount_conv': config.A_mount_conv,
            'A_TS': config.A_TS,
            'A_BS': config.A_BS,
            'A_bulkhead_face': config.A_bulkhead_face,
            'A_conv_plate': config.A_conv_plate,
            'A_conv_plateM': config.A_conv_plateM,
            
            # Contact areas and path lengths
            'A_contact_ESC_Mount': config.A_contact_ESC_Mount,
            'A_contact_Mount_BH1': config.A_contact_Mount_BH1,
            'A_contact_BH_Shell': config.A_contact_BH_Shell,
            'A_contact_batt_plate': config.A_contact_batt_plate,
            'L_path_ESC_Mount': config.L_path_ESC_Mount,
            't_bulkhead': config.t_bulkhead,
            't_cfrp': config.t_cfrp,
            't_plate': config.t_plate,
            
            # Emissivities
            'emis_batt': config.emis_batt,
            'emis_esc': config.emis_esc,
            'emis_mount': config.emis_mount,
            'emis_shell_int': config.emis_shell_int,
            'emis_shell_ext': config.emis_shell_ext,
            'emis_bulkhead': config.emis_bulkhead,
            'emis_plate': config.emis_plate,
            
            # Environmental
            'TARGET_ALTITUDE_KM': config.TARGET_ALTITUDE_KM,
            'velocity': config.velocity,
            'internal_air_velocity': config.internal_air_velocity,
            'initial_temp_K': config.initial_temp_K,
            'T_total': config.T_total,
            
            # Internal air volume
            'V_internal_air': config.V_internal_air,
        }
    
    def run_simulation_with_heat_flows(self, modified_params=None):
        """
        Run the thermal simulation and extract heat flow information
        """
        # Apply parameter modifications if provided
        if modified_params:
            original_values = {}
            for param, value in modified_params.items():
                if hasattr(config, param):
                    original_values[param] = getattr(config, param)
                    setattr(config, param, value)
        
        # Change to the correct directory for simulation
        original_dir = os.getcwd()
        sim_dir = os.path.join(os.path.dirname(__file__), '..', 'code_vM_1')
        os.chdir(sim_dir)
        
        try:
            from simulation_fix import f
            
            # Run simulation
            x0 = np.array([config.initial_temp_K] * len(config.labels))
            sol = solve_ivp(f, [0, config.T_total], x0, 
                           method='BDF', rtol=1e-6, atol=1e-8)
            
            if not sol.success:
                raise RuntimeError(f"Simulation failed: {sol.message}")
            
            # Extract final temperatures
            final_temps = sol.y[:, -1]
            
            # Calculate figures of merit
            fom = self.calculate_figures_of_merit(sol.y, sol.t)
            
            # Calculate heat flows at final state
            heat_flows = self.calculate_heat_flows_at_state(final_temps)
            
            return fom, heat_flows, final_temps
            
        finally:
            os.chdir(original_dir)
            # Restore original parameter values
            if modified_params:
                for param, original_value in original_values.items():
                    setattr(config, param, original_value)
        
    def calculate_figures_of_merit(self, temps, times):
        """
        Calculate various figures of merit from simulation results
        """
        # Analyze final 20% of simulation for steady-state values
        analysis_start_idx = int(0.8 * len(times))
        final_temps = temps[:, analysis_start_idx:]
        
        fom = {}
        
        # Maximum temperatures for each node
        for i, node in enumerate(config.labels):
            fom[f'max_temp_{node}'] = np.max(final_temps[i, :])
            fom[f'avg_temp_{node}'] = np.mean(final_temps[i, :])
        
        # Overall system metrics
        fom['max_battery_temp'] = max([fom[f'max_temp_{node}'] for node in config.labels if 'Batt' in node])
        fom['max_esc_temp'] = fom['max_temp_ESC']
        fom['max_system_temp'] = np.max(final_temps)
        fom['avg_system_temp'] = np.mean(final_temps)
        
        # Temperature gradients
        battery_temps = [fom[f'max_temp_{node}'] for node in config.labels if 'Batt' in node]
        fom['battery_temp_spread'] = max(battery_temps) - min(battery_temps)
        
        return fom
    
    def calculate_heat_flows_at_state(self, temperatures):
        """
        Calculate heat flows at a given temperature state using the heat flow analyzer
        """
        from heat_flow_analyzer import HeatFlowAnalyzer
        import pandas as pd
        
        # Create heat flow analyzer instance
        flow_analyzer = HeatFlowAnalyzer()
        
        # Get environment conditions from altitude data (same as simulation_fix.py)
        try:
            df_alt = pd.read_excel("altitude_data.xlsx")
            row = df_alt.iloc[(df_alt['Altitude'] - config.TARGET_ALTITUDE_KM).abs().idxmin()]
            T_E = row['Temperature']
            P_amb = row['Pressure']
        except Exception as e:
            # Fallback values
            T_E = 287.3  # Standard atmosphere at low altitude
            P_amb = 99832.0
        
        # Analyze heat flows
        heat_flows = flow_analyzer.analyze_heat_flows_at_state(temperatures, T_E, P_amb)
        
        return heat_flows
    
    def _calculate_detailed_heat_flows(self, temps):
        """
        Calculate detailed heat flow breakdown for each node
        """
        # Create temperature dictionary
        temp_dict = {label: temps[i] for i, label in enumerate(config.labels)}
        
        # This would contain detailed heat flow calculations
        # For now, return a placeholder structure
        heat_flows = {}
        
        for i, node in enumerate(config.labels):
            heat_flows[node] = {
                'generation': 0.0,
                'conduction_in': 0.0,
                'conduction_out': 0.0,
                'convection': 0.0,
                'radiation_in': 0.0,
                'radiation_out': 0.0,
                'net_heat_flow': 0.0
            }
            
            # Add heat generation for relevant nodes
            if 'Batt' in node:
                heat_flows[node]['generation'] = config.Q_batt_zone
            elif node == 'ESC':
                heat_flows[node]['generation'] = config.Q_ESC
        
        return heat_flows
    
    def _calculate_figures_of_merit(self, sol):
        """
        Calculate various figures of merit from simulation results
        """
        # Analyze final 20% of simulation for steady-state values
        analysis_start_idx = int(0.8 * len(sol.t))
        final_temps = sol.y[:, analysis_start_idx:]
        
        fom = {}
        
        # Maximum temperatures for each node
        for i, node in enumerate(config.labels):
            fom[f'max_temp_{node}'] = np.max(final_temps[i, :])
            fom[f'avg_temp_{node}'] = np.mean(final_temps[i, :])
        
        # Overall system metrics
        fom['max_battery_temp'] = max([fom[f'max_temp_{node}'] for node in config.labels if 'Batt' in node])
        fom['max_esc_temp'] = fom['max_temp_ESC']
        fom['max_system_temp'] = np.max(final_temps)
        fom['avg_system_temp'] = np.mean(final_temps)
        
        # Temperature gradients
        battery_temps = [fom[f'max_temp_{node}'] for node in config.labels if 'Batt' in node]
        fom['battery_temp_spread'] = max(battery_temps) - min(battery_temps)
        
        return fom
    
    def run_parameter_sensitivity_analysis(self, parameters_to_test=None):
        """
        Run comprehensive parameter sensitivity analysis
        """
        if parameters_to_test is None:
            parameters_to_test = [
                'Q_batt_zone', 'Q_ESC', 'k_eff_batt', 'k_cfrp', 'k_mount',
                'A_conv_batt_top', 'A_conv_esc_top', 'emis_batt', 'emis_esc',
                'internal_air_velocity', 'V_internal_air', 'TARGET_ALTITUDE_KM'
            ]
        
        print("=== COMPREHENSIVE SENSITIVITY ANALYSIS ===")
        print(f"Testing {len(parameters_to_test)} parameters with ±{self.perturbation:.0%} perturbation")
        print(f"Analyzing {len(config.labels)} thermal nodes\n")
        
        # Run baseline simulation
        print("Running baseline simulation...")
        baseline_fom, baseline_flows, baseline_temps = self.run_simulation_with_heat_flows()
        
        print(f"Baseline max battery temp: {baseline_fom['max_battery_temp']:.2f} K")
        print(f"Baseline max ESC temp: {baseline_fom['max_esc_temp']:.2f} K")
        print(f"Baseline max system temp: {baseline_fom['max_system_temp']:.2f} K\n")
        
        # Store baseline results
        self.results['baseline'] = {
            'fom': baseline_fom,
            'flows': baseline_flows,
            'temps': baseline_temps
        }
        
        # Parameter sensitivity analysis
        sensitivity_results = []
        
        for param in parameters_to_test:
            if param not in self.baseline_params:
                print(f"Warning: Parameter {param} not found in baseline parameters")
                continue
                
            print(f"Testing parameter: {param}")
            
            # Test positive perturbation
            modified_params = {param: self.baseline_params[param] * (1 + self.perturbation)}
            fom_plus, _, _ = self.run_simulation_with_heat_flows(modified_params)
            
            # Test negative perturbation
            modified_params = {param: self.baseline_params[param] * (1 - self.perturbation)}
            fom_minus, _, _ = self.run_simulation_with_heat_flows(modified_params)
            
            # Calculate sensitivities for all figures of merit
            param_sensitivity = {'Parameter': param}
            
            for fom_name, baseline_value in baseline_fom.items():
                if baseline_value != 0:
                    sensitivity = ((fom_plus[fom_name] - fom_minus[fom_name]) / baseline_value) / (2 * self.perturbation)
                    param_sensitivity[f'{fom_name}_sensitivity'] = sensitivity * 100  # Convert to percentage
                    param_sensitivity[f'{fom_name}_plus'] = fom_plus[fom_name]
                    param_sensitivity[f'{fom_name}_minus'] = fom_minus[fom_name]
            
            sensitivity_results.append(param_sensitivity)
        
        # Convert to DataFrame for analysis
        self.sensitivity_df = pd.DataFrame(sensitivity_results)
        
        return self.sensitivity_df
    
    def analyze_heat_transfer_modes(self):
        """
        Analyze the contribution of different heat transfer modes for each node
        """
        print("\n=== HEAT TRANSFER MODE ANALYSIS ===")
        
        baseline_flows = self.results['baseline']['flows']
        
        mode_analysis = {}
        
        for node in config.labels:
            flows = baseline_flows[node]
            
            total_heat_in = flows['generation'] + flows['conduction_in'] + flows['radiation_in']
            total_heat_out = abs(flows['conduction_out']) + abs(flows['convection']) + abs(flows['radiation_out'])
            
            if total_heat_out > 0:
                mode_analysis[node] = {
                    'heat_generation_W': flows['generation'],
                    'conduction_out_pct': abs(flows['conduction_out']) / total_heat_out * 100,
                    'convection_out_pct': abs(flows['convection']) / total_heat_out * 100,
                    'radiation_out_pct': abs(flows['radiation_out']) / total_heat_out * 100,
                    'total_heat_out_W': total_heat_out
                }
        
        self.mode_analysis_df = pd.DataFrame.from_dict(mode_analysis, orient='index')
        
        return self.mode_analysis_df
    
    def generate_comprehensive_report(self):
        """
        Generate comprehensive sensitivity analysis report
        """
        print("\n" + "="*60)
        print("COMPREHENSIVE THERMAL SENSITIVITY ANALYSIS REPORT")
        print("="*60)
        
        # Key sensitivity metrics
        key_sensitivities = self.sensitivity_df[['Parameter', 'max_battery_temp_sensitivity', 
                                                'max_esc_temp_sensitivity', 'max_system_temp_sensitivity']].copy()
        key_sensitivities['max_abs_sensitivity'] = key_sensitivities[['max_battery_temp_sensitivity', 
                                                                     'max_esc_temp_sensitivity', 
                                                                     'max_system_temp_sensitivity']].abs().max(axis=1)
        
        key_sensitivities = key_sensitivities.sort_values('max_abs_sensitivity', ascending=False)
        
        print("\nTOP 10 MOST SENSITIVE PARAMETERS:")
        print("-" * 40)
        for i, row in key_sensitivities.head(10).iterrows():
            print(f"{row['Parameter']:20s} | Max Sensitivity: {row['max_abs_sensitivity']:6.2f}%")
        
        # Heat transfer mode summary
        if hasattr(self, 'mode_analysis_df'):
            print("\nHEAT TRANSFER MODE BREAKDOWN (Top Heat Generating Nodes):")
            print("-" * 60)
            heat_gen_nodes = self.mode_analysis_df[self.mode_analysis_df['heat_generation_W'] > 0].sort_values('heat_generation_W', ascending=False)
            
            for node, data in heat_gen_nodes.head(5).iterrows():
                print(f"\n{node}:")
                print(f"  Generation: {data['heat_generation_W']:6.1f} W")
                print(f"  Conduction: {data['conduction_out_pct']:5.1f}%")
                print(f"  Convection: {data['convection_out_pct']:5.1f}%")
                print(f"  Radiation:  {data['radiation_out_pct']:5.1f}%")
        
        return key_sensitivities
    
    def create_visualizations(self):
        """
        Create comprehensive visualization suite
        """
        # Set up the plotting style
        plt.style.use('default')
        sns.set_palette("husl")
        
        # 1. Parameter Sensitivity Heatmap
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        
        # Sensitivity heatmap for key temperatures
        key_cols = ['max_battery_temp_sensitivity', 'max_esc_temp_sensitivity', 'max_system_temp_sensitivity']
        heatmap_data = self.sensitivity_df[['Parameter'] + key_cols].set_index('Parameter')
        
        sns.heatmap(heatmap_data.T, annot=True, fmt='.2f', cmap='RdBu_r', center=0, 
                   ax=axes[0,0], cbar_kws={'label': 'Sensitivity (%)'})
        axes[0,0].set_title('Temperature Sensitivity Heatmap')
        axes[0,0].set_xlabel('Parameters')
        
        # 2. Top sensitivities bar chart
        top_params = heatmap_data.abs().max(axis=0).sort_values(ascending=True).tail(10)
        top_params.plot(kind='barh', ax=axes[0,1], color='steelblue')
        axes[0,1].set_title('Top 10 Most Sensitive Parameters')
        axes[0,1].set_xlabel('Max Absolute Sensitivity (%)')
        
        # 3. Heat transfer mode pie charts (if available)
        if hasattr(self, 'mode_analysis_df'):
            # Find the node with highest heat generation
            max_gen_node = self.mode_analysis_df.loc[self.mode_analysis_df['heat_generation_W'].idxmax()]
            
            modes = ['conduction_out_pct', 'convection_out_pct', 'radiation_out_pct']
            values = [max_gen_node[mode] for mode in modes]
            labels = ['Conduction', 'Convection', 'Radiation']
            
            axes[1,0].pie(values, labels=labels, autopct='%1.1f%%', startangle=90)
            axes[1,0].set_title(f'Heat Transfer Modes\n({max_gen_node.name})')
        
        # 4. Temperature distribution
        baseline_temps = self.results['baseline']['temps']
        axes[1,1].bar(range(len(baseline_temps)), baseline_temps - 273.15)
        axes[1,1].set_title('Baseline Temperature Distribution')
        axes[1,1].set_xlabel('Node Index')
        axes[1,1].set_ylabel('Temperature (°C)')
        axes[1,1].tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        plt.show()
        
        return fig

# Main execution
if __name__ == "__main__":
    analyzer = ThermalSensitivityAnalyzer()
    
    # Run comprehensive analysis
    sensitivity_df = analyzer.run_parameter_sensitivity_analysis()
    mode_df = analyzer.analyze_heat_transfer_modes()
    
    # Generate report
    summary = analyzer.generate_comprehensive_report()
    
    # Create visualizations
    analyzer.create_visualizations()
    
    # Save results
    output_dir = os.path.dirname(os.path.abspath(__file__))
    sensitivity_df.to_excel(os.path.join(output_dir, "comprehensive_sensitivity_results.xlsx"), index=False)
    mode_df.to_excel(os.path.join(output_dir, "heat_transfer_mode_analysis.xlsx"))
    
    print(f"\nResults saved to: {output_dir}")
    print("Analysis complete!")
