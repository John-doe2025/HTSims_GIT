# robust_sensitivity_analyzer.py
# Robust Sensitivity Analysis for 20-Node UAV Thermal Model with Error Handling

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

class RobustThermalSensitivityAnalyzer:
    """
    Robust sensitivity analysis with error handling and adaptive perturbation sizing
    """
    
    def __init__(self, base_perturbation=0.05):  # Start with 5% instead of 10%
        self.base_perturbation = base_perturbation
        self.results = {}
        self.successful_runs = {}
        self.failed_runs = {}
        
        # Initialize environment
        environment_model.init()
    
    def get_baseline_parameters(self):
        """Get baseline parameter values for sensitivity analysis"""
        return {
            # Heat generation
            'Q_batt_zone': config.Q_batt_zone,
            'Q_ESC': config.Q_ESC,
            
            # Material properties
            'k_eff_batt': config.k_eff_batt,
            'k_cfrp': config.k_cfrp,
            'k_mount': config.k_mount,
            'k_plate': config.k_plate,
            'k_bulkhead': config.k_bulkhead,
            
            # Heat transfer areas (critical for convection)
            'A_conv_batt_top': config.A_conv_batt_top,
            'A_conv_esc_top': config.A_conv_esc_top,
            
            # Environmental/operational
            'internal_air_velocity': config.internal_air_velocity,
            'initial_temp_K': config.initial_temp_K,
        }
    
    def run_simulation_safely(self, modified_params=None, max_retries=3):
        """
        Run simulation with error handling and retry logic
        """
        original_values = {}
        
        # Apply parameter modifications if provided
        if modified_params:
            for param, value in modified_params.items():
                if hasattr(config, param):
                    original_values[param] = getattr(config, param)
                    setattr(config, param, value)
        
        # Change to simulation directory
        original_dir = os.getcwd()
        sim_dir = os.path.join(os.path.dirname(__file__), '..', 'code_vM_1')
        os.chdir(sim_dir)
        
        for attempt in range(max_retries):
            try:
                from simulation_fix import f
                
                # Create initial temperature array
                x0 = np.array([config.initial_temp_K] * len(config.labels))
                
                # Try different solver tolerances if needed
                if attempt == 0:
                    rtol, atol = 1e-6, 1e-8
                elif attempt == 1:
                    rtol, atol = 1e-5, 1e-7
                else:
                    rtol, atol = 1e-4, 1e-6
                
                # Run simulation with shorter time if needed
                sim_time = config.T_total if attempt == 0 else config.T_total * 0.5
                
                sol = solve_ivp(f, [0, sim_time], x0, 
                               method='BDF', rtol=rtol, atol=atol,
                               max_step=3600)  # Limit max step size
                
                if sol.success:
                    # Calculate figures of merit
                    fom = self.calculate_figures_of_merit(sol.y, sol.t)
                    return fom, sol.y[:, -1], True, None
                else:
                    continue
                    
            except Exception as e:
                if attempt == max_retries - 1:
                    return None, None, False, str(e)
                continue
            finally:
                os.chdir(original_dir)
                # Restore original parameter values
                if modified_params:
                    for param, original_value in original_values.items():
                        setattr(config, param, original_value)
        
        return None, None, False, "Max retries exceeded"
    
    def calculate_figures_of_merit(self, temps, times):
        """Calculate key figures of merit from simulation results"""
        # Use final 10% of simulation for steady-state analysis
        analysis_start_idx = max(1, int(0.9 * len(times)))
        final_temps = temps[:, analysis_start_idx:]
        
        fom = {}
        
        # Node-specific temperatures
        for i, node in enumerate(config.labels):
            fom[f'max_temp_{node}'] = np.max(final_temps[i, :])
            fom[f'avg_temp_{node}'] = np.mean(final_temps[i, :])
        
        # System-level metrics
        battery_nodes = [i for i, node in enumerate(config.labels) if 'Batt' in node]
        if battery_nodes:
            battery_temps = [np.max(final_temps[i, :]) for i in battery_nodes]
            fom['max_battery_temp'] = max(battery_temps)
            fom['avg_battery_temp'] = np.mean(battery_temps)
            fom['battery_temp_spread'] = max(battery_temps) - min(battery_temps)
        
        fom['max_esc_temp'] = fom.get('max_temp_ESC', 0)
        fom['max_system_temp'] = np.max(final_temps)
        fom['avg_system_temp'] = np.mean(final_temps)
        
        return fom
    
    def run_parameter_sensitivity_analysis(self):
        """
        Run comprehensive parameter sensitivity analysis with robust error handling
        """
        print("=== ROBUST SENSITIVITY ANALYSIS ===")
        parameters_to_test = self.get_baseline_parameters()
        print(f"Testing {len(parameters_to_test)} parameters with ±{self.base_perturbation*100:.1f}% perturbation")
        print(f"Analyzing {len(config.labels)} thermal nodes\n")
        
        # Run baseline simulation
        print("Running baseline simulation...")
        baseline_fom, baseline_temps, success, error = self.run_simulation_safely()
        
        if not success:
            raise RuntimeError(f"Baseline simulation failed: {error}")
        
        print(f"[OK] Baseline successful")
        print(f"  Max battery temp: {baseline_fom['max_battery_temp']:.2f} K ({baseline_fom['max_battery_temp']-273.15:.1f}C)")
        print(f"  Max ESC temp: {baseline_fom['max_esc_temp']:.2f} K ({baseline_fom['max_esc_temp']-273.15:.1f}C)")
        print(f"  Max system temp: {baseline_fom['max_system_temp']:.2f} K ({baseline_fom['max_system_temp']-273.15:.1f}C)\n")
        
        # Store baseline
        self.results['baseline'] = baseline_fom
        
        # Parameter sensitivity analysis
        sensitivity_data = []
        successful_params = []
        
        for param_name, baseline_value in parameters_to_test.items():
            print(f"Testing parameter: {param_name}")
            param_success_count = 0
            
            # Test positive and negative perturbations
            for direction in ['positive', 'negative']:
                perturbation = self.base_perturbation if direction == 'positive' else -self.base_perturbation
                perturbed_value = baseline_value * (1 + perturbation)
                
                # Skip if perturbation would create invalid values
                if perturbed_value <= 0 and baseline_value > 0:
                    print(f"  [WARN] Skipping {direction} perturbation (would create negative value)")
                    continue
                
                modified_params = {param_name: perturbed_value}
                fom, temps, success, error = self.run_simulation_safely(modified_params)
                
                if success:
                    param_success_count += 1
                    
                    # Calculate sensitivity for key metrics
                    for metric in ['max_battery_temp', 'max_esc_temp', 'max_system_temp']:
                        if metric in baseline_fom and metric in fom:
                            baseline_val = baseline_fom[metric]
                            perturbed_val = fom[metric]
                            
                            # Calculate normalized sensitivity: (ΔT/T_baseline) / (Δparam/param_baseline)
                            if baseline_val > 0 and baseline_value > 0:
                                temp_change_pct = (perturbed_val - baseline_val) / baseline_val
                                param_change_pct = perturbation
                                sensitivity = temp_change_pct / param_change_pct
                                
                                sensitivity_data.append({
                                    'Parameter': param_name,
                                    'Direction': direction,
                                    'Metric': metric,
                                    'Baseline_Value': baseline_val,
                                    'Perturbed_Value': perturbed_val,
                                    'Temperature_Change_K': perturbed_val - baseline_val,
                                    'Temperature_Change_Pct': temp_change_pct * 100,
                                    'Parameter_Change_Pct': perturbation * 100,
                                    'Sensitivity': sensitivity,
                                    'Abs_Sensitivity': abs(sensitivity)
                                })
                    
                    print(f"  [OK] {direction} perturbation successful")
                else:
                    print(f"  [FAIL] {direction} perturbation failed: {error}")
                    self.failed_runs[f"{param_name}_{direction}"] = error
            
            if param_success_count > 0:
                successful_params.append(param_name)
                self.successful_runs[param_name] = param_success_count
            
            print()
        
        # Create results DataFrame
        if sensitivity_data:
            sensitivity_df = pd.DataFrame(sensitivity_data)
            
            # Create summary statistics
            summary_stats = sensitivity_df.groupby(['Parameter', 'Metric']).agg({
                'Abs_Sensitivity': ['mean', 'max'],
                'Temperature_Change_K': ['mean', 'std']
            }).round(4)
            
            print(f"=== SENSITIVITY ANALYSIS RESULTS ===")
            print(f"Successful parameters: {len(successful_params)}/{len(parameters_to_test)}")
            print(f"Failed runs: {len(self.failed_runs)}")
            print(f"Total sensitivity measurements: {len(sensitivity_data)}\n")
            
            # Show top sensitivities
            top_sensitivities = sensitivity_df.nlargest(10, 'Abs_Sensitivity')[
                ['Parameter', 'Metric', 'Temperature_Change_K', 'Sensitivity']
            ]
            print("Top 10 Parameter Sensitivities:")
            print(top_sensitivities.to_string(index=False))
            
            # Create visualizations
            self.create_sensitivity_visualizations(sensitivity_df)
            
            # Save detailed results
            sensitivity_df.to_csv('robust_sensitivity_results.csv', index=False)
            summary_stats.to_csv('sensitivity_summary_stats.csv')
            
            return sensitivity_df
        else:
            print("[WARN] No successful sensitivity measurements obtained")
            return pd.DataFrame()
    
    def create_sensitivity_visualizations(self, sensitivity_df):
        """Create comprehensive sensitivity visualizations"""
        
        # 1. Parameter sensitivity ranking
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        
        # Top sensitivities by parameter
        param_sensitivity = sensitivity_df.groupby('Parameter')['Abs_Sensitivity'].mean().sort_values(ascending=False)
        axes[0,0].barh(range(len(param_sensitivity)), param_sensitivity.values)
        axes[0,0].set_yticks(range(len(param_sensitivity)))
        axes[0,0].set_yticklabels(param_sensitivity.index)
        axes[0,0].set_xlabel('Average Absolute Sensitivity')
        axes[0,0].set_title('Parameter Sensitivity Ranking')
        
        # Temperature change distribution
        axes[0,1].hist(sensitivity_df['Temperature_Change_K'], bins=20, alpha=0.7, edgecolor='black')
        axes[0,1].set_xlabel('Temperature Change (K)')
        axes[0,1].set_ylabel('Frequency')
        axes[0,1].set_title('Distribution of Temperature Changes')
        axes[0,1].axvline(0, color='red', linestyle='--', alpha=0.7)
        
        # Sensitivity by metric
        metric_sensitivity = sensitivity_df.groupby('Metric')['Abs_Sensitivity'].mean()
        axes[1,0].bar(metric_sensitivity.index, metric_sensitivity.values)
        axes[1,0].set_ylabel('Average Absolute Sensitivity')
        axes[1,0].set_title('Sensitivity by Temperature Metric')
        axes[1,0].tick_params(axis='x', rotation=45)
        
        # Success rate by parameter
        success_rates = []
        param_names = []
        for param in sensitivity_df['Parameter'].unique():
            total_attempts = len(sensitivity_df[sensitivity_df['Parameter'] == param])
            success_rate = total_attempts / 2.0  # 2 directions per parameter
            success_rates.append(success_rate)
            param_names.append(param)
        
        axes[1,1].bar(param_names, success_rates)
        axes[1,1].set_ylabel('Success Rate')
        axes[1,1].set_title('Parameter Test Success Rate')
        axes[1,1].tick_params(axis='x', rotation=45)
        axes[1,1].set_ylim(0, 1)
        
        plt.tight_layout()
        plt.savefig('robust_sensitivity_analysis.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        # 2. Detailed sensitivity heatmap for successful parameters
        if len(sensitivity_df) > 0:
            # Create pivot table for heatmap
            heatmap_data = sensitivity_df.pivot_table(
                values='Abs_Sensitivity', 
                index='Parameter', 
                columns='Metric', 
                aggfunc='mean'
            )
            
            plt.figure(figsize=(10, 8))
            if HAS_SEABORN:
                sns.heatmap(heatmap_data, annot=True, fmt='.3f', cmap='YlOrRd')
            else:
                im = plt.imshow(heatmap_data.values, cmap='YlOrRd', aspect='auto')
                plt.colorbar(im)
                plt.xticks(range(len(heatmap_data.columns)), heatmap_data.columns)
                plt.yticks(range(len(heatmap_data.index)), heatmap_data.index)
            
            plt.title('Parameter Sensitivity Heatmap\n(Average Absolute Sensitivity)')
            plt.xlabel('Temperature Metrics')
            plt.ylabel('Parameters')
            plt.tight_layout()
            plt.savefig('sensitivity_heatmap_robust.png', dpi=300, bbox_inches='tight')
            plt.show()

# Example usage and testing
if __name__ == "__main__":
    analyzer = RobustThermalSensitivityAnalyzer(base_perturbation=0.05)
    
    try:
        sensitivity_df = analyzer.run_parameter_sensitivity_analysis()
        
        if not sensitivity_df.empty:
            print(f"\n=== ANALYSIS COMPLETE ===")
            print(f"Results saved to:")
            print("  - robust_sensitivity_results.csv")
            print("  - sensitivity_summary_stats.csv") 
            print("  - robust_sensitivity_analysis.png")
            print("  - sensitivity_heatmap_robust.png")
        else:
            print("\n[WARN] Analysis completed but no successful measurements obtained")
            print("Consider:")
            print("  - Reducing perturbation size further")
            print("  - Checking parameter bounds")
            print("  - Reviewing simulation stability")
            
    except Exception as e:
        print(f"\n[ERROR] Analysis failed: {e}")
        print("Check simulation setup and parameter definitions")
