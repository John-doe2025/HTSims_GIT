# UAV Thermal Simulation Sensitivity Analysis Framework

## Overview
This document summarizes the comprehensive sensitivity analysis framework developed for the 20-node UAV thermal simulation model in `code_vM_1`. The framework provides detailed quantitative analysis of parameter sensitivity, heat transfer mode contributions, and error margins for thermal behavior insights.

## Framework Components

### 1. Heat Flow Analyzer (`heat_flow_analyzer.py`)
**Purpose**: Detailed analysis of heat transfer mechanisms for individual thermal states

**Key Features**:
- Calculates heat flows for all 20 thermal nodes at any temperature state
- Breaks down heat transfer into conduction, convection, and radiation components
- Provides thermal coefficient calculations for all node connections
- Generates comprehensive heat flow visualizations

**Key Methods**:
- `analyze_heat_flows_at_state()`: Main analysis function for temperature states
- `create_heat_flow_summary()`: Generates DataFrame summary of all heat flows
- `create_heat_flow_visualization()`: Creates 4-panel visualization plots

**Output**:
- Heat generation vs rejection analysis
- Heat transfer mode breakdowns (pie charts)
- Temperature distribution across nodes
- Net heat flow analysis

### 2. Comprehensive Sensitivity Analyzer (`comprehensive_sensitivity_analyzer.py`)
**Purpose**: Full parameter sensitivity analysis with heat flow integration

**Key Features**:
- Tests 12+ critical thermal parameters with ±10% perturbations
- Integrates with heat flow analyzer for detailed thermal insights
- Calculates multiple figures of merit (FOM) for system performance
- Generates sensitivity heatmaps and detailed reports

**Parameters Analyzed**:
- Heat generation: `Q_batt_zone`, `Q_ESC`
- Material properties: `k_eff_batt`, `k_cfrp`, `k_mount`, `k_plate`, `k_bulkhead`
- Heat transfer areas: `A_conv_batt_top`, `A_conv_esc_top`
- Operational: `internal_air_velocity`, `initial_temp_K`, `T_total`
- Environmental: `V_internal_air`

**Figures of Merit**:
- Maximum temperatures per node
- Average temperatures per node
- System-level metrics (max battery, ESC, system temps)
- Temperature gradients and spreads

### 3. Robust Sensitivity Analyzer (`robust_sensitivity_analyzer.py`)
**Purpose**: Numerically stable sensitivity analysis with error handling

**Key Features**:
- Uses smaller perturbations (±5%) to avoid numerical instabilities
- Implements retry logic with adaptive solver tolerances
- Comprehensive error handling and reporting
- Success rate tracking for parameter tests

**Robustness Measures**:
- Multiple solver tolerance levels (1e-6 to 1e-4)
- Adaptive simulation time reduction for unstable cases
- Maximum step size limiting (3600s)
- Parameter bounds checking to avoid invalid values

**Error Handling**:
- Tracks successful vs failed parameter tests
- Reports specific failure reasons
- Provides recommendations for parameter adjustment
- Generates partial results even with some failures

## Analysis Capabilities

### Heat Transfer Mode Analysis
The framework quantifies the contribution of each heat transfer mechanism:

1. **Conduction**: 
   - Battery-to-plate connections
   - ESC-to-mount thermal paths
   - Bulkhead-to-shell connections
   - Internal structural heat paths

2. **Convection**:
   - Internal air circulation (natural/forced)
   - External ambient convection
   - Temperature-dependent heat transfer coefficients

3. **Radiation**:
   - Internal surface-to-surface radiation
   - External environmental radiation loads
   - Temperature^4 dependent heat exchange

### Parameter Sensitivity Metrics
For each parameter perturbation, the framework calculates:

- **Absolute Temperature Changes**: ΔT in Kelvin
- **Relative Temperature Changes**: ΔT/T_baseline as percentage
- **Normalized Sensitivity**: (ΔT/T_baseline) / (Δparam/param_baseline)
- **Error Margins**: Statistical analysis of temperature variations

### Visualization Outputs
The framework generates multiple visualization types:

1. **Sensitivity Heatmaps**: Parameter vs node temperature sensitivity
2. **Heat Flow Diagrams**: Visual representation of thermal paths
3. **Temperature Distributions**: Spatial temperature mapping
4. **Parameter Ranking**: Ordered sensitivity importance
5. **Success Rate Charts**: Numerical stability assessment

## Usage Examples

### Basic Heat Flow Analysis
```python
from heat_flow_analyzer import HeatFlowAnalyzer

analyzer = HeatFlowAnalyzer()
# Example temperatures from simulation
temperatures = np.array([315.0] * 20)  # All nodes at 315K
heat_flows = analyzer.analyze_heat_flows_at_state(
    temperatures, ambient_temp=287.3, ambient_pressure=99832.0
)
summary_df = analyzer.create_heat_flow_summary(heat_flows)
```

### Robust Sensitivity Analysis
```python
from robust_sensitivity_analyzer import RobustThermalSensitivityAnalyzer

analyzer = RobustThermalSensitivityAnalyzer(base_perturbation=0.05)
sensitivity_df = analyzer.run_parameter_sensitivity_analysis()
```

## Key Findings and Insights

### Thermal Behavior Characteristics
Based on baseline analysis:
- **Battery temperatures**: ~315-316K (42-43°C) under normal conditions
- **ESC temperature**: ~397K (124°C) - highest temperature component
- **Internal air**: Acts as thermal buffer, ~318K (45°C)
- **Heat rejection**: Primarily through external shell convection

### Critical Parameters Identified
Most sensitive parameters affecting thermal performance:
1. **Heat generation rates** (`Q_batt_zone`, `Q_ESC`)
2. **Convection areas** (`A_conv_batt_top`, `A_conv_esc_top`)
3. **Material conductivities** (`k_eff_batt`, `k_cfrp`)
4. **Internal air circulation** (`internal_air_velocity`)

### Numerical Stability Insights
- Large perturbations (±10%) can cause solver instabilities
- High-altitude conditions exacerbate numerical issues
- Smaller perturbations (±5%) provide more reliable results
- Adaptive solver tolerances essential for robust analysis

## File Structure
```
sens_analysis_v2/
├── heat_flow_analyzer.py              # Individual state heat flow analysis
├── comprehensive_sensitivity_analyzer.py  # Full sensitivity framework
├── robust_sensitivity_analyzer.py     # Numerically stable version
├── SENSITIVITY_ANALYSIS_SUMMARY.md    # This documentation
└── [Generated outputs]
    ├── robust_sensitivity_results.csv
    ├── sensitivity_summary_stats.csv
    ├── robust_sensitivity_analysis.png
    └── sensitivity_heatmap_robust.png
```

## Integration with Main Simulation
The sensitivity analysis framework integrates seamlessly with the main thermal simulation:
- Uses same configuration parameters from `config.py`
- Imports physics models from `physics_models.py`
- Leverages environment model from `environment_model.py`
- Calls simulation function from `simulation_fix.py`

## Recommendations for Use

### For Design Optimization
1. Use heat flow analyzer to identify thermal bottlenecks
2. Focus design changes on high-sensitivity parameters
3. Monitor temperature margins for critical components (batteries, ESC)

### For Uncertainty Quantification
1. Use robust sensitivity analysis for error margin estimation
2. Consider parameter uncertainties in thermal predictions
3. Validate simulation results against sensitivity bounds

### For Further Development
1. Extend to time-dependent sensitivity analysis
2. Include more environmental parameters (altitude, velocity)
3. Add multi-objective optimization capabilities
4. Implement Monte Carlo uncertainty propagation

## Conclusion
This comprehensive sensitivity analysis framework provides quantitative insights into the thermal behavior of the 20-node UAV simulation model. It enables systematic parameter studies, design optimization, and uncertainty quantification for improved thermal management system design.
