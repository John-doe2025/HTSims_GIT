# Comprehensive Heat Transfer Analysis Design

## Overview
Design for complete heat transfer tracking showing how each node gains and loses heat through all modes and pathways.

## Analysis Structure

### 1. Node Heat Balance Framework
For each node, track:
- **Heat Sources (Gains)**
  - Internal generation
  - Conduction FROM other nodes
  - Convection FROM fluid/ambient
  - Radiation FROM other surfaces
- **Heat Sinks (Losses)**
  - Conduction TO other nodes
  - Convection TO fluid/ambient
  - Radiation TO other surfaces

### 2. Heat Transfer Mode Modules

#### A. Conduction Analysis (`conduction_analyzer.py`)
- Track all conduction paths between nodes
- Show heat flow magnitude and direction
- List: Node1 → Node2, Heat_Flow_W, Thermal_Resistance

#### B. Convection Analysis (`convection_analyzer.py`)
- Internal convection (nodes ↔ internal air)
- External convection (shell exteriors ↔ ambient)
- Show heat transfer coefficients and areas

#### C. Radiation Analysis (`radiation_analyzer.py`)
- All surface-to-surface radiation exchanges
- Temperature^4 dependent heat flows
- View factors and emissivity effects

### 3. Node-to-Node Heat Flow Mapping (`node_flow_mapper.py`)
- Complete connectivity matrix showing all heat paths
- Direction and magnitude of each connection
- Identify dominant heat transfer paths

### 4. Heat Balance Verification (`heat_balance_checker.py`)
- Verify energy conservation for each node
- Check: Generation + Heat_In = Heat_Out
- Identify any calculation errors or missing paths

## Implementation Files Structure

```
sens_analysis_v2/
├── heat_transfer_master.py          # Main coordinator
├── conduction_analyzer.py           # Conduction-specific analysis
├── convection_analyzer.py           # Convection-specific analysis  
├── radiation_analyzer.py            # Radiation-specific analysis
├── node_flow_mapper.py              # Node-to-node flow mapping
├── heat_balance_checker.py          # Energy balance verification
└── comprehensive_reporter.py        # Unified reporting
```

## Output Format Design

### Per-Node Heat Balance Report
```
NODE: Batt_BF_Top
HEAT SOURCES (Gains):
  Generation:           +2.100 W
  Conduction from:
    - plateT:          +1.250 W
    - plateM:          +0.890 W
  Radiation from:
    - ESC:             +5.576 W
  Convection from:
    - Internal_air:    +0.254 W
  TOTAL GAINS:         +10.070 W

HEAT SINKS (Losses):
  Conduction to:        (none)
  Radiation to:
    - Top_Shell_Int:   -0.850 W
    - Other_batteries: -0.250 W
  Convection to:        (none)
  TOTAL LOSSES:        -1.100 W

NET HEAT FLOW:         +8.970 W
```

### Heat Transfer Mode Summary
```
CONDUCTION FLOWS:
From_Node → To_Node        Heat_Flow_W    Thermal_Resistance
ESC → ESC_Mount            -489.975       0.002 K/W
ESC_Mount → BH_1           +489.975       0.001 K/W
...

CONVECTION FLOWS:
Node → Target              Heat_Flow_W    h_coeff    Area_m2
Top_Shell_Ext → Ambient    -186.504       25.5       0.145
ESC → Internal_air         -4.556         12.3       0.030
...

RADIATION FLOWS:
From_Node → To_Node        Heat_Flow_W    T1_K    T2_K
Batt_BF_Top → ESC          +5.576         315     390
ESC → Top_Shell_Int        -5.720         390     315
...
```

### Node Connectivity Matrix
```
           ESC  ESC_Mount  BH_1  Batt_BF_Top  ...
ESC         0    -489.9     0      +5.6       ...
ESC_Mount  +489.9   0     +489.9    0        ...
BH_1        0    -489.9     0       0        ...
...
```

## Key Features

1. **Complete Accounting**: Every joule tracked from source to sink
2. **Directional Clarity**: Clear indication of heat flow direction
3. **Mode Separation**: Individual analysis for each heat transfer mechanism
4. **Balance Verification**: Energy conservation checks
5. **Pathway Identification**: Dominant heat transfer routes highlighted
6. **Quantitative Focus**: Exact values for all heat flows

## Implementation Priority

1. **Phase 1**: Conduction analyzer (largest heat flows)
2. **Phase 2**: Convection analyzer (ambient rejection)
3. **Phase 3**: Radiation analyzer (internal exchanges)
4. **Phase 4**: Integration and verification
5. **Phase 5**: Comprehensive reporting

This design ensures we capture every heat transfer pathway and can answer questions like:
- Where does each node get its heat from?
- Where does each node send its heat to?
- What are the dominant thermal pathways?
- Is energy conserved in the system?
