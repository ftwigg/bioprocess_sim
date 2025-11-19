# Batch Reactor Simulation - Architecture Overview

## System Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                     USER INTERFACE LAYER                         │
├─────────────────────────────────────────────────────────────────┤
│  example_batch_simulation.py  │  Custom User Scripts            │
│  - Pre-built examples         │  - Domain-specific scenarios     │
│  - Parameter scans            │  - Integration with experiments  │
└──────────────────┬────────────────────────────────┬─────────────┘
                   │                                │
                   v                                v
┌─────────────────────────────────┐  ┌────────────────────────────┐
│     SIMULATOR MODULE             │  │     UTILS MODULE            │
├─────────────────────────────────┤  ├────────────────────────────┤
│  simulator.py                    │  │  utils.py                   │
│  - ODE integration               │  │  - Visualization            │
│  - Time-stepping                 │  │  - Analysis functions       │
│  - Event detection               │  │  - Data export              │
│  - Multiple solvers              │  │  - Yield calculations       │
└──────────────┬──────────────────┘  └────────────────────────────┘
               │
               v
┌──────────────────────────────────────────────────────────────────┐
│                     REACTOR MODULE                                │
├──────────────────────────────────────────────────────────────────┤
│  reactor.py                                                       │
│  - BatchReactor class (main integration point)                   │
│  - State management                                               │
│  - Coordinates all subsystems                                     │
│  - Implements derivatives() for ODE solver                        │
└───┬─────────┬─────────┬─────────┬─────────┬──────────┬──────────┘
    │         │         │         │         │          │
    v         v         v         v         v          v
┌────────┐┌──────┐┌────────┐┌────────┐┌────────┐┌────────────┐
│KINETICS││MASS  ││THERMO- ││CHEM-   ││BASE    ││(others)    │
│        ││BALANCE│DYNAMICS││ISTRY   ││MODELS  ││            │
└────────┘└──────┘└────────┘└────────┘└────────┘└────────────┘
```

## Module Dependency Graph

```
base_models.py (0 dependencies)
    ↓
    ├─→ kinetics.py (base_models)
    ├─→ mass_balance.py (base_models)
    ├─→ thermodynamics.py (base_models)
    └─→ chemistry.py (base_models)
         ↓
         └─→ reactor.py (all above)
              ↓
              ├─→ simulator.py (reactor, base_models)
              └─→ utils.py (base_models)
                   ↓
                   └─→ example_batch_simulation.py (all)
```

## Data Flow

```
┌──────────────────────────────────────────────────────────────┐
│ INPUT PARAMETERS                                              │
├──────────────────────────────────────────────────────────────┤
│                                                               │
│  CellParameters               ReactorConfig                  │
│  ├─ μ_max, Ks               ├─ V_reactor, V_working        │
│  ├─ Yx_s_max, Yp_s          ├─ kLa_O2, kLa_CO2             │
│  ├─ ms (maintenance)        ├─ T_set                        │
│  ├─ α, β (production)       └─ X0, S0, P0, pH0             │
│  └─ RQ                                                       │
└──────────────────────────────────────────────────────────────┘
                              ↓
┌──────────────────────────────────────────────────────────────┐
│ STATE VECTOR (at time t)                                      │
├──────────────────────────────────────────────────────────────┤
│  y(t) = [X, S, P, dissolved_O2, dissolved_CO2, T]           │
└──────────────────────────────────────────────────────────────┘
                              ↓
┌──────────────────────────────────────────────────────────────┐
│ KINETICS CALCULATIONS                                         │
├──────────────────────────────────────────────────────────────┤
│  kinetics.py                                                  │
│  ├─ μ = f(S, Ks, μ_max)           [Monod]                   │
│  ├─ dX/dt = μ * X                 [Growth]                   │
│  ├─ dS/dt = -(μ/Yx_s + ms) * X   [Substrate]               │
│  ├─ dP/dt = (α*μ + β) * X        [Product]                 │
│  └─ OUR, CER                      [Respiration]             │
└──────────────────────────────────────────────────────────────┘
                              ↓
┌──────────────────────────────────────────────────────────────┐
│ MASS BALANCES                                                 │
├──────────────────────────────────────────────────────────────┤
│  mass_balance.py                                              │
│  ├─ Liquid phase: dX/dt, dS/dt, dP/dt                       │
│  ├─ Gas transfer: OTR, CTR                                   │
│  ├─ Dissolved gases: dDO/dt, dDCO2/dt                       │
│  └─ Gas composition: outlet = f(inlet, OUR, CER)            │
└──────────────────────────────────────────────────────────────┘
                              ↓
┌──────────────────────────────────────────────────────────────┐
│ THERMODYNAMICS                                                │
├──────────────────────────────────────────────────────────────┤
│  thermodynamics.py                                            │
│  ├─ Q_gen = OUR * V * ΔH_ox     [Heat generation]          │
│  ├─ Q_removed = UA*(T-T_jacket)  [Cooling]                  │
│  └─ dT/dt = (Q_gen - Q_removed)/(m*Cp)                      │
└──────────────────────────────────────────────────────────────┘
                              ↓
┌──────────────────────────────────────────────────────────────┐
│ CHEMISTRY                                                     │
├──────────────────────────────────────────────────────────────┤
│  chemistry.py                                                 │
│  ├─ pH = f(P, buffers)           [Henderson-Hasselbalch]    │
│  ├─ Buffer capacity                                          │
│  └─ Ionic strength effects                                   │
└──────────────────────────────────────────────────────────────┘
                              ↓
┌──────────────────────────────────────────────────────────────┐
│ DERIVATIVES VECTOR                                            │
├──────────────────────────────────────────────────────────────┤
│  dy/dt = [dX/dt, dS/dt, dP/dt, dDO/dt, dDCO2/dt, dT/dt]    │
└──────────────────────────────────────────────────────────────┘
                              ↓
                      ODE SOLVER (scipy)
                              ↓
┌──────────────────────────────────────────────────────────────┐
│ OUTPUT RESULTS                                                │
├──────────────────────────────────────────────────────────────┤
│  SimulationResults                                            │
│  ├─ Time series: X(t), S(t), P(t), pH(t), μ(t)             │
│  ├─ Rates: OUR(t), CER(t), Q_gen(t)                         │
│  ├─ Cumulative: O2_consumed, CO2_produced, heat_total       │
│  └─ Metrics: yields, productivities, phase analysis          │
└──────────────────────────────────────────────────────────────┘
```

## Key Classes and Their Responsibilities

### base_models.py
```
CellParameters
├─ Stores: μ_max, Ks, Yx_s_max, ms, yields, production model
└─ Validates parameter reasonableness

ReactorConfig
├─ Stores: volumes, kLa values, T_set, initial conditions
└─ Validates configuration

ReactorState
├─ Current values: X, S, P, pH, T, dissolved gases
└─ Conversion to/from arrays

SimulationResults
├─ Time series arrays
├─ get_summary() → key metrics
└─ Extra variables dict
```

### kinetics.py
```
PirtKinetics
├─ specific_growth_rate(S) → μ
├─ effective_yield(μ) → Yx_s
├─ substrate_consumption_rate(μ, X) → dS/dt
├─ product_formation_rate(μ, X) → dP/dt
└─ biomass_growth_rate(μ, X) → dX/dt

SubstrateBalance
├─ substrate_to_biomass(dX_dt)
├─ substrate_to_product(dP_dt)
├─ substrate_for_maintenance(X)
└─ total_substrate_consumption()

RespiratoryQuotient
├─ oxygen_uptake_rate() → OUR
└─ co2_evolution_rate() → CER
```

### mass_balance.py
```
LiquidPhaseMassBalance
├─ biomass_balance()
├─ substrate_balance()
├─ product_balance()
└─ volume_balance()

GasPhaseMassBalance
├─ saturation_concentration(P, H)
├─ oxygen_mass_transfer() → OTR
├─ co2_mass_transfer() → CTR
├─ dissolved_oxygen_balance()
├─ dissolved_co2_balance()
└─ outlet_gas_composition()
```

### thermodynamics.py
```
MetabolicHeat
├─ heat_from_our(OUR, V) → Q
├─ heat_from_substrate(qs, X, V) → Q
├─ heat_from_growth(μ, X, V) → Q
├─ maintenance_heat(ms, X, V) → Q
└─ total_heat_generation() → Q

HeatBalance
├─ heat_removal(T, T_jacket) → Q_removed
├─ temperature_change(Q_gen, Q_removed) → dT/dt
└─ required_cooling(Q_gen, T, T_set)
```

### chemistry.py
```
pHCalculator
├─ henderson_hasselbalch(pKa, acid, base) → pH
├─ weak_acid_pH(Ca, pKa) → pH
├─ buffer_pH(buffer_conc, acid_conc) → pH
├─ co2_pH_effect(CO2_dissolved) → pH
├─ organic_acid_pH(acid_conc, acid_type) → pH
└─ calculate_pH_from_metabolites() → pH

BufferSystem
├─ buffer_capacity(pH) → β
└─ pH_after_acid_addition() → pH_new
```

### reactor.py
```
BatchReactor
├─ __init__(cell_params, config)
├─ update_state(t, y)
├─ derivatives(t, y) → dy/dt
├─ get_state_vector() → y
├─ record_state()
├─ check_limits() → dict
├─ is_finished() → bool
├─ get_summary() → dict
└─ print_summary()
```

### simulator.py
```
BatchSimulator
├─ simulate(t_end, method) → SimulationResults
├─ simulate_with_control(control_fn) → SimulationResults
└─ _process_results(t, y) → SimulationResults

Helper Functions:
└─ run_quick_simulation() → SimulationResults
```

### utils.py
```
Visualization:
├─ plot_batch_results(results)
├─ plot_comparison(results_list, labels)

Analysis:
├─ calculate_yields(results, X0, S0)
├─ calculate_productivities(results)
├─ analyze_phase(results, t_start, t_end)
├─ find_exponential_phase(results)

Export:
├─ export_to_csv(results, filename)
├─ export_to_json(results, filename)
└─ print_summary_table(results)
```

## Design Patterns Used

### 1. **Separation of Concerns**
- Each module handles one aspect: kinetics, mass balance, etc.
- Clear boundaries between subsystems

### 2. **Dependency Injection**
- `BatchReactor` receives `CellParameters` and `ReactorConfig`
- Easy to swap implementations (e.g., different kinetics models)

### 3. **Factory Pattern**
- `create_example_reactor()` creates pre-configured instances
- `run_quick_simulation()` for rapid prototyping

### 4. **Dataclass Pattern**
- Immutable configuration objects
- Automatic validation via `__post_init__()`

### 5. **Single Responsibility**
- Each class has one job
- E.g., `PirtKinetics` only calculates rates, doesn't handle ODE solving

## Extensibility Points

### Add New Kinetics Model
```python
# In kinetics.py
class ContoisKinetics(PirtKinetics):
    def specific_growth_rate(self, S, X):
        return self.params.mu_max * S / (self.params.Ks * X + S)
```

### Add New Metabolite
```python
# In reactor.py - extend state vector
def derivatives(self, t, y):
    # y = [X, S, P, Acetate, DO, DCO2, T]
    acetate = y[3]
    dAcetate_dt = ...  # Add acetate kinetics
```

### Add New Control Strategy
```python
# In simulator.py
def ph_control(t, reactor):
    if reactor.state.pH < 6.5:
        # Add base
        reactor.state.pH += 0.1
```

## Performance Considerations

### Computational Bottlenecks
1. **ODE integration** (70-80% of time)
   - Use 'LSODA' for adaptive stiffness
   - Adjust rtol/atol for speed vs. accuracy

2. **Derivative calculations** (15-20% of time)
   - Minimize function calls
   - Cache repeated calculations

3. **Visualization** (5-10% of time)
   - Render only when needed
   - Use downsampled data for large datasets

### Memory Usage
- State history: O(n_timepoints * n_variables)
- Typical: 120 points × 10 vars × 8 bytes = ~10 KB
- Negligible for most applications

## Testing Strategy

### Unit Tests (by module)
```python
# Test kinetics
assert PirtKinetics.specific_growth_rate(S=10, Ks=0.1) > 0

# Test mass balance
assert LiquidPhaseMassBalance.biomass_balance(dX_dt > 0) > 0

# Test thermodynamics
assert MetabolicHeat.heat_from_our(OUR=10) > 0
```

### Integration Tests
```python
# Test full simulation
results = run_quick_simulation(t_end=1.0)
assert results.X[-1] > results.X[0]  # Biomass grows
assert results.S[-1] < results.S[0]  # Substrate depletes
```

### Validation Tests
```python
# Compare to experimental data
assert abs(predicted_X - measured_X) / measured_X < 0.10  # 10% error
```

## Error Handling

### Parameter Validation
- `CellParameters.__post_init__()` checks ranges
- `ReactorConfig.__post_init__()` validates geometry

### Numerical Issues
- Negative concentrations → clamp to zero
- Division by zero → check denominators
- Integration failure → try different solver

### Physical Constraints
- Mass conservation → elemental balance check
- Energy conservation → enthalpy balance
- Thermodynamic limits → yield bounds

---

This architecture provides a **robust, extensible, and maintainable** framework for batch reactor simulation! 🏗️
