# Fed-Batch Bioprocess Simulator

A first-principles simulator for fed-batch fermentation processes built from engineering fundamentals.

## Features

### Core Capabilities
- **9 coupled ODEs** for complete fed-batch dynamics
- **Multiple substrate limitation** (carbon, nitrogen, oxygen)
- **Generic carbon sources** (glucose, glycerol, acetate, methanol, or custom)
- **Elemental balance** for automatic yield calculation
- **Dynamic kLa** with biomass correction
- **DO cascade control** (agitation → aeration)
- **Temperature control** (PID)
- **Feed strategies** (constant, exponential, DO-stat, piecewise)
- **pH modeling** (simplified empirical, extensible to mechanistic)

### Engineering First Principles

**Stoichiometry:**
- Electron balance: γ_substrate = γ_biomass × c + 4a
- Automatic calculation of Y_x/O2, Y_x/N from Y_x/S
- Works for any carbon source

**Kinetics:**
- Multiple substrate Monod (or Haldane with inhibition)
- Pirt maintenance model
- Luedeking-Piret product formation
- Stoichiometric OUR and CER

**Mass Transfer:**
- Temperature-dependent Henry's constants
- Dynamic kLa: kLa = k × N^a × (Q/V)^b × exp(-k_X × X)
- Gas phase material balances
- Average partial pressures

**Thermodynamics:**
- Metabolic heat from OUR: Q = -ΔH_ox × OUR × V
- Agitation heat from power input
- Feed heat from temperature difference
- Jacket cooling with dynamic area
- PID temperature control

**Control:**
- DO cascade: Try agitation first, then aeration
- Detects oxygen limitation
- Temperature PID with anti-windup
- Multiple feed strategies

## Installation

### Requirements
- Python 3.11+
- NumPy
- SciPy

### Install dependencies
```bash
pip install numpy scipy
```

### Quick start
```python
from fedbatch_simulator import *
from fedbatch_simulator.kinetics import CellParameters
from fedbatch_simulator.reactor import FedBatchReactor
from fedbatch_simulator.simulator import FedBatchSimulator

# See examples/ for complete working examples
```

## State Vector

The simulator solves 9 coupled ODEs:

```
y = [X, S_carbon, S_nitrogen, P, DO, DCO2, V, T, N]
```

Where:
- `X`: Biomass concentration (g/L)
- `S_carbon`: Carbon substrate concentration (g/L)
- `S_nitrogen`: Nitrogen substrate concentration (g/L)
- `P`: Product concentration (g/L)
- `DO`: Dissolved oxygen (mmol/L)
- `DCO2`: Dissolved CO2 (mmol/L)
- `V`: Reactor volume (L)
- `T`: Temperature (°C)
- `N`: Agitation speed (rpm)

## Mass Balances

**General form:**
```
dC/dt = D × (C_in - C) + r
```

Where:
- `D = F_in/V`: Dilution rate
- `C_in`: Feed concentration
- `r`: Reaction rate

**Specific balances:**
```
dX/dt = μ × X - D × X
dS/dt = D × (S_in - S) - (μ/Y_xs + ms) × X
dP/dt = (α×μ + β) × X - D × P
dDO/dt = OTR - OUR - D × DO
dDCO2/dt = CER - CTR - D × DCO2
dV/dt = F_in
dT/dt = Q_net / (ρ × V × Cp)
dN/dt = (N_target - N) / τ_N
```

## Examples

### Example 1: Exponential Fed-Batch

```python
# Define carbon source
glucose = GLUCOSE

# Define cell parameters
carbon_params = SubstrateParameters(
    name='carbon',
    Ks=0.1,  # g/L
    Y_xs=0.5,  # g/g
    ms=0.03   # g/g/h
)

cell_params = CellParameters(
    mu_max=0.5,  # 1/h
    carbon_source=glucose,
    biomass_composition=STANDARD_BIOMASS,
    substrates={'carbon': carbon_params},
    Y_xs=0.5
)

# Initial conditions
initial_state = ReactorState(
    X=0.5, S_carbon=10.0, V=2.0, T=37.0, N=300
)

# Feed strategy: Batch → Exponential
feed_comp = FeedComposition(S_carbon=500.0, temperature=25.0)
batch_feed = ConstantFeed(feed_comp, F_constant=0.0)
exp_feed = ExponentialFeed(feed_comp, F0=0.01, mu_set=0.2, F_max=0.5)
feed_strategy = PiecewiseFeed([(0.0, batch_feed), (10.0, exp_feed)])

# DO cascade control
do_control = SimplifiedCascade(
    DO_setpoint=0.06,
    N_min=200, N_max=1000,
    Q_gas_min=60, Q_gas_max=240,
    kLa_correlation=DynamicKLa()
)

# Create and run reactor
reactor = FedBatchReactor(
    cell_params=cell_params,
    reactor_config=LAB_STR_5L,
    initial_state=initial_state,
    feed_strategy=feed_strategy,
    do_control=do_control
)

simulator = FedBatchSimulator(reactor)
results = simulator.simulate(t_end=30.0)

print(results.get_summary())
```

### Example 2: Custom Carbon Source

```python
# Define custom carbon source
glycerol = CarbonSource(
    name='glycerol',
    formula='C3H8O3',
    MW=92.09,
    C=3, H=8, O=3
)

# Yields are automatically calculated from elemental balance!
cell_params = CellParameters(
    mu_max=0.4,
    carbon_source=glycerol,  # Use glycerol instead of glucose
    biomass_composition=STANDARD_BIOMASS,
    substrates={'carbon': carbon_params},
    Y_xs=0.6
)

# Y_x/O2 and Y_x/N are calculated automatically
print(f"Y_x/O2: {cell_params.Y_x_O2:.3f} g/g")
print(f"RQ: {cell_params.RQ:.3f}")
```

## Module Structure

```
fedbatch_simulator/
├── __init__.py              # Package initialization
├── base_models.py           # Data structures
├── stoichiometry.py         # Elemental balance
├── kinetics.py              # Growth kinetics
├── mass_transfer.py         # Gas-liquid transfer
├── thermodynamics.py        # Heat balance
├── feed_strategies.py       # Feed control
├── control.py               # DO cascade
├── chemistry.py             # pH models
├── reactor.py               # Main reactor class
├── simulator.py             # ODE integration
└── examples/
    └── exponential_fedbatch.py
```

## Predefined Configurations

### Carbon Sources
- `GLUCOSE`: C6H12O6
- `GLYCEROL`: C3H8O3
- `ACETATE`: C2H4O2
- `METHANOL`: CH4O

### Biomass Compositions
- `STANDARD_BIOMASS`: CH1.8O0.5N0.2
- `ECOLI_BIOMASS`: CH1.77O0.49N0.24
- `YEAST_BIOMASS`: CH1.83O0.55N0.17

### Reactor Configurations
- `LAB_STR_5L`: 5L stirred tank
- `PILOT_STR_100L`: 100L stirred tank
- `PRODUCTION_STR_10000L`: 10,000L stirred tank

## Key Design Decisions

### 1. Concentration-Based Mass Balances
- State variables are concentrations (g/L, mmol/L)
- Dilution term D = F_in/V appears in each balance
- More intuitive than amount-based (n = C×V)

### 2. Generic Carbon Source
- Not locked to glucose
- Works for any carbon source with C, H, O, N composition
- Yields calculated from elemental balance

### 3. Dynamic kLa with DO Cascade
- kLa changes with N, Q_gas, X, T
- Control logic: Increase N first, then Q_gas
- Detects oxygen limitation

### 4. Simplified pH (Extensible)
- Default: pH = pH0 - k_acid × P
- Fast, sufficient for many applications
- Mechanistic model can be added later

### 5. Temperature-Dependent Properties
- Henry's constants: H(T) via van't Hoff
- Constant ρ, Cp (extensible)

## Validation

The simulator has been validated against:
- Elemental balance (closure < 1e-15)
- Stoichiometric consistency
- Physical constraints (DO ≥ 0, V > 0, etc.)
- Typical fermentation behavior

## Extensibility

The architecture is designed for easy extension:

### Add new carbon source
```python
my_substrate = CarbonSource(
    name='my_substrate',
    formula='C4H8O2',
    MW=88.0,
    C=4, H=8, O=2
)
```

### Add new feed strategy
```python
class MyFeedStrategy(FeedStrategy):
    def get_feed_rate(self, t, state):
        # Your logic here
        return F
```

### Add mechanistic pH model
```python
class MechanisticpHModel(pHModel):
    def calculate_pH(self, P):
        # Charge balance solution
        return pH
```

## Performance

Typical simulation (30 hours):
- **Time**: ~2-5 seconds
- **Function evaluations**: ~2000-3000
- **Solver**: BDF (stiff ODE solver)
- **Accuracy**: rtol=1e-6, atol=1e-8

## Limitations

Current version:
- Single product (extensible to multiple)
- No cell death kinetics (can be added)
- No overflow metabolism (can be added)
- Simplified pH (mechanistic version planned)
- No sampling (volume decrease can be added)

## Future Enhancements

Planned features:
- Multiple products
- Cell death and lysis
- Overflow metabolism (acetate, ethanol)
- Mechanistic pH with charge balance
- Full PID cascade for DO (with dynamics)
- Sampling and volume decrease
- Parameter estimation from data
- Sensitivity analysis
- Optimization (feed profile, setpoints)

## References

### Stoichiometry
- Roels, J.A. (1983). Energetics and Kinetics in Biotechnology.
- Nielsen, J. et al. (2003). Bioreaction Engineering Principles.

### Kinetics
- Monod, J. (1949). The growth of bacterial cultures.
- Pirt, S.J. (1965). The maintenance energy of bacteria.
- Luedeking & Piret (1959). Product formation kinetics.

### Mass Transfer
- Van't Riet, K. (1979). Review of kLa measurement.
- Garcia-Ochoa & Gomez (2009). Bioreactor scale-up.

### Fed-Batch Control
- Lee, J. et al. (1999). Fed-batch cultures.
- Yamané & Shimizu (1984). Fed-batch techniques.

## License

MIT License

## Authors

Created as a first-principles bioprocess simulator for educational and research purposes.

## Citation

If you use this simulator in your research, please cite:
```
Fed-Batch Bioprocess Simulator (2024)
https://github.com/yourusername/fedbatch_simulator
```

## Contact

For questions, issues, or contributions, please open an issue on GitHub.
