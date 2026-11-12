# Quick Start Guide - Fed-Batch Simulator

Get started with the fed-batch bioprocess simulator in 5 minutes!

## Installation

```bash
# Install dependencies
pip install numpy scipy

# Or use requirements.txt
pip install -r requirements.txt
```

## Basic Usage

### 1. Import the simulator

```python
from fedbatch_simulator import *
from fedbatch_simulator.kinetics import CellParameters
from fedbatch_simulator.feed_strategies import ConstantFeed, ExponentialFeed, PiecewiseFeed
from fedbatch_simulator.control import PICascade
from fedbatch_simulator.mass_transfer import DynamicKLa
from fedbatch_simulator.reactor import FedBatchReactor
from fedbatch_simulator.simulator import FedBatchSimulator
```

### 2. Define cell parameters

```python
# Define substrate parameters
carbon_params = SubstrateParameters(
    name='carbon',
    Ks=0.1,      # Half-saturation constant (g/L)
    Y_xs=0.5,    # Yield (g biomass / g substrate)
    ms=0.03      # Maintenance (g substrate / g biomass / h)
)

# Create cell parameters (yields auto-calculated!)
cell_params = CellParameters(
    mu_max=0.5,                      # Max growth rate (1/h)
    carbon_source=GLUCOSE,           # Use predefined or custom
    biomass_composition=STANDARD_BIOMASS,
    substrates={'carbon': carbon_params},
    Y_xs=0.5
)

# Check auto-calculated values
print(f"Y_x/O2: {cell_params.Y_x_O2:.3f} g/g")  # Calculated!
print(f"RQ: {cell_params.RQ:.3f}")               # Calculated!
```

### 3. Set initial conditions

```python
initial_state = ReactorState(
    X=0.5,          # Biomass (g/L)
    S_carbon=10.0,  # Substrate (g/L)
    S_nitrogen=2.0, # Nitrogen (g/L)
    V=2.0,          # Volume (L)
    T=37.0,         # Temperature (°C)
    DO=0.2,         # Dissolved O2 (mmol/L)
    N=300           # Agitation (rpm)
)
```

### 4. Define feed strategy

```python
# Feed composition
feed_comp = FeedComposition(
    S_carbon=500.0,    # Concentrated glucose (g/L)
    S_nitrogen=50.0,   # Nitrogen source (g/L)
    temperature=25.0
)

# Batch first, then exponential feeding. The 10 g/L of batch glucose runs out
# at about 5.3 h and leaves about 5.3 g/L of biomass.
mu_set = 0.12  # Target growth rate (1/h)
F0 = ExponentialFeed.initial_rate(   # what that biomass needs to grow at mu_set
    mu_set=mu_set, X=5.3, V=2.0, S_feed=500.0, Y_xs=0.5, ms=0.03)

exp_feed = ExponentialFeed(
    feed_composition=feed_comp,
    F0=F0,                 # Initial rate (L/h)
    mu_set=mu_set,
    F_max=4.5 / 500.0      # Cap (L/h): 4.5 g glucose/h is what the oxygen supply covers
)
exp_feed.t_start = 5.5

feed_strategy = PiecewiseFeed([
    (0.0, ConstantFeed(feed_comp, F_constant=0.0)),
    (5.5, exp_feed)
])
```

Size the feed from oxygen, not from the pump: feeding faster than the reactor's
oxygen transfer can burn only accumulates glucose and drives DO to zero.

### 5. Set up DO control

```python
do_control = PICascade(
    DO_setpoint=0.06,  # Target DO (mmol/L)
    N_min=200,         # Min agitation (rpm)
    N_max=1000,        # Max agitation (rpm)
    Q_gas_min=60,      # Min aeration (L/h)
    Q_gas_max=240,     # Max aeration (L/h)
    kLa_correlation=DynamicKLa()
)
```

### 6. Create and run reactor

```python
# Create reactor
reactor = FedBatchReactor(
    cell_params=cell_params,
    reactor_config=LAB_STR_5L,      # Predefined config
    initial_state=initial_state,
    feed_strategy=feed_strategy,
    do_control=do_control
)

# Run simulation
simulator = FedBatchSimulator(reactor)
results = simulator.simulate(t_end=22.0)

# View results
print(results.get_summary())
```

## Complete Minimal Example

```python
from fedbatch_simulator import *
from fedbatch_simulator.kinetics import CellParameters
from fedbatch_simulator.feed_strategies import ConstantFeed, ExponentialFeed, PiecewiseFeed
from fedbatch_simulator.control import PICascade
from fedbatch_simulator.mass_transfer import DynamicKLa
from fedbatch_simulator.reactor import FedBatchReactor
from fedbatch_simulator.simulator import FedBatchSimulator

# Cell parameters
cell_params = CellParameters(
    mu_max=0.5,
    carbon_source=GLUCOSE,
    biomass_composition=STANDARD_BIOMASS,
    substrates={'carbon': SubstrateParameters('carbon', Ks=0.1, Y_xs=0.5, ms=0.03)},
    Y_xs=0.5
)

# Initial state
initial_state = ReactorState(X=0.5, S_carbon=10.0, S_nitrogen=2.0, 
                             V=2.0, T=37.0, DO=0.2, N=300)

# Feed: batch until the glucose runs out (~5.3 h), then exponential, capped by oxygen
feed_comp = FeedComposition(S_carbon=500.0, S_nitrogen=50.0, temperature=25.0)
exp_feed = ExponentialFeed(
    feed_comp,
    F0=ExponentialFeed.initial_rate(mu_set=0.12, X=5.3, V=2.0, S_feed=500.0, Y_xs=0.5, ms=0.03),
    mu_set=0.12, F_max=4.5 / 500.0
)
exp_feed.t_start = 5.5
feed_strategy = PiecewiseFeed([(0.0, ConstantFeed(feed_comp, 0.0)), (5.5, exp_feed)])

# DO control
do_control = PICascade(DO_setpoint=0.06, N_min=200, N_max=1000,
                               Q_gas_min=60, Q_gas_max=240,
                               kLa_correlation=DynamicKLa())

# Run
reactor = FedBatchReactor(cell_params, LAB_STR_5L, initial_state, 
                          feed_strategy, do_control)
results = FedBatchSimulator(reactor).simulate(t_end=22.0)
print(results.get_summary())
print(format_balances(check_balances(results, cell_params, feed_strategy)))
```

## Examples

See `examples/` directory for complete working examples:

1. **exponential_fedbatch.py** - Exponential feeding with DO cascade
2. **do_stat_fedbatch.py** - DO-stat feedback control
3. **glycerol_fedbatch.py** - Custom carbon source (glycerol)

Run an example:
```bash
python examples/exponential_fedbatch.py
```

The same runs with plots are in `notebooks/fedbatch_simulator_examples.ipynb`.

## Common Patterns

### Batch → Fed-batch transition

```python
from fedbatch_simulator.feed_strategies import ConstantFeed, PiecewiseFeed

batch_feed = ConstantFeed(feed_comp, F_constant=0.0)
exp_feed = ExponentialFeed(feed_comp, F0=0.01, mu_set=0.2)
exp_feed.t_start = 10.0

feed_strategy = PiecewiseFeed([
    (0.0, batch_feed),    # Batch phase
    (10.0, exp_feed)      # Fed-batch phase
])
```

### Custom carbon source

```python
my_substrate = CarbonSource(
    name='my_substrate',
    formula='C4H8O2',
    MW=88.0,
    C=4, H=8, O=2
)

cell_params = CellParameters(
    mu_max=0.4,
    carbon_source=my_substrate,  # Use custom substrate
    biomass_composition=STANDARD_BIOMASS,
    substrates={'carbon': carbon_params},
    Y_xs=0.6
)
# Yields calculated automatically from elemental balance!
```

### Temperature control

```python
from fedbatch_simulator.thermodynamics import TemperatureController

temp_controller = TemperatureController(
    T_setpoint=37.0,
    K_p=10.0,
    K_i=0.5
)

reactor = FedBatchReactor(
    ...,
    temp_controller=temp_controller  # Add to reactor
)
```

### DO-stat feeding

```python
from fedbatch_simulator.feed_strategies import DOStatFeed

from fedbatch_simulator.control import FixedAerationAgitation

feed_strategy = DOStatFeed(
    feed_composition=feed_comp,
    DO_setpoint=0.06,
    K_p=0.1,       # Positive: high DO → increase feed
    K_i=1.0,
    F_min=0.0,
    F_max=0.03,
    F_base=0.0
)

# Hold the oxygen supply fixed so the feed is the only thing controlling DO
do_control = FixedAerationAgitation(N=1000, Q_gas=240, kLa_correlation=DynamicKLa())
```

See `examples/do_stat_fedbatch.py` for the batch phase that precedes it.

## Accessing Results

```python
results = simulator.simulate(t_end=30.0)

# Time series data
time = results.time
biomass = results.X
substrate = results.S_carbon
DO = results.DO
volume = results.V
temperature = results.T

# Rates
growth_rate = results.mu
OUR = results.OUR
CER = results.CER

# Control variables
agitation = results.N
aeration = results.Q_gas
feed_rate = results.F_feed

# Summary
print(results.get_summary())
```

## Plotting Results

```python
import matplotlib.pyplot as plt
from fedbatch_simulator.plotting import plot_fedbatch_results, plot_comparison

# Twelve panels for one run
plot_fedbatch_results(results, "My run", DO_setpoint=0.06, marks={'feed on': 5.5})

# Up to three runs overlaid
plot_comparison({'Run A': results_a, 'Run B': results_b})

plt.show()
```

The notebooks in `notebooks/` do this for all three examples and open directly in Google Colab.

## Tips

1. **Start simple**: Use predefined configurations (GLUCOSE, LAB_STR_5L)
2. **Check yields**: Print auto-calculated Y_x/O2, RQ to verify stoichiometry
3. **Monitor DO**: Check for oxygen limitation in results
4. **Adjust control**: Tune K_p for DO-stat, PID gains for temperature
5. **Volume limits**: Simulation stops when V reaches V_working_max

## Troubleshooting

**Simulation fails:**
- Check initial conditions (all positive, reasonable values)
- Reduce max_step (default 0.1 h)
- Check feed composition (S_carbon > 0)

**Oxygen limitation:**
- Increase N_max or Q_gas_max
- Reduce mu_max or feed rate
- Check kLa correlation parameters

**Temperature oscillations:**
- Reduce PID gains (K_p, K_i)
- Check heat transfer coefficient U

## Next Steps

- Read full README.md for detailed documentation
- Explore examples/ for complete simulations
- Modify parameters for your specific organism/process
- Add custom feed strategies or control logic

## Support

For issues or questions:
- Check examples/ for working code
- Review module docstrings
- See IMPLEMENTATION_SUMMARY.md for design decisions
