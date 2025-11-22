"""
Example: Exponential fed-batch simulation.

Demonstrates:
- Batch phase (10 h)
- Exponential feeding phase
- DO cascade control
- Temperature control
- Complete first-principles simulation
"""

import sys
sys.path.insert(0, '/home/ubuntu')

from fedbatch_simulator import *
from fedbatch_simulator.kinetics import CellParameters
from fedbatch_simulator.mass_transfer import DynamicKLa
from fedbatch_simulator.thermodynamics import TemperatureController
from fedbatch_simulator.feed_strategies import ConstantFeed, ExponentialFeed, PiecewiseFeed
from fedbatch_simulator.control import SimplifiedCascade
from fedbatch_simulator.reactor import FedBatchReactor
from fedbatch_simulator.simulator import FedBatchSimulator


def main():
    print("=" * 70)
    print("EXPONENTIAL FED-BATCH SIMULATION")
    print("=" * 70)
    
    # ========== 1. DEFINE CARBON SOURCE ==========
    glucose = GLUCOSE
    print(f"\nCarbon source: {glucose.name}")
    print(f"  Formula: {glucose.formula}")
    print(f"  MW: {glucose.MW} g/mol")
    print(f"  Degree of reduction: {glucose.degree_of_reduction:.2f}")
    
    # ========== 2. DEFINE CELL PARAMETERS ==========
    carbon_params = SubstrateParameters(
        name='carbon',
        Ks=0.1,  # g/L
        Ki=None,  # No inhibition (Monod)
        Y_xs=0.5,  # g biomass / g glucose
        ms=0.03   # g glucose / g biomass / h
    )
    
    nitrogen_params = SubstrateParameters(
        name='nitrogen',
        Ks=0.05,  # g/L
        Ki=None,
        Y_xs=10.0,  # g biomass / g nitrogen
        ms=0.001
    )
    
    cell_params = CellParameters(
        mu_max=0.5,  # 1/h
        carbon_source=glucose,
        biomass_composition=STANDARD_BIOMASS,
        substrates={
            'carbon': carbon_params,
            'nitrogen': nitrogen_params
        },
        Y_xs=0.5,
        Ks_O2=0.003,  # mmol/L
        m_O2=0.01,
        RQ=None,  # Will be calculated
        alpha=0.0,  # No product
        beta=0.0
    )
    
    print(f"\nCell parameters:")
    print(f"  μ_max: {cell_params.mu_max} 1/h")
    print(f"  Y_x/S: {cell_params.Y_xs} g/g")
    print(f"  Y_x/O2: {cell_params.Y_x_O2:.3f} g/g (calculated)")
    print(f"  Y_x/N: {cell_params.Y_x_N:.3f} g/g (calculated)")
    print(f"  RQ: {cell_params.RQ:.3f} (calculated)")
    
    # ========== 3. REACTOR CONFIGURATION ==========
    reactor_config = LAB_STR_5L
    print(f"\nReactor: {reactor_config.name}")
    print(f"  Type: {reactor_config.reactor_type}")
    print(f"  Volume: {reactor_config.V_reactor} L (max {reactor_config.V_working_max} L)")
    print(f"  Agitation: {reactor_config.N_min}-{reactor_config.N_max} rpm")
    
    # ========== 4. INITIAL CONDITIONS ==========
    initial_state = ReactorState(
        time=0.0,
        X=0.5,  # g/L
        S_carbon=10.0,  # g/L (batch phase glucose)
        S_nitrogen=2.0,  # g/L
        P=0.0,
        DO=0.2,  # mmol/L (~80% saturation)
        DCO2=0.5,  # mmol/L
        V=2.0,  # L (starting volume)
        T=37.0,  # °C
        N=300,  # rpm (initial agitation)
        pH=7.0
    )
    
    print(f"\nInitial conditions:")
    print(f"  X: {initial_state.X} g/L")
    print(f"  S_carbon: {initial_state.S_carbon} g/L")
    print(f"  S_nitrogen: {initial_state.S_nitrogen} g/L")
    print(f"  V: {initial_state.V} L")
    print(f"  T: {initial_state.T}°C")
    
    # ========== 5. FEED STRATEGY ==========
    feed_composition = FeedComposition(
        S_carbon=500.0,  # g/L concentrated glucose
        S_nitrogen=50.0,  # g/L ammonium sulfate
        temperature=25.0,  # °C
        pH=5.0
    )
    
    # Batch phase then exponential feed
    batch_feed = ConstantFeed(feed_composition, F_constant=0.0)
    exp_feed = ExponentialFeed(
        feed_composition=feed_composition,
        F0=0.01,  # L/h
        mu_set=0.2,  # 1/h
        F_max=0.5  # L/h
    )
    exp_feed.t_start = 10.0  # Start at 10 h
    
    feed_strategy = PiecewiseFeed([
        (0.0, batch_feed),
        (10.0, exp_feed)
    ])
    
    print(f"\nFeed strategy:")
    print(f"  0-10 h: Batch (F = 0)")
    print(f"  10+ h: Exponential (F0 = {exp_feed.F0} L/h, μ_set = {exp_feed.mu_set} 1/h)")
    print(f"  Feed composition: {feed_composition.S_carbon} g/L glucose")
    
    # ========== 6. DO CASCADE CONTROL ==========
    do_control = SimplifiedCascade(
        DO_setpoint=0.06,  # mmol/L (~30% saturation)
        N_min=reactor_config.N_min,
        N_max=reactor_config.N_max,
        Q_gas_min=reactor_config.Q_gas_min * initial_state.V * 60,  # L/h
        Q_gas_max=reactor_config.Q_gas_max * initial_state.V * 60,
        kLa_correlation=DynamicKLa(
            k_O2=reactor_config.k_O2,
            a=reactor_config.a,
            b=reactor_config.b,
            k_X=reactor_config.k_X
        )
    )
    
    print(f"\nDO control:")
    print(f"  Setpoint: {do_control.DO_sp} mmol/L")
    print(f"  Agitation: {do_control.N_min}-{do_control.N_max} rpm")
    print(f"  Aeration: {do_control.Q_min:.0f}-{do_control.Q_max:.0f} L/h")
    
    # ========== 7. TEMPERATURE CONTROL ==========
    temp_controller = TemperatureController(
        T_setpoint=37.0,  # °C
        K_p=10.0,
        K_i=0.5,
        T_jacket_min=4.0,
        T_jacket_max=80.0
    )
    
    print(f"\nTemperature control:")
    print(f"  Setpoint: {temp_controller.T_sp}°C")
    print(f"  PID: K_p={temp_controller.K_p}, K_i={temp_controller.K_i}")
    
    # ========== 8. CREATE REACTOR ==========
    print(f"\n" + "=" * 70)
    print("Creating reactor...")
    
    reactor = FedBatchReactor(
        cell_params=cell_params,
        reactor_config=reactor_config,
        initial_state=initial_state,
        feed_strategy=feed_strategy,
        do_control=do_control,
        temp_controller=temp_controller
    )
    
    # ========== 9. RUN SIMULATION ==========
    print("=" * 70)
    
    simulator = FedBatchSimulator(reactor)
    results = simulator.simulate(t_end=30.0, max_step=0.1)
    
    # ========== 10. ANALYZE RESULTS ==========
    print("\n" + "=" * 70)
    print(results.get_summary())
    print("=" * 70)
    
    # Additional analysis
    print("\nProcess performance:")
    
    # Productivity
    final_X = results.X[-1]
    final_V = results.V[-1]
    final_t = results.time[-1]
    total_biomass = final_X * final_V
    productivity = total_biomass / final_t
    
    print(f"  Total biomass produced: {total_biomass:.2f} g")
    print(f"  Volumetric productivity: {productivity:.2f} g/L/h")
    
    # Yield
    initial_substrate = initial_state.S_carbon * initial_state.V
    final_substrate = results.S_carbon[-1] * final_V
    substrate_consumed = initial_substrate - final_substrate
    
    # Add fed substrate
    import numpy as np
    fed_substrate = np.trapz(results.F_feed * feed_composition.S_carbon, results.time)
    total_substrate = substrate_consumed + fed_substrate
    
    overall_yield = total_biomass / total_substrate if total_substrate > 0 else 0
    
    print(f"  Total substrate consumed: {total_substrate:.2f} g")
    print(f"  Overall yield: {overall_yield:.3f} g biomass / g substrate")
    
    # Oxygen limitation check
    if np.any(results.DO < 0.01):
        print(f"  ⚠️  Oxygen limitation detected!")
        t_limited = results.time[results.DO < 0.01][0]
        print(f"      First occurred at t = {t_limited:.1f} h")
    
    print("\n" + "=" * 70)
    print("Simulation complete!")
    print("=" * 70)
    
    return results


if __name__ == "__main__":
    results = main()
