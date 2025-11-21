"""
Example: DO-stat fed-batch simulation.

Demonstrates:
- DO-stat feeding (feedback control)
- Substrate-limited growth
- Realistic industrial scenario
"""

import sys
sys.path.insert(0, '/home/ubuntu')

from fedbatch_simulator import *
from fedbatch_simulator.kinetics import CellParameters
from fedbatch_simulator.mass_transfer import DynamicKLa
from fedbatch_simulator.thermodynamics import TemperatureController
from fedbatch_simulator.feed_strategies import ConstantFeed, DOStatFeed, PiecewiseFeed
from fedbatch_simulator.control import SimplifiedCascade
from fedbatch_simulator.reactor import FedBatchReactor
from fedbatch_simulator.simulator import FedBatchSimulator


def main():
    print("=" * 70)
    print("DO-STAT FED-BATCH SIMULATION")
    print("=" * 70)
    print("\nStrategy: Feed rate controlled by DO feedback")
    print("  - High DO → Increase feed (more substrate, more OUR)")
    print("  - Low DO → Decrease feed (less substrate, less OUR)")
    print("  - Maintains DO near setpoint via substrate limitation")
    
    # ========== CELL PARAMETERS ==========
    carbon_params = SubstrateParameters(
        name='carbon',
        Ks=0.1,
        Y_xs=0.5,
        ms=0.03
    )
    
    nitrogen_params = SubstrateParameters(
        name='nitrogen',
        Ks=0.05,
        Y_xs=10.0,
        ms=0.001
    )
    
    cell_params = CellParameters(
        mu_max=0.5,
        carbon_source=GLUCOSE,
        biomass_composition=STANDARD_BIOMASS,
        substrates={
            'carbon': carbon_params,
            'nitrogen': nitrogen_params
        },
        Y_xs=0.5,
        Ks_O2=0.003,
        m_O2=0.01
    )
    
    print(f"\nCell parameters:")
    print(f"  μ_max: {cell_params.mu_max} 1/h")
    print(f"  Y_x/S: {cell_params.Y_xs} g/g")
    print(f"  Y_x/O2: {cell_params.Y_x_O2:.3f} g/g")
    
    # ========== INITIAL CONDITIONS ==========
    initial_state = ReactorState(
        time=0.0,
        X=2.0,  # Higher initial biomass
        S_carbon=5.0,  # Lower initial substrate
        S_nitrogen=2.0,
        P=0.0,
        DO=0.06,  # Start at setpoint
        DCO2=0.5,
        V=2.0,
        T=37.0,
        N=300,
        pH=7.0
    )
    
    print(f"\nInitial conditions:")
    print(f"  X: {initial_state.X} g/L (higher than exponential example)")
    print(f"  S_carbon: {initial_state.S_carbon} g/L (lower, will need feeding)")
    print(f"  V: {initial_state.V} L")
    
    # ========== FEED STRATEGY: DO-STAT ==========
    feed_composition = FeedComposition(
        S_carbon=500.0,
        S_nitrogen=50.0,
        temperature=25.0,
        pH=5.0
    )
    
    # Batch phase then DO-stat
    batch_feed = ConstantFeed(feed_composition, F_constant=0.0)
    
    do_stat_feed = DOStatFeed(
        feed_composition=feed_composition,
        DO_setpoint=0.06,  # mmol/L
        K_p=-2.0,  # Negative: high DO → higher feed → lower DO
        F_min=0.0,
        F_max=0.3
    )
    
    feed_strategy = PiecewiseFeed([
        (0.0, batch_feed),
        (5.0, do_stat_feed)  # Start DO-stat at 5h
    ])
    
    print(f"\nFeed strategy:")
    print(f"  0-5 h: Batch (F = 0)")
    print(f"  5+ h: DO-stat")
    print(f"    DO setpoint: {do_stat_feed.DO_setpoint} mmol/L")
    print(f"    K_p: {do_stat_feed.K_p} (negative feedback)")
    print(f"    F range: {do_stat_feed.F_min}-{do_stat_feed.F_max} L/h")
    
    # ========== DO CASCADE CONTROL ==========
    reactor_config = LAB_STR_5L
    
    do_control = SimplifiedCascade(
        DO_setpoint=0.06,
        N_min=reactor_config.N_min,
        N_max=reactor_config.N_max,
        Q_gas_min=reactor_config.Q_gas_min * initial_state.V * 60,
        Q_gas_max=reactor_config.Q_gas_max * initial_state.V * 60,
        kLa_correlation=DynamicKLa(
            k_O2=reactor_config.k_O2,
            a=reactor_config.a,
            b=reactor_config.b,
            k_X=reactor_config.k_X
        )
    )
    
    print(f"\nDO cascade control:")
    print(f"  Setpoint: {do_control.DO_sp} mmol/L")
    print(f"  Note: Both feed control AND agitation/aeration control DO!")
    
    # ========== TEMPERATURE CONTROL ==========
    temp_controller = TemperatureController(
        T_setpoint=37.0,
        K_p=10.0,
        K_i=0.5
    )
    
    # ========== CREATE AND RUN REACTOR ==========
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
    
    print("=" * 70)
    
    simulator = FedBatchSimulator(reactor)
    results = simulator.simulate(t_end=25.0, max_step=0.1)
    
    # ========== ANALYZE RESULTS ==========
    print("\n" + "=" * 70)
    print(results.get_summary())
    print("=" * 70)
    
    # DO-stat performance
    import numpy as np
    
    # Find DO-stat period
    do_stat_period = results.time >= 5.0
    
    if np.any(do_stat_period):
        DO_during_dostat = results.DO[do_stat_period]
        F_during_dostat = results.F_feed[do_stat_period]
        
        print("\nDO-stat performance (t > 5h):")
        print(f"  DO mean: {np.mean(DO_during_dostat):.4f} mmol/L")
        print(f"  DO std:  {np.std(DO_during_dostat):.4f} mmol/L")
        print(f"  DO setpoint: {do_stat_feed.DO_setpoint} mmol/L")
        print(f"  Feed rate mean: {np.mean(F_during_dostat):.3f} L/h")
        print(f"  Feed rate range: {np.min(F_during_dostat):.3f}-{np.max(F_during_dostat):.3f} L/h")
    
    # Final performance
    final_X = results.X[-1]
    final_V = results.V[-1]
    final_t = results.time[-1]
    total_biomass = final_X * final_V
    productivity = total_biomass / final_t
    
    print(f"\nProcess performance:")
    print(f"  Total biomass: {total_biomass:.2f} g")
    print(f"  Productivity: {productivity:.2f} g/L/h")
    
    print("\n" + "=" * 70)
    print("DO-stat simulation complete!")
    print("=" * 70)
    
    return results


if __name__ == "__main__":
    results = main()
