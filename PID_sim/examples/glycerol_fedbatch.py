"""
Example: Glycerol fed-batch simulation.

Demonstrates:
- Custom carbon source (glycerol instead of glucose)
- Automatic yield calculation from elemental balance
- Different stoichiometry and RQ
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
    print("GLYCEROL FED-BATCH SIMULATION")
    print("=" * 70)
    print("\nDemonstrates: Generic carbon source architecture")
    print("  - Glycerol (C3H8O3) instead of glucose (C6H12O6)")
    print("  - Yields calculated automatically from elemental balance")
    print("  - Different stoichiometry and RQ")
    
    # ========== COMPARE CARBON SOURCES ==========
    glucose = GLUCOSE
    glycerol = GLYCEROL
    
    print(f"\n" + "=" * 70)
    print("CARBON SOURCE COMPARISON")
    print("=" * 70)
    
    print(f"\nGlucose:")
    print(f"  Formula: {glucose.formula}")
    print(f"  MW: {glucose.MW} g/mol")
    print(f"  Degree of reduction: {glucose.degree_of_reduction:.2f}")
    
    print(f"\nGlycerol:")
    print(f"  Formula: {glycerol.formula}")
    print(f"  MW: {glycerol.MW} g/mol")
    print(f"  Degree of reduction: {glycerol.degree_of_reduction:.2f}")
    
    # ========== CALCULATE YIELDS FOR BOTH ==========
    from fedbatch_simulator.stoichiometry import (
        calculate_oxygen_yield,
        calculate_nitrogen_yield,
        calculate_RQ
    )
    
    Y_xs = 0.5  # Same yield on both substrates
    biomass = STANDARD_BIOMASS
    
    # Glucose
    Y_x_O2_glucose = calculate_oxygen_yield(glucose, biomass, Y_xs)
    Y_x_N_glucose = calculate_nitrogen_yield(biomass, Y_xs, glucose)
    RQ_glucose = calculate_RQ(glucose, biomass, Y_xs)
    
    # Glycerol
    Y_x_O2_glycerol = calculate_oxygen_yield(glycerol, biomass, Y_xs)
    Y_x_N_glycerol = calculate_nitrogen_yield(biomass, Y_xs, glycerol)
    RQ_glycerol = calculate_RQ(glycerol, biomass, Y_xs)
    
    print(f"\n" + "=" * 70)
    print("STOICHIOMETRY COMPARISON (Y_x/S = 0.5 g/g for both)")
    print("=" * 70)
    
    print(f"\n{'Parameter':<20} {'Glucose':<15} {'Glycerol':<15}")
    print("-" * 70)
    print(f"{'Y_x/O2 (g/g)':<20} {Y_x_O2_glucose:<15.3f} {Y_x_O2_glycerol:<15.3f}")
    print(f"{'Y_x/N (g/g)':<20} {Y_x_N_glucose:<15.3f} {Y_x_N_glycerol:<15.3f}")
    print(f"{'RQ (mol/mol)':<20} {RQ_glucose:<15.3f} {RQ_glycerol:<15.3f}")
    
    print(f"\nKey insight:")
    print(f"  - Glycerol is more reduced (γ={glycerol.degree_of_reduction:.2f} vs {glucose.degree_of_reduction:.2f})")
    print(f"  - Requires MORE oxygen per biomass (lower Y_x/O2)")
    print(f"  - Higher RQ (more CO2 per O2)")
    
    # ========== CELL PARAMETERS WITH GLYCEROL ==========
    carbon_params = SubstrateParameters(
        name='carbon',
        Ks=0.15,  # Slightly higher for glycerol
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
        mu_max=0.4,  # Slightly lower for glycerol
        carbon_source=glycerol,  # ← Using glycerol!
        biomass_composition=STANDARD_BIOMASS,
        substrates={
            'carbon': carbon_params,
            'nitrogen': nitrogen_params
        },
        Y_xs=0.5,
        Ks_O2=0.003,
        m_O2=0.01
    )
    
    print(f"\n" + "=" * 70)
    print("CELL PARAMETERS (GLYCEROL)")
    print("=" * 70)
    print(f"  μ_max: {cell_params.mu_max} 1/h")
    print(f"  Y_x/S: {cell_params.Y_xs} g/g")
    print(f"  Y_x/O2: {cell_params.Y_x_O2:.3f} g/g (auto-calculated)")
    print(f"  Y_x/N: {cell_params.Y_x_N:.3f} g/g (auto-calculated)")
    print(f"  RQ: {cell_params.RQ:.3f} (auto-calculated)")
    
    # ========== INITIAL CONDITIONS ==========
    initial_state = ReactorState(
        time=0.0,
        X=0.5,
        S_carbon=10.0,  # g/L glycerol
        S_nitrogen=2.0,
        P=0.0,
        DO=0.2,
        DCO2=0.5,
        V=2.0,
        T=37.0,
        N=300,
        pH=7.0
    )
    
    # ========== FEED STRATEGY ==========
    feed_composition = FeedComposition(
        S_carbon=600.0,  # g/L concentrated glycerol (higher density than glucose)
        S_nitrogen=50.0,
        temperature=25.0,
        pH=5.0
    )
    
    batch_feed = ConstantFeed(feed_composition, F_constant=0.0)
    exp_feed = ExponentialFeed(
        feed_composition=feed_composition,
        F0=0.01,
        mu_set=0.15,  # Lower for glycerol
        F_max=0.4
    )
    exp_feed.t_start = 10.0
    
    feed_strategy = PiecewiseFeed([
        (0.0, batch_feed),
        (10.0, exp_feed)
    ])
    
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
    
    # ========== TEMPERATURE CONTROL ==========
    temp_controller = TemperatureController(
        T_setpoint=37.0,
        K_p=10.0,
        K_i=0.5
    )
    
    # ========== CREATE AND RUN REACTOR ==========
    print(f"\n" + "=" * 70)
    print("Creating reactor with GLYCEROL as carbon source...")
    print("=" * 70)
    
    reactor = FedBatchReactor(
        cell_params=cell_params,
        reactor_config=reactor_config,
        initial_state=initial_state,
        feed_strategy=feed_strategy,
        do_control=do_control,
        temp_controller=temp_controller
    )
    
    simulator = FedBatchSimulator(reactor)
    results = simulator.simulate(t_end=30.0, max_step=0.1)
    
    # ========== ANALYZE RESULTS ==========
    print("\n" + "=" * 70)
    print(results.get_summary())
    print("=" * 70)
    
    # Compare OUR and CER
    import numpy as np
    
    # Average during fed-batch phase
    fedbatch_period = results.time >= 10.0
    if np.any(fedbatch_period):
        OUR_avg = np.mean(results.OUR[fedbatch_period])
        CER_avg = np.mean(results.CER[fedbatch_period])
        RQ_measured = CER_avg / OUR_avg if OUR_avg > 0 else 0
        
        print(f"\nGlycerol fermentation characteristics:")
        print(f"  Average OUR (fed-batch): {OUR_avg:.2f} mmol/L/h")
        print(f"  Average CER (fed-batch): {CER_avg:.2f} mmol/L/h")
        print(f"  Measured RQ: {RQ_measured:.3f}")
        print(f"  Theoretical RQ: {cell_params.RQ:.3f}")
        print(f"  Match: {'✓' if abs(RQ_measured - cell_params.RQ) < 0.1 else '✗'}")
    
    # Final performance
    final_X = results.X[-1]
    final_V = results.V[-1]
    final_t = results.time[-1]
    total_biomass = final_X * final_V
    
    print(f"\nProcess performance:")
    print(f"  Total biomass: {total_biomass:.2f} g")
    print(f"  Final concentration: {final_X:.2f} g/L")
    
    print("\n" + "=" * 70)
    print("Glycerol fed-batch simulation complete!")
    print("Demonstrates: Simulator works for ANY carbon source!")
    print("=" * 70)
    
    return results


if __name__ == "__main__":
    results = main()
