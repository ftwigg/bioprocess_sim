# example_batch_simulation.py
"""
Example script demonstrating batch reactor simulation capabilities.

This script shows how to:
1. Set up a batch reactor with E. coli parameters
2. Run a simulation
3. Visualize results
4. Compare different conditions
5. Export data
"""

import numpy as np
import matplotlib.pyplot as plt

from base_models import CellParameters, ReactorConfig, ProductionModel
from reactor import BatchReactor
from simulator import BatchSimulator, run_quick_simulation
from utils import (
    plot_batch_results, plot_comparison, print_summary_table,
    calculate_yields, analyze_phase, find_exponential_phase,
    export_to_csv
)


def example_1_basic_simulation():
    """
    Example 1: Basic batch fermentation with E. coli on glucose.
    """
    print("\n" + "="*70)
    print("EXAMPLE 1: Basic E. coli Batch Fermentation")
    print("="*70)
    
    # Define cell parameters (typical E. coli on glucose)
    cell_params = CellParameters(
        mu_max=0.7,  # 1/h - maximum growth rate
        Ks=0.1,  # g/L - Monod constant
        Yx_s_max=0.5,  # g/g - maximum biomass yield
        ms=0.03,  # g/g/h - maintenance coefficient
        Yp_s=0.0,  # No product formation
        production_model=ProductionModel.GROWTH_ASSOCIATED,
        RQ=1.0  # Respiratory quotient
    )
    
    # Define reactor configuration
    config = ReactorConfig(
        V_reactor=2.0,  # L - total volume
        V_working=1.5,  # L - working volume
        kLa_O2=100.0,  # 1/h - oxygen mass transfer coefficient
        kLa_CO2=80.0,  # 1/h - CO2 mass transfer coefficient
        T_set=37.0,  # °C - temperature setpoint
        X0=0.1,  # g/L - initial biomass
        S0=20.0,  # g/L - initial glucose
        P0=0.0,  # g/L - initial product
        pH0=7.0  # Initial pH
    )
    
    # Create reactor
    reactor = BatchReactor(cell_params, config)
    
    # Create simulator
    simulator = BatchSimulator(reactor)
    
    # Run simulation
    results = simulator.simulate(
        t_end=24.0,  # hours
        dt=0.1,  # time step for output
        method='LSODA',  # integration method
        verbose=True
    )
    
    # Print summary
    print_summary_table(results, config.X0, config.S0, config.P0)
    
    # Analyze exponential phase
    t_start, t_end, avg_mu = find_exponential_phase(results)
    print(f"Exponential Phase:")
    print(f"  Duration: {t_start:.2f} - {t_end:.2f} h")
    print(f"  Average μ: {avg_mu:.4f} 1/h")
    
    # Plot results
    plot_batch_results(results, save_path='example1_results.png')
    
    # Export data
    export_to_csv(results, 'example1_data.csv')
    
    return results


def example_2_product_formation():
    """
    Example 2: Fermentation with product formation (e.g., lactate).
    """
    print("\n" + "="*70)
    print("EXAMPLE 2: Lactic Acid Fermentation")
    print("="*70)
    
    # Lactic acid bacteria parameters
    cell_params = CellParameters(
        mu_max=0.5,
        Ks=0.2,
        Yx_s_max=0.15,  # Lower yield - more goes to product
        Yp_s=0.9,  # High product yield
        ms=0.01,
        production_model=ProductionModel.MIXED,
        alpha=1.5,  # Growth-associated production
        beta=0.1,  # Non-growth-associated production
        RQ=0.0  # Fermentative - no oxygen consumption
    )
    
    config = ReactorConfig(
        V_working=1.0,
        X0=0.1,
        S0=50.0,  # High glucose for lactic acid production
        P0=0.0,
        pH0=6.5,
        T_set=37.0
    )
    
    reactor = BatchReactor(cell_params, config)
    simulator = BatchSimulator(reactor)
    
    results = simulator.simulate(t_end=30.0, verbose=True)
    
    print_summary_table(results, config.X0, config.S0, config.P0)
    
    plot_batch_results(results, save_path='example2_lactate.png')
    
    return results


def example_3_parameter_comparison():
    """
    Example 3: Compare different growth rates.
    """
    print("\n" + "="*70)
    print("EXAMPLE 3: Parameter Sensitivity - Growth Rate")
    print("="*70)
    
    # Test different mu_max values
    mu_values = [0.3, 0.5, 0.7, 0.9]
    results_list = []
    labels = []
    
    for mu_max in mu_values:
        print(f"\nSimulating with μ_max = {mu_max} 1/h...")
        
        results = run_quick_simulation(
            mu_max=mu_max,
            Ks=0.1,
            Yx_s=0.5,
            ms=0.03,
            S0=20.0,
            X0=0.1,
            t_end=24.0
        )
        
        results_list.append(results)
        labels.append(f'μ_max = {mu_max}')
        
        yields = calculate_yields(results, 0.1, 20.0)
        print(f"  Final biomass: {yields['final_biomass']:.3f} g/L")
        print(f"  Yx/s: {yields['Yx_s']:.3f}")
    
    # Compare biomass profiles
    plot_comparison(results_list, labels, variable='X')
    
    # Compare growth rates
    plot_comparison(results_list, labels, variable='mu')
    
    return results_list


def example_4_temperature_effect():
    """
    Example 4: Effect of temperature (via Arrhenius relationship).
    
    Note: This is simplified - real implementation would modify mu_max
    with temperature dependence.
    """
    print("\n" + "="*70)
    print("EXAMPLE 4: Temperature Effects")
    print("="*70)
    
    # Simulate at different temperatures
    # Assume μ_max follows Arrhenius: μ(T) = μ_ref * exp[Ea/R * (1/T_ref - 1/T)]
    
    temperatures = [25, 30, 37, 42]  # °C
    mu_ref = 0.7  # at 37°C
    T_ref = 37 + 273.15  # K
    Ea_R = 5000  # Activation energy / R (K)
    
    results_list = []
    labels = []
    
    for T in temperatures:
        T_K = T + 273.15
        
        # Temperature-adjusted growth rate
        mu_max_T = mu_ref * np.exp(Ea_R * (1/T_ref - 1/T_K))
        
        print(f"\nT = {T}°C, μ_max = {mu_max_T:.4f} 1/h")
        
        results = run_quick_simulation(
            mu_max=mu_max_T,
            Ks=0.1,
            Yx_s=0.5,
            ms=0.03,
            S0=20.0,
            t_end=24.0
        )
        
        results_list.append(results)
        labels.append(f'{T}°C')
    
    plot_comparison(results_list, labels, variable='X')
    
    return results_list


def example_5_substrate_limitation():
    """
    Example 5: Different initial substrate concentrations.
    """
    print("\n" + "="*70)
    print("EXAMPLE 5: Substrate Limitation")
    print("="*70)
    
    S0_values = [5, 10, 20, 40]  # g/L
    
    results_list = []
    labels = []
    
    for S0 in S0_values:
        print(f"\nSimulating with S0 = {S0} g/L...")
        
        results = run_quick_simulation(
            mu_max=0.7,
            Ks=0.1,
            Yx_s=0.5,
            ms=0.03,
            S0=S0,
            X0=0.1,
            t_end=30.0
        )
        
        results_list.append(results)
        labels.append(f'S₀ = {S0} g/L')
        
        yields = calculate_yields(results, 0.1, S0)
        print(f"  Final biomass: {yields['final_biomass']:.3f} g/L")
        print(f"  Final substrate: {yields['final_substrate']:.3f} g/L")
    
    # Compare substrate consumption
    plt.figure(figsize=(10, 6))
    for results, label in zip(results_list, labels):
        plt.plot(results.time, results.S, linewidth=2, label=label)
    plt.xlabel('Time (h)', fontsize=12)
    plt.ylabel('Substrate (g/L)', fontsize=12)
    plt.title('Substrate Profiles', fontsize=14, fontweight='bold')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()
    
    # Compare final biomass
    final_biomass = [r.X[-1] for r in results_list]
    
    plt.figure(figsize=(8, 6))
    plt.bar(range(len(S0_values)), final_biomass, tick_label=[f'{s}' for s in S0_values])
    plt.xlabel('Initial Substrate (g/L)', fontsize=12)
    plt.ylabel('Final Biomass (g/L)', fontsize=12)
    plt.title('Substrate Limitation Effect', fontsize=14, fontweight='bold')
    plt.grid(True, alpha=0.3, axis='y')
    plt.tight_layout()
    plt.show()
    
    return results_list


def example_6_maintenance_effect():
    """
    Example 6: Effect of maintenance coefficient.
    """
    print("\n" + "="*70)
    print("EXAMPLE 6: Maintenance Energy Effect")
    print("="*70)
    
    ms_values = [0.0, 0.01, 0.03, 0.05]  # g/g/h
    
    results_list = []
    labels = []
    
    for ms in ms_values:
        print(f"\nSimulating with ms = {ms} g/g/h...")
        
        results = run_quick_simulation(
            mu_max=0.7,
            Ks=0.1,
            Yx_s=0.5,
            ms=ms,
            S0=20.0,
            X0=0.1,
            t_end=24.0
        )
        
        results_list.append(results)
        labels.append(f'ms = {ms}')
        
        yields = calculate_yields(results, 0.1, 20.0)
        print(f"  Yx/s: {yields['Yx_s']:.4f}")
        print(f"  Final biomass: {yields['final_biomass']:.3f} g/L")
    
    # Compare biomass and yields
    plot_comparison(results_list, labels, variable='X')
    
    # Plot yield vs maintenance
    actual_yields = [calculate_yields(r, 0.1, 20.0)['Yx_s'] for r in results_list]
    
    plt.figure(figsize=(8, 6))
    plt.plot(ms_values, actual_yields, 'bo-', linewidth=2, markersize=10)
    plt.axhline(y=0.5, color='r', linestyle='--', label='Yx/s,max')
    plt.xlabel('Maintenance Coefficient (g/g/h)', fontsize=12)
    plt.ylabel('Actual Yield Yx/s (g/g)', fontsize=12)
    plt.title('Effect of Maintenance on Yield', fontsize=14, fontweight='bold')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()
    
    return results_list


def main():
    """
    Run all examples.
    """
    print("\n")
    print("*" * 70)
    print("*" + " " * 68 + "*")
    print("*" + "  BATCH REACTOR SIMULATION - EXAMPLES".center(68) + "*")
    print("*" + " " * 68 + "*")
    print("*" * 70)
    
    # Run examples
    try:
        results1 = example_1_basic_simulation()
        results2 = example_2_product_formation()
        results_list3 = example_3_parameter_comparison()
        results_list4 = example_4_temperature_effect()
        results_list5 = example_5_substrate_limitation()
        results_list6 = example_6_maintenance_effect()
        
        print("\n" + "="*70)
        print("ALL EXAMPLES COMPLETED SUCCESSFULLY!")
        print("="*70 + "\n")
        
    except Exception as e:
        print(f"\nError during simulation: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    # Check if matplotlib is available
    try:
        import matplotlib.pyplot as plt
        print("✓ Matplotlib available - visualizations enabled")
    except ImportError:
        print("⚠ Matplotlib not available - visualizations disabled")
    
    # Check if pandas is available
    try:
        import pandas as pd
        print("✓ Pandas available - CSV export enabled")
    except ImportError:
        print("⚠ Pandas not available - CSV export disabled")
    
    main()
