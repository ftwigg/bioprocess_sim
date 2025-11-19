# utils.py
"""
Utility functions for visualization, analysis, and data export.
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import Optional, List, Dict
import json

from base_models import SimulationResults


def plot_batch_results(results: SimulationResults, 
                      figsize: tuple = (15, 10),
                      save_path: Optional[str] = None):
    """
    Create comprehensive visualization of batch simulation results.
    
    Args:
        results: SimulationResults object
        figsize: Figure size (width, height)
        save_path: Path to save figure (optional)
    """
    fig, axes = plt.subplots(3, 3, figsize=figsize)
    fig.suptitle('Batch Fermentation Simulation Results', fontsize=16, fontweight='bold')
    
    t = results.time
    
    # 1. Biomass, Substrate, Product
    ax = axes[0, 0]
    ax.plot(t, results.X, 'b-', linewidth=2, label='Biomass (X)')
    ax.plot(t, results.S, 'r--', linewidth=2, label='Substrate (S)')
    ax.plot(t, results.P, 'g-.', linewidth=2, label='Product (P)')
    ax.set_xlabel('Time (h)')
    ax.set_ylabel('Concentration (g/L)')
    ax.set_title('Concentrations vs Time')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 2. Growth rate
    ax = axes[0, 1]
    ax.plot(t, results.mu, 'b-', linewidth=2)
    ax.set_xlabel('Time (h)')
    ax.set_ylabel('Specific Growth Rate (1/h)')
    ax.set_title('Growth Rate (μ)')
    ax.grid(True, alpha=0.3)
    
    # 3. pH
    ax = axes[0, 2]
    ax.plot(t, results.pH, 'purple', linewidth=2)
    ax.axhline(y=7.0, color='k', linestyle=':', alpha=0.5, label='Neutral')
    ax.set_xlabel('Time (h)')
    ax.set_ylabel('pH')
    ax.set_title('pH Profile')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 4. OUR and CER
    ax = axes[1, 0]
    ax.plot(t, results.OUR, 'r-', linewidth=2, label='OUR')
    ax.plot(t, results.CER, 'b--', linewidth=2, label='CER')
    ax.set_xlabel('Time (h)')
    ax.set_ylabel('Rate (mmol/L/h)')
    ax.set_title('Oxygen Uptake & CO₂ Evolution')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 5. RQ (Respiratory Quotient)
    ax = axes[1, 1]
    RQ = np.divide(results.CER, results.OUR, 
                   out=np.ones_like(results.CER), 
                   where=results.OUR>0)
    ax.plot(t, RQ, 'g-', linewidth=2)
    ax.axhline(y=1.0, color='k', linestyle=':', alpha=0.5, label='RQ = 1')
    ax.set_xlabel('Time (h)')
    ax.set_ylabel('RQ (CER/OUR)')
    ax.set_title('Respiratory Quotient')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_ylim([0, 2])
    
    # 6. Heat generation
    ax = axes[1, 2]
    ax.plot(t, results.Q_gen, 'orange', linewidth=2)
    ax.set_xlabel('Time (h)')
    ax.set_ylabel('Heat Generation (W)')
    ax.set_title('Metabolic Heat Generation')
    ax.grid(True, alpha=0.3)
    
    # 7. Cumulative heat
    ax = axes[2, 0]
    cumulative_heat = np.cumsum(results.Q_gen) * np.diff(t, prepend=0) * 3600 / 1000  # kJ
    ax.plot(t, cumulative_heat, 'red', linewidth=2)
    ax.set_xlabel('Time (h)')
    ax.set_ylabel('Cumulative Heat (kJ)')
    ax.set_title('Total Heat Generated')
    ax.grid(True, alpha=0.3)
    
    # 8. Substrate consumption rate
    ax = axes[2, 1]
    dS_dt = -np.gradient(results.S, t)
    ax.plot(t, dS_dt, 'r-', linewidth=2)
    ax.set_xlabel('Time (h)')
    ax.set_ylabel('dS/dt (g/L/h)')
    ax.set_title('Substrate Consumption Rate')
    ax.grid(True, alpha=0.3)
    
    # 9. Phase plot: X vs S
    ax = axes[2, 2]
    ax.plot(results.S, results.X, 'b-', linewidth=2)
    ax.scatter(results.S[0], results.X[0], c='green', s=100, 
               marker='o', label='Start', zorder=5)
    ax.scatter(results.S[-1], results.X[-1], c='red', s=100, 
               marker='s', label='End', zorder=5)
    ax.set_xlabel('Substrate (g/L)')
    ax.set_ylabel('Biomass (g/L)')
    ax.set_title('Phase Plot: X vs S')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Figure saved to {save_path}")
    
    plt.show()


def plot_comparison(results_list: List[SimulationResults],
                   labels: List[str],
                   variable: str = 'X',
                   figsize: tuple = (10, 6)):
    """
    Compare multiple simulation results.
    
    Args:
        results_list: List of SimulationResults
        labels: Labels for each simulation
        variable: Variable to plot ('X', 'S', 'P', 'mu', 'OUR', etc.)
        figsize: Figure size
    """
    plt.figure(figsize=figsize)
    
    for results, label in zip(results_list, labels):
        if variable == 'X':
            plt.plot(results.time, results.X, linewidth=2, label=label)
            ylabel = 'Biomass (g/L)'
        elif variable == 'S':
            plt.plot(results.time, results.S, linewidth=2, label=label)
            ylabel = 'Substrate (g/L)'
        elif variable == 'P':
            plt.plot(results.time, results.P, linewidth=2, label=label)
            ylabel = 'Product (g/L)'
        elif variable == 'mu':
            plt.plot(results.time, results.mu, linewidth=2, label=label)
            ylabel = 'Growth Rate (1/h)'
        elif variable == 'OUR':
            plt.plot(results.time, results.OUR, linewidth=2, label=label)
            ylabel = 'OUR (mmol/L/h)'
        else:
            raise ValueError(f"Unknown variable: {variable}")
    
    plt.xlabel('Time (h)', fontsize=12)
    plt.ylabel(ylabel, fontsize=12)
    plt.title(f'Comparison: {ylabel}', fontsize=14, fontweight='bold')
    plt.legend(fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()


def calculate_yields(results: SimulationResults, 
                    X0: float, S0: float, P0: float = 0.0) -> Dict[str, float]:
    """
    Calculate overall yields from simulation results.
    
    Args:
        results: SimulationResults object
        X0: Initial biomass (g/L)
        S0: Initial substrate (g/L)
        P0: Initial product (g/L)
        
    Returns:
        Dictionary of yields
    """
    # Final values
    X_final = results.X[-1]
    S_final = results.S[-1]
    P_final = results.P[-1]
    
    # Changes
    delta_X = X_final - X0
    delta_S = S0 - S_final
    delta_P = P_final - P0
    
    # Yields
    yields = {
        'Yx_s': delta_X / delta_S if delta_S > 0 else 0.0,
        'Yp_s': delta_P / delta_S if delta_S > 0 else 0.0,
        'Yp_x': delta_P / delta_X if delta_X > 0 else 0.0,
        'substrate_consumed': delta_S,
        'biomass_formed': delta_X,
        'product_formed': delta_P,
        'final_biomass': X_final,
        'final_substrate': S_final,
        'final_product': P_final
    }
    
    return yields


def calculate_productivities(results: SimulationResults,
                            X0: float, S0: float) -> Dict[str, float]:
    """
    Calculate volumetric and specific productivities.
    
    Args:
        results: SimulationResults object
        X0: Initial biomass (g/L)
        S0: Initial substrate (g/L)
        
    Returns:
        Dictionary of productivities
    """
    t_final = results.time[-1]
    X_final = results.X[-1]
    P_final = results.P[-1]
    
    delta_X = X_final - X0
    
    productivities = {
        'Qx': delta_X / t_final,  # Biomass productivity (g/L/h)
        'Qp': P_final / t_final,  # Product productivity (g/L/h)
        'qp': P_final / (X_final * t_final) if X_final > 0 else 0.0,  # Specific productivity
        'duration': t_final
    }
    
    return productivities


def export_to_csv(results: SimulationResults, filename: str):
    """
    Export simulation results to CSV file.
    
    Args:
        results: SimulationResults object
        filename: Output CSV filename
    """
    import pandas as pd
    
    data = {
        'time_h': results.time,
        'biomass_g_L': results.X,
        'substrate_g_L': results.S,
        'product_g_L': results.P,
        'pH': results.pH,
        'growth_rate_1_h': results.mu,
        'OUR_mmol_L_h': results.OUR,
        'CER_mmol_L_h': results.CER,
        'heat_generation_W': results.Q_gen
    }
    
    # Add extra variables
    for key, values in results.extra_vars.items():
        data[key] = values
    
    df = pd.DataFrame(data)
    df.to_csv(filename, index=False)
    print(f"Results exported to {filename}")


def export_to_json(results: SimulationResults, filename: str):
    """
    Export simulation results to JSON file.
    
    Args:
        results: SimulationResults object
        filename: Output JSON filename
    """
    data = {
        'time': results.time.tolist(),
        'biomass': results.X.tolist(),
        'substrate': results.S.tolist(),
        'product': results.P.tolist(),
        'pH': results.pH.tolist(),
        'growth_rate': results.mu.tolist(),
        'OUR': results.OUR.tolist(),
        'CER': results.CER.tolist(),
        'heat_generation': results.Q_gen.tolist(),
        'summary': results.get_summary()
    }
    
    with open(filename, 'w') as f:
        json.dump(data, f, indent=2)
    
    print(f"Results exported to {filename}")


def print_summary_table(results: SimulationResults,
                       X0: float, S0: float, P0: float = 0.0):
    """
    Print a formatted summary table of results.
    
    Args:
        results: SimulationResults object
        X0: Initial biomass
        S0: Initial substrate
        P0: Initial product
    """
    summary = results.get_summary()
    yields = calculate_yields(results, X0, S0, P0)
    productivities = calculate_productivities(results, X0, S0)
    
    print("\n" + "="*70)
    print("BATCH FERMENTATION SUMMARY")
    print("="*70)
    
    print("\nFinal Concentrations:")
    print(f"  Biomass:    {summary['final_biomass']:8.3f} g/L")
    print(f"  Substrate:  {summary['final_substrate']:8.3f} g/L")
    print(f"  Product:    {summary['final_product']:8.3f} g/L")
    
    print("\nYields:")
    print(f"  Yx/s:       {yields['Yx_s']:8.4f} g biomass / g substrate")
    print(f"  Yp/s:       {yields['Yp_s']:8.4f} g product / g substrate")
    if yields['Yp_x'] > 0:
        print(f"  Yp/x:       {yields['Yp_x']:8.4f} g product / g biomass")
    
    print("\nProductivities:")
    print(f"  Biomass:    {productivities['Qx']:8.4f} g/L/h")
    print(f"  Product:    {productivities['Qp']:8.4f} g/L/h")
    if productivities['qp'] > 0:
        print(f"  Specific:   {productivities['qp']:8.4f} g product / g biomass / h")
    
    print("\nKinetics:")
    print(f"  Max μ:      {summary['max_growth_rate']:8.4f} 1/h")
    print(f"  Avg OUR:    {summary['avg_OUR']:8.2f} mmol/L/h")
    
    print("\nEnergetics:")
    print(f"  Total heat: {summary['total_heat']:8.2f} kJ")
    
    print("\nDuration:")
    print(f"  Time:       {productivities['duration']:8.2f} h")
    
    print("="*70 + "\n")


def analyze_phase(results: SimulationResults, 
                 t_start: float = 0, 
                 t_end: Optional[float] = None) -> Dict[str, float]:
    """
    Analyze a specific phase of the fermentation.
    
    Args:
        results: SimulationResults object
        t_start: Start time of phase (h)
        t_end: End time of phase (h), None for end of simulation
        
    Returns:
        Dictionary with phase-specific metrics
    """
    if t_end is None:
        t_end = results.time[-1]
    
    # Find indices for time window
    idx = (results.time >= t_start) & (results.time <= t_end)
    
    if not np.any(idx):
        return {}
    
    phase_analysis = {
        'duration': t_end - t_start,
        'avg_growth_rate': np.mean(results.mu[idx]),
        'max_growth_rate': np.max(results.mu[idx]),
        'avg_OUR': np.mean(results.OUR[idx]),
        'max_OUR': np.max(results.OUR[idx]),
        'biomass_change': results.X[idx][-1] - results.X[idx][0],
        'substrate_consumed': results.S[idx][0] - results.S[idx][-1],
        'product_formed': results.P[idx][-1] - results.P[idx][0],
        'heat_generated': np.trapz(results.Q_gen[idx], results.time[idx]) * 3600 / 1000  # kJ
    }
    
    return phase_analysis


def find_exponential_phase(results: SimulationResults,
                          mu_threshold: float = 0.9) -> tuple:
    """
    Identify exponential growth phase.
    
    Args:
        results: SimulationResults object
        mu_threshold: Fraction of mu_max to consider exponential (default 0.9)
        
    Returns:
        Tuple of (start_time, end_time, avg_mu)
    """
    mu_max = np.max(results.mu)
    threshold = mu_threshold * mu_max
    
    # Find where mu > threshold
    in_exp_phase = results.mu >= threshold
    
    if not np.any(in_exp_phase):
        return (0, 0, 0)
    
    # Find continuous region
    indices = np.where(in_exp_phase)[0]
    start_idx = indices[0]
    end_idx = indices[-1]
    
    t_start = results.time[start_idx]
    t_end = results.time[end_idx]
    avg_mu = np.mean(results.mu[start_idx:end_idx+1])
    
    return (t_start, t_end, avg_mu)
