# simulator.py
"""
Simulation engine for batch reactor.
Handles ODE integration and time-stepping.
"""

import numpy as np
from scipy.integrate import solve_ivp, odeint
from typing import Optional, Callable, Dict
import time as time_module

from reactor import BatchReactor
from base_models import SimulationResults


class BatchSimulator:
    """
    Simulator for batch fermentation.
    
    Uses scipy's ODE solvers to integrate the reactor equations over time.
    Provides options for different integration methods and adaptive time-stepping.
    """
    
    def __init__(self, reactor: BatchReactor):
        """
        Initialize simulator with a reactor.
        
        Args:
            reactor: BatchReactor instance to simulate
        """
        self.reactor = reactor
        self.solver_stats = {}
    
    def simulate(self,
                 t_end: float = 24.0,
                 dt: float = 0.1,
                 method: str = 'LSODA',
                 events: Optional[list] = None,
                 verbose: bool = True) -> SimulationResults:
        """
        Run batch simulation.
        
        Args:
            t_end: End time (hours)
            dt: Time step for recording (hours)
            method: Integration method ('LSODA', 'RK45', 'BDF', etc.)
            events: List of event functions to detect
            verbose: Print progress
            
        Returns:
            SimulationResults with trajectories
        """
        if verbose:
            print(f"\nStarting batch simulation...")
            print(f"Method: {method}, Duration: {t_end} h")
            self.reactor.print_summary()
        
        # Set up time points for output
        t_eval = np.arange(0, t_end + dt, dt)
        
        # Initial conditions
        y0 = self.reactor.get_state_vector()
        
        # Define events if substrate depletion should stop simulation
        # Note: Events are complex with scipy solve_ivp, so we'll use None for now
        # and handle termination in post-processing
        events_to_use = events  # Use provided events or None
        
        # Record start time
        start_time = time_module.time()
        
        # Solve ODE system
        try:
            sol = solve_ivp(
                fun=self.reactor.derivatives,
                t_span=(0, t_end),
                y0=y0,
                method=method,
                t_eval=t_eval,
                events=events_to_use,
                dense_output=True,
                vectorized=False,
                rtol=1e-6,
                atol=1e-8
            )
            
            # Store solver statistics
            self.solver_stats = {
                'success': sol.success,
                'message': sol.message,
                'nfev': sol.nfev,  # Number of function evaluations
                'njev': sol.njev if hasattr(sol, 'njev') else 0,
                'nlu': sol.nlu if hasattr(sol, 'nlu') else 0,
                'status': sol.status
            }
            
        except Exception as e:
            print(f"Integration error: {e}")
            raise
        
        # Calculate elapsed time
        elapsed = time_module.time() - start_time
        
        if verbose:
            print(f"\nSimulation completed in {elapsed:.2f} seconds")
            print(f"Function evaluations: {self.solver_stats['nfev']}")
            if not sol.success:
                print(f"Warning: {sol.message}")
        
        # Extract solution
        t = sol.t
        y = sol.y
        
        # Process results
        results = self._process_results(t, y, verbose=verbose)
        
        return results
    
    def _substrate_depleted_event(self, t: float, y: np.ndarray) -> float:
        """
        Event function to detect substrate depletion.
        
        Args:
            t: Current time
            y: Current state vector
            
        Returns:
            Value that crosses zero when substrate depleted
        """
        S = y[1]  # Substrate concentration
        return S - 0.01  # Trigger when S < 0.01 g/L
    
    def _process_results(self, t: np.ndarray, y: np.ndarray, 
                        verbose: bool = True) -> SimulationResults:
        """
        Process raw ODE solution into SimulationResults.
        
        Args:
            t: Time array
            y: Solution array (6 x N)
            verbose: Print final summary
            
        Returns:
            SimulationResults object
        """
        n_points = len(t)
        
        # Extract state variables
        X = y[0, :]
        S = y[1, :]
        P = y[2, :]
        DO = y[3, :]
        DCO2 = y[4, :]
        T = y[5, :]
        
        # Calculate derived quantities at each time point
        mu = np.zeros(n_points)
        OUR = np.zeros(n_points)
        CER = np.zeros(n_points)
        Q_gen = np.zeros(n_points)
        pH = np.zeros(n_points)
        
        for i in range(n_points):
            # Update reactor state for this time point
            self.reactor.update_state(t[i], y[:, i])
            
            # Calculate rates
            from kinetics import calculate_all_rates
            rates = calculate_all_rates(
                self.reactor.cell_params,
                S[i],
                X[i],
                P[i]
            )
            
            mu[i] = rates['mu']
            OUR[i] = rates['OUR']
            CER[i] = rates['CER']
            
            # Heat generation
            Q_gen[i] = self.reactor.heat_gen.heat_from_our(OUR[i], self.reactor.state.V)
            
            # pH
            pH[i] = self.reactor.state.pH
        
        # Create results object
        results = SimulationResults(
            time=t,
            X=X,
            S=S,
            P=P,
            pH=pH,
            mu=mu,
            OUR=OUR,
            CER=CER,
            Q_gen=Q_gen,
            extra_vars={
                'dissolved_O2': DO,
                'dissolved_CO2': DCO2,
                'temperature': T
            }
        )
        
        if verbose:
            print("\nFinal State:")
            self.reactor.print_summary()
            
            summary = results.get_summary()
            print("\nSimulation Summary:")
            print(f"  Final biomass: {summary['final_biomass']:.3f} g/L")
            print(f"  Final substrate: {summary['final_substrate']:.3f} g/L")
            print(f"  Final product: {summary['final_product']:.3f} g/L")
            print(f"  Max growth rate: {summary['max_growth_rate']:.4f} 1/h")
            print(f"  Avg OUR: {summary['avg_OUR']:.2f} mmol/L/h")
            print(f"  Total heat: {summary['total_heat']:.2f} kJ")
            print(f"  Biomass productivity: {summary['biomass_productivity']:.4f} g/L/h")
        
        return results
    
    def simulate_with_control(self,
                            t_end: float = 24.0,
                            dt: float = 0.1,
                            control_function: Optional[Callable] = None,
                            verbose: bool = True) -> SimulationResults:
        """
        Simulate with control actions (e.g., pH control, feed addition).
        
        This is a more advanced simulation that allows external control.
        
        Args:
            t_end: End time (hours)
            dt: Time step (hours)
            control_function: Function(t, reactor) that modifies reactor state
            verbose: Print progress
            
        Returns:
            SimulationResults
        """
        # Time points
        t_span = np.arange(0, t_end + dt, dt)
        n_points = len(t_span)
        
        # Initialize arrays
        X = np.zeros(n_points)
        S = np.zeros(n_points)
        P = np.zeros(n_points)
        pH = np.zeros(n_points)
        mu = np.zeros(n_points)
        OUR = np.zeros(n_points)
        CER = np.zeros(n_points)
        Q_gen = np.zeros(n_points)
        
        # Initial conditions
        X[0] = self.reactor.state.X
        S[0] = self.reactor.state.S
        P[0] = self.reactor.state.P
        pH[0] = self.reactor.state.pH
        
        if verbose:
            print(f"\nStarting controlled simulation...")
            print(f"Duration: {t_end} h, Time step: {dt} h")
        
        # Time-stepping loop
        for i in range(n_points - 1):
            t_current = t_span[i]
            
            # Apply control action if provided
            if control_function is not None:
                control_function(t_current, self.reactor)
            
            # Get current state
            y_current = self.reactor.get_state_vector()
            
            # Integrate one step using solve_ivp
            sol = solve_ivp(
                fun=self.reactor.derivatives,
                t_span=(t_current, t_current + dt),
                y0=y_current,
                method='RK45',
                dense_output=False,
                rtol=1e-6,
                atol=1e-8
            )
            
            # Update reactor state
            y_next = sol.y[:, -1]
            self.reactor.update_state(t_span[i+1], y_next)
            
            # Record state
            X[i+1] = self.reactor.state.X
            S[i+1] = self.reactor.state.S
            P[i+1] = self.reactor.state.P
            pH[i+1] = self.reactor.state.pH
            
            # Calculate rates
            from kinetics import calculate_all_rates
            rates = calculate_all_rates(
                self.reactor.cell_params,
                S[i+1],
                X[i+1],
                P[i+1]
            )
            
            mu[i+1] = rates['mu']
            OUR[i+1] = rates['OUR']
            CER[i+1] = rates['CER']
            Q_gen[i+1] = self.reactor.heat_gen.heat_from_our(
                OUR[i+1], 
                self.reactor.state.V
            )
            
            # Print progress
            if verbose and (i+1) % max(1, n_points // 10) == 0:
                pct = 100 * (i+1) / n_points
                print(f"  Progress: {pct:.0f}% - t={t_span[i+1]:.1f}h, X={X[i+1]:.3f}, S={S[i+1]:.3f}")
        
        # Create results
        results = SimulationResults(
            time=t_span,
            X=X,
            S=S,
            P=P,
            pH=pH,
            mu=mu,
            OUR=OUR,
            CER=CER,
            Q_gen=Q_gen
        )
        
        if verbose:
            print("\nSimulation completed!")
            summary = results.get_summary()
            print(f"  Final biomass: {summary['final_biomass']:.3f} g/L")
            print(f"  Final substrate: {summary['final_substrate']:.3f} g/L")
        
        return results


def run_quick_simulation(
    mu_max: float = 0.7,
    Ks: float = 0.1,
    Yx_s: float = 0.5,
    ms: float = 0.03,
    S0: float = 20.0,
    X0: float = 0.1,
    t_end: float = 24.0
) -> SimulationResults:
    """
    Convenience function for quick simulations with default parameters.
    
    Args:
        mu_max: Maximum growth rate (1/h)
        Ks: Monod constant (g/L)
        Yx_s: Biomass yield (g/g)
        ms: Maintenance coefficient (g/g/h)
        S0: Initial substrate (g/L)
        X0: Initial biomass (g/L)
        t_end: Simulation duration (h)
        
    Returns:
        SimulationResults
    """
    from base_models import CellParameters, ReactorConfig, ProductionModel
    from reactor import BatchReactor
    
    # Create parameters
    cell_params = CellParameters(
        mu_max=mu_max,
        Ks=Ks,
        Yx_s_max=Yx_s,
        ms=ms,
        production_model=ProductionModel.GROWTH_ASSOCIATED,
        Yp_s=0.0
    )
    
    config = ReactorConfig(
        V_working=1.0,
        X0=X0,
        S0=S0,
        P0=0.0
    )
    
    # Create and simulate
    reactor = BatchReactor(cell_params, config)
    simulator = BatchSimulator(reactor)
    
    results = simulator.simulate(t_end=t_end, verbose=False)
    
    return results
