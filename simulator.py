"""
Fed-batch simulator with ODE integration.

Provides high-level interface for running simulations.
"""

import numpy as np
from scipy.integrate import solve_ivp
from typing import Optional

from .reactor import FedBatchReactor
from .base_models import SimulationResults


class FedBatchSimulator:
    """ODE integration and simulation control."""
    
    def __init__(self, reactor: FedBatchReactor):
        """
        Initialize simulator.
        
        Args:
            reactor: FedBatchReactor instance
        """
        self.reactor = reactor
    
    def simulate(self,
                t_end: float,
                method: str = 'BDF',
                rtol: float = 1e-6,
                atol: float = 1e-8,
                max_step: float = 0.1) -> SimulationResults:
        """
        Run fed-batch simulation.
        
        Args:
            t_end: End time (hours)
            method: ODE solver ('BDF' for stiff, 'RK45' for non-stiff)
            rtol: Relative tolerance
            atol: Absolute tolerance
            max_step: Maximum time step (hours)
        
        Returns:
            SimulationResults object
        """
        # Initial state
        y0 = self.reactor.get_state_vector()
        t_span = (0, t_end)
        
        # Event function for termination
        def termination_event(t, y):
            return -1 if self.reactor.check_termination(t, y) else 1
        termination_event.terminal = True
        
        # Solve ODE
        print(f"Starting simulation: 0 to {t_end} h")
        print(f"Solver: {method}, rtol={rtol}, atol={atol}")
        print(f"Initial state: X={y0[0]:.2f} g/L, V={y0[6]:.2f} L, T={y0[7]:.1f}°C")
        
        sol = solve_ivp(
            fun=self.reactor.derivatives,
            t_span=t_span,
            y0=y0,
            method=method,
            rtol=rtol,
            atol=atol,
            max_step=max_step,
            events=termination_event,
            dense_output=False
        )
        
        if not sol.success:
            raise RuntimeError(f"ODE solver failed: {sol.message}")
        
        print(f"Simulation completed: {sol.t[-1]:.2f} h")
        print(f"Function evaluations: {sol.nfev}")
        
        # Process results
        results = self._process_results(sol)
        
        return results
    
    def _process_results(self, sol) -> SimulationResults:
        """Convert solver output to SimulationResults."""
        # Convert history to arrays
        history_arrays = {
            key: np.array(val) for key, val in self.reactor.history.items()
        }
        
        results = SimulationResults(
            time=history_arrays['time'],
            X=history_arrays['X'],
            S_carbon=history_arrays['S_carbon'],
            S_nitrogen=history_arrays['S_nitrogen'],
            P=history_arrays['P'],
            DO=history_arrays['DO'],
            DCO2=history_arrays['DCO2'],
            V=history_arrays['V'],
            T=history_arrays['T'],
            N=history_arrays['N'],
            mu=history_arrays['mu'],
            OUR=history_arrays['OUR'],
            CER=history_arrays['CER'],
            OTR=history_arrays['OTR'],
            CTR=history_arrays['CTR'],
            Q_gas=history_arrays['Q_gas'],
            kLa=history_arrays['kLa'],
            pH=history_arrays['pH'],
            T_jacket=history_arrays['T_jacket'],
            F_feed=history_arrays['F_feed'],
            D=history_arrays['D']
        )
        
        return results
