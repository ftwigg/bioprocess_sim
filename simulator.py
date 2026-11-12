"""
Fed-batch simulator with ODE integration.

Provides high-level interface for running simulations.
"""

import numpy as np
from scipy.integrate import solve_ivp
from typing import List, Optional

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
        self.termination_message: Optional[str] = None
    
    def _segment_times(self, t_end: float) -> List[float]:
        """
        Integration break points: start, end, and every time the feed
        strategy switches in between. Restarting the solver at a switch keeps
        it from stepping across the discontinuity.
        """
        switches = getattr(self.reactor.feed_strategy, 'switch_times', [])
        inner = sorted({float(ts) for ts in switches if 0.0 < ts < t_end})
        return [0.0] + inner + [t_end]
    
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
            SimulationResults object. Every row is an accepted solver step,
            so time is monotonic and all states are ones the model actually
            passed through.
        """
        reactor = self.reactor
        y = reactor.get_state_vector()
        
        events = reactor.termination_events()
        event_fns = [fn for _, fn in events]
        
        print(f"Starting simulation: 0 to {t_end} h")
        print(f"Solver: {method}, rtol={rtol}, atol={atol}")
        print(f"Initial state: X={y[0]:.2f} g/L, V={y[6]:.2f} L, T={y[7]:.1f}°C")
        
        times = self._segment_times(t_end)
        t_parts, y_parts, out_parts = [], [], []
        nfev = 0
        self.termination_message = None
        
        for k, (t0, t1) in enumerate(zip(times[:-1], times[1:])):
            sol = solve_ivp(
                fun=reactor.derivatives,
                t_span=(t0, t1),
                y0=y,
                method=method,
                rtol=rtol,
                atol=atol,
                max_step=max_step,
                events=event_fns,
                dense_output=False
            )
            
            if not sol.success:
                raise RuntimeError(f"ODE solver failed: {sol.message}")
            
            nfev += sol.nfev
            
            # Algebraic outputs on the accepted steps. The last point of a
            # segment that ends at a feed switch is evaluated just before the
            # switch, so the step change shows up as two rows at one time.
            t_eval = sol.t.copy()
            ends_at_switch = (sol.status == 0 and k < len(times) - 2)
            if ends_at_switch:
                t_eval[-1] = np.nextafter(t_eval[-1], -np.inf)
            outs = [reactor.outputs(te, sol.y[:, i]) for i, te in enumerate(t_eval)]
            
            t_parts.append(sol.t)
            y_parts.append(sol.y)
            out_parts.extend(outs)
            y = sol.y[:, -1]
            
            if sol.status == 1:
                fired = [desc for (desc, _), te in zip(events, sol.t_events) if len(te) > 0]
                self.termination_message = "; ".join(fired)
                print(f"Termination at t={sol.t[-1]:.2f} h: {self.termination_message}")
                break
        
        t_all = np.concatenate(t_parts)
        y_all = np.concatenate(y_parts, axis=1)
        
        print(f"Simulation completed: {t_all[-1]:.2f} h")
        print(f"Function evaluations: {nfev}")
        
        reactor.update_state(t_all[-1], y_all[:, -1])
        
        return self._process_results(t_all, y_all, out_parts)
    
    def _process_results(self, t, y, outs) -> SimulationResults:
        """Assemble SimulationResults from accepted steps and their outputs."""
        
        def column(key):
            return np.array([np.nan if o[key] is None else o[key] for o in outs], dtype=float)
        
        results = SimulationResults(
            time=t,
            X=y[0],
            S_carbon=y[1],
            S_nitrogen=y[2],
            P=y[3],
            DO=y[4],
            DCO2=y[5],
            V=y[6],
            T=y[7],
            N=y[8],
            mu=column('mu'),
            OUR=column('OUR'),
            CER=column('CER'),
            OTR=column('OTR'),
            CTR=column('CTR'),
            Q_gas=column('Q_gas'),
            kLa=column('kLa'),
            pH=column('pH'),
            T_jacket=column('T_jacket'),
            F_feed=column('F_feed'),
            D=column('D')
        )
        
        # Keep reactor.history in step with the results
        self.reactor.history = {
            name: list(getattr(results, name)) for name in self.reactor.history
        }
        
        return results
