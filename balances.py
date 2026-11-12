"""
Conservation checks on a finished simulation.

These do not feed back into the model. They integrate what went in, what
came out and what is left in the broth, and report how well carbon and
available electrons are accounted for. Use them after changing kinetics or
adding a term to a balance: closure away from 100% means mass is being
created or destroyed somewhere.
"""

from typing import Dict
import numpy as np

from .base_models import SimulationResults
from .kinetics import CellParameters
from .feed_strategies import FeedStrategy


def _integrate(f: np.ndarray, t: np.ndarray) -> float:
    """Trapezoidal integral; rows that share a time (feed switches) add nothing."""
    return float(np.sum(0.5 * (f[1:] + f[:-1]) * np.diff(t)))


def check_balances(results: SimulationResults,
                   cell_params: CellParameters,
                   feed_strategy: FeedStrategy) -> Dict[str, float]:
    """
    Carbon and electron closure over the whole run, plus the lowest value
    each concentration reached.
    
    Carbon (C-mol):
        in      = carbon in the initial broth + carbon fed
        out     = CO2 stripped to the gas phase
        closure = (carbon in final broth + out) / in
    
    Available electrons (degree of reduction):
        donated  = gamma_S × substrate consumed
        accepted = gamma_X × biomass formed + 4 × O2 consumed
        closure  = accepted / donated
    
    Product is not included: the model does not yet charge substrate for
    product formation, so runs with alpha or beta > 0 will not close.
    
    Args:
        results: Output of FedBatchSimulator.simulate
        cell_params: The cell parameters used for the run
        feed_strategy: The feed strategy used for the run
    
    Returns:
        Dictionary with 'carbon_closure_pct', 'electron_closure_pct',
        'min_S_carbon', 'min_S_nitrogen', 'min_DO', 'min_DCO2', 'min_X'
    """
    r = results
    cs = cell_params.carbon_source
    bm = cell_params.biomass_composition
    C_s = cs.C / cs.MW              # C-mol per g substrate
    C_x = 1.0 / bm.MW_per_Cmol      # C-mol per g biomass
    
    S_feed = np.array([feed_strategy.get_composition(t).S_carbon for t in r.time])
    fed_g = _integrate(r.F_feed * S_feed, r.time)
    
    # Carbon
    def broth_carbon(i):
        return r.V[i] * (r.S_carbon[i] * C_s + r.X[i] * C_x + r.DCO2[i] / 1000.0)
    
    carbon_in = broth_carbon(0) + fed_g * C_s
    carbon_out = _integrate(r.CTR * r.V / 1000.0, r.time)
    carbon_closure = 100.0 * (broth_carbon(-1) + carbon_out) / carbon_in
    
    # Electrons
    consumed_g = r.S_carbon[0] * r.V[0] + fed_g - r.S_carbon[-1] * r.V[-1]
    formed_g = r.X[-1] * r.V[-1] - r.X[0] * r.V[0]
    O2_mol = _integrate(r.OUR * r.V / 1000.0, r.time)
    donated = cs.degree_of_reduction * consumed_g * C_s
    accepted = bm.degree_of_reduction * formed_g * C_x + 4.0 * O2_mol
    electron_closure = 100.0 * accepted / donated if donated > 0 else float('nan')
    
    return {
        'carbon_closure_pct': carbon_closure,
        'electron_closure_pct': electron_closure,
        'min_X': float(r.X.min()),
        'min_S_carbon': float(r.S_carbon.min()),
        'min_S_nitrogen': float(r.S_nitrogen.min()),
        'min_DO': float(r.DO.min()),
        'min_DCO2': float(r.DCO2.min()),
    }


def format_balances(report: Dict[str, float]) -> str:
    """Human-readable version of a check_balances report."""
    lines = [
        "Conservation check:",
        f"  Carbon accounted for:    {report['carbon_closure_pct']:.2f} %",
        f"  Electrons accounted for: {report['electron_closure_pct']:.2f} %",
        "  Lowest value reached:    "
        f"carbon substrate {report['min_S_carbon']:.2e} g/L, "
        f"DO {report['min_DO']:.2e} mmol/L, "
        f"nitrogen {report['min_S_nitrogen']:.2e} g/L",
    ]
    return "\n".join(lines)
