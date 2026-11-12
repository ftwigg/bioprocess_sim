"""
Growth kinetics models for bioprocesses.

Uptake-driven Pirt model with multiple-substrate Monod/Haldane limitation.
Substrate uptake is the driver and growth is what is left after maintenance:

    q_S  = q_S,max * f_C * f_N * ... * f_O2          (g substrate / g biomass / h)
    mu   = Y_xs * (q_S - m_S * f_O2)                 (1/h, negative when starved)

with q_S,max = mu_max / Y_xs + m_S, so mu -> mu_max when nothing is limiting.

Because uptake carries the Monod term, consumption vanishes as S -> 0 and the
substrate balance cannot go negative. When uptake no longer covers maintenance
the deficit is met by endogenous metabolism: mu becomes negative (down to
-Y_xs * m_S) and biomass is oxidised instead.

Oxygen uptake and CO2 evolution are not independent parameters. They follow
from the electron (degree-of-reduction) and carbon balances over the same
q_S and mu, so both balances close at every instant, including during decay:

    q_O2  = (gamma_S * q_S[C-mol] - gamma_X * mu[C-mol]) / 4
    q_CO2 =  q_S[C-mol] - mu[C-mol]
"""

import warnings
from typing import Dict, Optional
from dataclasses import dataclass
import numpy as np

from .base_models import CarbonSource, BiomassComposition, SubstrateParameters
from .stoichiometry import (
    calculate_oxygen_yield,
    calculate_nitrogen_yield,
    calculate_RQ
)

MW_O2 = 32.0  # g/mol


@dataclass
class CellParameters:
    """
    Complete cellular kinetic parameters with stoichiometry.
    
    Attributes:
        mu_max: Maximum specific growth rate (1/h)
        carbon_source: Carbon source specification
        biomass_composition: Biomass elemental composition
        substrates: Dictionary of substrate parameters (carbon, nitrogen, etc.)
        Y_xs: True biomass yield on carbon substrate (g/g). If a 'carbon'
              entry is present in `substrates`, its Y_xs is the one used and
              this field is synchronised to it.
        Ks_O2: Oxygen half-saturation constant (mmol/L)
        Ki_O2: Oxygen inhibition constant (mmol/L), None for no inhibition
        alpha: Growth-associated product formation (g product / g biomass)
        beta: Non-growth-associated product formation (g product / g biomass / h)
    
    Derived (reported, not inputs):
        Y_x_O2: Biomass yield on oxygen for growth alone (g/g)
        Y_x_N:  Biomass yield on nitrogen from biomass composition (g/g)
        m_O2:   Oxygen maintenance coefficient implied by m_S (g O2 / g biomass / h)
        RQ:     Respiratory quotient of growth alone (mol CO2 / mol O2)
    
    Y_x_O2, m_O2 and RQ used to be free inputs. They are now fixed by the
    electron and carbon balances (see GrowthKinetics); a value passed in that
    disagrees with the balance is ignored with a warning, because honouring it
    would create or destroy carbon and electrons.
    """
    mu_max: float
    carbon_source: CarbonSource
    biomass_composition: BiomassComposition
    substrates: Dict[str, SubstrateParameters]
    Y_xs: float = 0.5
    Y_x_O2: Optional[float] = None
    Y_x_N: Optional[float] = None
    Ks_O2: float = 0.003
    Ki_O2: Optional[float] = None
    m_O2: Optional[float] = None
    RQ: Optional[float] = None
    alpha: float = 0.0
    beta: float = 0.0
    
    def __post_init__(self):
        """Derive oxygen yield, oxygen maintenance and RQ from stoichiometry."""
        carbon = self.substrates.get('carbon')
        if carbon is not None and abs(carbon.Y_xs - self.Y_xs) > 1e-9:
            warnings.warn(
                f"CellParameters.Y_xs ({self.Y_xs}) differs from "
                f"substrates['carbon'].Y_xs ({carbon.Y_xs}); using the latter "
                f"for both kinetics and stoichiometry.", stacklevel=2)
            self.Y_xs = carbon.Y_xs
        ms = carbon.ms if carbon is not None else 0.0
        
        cs, bm = self.carbon_source, self.biomass_composition
        C_s = cs.C / cs.MW                  # C-mol per g substrate
        C_x = 1.0 / bm.MW_per_Cmol          # C-mol per g biomass
        Y_e = cs.degree_of_reduction * C_s / (bm.degree_of_reduction * C_x)
        Y_c = C_s / C_x
        if self.Y_xs >= min(Y_e, Y_c):
            raise ValueError(
                f"Y_xs = {self.Y_xs} g/g is not possible for aerobic growth on "
                f"{cs.name}: the electron balance allows at most {Y_e:.3f} and "
                f"the carbon balance {Y_c:.3f} g/g.")
        
        derived = {
            'Y_x_O2': calculate_oxygen_yield(cs, bm, self.Y_xs),
            'm_O2': ms * cs.degree_of_reduction * C_s / 4.0 * MW_O2,
            'RQ': calculate_RQ(cs, bm, self.Y_xs),
        }
        for name, value in derived.items():
            given = getattr(self, name)
            if given is not None and abs(given - value) > 1e-3 * max(abs(value), 1e-12):
                warnings.warn(
                    f"CellParameters.{name}={given} is ignored: the electron and "
                    f"carbon balances fix it at {value:.4g} for Y_xs={self.Y_xs} "
                    f"and m_S={ms}.", stacklevel=2)
            setattr(self, name, value)
        
        if self.Y_x_N is None:
            self.Y_x_N = calculate_nitrogen_yield(
                self.biomass_composition,
                self.Y_xs,
                self.carbon_source
            )


class GrowthKinetics:
    """
    Uptake-driven Pirt kinetics with multiple-substrate limitation.
    
    Implements:
    - Multiple substrate limitation (carbon, nitrogen, oxygen), Monod or Haldane
    - Pirt maintenance, met from substrate uptake or, failing that, from biomass
    - Oxygen uptake and CO2 evolution from the electron and carbon balances
    - Luedeking-Piret product formation
    
    `specific_rates` / `rates` are the consistent entry points. The older
    per-quantity methods are kept for compatibility.
    """
    
    def __init__(self, cell_params: CellParameters):
        """
        Initialize growth kinetics model.
        
        Args:
            cell_params: Complete cellular parameters
        """
        self.params = cell_params
        p = cell_params
        carbon = p.substrates.get('carbon')
        self.Y = carbon.Y_xs if carbon is not None else p.Y_xs
        self.ms = carbon.ms if carbon is not None else 0.0
        self.qs_max = p.mu_max / self.Y + self.ms
        # C-mol per gram, and degrees of reduction, for the balances
        self._C_s = p.carbon_source.C / p.carbon_source.MW
        self._C_x = 1.0 / p.biomass_composition.MW_per_Cmol
        self._gamma_s = p.carbon_source.degree_of_reduction
        self._gamma_x = p.biomass_composition.degree_of_reduction
    
    def monod_term(self, S: float, Ks: float, Ki: Optional[float] = None) -> float:
        """
        Calculate Monod or Haldane limitation term.
        
        Monod: f = S / (Ks + S)
        Haldane: f = S / (Ks + S + S²/Ki)
        
        Args:
            S: Substrate concentration
            Ks: Half-saturation constant
            Ki: Inhibition constant, None for Monod
        
        Returns:
            Limitation factor (0-1)
        """
        if S <= 0:
            return 0.0
        
        if Ki is None:
            # Monod
            return S / (Ks + S)
        else:
            # Haldane
            return S / (Ks + S + S**2 / Ki)
    
    def limitation_terms(self, substrates: Dict[str, float], DO: float) -> Dict[str, float]:
        """Monod/Haldane factor (0-1) for each substrate and for oxygen."""
        f = {}
        for name, sp in self.params.substrates.items():
            f[name] = self.monod_term(substrates.get(name, 0.0), sp.Ks, sp.Ki)
        f['oxygen'] = self.monod_term(DO, self.params.Ks_O2, self.params.Ki_O2)
        return f
    
    def specific_rates(self, substrates: Dict[str, float], DO: float) -> Dict:
        """
        All specific rates (per g biomass) from one consistent calculation.
        
        Args:
            substrates: Substrate concentrations (g/L), e.g. {'carbon': 10.0, 'nitrogen': 2.0}
            DO: Dissolved oxygen (mmol/L)
        
        Returns:
            Dictionary with
                mu:   net specific growth rate (1/h), negative during starvation
                q_substrate: {name: uptake (g/g/h)}; a negative value is a release
                q_O2:  oxygen uptake (mmol/g/h)
                q_CO2: CO2 evolution (mmol/g/h)
                f:     limitation factors
        """
        p = self.params
        f = self.limitation_terms(substrates, DO)
        f_O2 = f['oxygen']
        
        limitation = 1.0
        for value in f.values():
            limitation *= value
        
        # Uptake is the driver; maintenance is aerobic, so it scales with f_O2
        q_s = self.qs_max * limitation
        m_eff = self.ms * f_O2
        mu = self.Y * (q_s - m_eff)
        
        # Other substrates follow biomass stoichiometry. Decay (mu < 0) returns
        # them to the broth; their maintenance term fades as they run out.
        q_sub = {}
        for name, sp in p.substrates.items():
            if name == 'carbon':
                q_sub[name] = q_s
            else:
                q_sub[name] = mu / sp.Y_xs + sp.ms * f[name]
        if 'carbon' not in q_sub:
            q_sub['carbon'] = q_s
        
        # Electron and carbon balances (C-mol / g biomass / h)
        s_C = q_s * self._C_s
        x_C = mu * self._C_x
        q_O2 = (self._gamma_s * s_C - self._gamma_x * x_C) / 4.0 * 1000.0   # mmol/g/h
        q_CO2 = (s_C - x_C) * 1000.0                                         # mmol/g/h
        
        return {'mu': mu, 'q_substrate': q_sub, 'q_O2': q_O2, 'q_CO2': q_CO2, 'f': f}
    
    def rates(self, substrates: Dict[str, float], DO: float, X: float) -> Dict:
        """
        Volumetric reaction rates at the given state.
        
        Returns:
            Dictionary with mu (1/h), dX_dt (g/L/h), substrate_rates
            ({name: g/L/h}, negative = consumption), OUR and CER (mmol/L/h),
            dP_dt (g/L/h) and the limitation factors f.
        """
        q = self.specific_rates(substrates, DO)
        mu = q['mu']
        return {
            'mu': mu,
            'dX_dt': mu * X,
            'substrate_rates': {name: -rate * X for name, rate in q['q_substrate'].items()},
            'OUR': q['q_O2'] * X,
            'CER': q['q_CO2'] * X,
            'dP_dt': self.product_formation_rate(mu, X),
            'f': q['f'],
        }
    
    def specific_growth_rate(self, substrates: Dict[str, float], DO: float) -> float:
        """
        Net specific growth rate with multiple substrate limitation.
        
        μ = Y_xs × (q_S − m_S × f_O2),  q_S = q_S,max × f_carbon × f_nitrogen × ... × f_oxygen
        
        Args:
            substrates: Dictionary of substrate concentrations (g/L)
                       e.g., {'carbon': 10.0, 'nitrogen': 2.0}
            DO: Dissolved oxygen concentration (mmol/L)
        
        Returns:
            μ: Specific growth rate (1/h); negative when uptake is below maintenance
        """
        return self.specific_rates(substrates, DO)['mu']
    
    def biomass_growth_rate(self, mu: float, X: float) -> float:
        """
        Calculate biomass growth rate.
        
        dX/dt = μ × X
        
        Args:
            mu: Specific growth rate (1/h)
            X: Biomass concentration (g/L)
        
        Returns:
            dX/dt (g/L/h)
        """
        return mu * X
    
    def substrate_consumption_rates(self, mu: float, X: float) -> Dict[str, float]:
        """
        Substrate consumption rates from μ alone (compatibility wrapper).
        
        dS/dt = -(μ/Y_xs + ms) × X
        
        Exact when oxygen is not limiting. Prefer `rates`, which also handles
        oxygen limitation and the fading of non-carbon maintenance terms.
        
        Args:
            mu: Specific growth rate (1/h)
            X: Biomass concentration (g/L)
        
        Returns:
            Dictionary of consumption rates (g/L/h), negative values
        """
        rates = {}
        
        for substrate_name, substrate_params in self.params.substrates.items():
            rate = -(mu / substrate_params.Y_xs + substrate_params.ms) * X
            rates[substrate_name] = rate
        
        return rates
    
    def oxygen_uptake_rate(self, mu: float, X: float) -> float:
        """
        Oxygen uptake rate from μ alone (compatibility wrapper).
        
        OUR = (μ/Y_x_O2 + m_O2) × X
        
        With the stoichiometric Y_x_O2 and m_O2 this is the electron balance
        written in terms of μ; it matches `rates` when oxygen is not limiting.
        
        Args:
            mu: Specific growth rate (1/h)
            X: Biomass concentration (g/L)
        
        Returns:
            OUR (mmol/L/h)
        """
        # Calculate in g O2 / L / h
        OUR_g = (mu / self.params.Y_x_O2 + self.params.m_O2) * X
        
        # Convert to mmol/L/h (MW of O2 = 32 g/mol)
        OUR_mmol = OUR_g / MW_O2 * 1000.0
        
        return OUR_mmol
    
    def co2_evolution_rate(self, OUR: float) -> float:
        """
        CO2 evolution rate from OUR and the growth RQ (compatibility wrapper).
        
        CER = RQ × OUR
        
        Approximate: maintenance has its own RQ. `rates` takes CER from the
        carbon balance instead.
        
        Args:
            OUR: Oxygen uptake rate (mmol/L/h)
        
        Returns:
            CER (mmol/L/h)
        """
        CER = self.params.RQ * OUR
        return CER
    
    def product_formation_rate(self, mu: float, X: float) -> float:
        """
        Calculate product formation rate using Luedeking-Piret model.
        
        dP/dt = (α × max(μ, 0) + β) × X
        
        The growth-associated term stops when the culture is decaying.
        
        Args:
            mu: Specific growth rate (1/h)
            X: Biomass concentration (g/L)
        
        Returns:
            dP/dt (g/L/h)
        """
        dP_dt = (self.params.alpha * max(mu, 0.0) + self.params.beta) * X
        return dP_dt


# Example usage and testing
if __name__ == "__main__":
    from .base_models import GLUCOSE, STANDARD_BIOMASS
    
    print("=" * 60)
    print("GROWTH KINETICS VALIDATION")
    print("=" * 60)
    
    # Define substrate parameters
    carbon_params = SubstrateParameters(
        name='carbon',
        Ks=0.1,  # g/L
        Ki=None,  # No inhibition
        Y_xs=0.5,
        ms=0.03
    )
    
    nitrogen_params = SubstrateParameters(
        name='nitrogen',
        Ks=0.05,  # g/L
        Ki=None,
        Y_xs=10.0,
        ms=0.001
    )
    
    # Create cell parameters
    cell_params = CellParameters(
        mu_max=0.5,  # 1/h
        carbon_source=GLUCOSE,
        biomass_composition=STANDARD_BIOMASS,
        substrates={
            'carbon': carbon_params,
            'nitrogen': nitrogen_params
        },
        Y_xs=0.5,
        Ks_O2=0.003,  # mmol/L
        alpha=0.1,  # Product formation
        beta=0.01
    )
    
    print(f"\nCell parameters:")
    print(f"  μ_max: {cell_params.mu_max} 1/h")
    print(f"  Y_x/S: {cell_params.Y_xs} g/g")
    print(f"  Y_x/O2: {cell_params.Y_x_O2:.3f} g/g (calculated)")
    print(f"  Y_x/N: {cell_params.Y_x_N:.3f} g/g (calculated)")
    print(f"  m_O2: {cell_params.m_O2:.4f} g/g/h (calculated from m_S)")
    print(f"  RQ: {cell_params.RQ:.3f} (calculated, growth only)")
    
    # Create kinetics model
    kinetics = GrowthKinetics(cell_params)
    
    # Test conditions
    substrates = {
        'carbon': 10.0,  # g/L
        'nitrogen': 2.0   # g/L
    }
    DO = 0.2  # mmol/L (~80% saturation)
    X = 5.0   # g/L
    
    print(f"\nTest conditions:")
    print(f"  S_carbon: {substrates['carbon']} g/L")
    print(f"  S_nitrogen: {substrates['nitrogen']} g/L")
    print(f"  DO: {DO} mmol/L")
    print(f"  X: {X} g/L")
    
    # Calculate all rates consistently
    r = kinetics.rates(substrates, DO, X)
    mu = r['mu']
    print(f"\nCalculated μ: {mu:.4f} 1/h")
    
    dX_dt = r['dX_dt']
    substrate_rates = r['substrate_rates']
    OUR = r['OUR']
    CER = r['CER']
    dP_dt = r['dP_dt']
    
    print(f"\nReaction rates:")
    print(f"  dX/dt: {dX_dt:.3f} g/L/h")
    print(f"  dS_carbon/dt: {substrate_rates['carbon']:.3f} g/L/h")
    print(f"  dS_nitrogen/dt: {substrate_rates['nitrogen']:.3f} g/L/h")
    print(f"  OUR: {OUR:.3f} mmol/L/h")
    print(f"  CER: {CER:.3f} mmol/L/h")
    print(f"  dP/dt: {dP_dt:.3f} g/L/h")
    
    # Test substrate limitation
    print(f"\n" + "=" * 60)
    print("SUBSTRATE LIMITATION TESTS")
    print("=" * 60)
    
    test_cases = [
        ({'carbon': 10.0, 'nitrogen': 2.0}, 0.2, "Normal"),
        ({'carbon': 0.05, 'nitrogen': 2.0}, 0.2, "Carbon limited"),
        ({'carbon': 10.0, 'nitrogen': 0.02}, 0.2, "Nitrogen limited"),
        ({'carbon': 10.0, 'nitrogen': 2.0}, 0.001, "Oxygen limited"),
        ({'carbon': 0.05, 'nitrogen': 0.02}, 0.001, "All limited"),
        ({'carbon': 0.0, 'nitrogen': 2.0}, 0.2, "Starved (decay)"),
    ]
    
    for substrates_test, DO_test, description in test_cases:
        mu_test = kinetics.specific_growth_rate(substrates_test, DO_test)
        print(f"{description:20s}: μ = {mu_test:.4f} 1/h ({mu_test/cell_params.mu_max*100:.1f}% of μ_max)")
    
    print("=" * 60)
