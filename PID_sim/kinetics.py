"""
Growth kinetics models for bioprocesses.

Implements multiple substrate Monod kinetics with optional Haldane inhibition,
Pirt maintenance model, and Luedeking-Piret product formation.
"""

from typing import Dict, Optional
from dataclasses import dataclass
import numpy as np

from .base_models import CarbonSource, BiomassComposition, SubstrateParameters
from .stoichiometry import (
    calculate_oxygen_yield,
    calculate_nitrogen_yield,
    calculate_RQ
)


@dataclass
class CellParameters:
    """
    Complete cellular kinetic parameters with stoichiometry.
    
    Attributes:
        mu_max: Maximum specific growth rate (1/h)
        carbon_source: Carbon source specification
        biomass_composition: Biomass elemental composition
        substrates: Dictionary of substrate parameters (carbon, nitrogen, etc.)
        Y_xs: Biomass yield on carbon substrate (g/g)
        Y_x_O2: Biomass yield on oxygen (g/g), calculated if None
        Y_x_N: Biomass yield on nitrogen (g/g), calculated if None
        Ks_O2: Oxygen half-saturation constant (mmol/L)
        Ki_O2: Oxygen inhibition constant (mmol/L), None for no inhibition
        m_O2: Oxygen maintenance coefficient (g O2 / g biomass / h)
        RQ: Respiratory quotient (mol CO2 / mol O2), calculated if None
        alpha: Growth-associated product formation (g product / g biomass)
        beta: Non-growth-associated product formation (g product / g biomass / h)
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
    m_O2: float = 0.01
    RQ: Optional[float] = None
    alpha: float = 0.0
    beta: float = 0.0
    
    def __post_init__(self):
        """Calculate yields and RQ from stoichiometry if not provided."""
        if self.Y_x_O2 is None:
            self.Y_x_O2 = calculate_oxygen_yield(
                self.carbon_source,
                self.biomass_composition,
                self.Y_xs
            )
        
        if self.Y_x_N is None:
            self.Y_x_N = calculate_nitrogen_yield(
                self.biomass_composition,
                self.Y_xs,
                self.carbon_source
            )
        
        if self.RQ is None:
            self.RQ = calculate_RQ(
                self.carbon_source,
                self.biomass_composition,
                self.Y_xs
            )


class GrowthKinetics:
    """
    Multiple substrate Monod kinetics with optional Haldane inhibition.
    
    Implements:
    - Multiple substrate limitation (carbon, nitrogen, oxygen)
    - Monod or Haldane kinetics for each substrate
    - Pirt maintenance model
    - Luedeking-Piret product formation
    """
    
    def __init__(self, cell_params: CellParameters):
        """
        Initialize growth kinetics model.
        
        Args:
            cell_params: Complete cellular parameters
        """
        self.params = cell_params
    
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
    
    def specific_growth_rate(self, substrates: Dict[str, float], DO: float) -> float:
        """
        Calculate specific growth rate with multiple substrate limitation.
        
        μ = μ_max × f_carbon × f_nitrogen × ... × f_oxygen
        
        Args:
            substrates: Dictionary of substrate concentrations (g/L)
                       e.g., {'carbon': 10.0, 'nitrogen': 2.0}
            DO: Dissolved oxygen concentration (mmol/L)
        
        Returns:
            μ: Specific growth rate (1/h)
        """
        mu = self.params.mu_max
        
        # Multiply limitation terms for each substrate
        for substrate_name, S in substrates.items():
            if substrate_name in self.params.substrates:
                substrate_params = self.params.substrates[substrate_name]
                f = self.monod_term(S, substrate_params.Ks, substrate_params.Ki)
                mu *= f
        
        # Oxygen limitation
        f_O2 = self.monod_term(DO, self.params.Ks_O2, self.params.Ki_O2)
        mu *= f_O2
        
        return mu
    
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
        Calculate substrate consumption rates using Pirt model.
        
        dS/dt = -(μ/Y_xs + ms) × X
        
        Args:
            mu: Specific growth rate (1/h)
            X: Biomass concentration (g/L)
        
        Returns:
            Dictionary of consumption rates (g/L/h), negative values
        """
        rates = {}
        
        for substrate_name, substrate_params in self.params.substrates.items():
            # Pirt model: growth + maintenance
            rate = -(mu / substrate_params.Y_xs + substrate_params.ms) * X
            rates[substrate_name] = rate
        
        return rates
    
    def oxygen_uptake_rate(self, mu: float, X: float) -> float:
        """
        Calculate oxygen uptake rate (OUR).
        
        OUR = (μ/Y_x_O2 + m_O2) × X
        
        Args:
            mu: Specific growth rate (1/h)
            X: Biomass concentration (g/L)
        
        Returns:
            OUR (mmol/L/h)
        """
        # Calculate in g O2 / L / h
        OUR_g = (mu / self.params.Y_x_O2 + self.params.m_O2) * X
        
        # Convert to mmol/L/h (MW of O2 = 32 g/mol)
        OUR_mmol = OUR_g / 32.0 * 1000.0
        
        return OUR_mmol
    
    def co2_evolution_rate(self, OUR: float) -> float:
        """
        Calculate CO2 evolution rate (CER) from OUR using RQ.
        
        CER = RQ × OUR
        
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
        
        dP/dt = (α × μ + β) × X
        
        Args:
            mu: Specific growth rate (1/h)
            X: Biomass concentration (g/L)
        
        Returns:
            dP/dt (g/L/h)
        """
        dP_dt = (self.params.alpha * mu + self.params.beta) * X
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
        m_O2=0.01,
        alpha=0.1,  # Product formation
        beta=0.01
    )
    
    print(f"\nCell parameters:")
    print(f"  μ_max: {cell_params.mu_max} 1/h")
    print(f"  Y_x/S: {cell_params.Y_xs} g/g")
    print(f"  Y_x/O2: {cell_params.Y_x_O2:.3f} g/g (calculated)")
    print(f"  Y_x/N: {cell_params.Y_x_N:.3f} g/g (calculated)")
    print(f"  RQ: {cell_params.RQ:.3f} (calculated)")
    
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
    
    # Calculate growth rate
    mu = kinetics.specific_growth_rate(substrates, DO)
    print(f"\nCalculated μ: {mu:.4f} 1/h")
    
    # Calculate rates
    dX_dt = kinetics.biomass_growth_rate(mu, X)
    substrate_rates = kinetics.substrate_consumption_rates(mu, X)
    OUR = kinetics.oxygen_uptake_rate(mu, X)
    CER = kinetics.co2_evolution_rate(OUR)
    dP_dt = kinetics.product_formation_rate(mu, X)
    
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
    ]
    
    for substrates_test, DO_test, description in test_cases:
        mu_test = kinetics.specific_growth_rate(substrates_test, DO_test)
        print(f"{description:20s}: μ = {mu_test:.4f} 1/h ({mu_test/cell_params.mu_max*100:.1f}% of μ_max)")
    
    print("=" * 60)
