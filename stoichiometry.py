"""
Stoichiometry and elemental balance calculations.

This module provides functions to calculate yield coefficients from first principles
using elemental balance and electron balance equations.
"""

from typing import Tuple
from .base_models import CarbonSource, BiomassComposition


def calculate_oxygen_yield(
    carbon_source: CarbonSource,
    biomass: BiomassComposition,
    Y_xs: float
) -> float:
    """
    Calculate Y_x/O2 from Y_x/S using electron balance.
    
    Electron balance:
    γ_substrate = γ_biomass × c + 4a
    
    Where:
        c = Y_xc (C-mol biomass / C-mol substrate)
        a = mol O2 / C-mol substrate
    
    Args:
        carbon_source: Carbon source specification
        biomass: Biomass composition
        Y_xs: Biomass yield on substrate (g biomass / g substrate)
    
    Returns:
        Y_x/O2: Biomass yield on oxygen (g biomass / g O2)
    """
    # Convert to C-mole basis
    substrate_Cmol = carbon_source.to_c_mole_basis()
    MW_substrate_per_Cmol = substrate_Cmol['MW_per_Cmol']
    
    # Y_xc: C-mol biomass / C-mol substrate
    Y_xc = Y_xs * MW_substrate_per_Cmol / biomass.MW_per_Cmol
    c = Y_xc
    
    # Electron balance: γ_s = γ_x × c + 4a
    gamma_s = carbon_source.degree_of_reduction
    gamma_x = biomass.degree_of_reduction
    
    a = (gamma_s - gamma_x * c) / 4.0  # mol O2 / C-mol substrate
    
    if a <= 0:
        # Anaerobic or substrate more reduced than biomass
        # Return very large value (no oxygen required)
        return float('inf')
    
    # Convert to mass basis (g biomass / g O2)
    # a is mol O2 / C-mol substrate
    # c is C-mol biomass / C-mol substrate
    Y_x_O2 = (c * biomass.MW_per_Cmol) / (a * 32.0)
    
    return Y_x_O2


def calculate_nitrogen_yield(
    biomass: BiomassComposition,
    Y_xs: float,
    carbon_source: CarbonSource
) -> float:
    """
    Calculate Y_x/N from Y_x/S using nitrogen balance.
    
    Args:
        biomass: Biomass composition
        Y_xs: Biomass yield on carbon substrate (g/g)
        carbon_source: Carbon source specification
    
    Returns:
        Y_x/N: Biomass yield on nitrogen (g biomass / g N)
    """
    # Convert to C-mole basis
    substrate_Cmol = carbon_source.to_c_mole_basis()
    MW_substrate_per_Cmol = substrate_Cmol['MW_per_Cmol']
    
    Y_xc = Y_xs * MW_substrate_per_Cmol / biomass.MW_per_Cmol
    c = Y_xc
    
    # Nitrogen requirement (mol NH3 / C-mol substrate)
    # Assuming nitrogen source is NH3 or NH4+
    b = biomass.N * c
    
    if b <= 0:
        # No nitrogen required
        return float('inf')
    
    # g biomass / g N
    # b is mol NH3 / C-mol substrate
    # c is C-mol biomass / C-mol substrate
    # MW_N = 14 g/mol (as nitrogen)
    
    Y_x_N = (c * biomass.MW_per_Cmol) / (b * 14.0)
    
    return Y_x_N


def calculate_stoichiometric_coefficients(
    carbon_source: CarbonSource,
    biomass: BiomassComposition,
    Y_xs: float
) -> Tuple[float, float, float, float]:
    """
    Calculate all stoichiometric coefficients from Y_x/S.
    
    Returns stoichiometric coefficients for the reaction:
    1 C-mol substrate + a O2 + b NH3 → c C-mol biomass + d CO2 + e H2O
    
    Args:
        carbon_source: Carbon source specification
        biomass: Biomass composition
        Y_xs: Biomass yield on substrate (g/g)
    
    Returns:
        Tuple of (a, b, c, d) where:
            a: mol O2 / C-mol substrate
            b: mol NH3 / C-mol substrate
            c: C-mol biomass / C-mol substrate
            d: C-mol CO2 / C-mol substrate
    """
    # Convert to C-mole basis
    substrate_Cmol = carbon_source.to_c_mole_basis()
    MW_substrate_per_Cmol = substrate_Cmol['MW_per_Cmol']
    
    # Y_xc: C-mol biomass / C-mol substrate
    Y_xc = Y_xs * MW_substrate_per_Cmol / biomass.MW_per_Cmol
    c = Y_xc
    
    # Carbon balance: 1 = c + d
    d = 1.0 - c
    
    # Nitrogen balance
    b = biomass.N * c
    
    # Electron balance: γ_s = γ_x × c + 4a
    gamma_s = carbon_source.degree_of_reduction
    gamma_x = biomass.degree_of_reduction
    a = (gamma_s - gamma_x * c) / 4.0
    
    return a, b, c, d


def validate_elemental_balance(
    carbon_source: CarbonSource,
    biomass: BiomassComposition,
    a: float,
    b: float,
    c: float,
    d: float
) -> dict:
    """
    Validate elemental balance for calculated stoichiometric coefficients.
    
    Reaction: 1 C-mol substrate + a O2 + b NH3 → c C-mol biomass + d CO2 + e H2O
    
    Args:
        carbon_source: Carbon source specification
        biomass: Biomass composition
        a: mol O2 / C-mol substrate
        b: mol NH3 / C-mol substrate
        c: C-mol biomass / C-mol substrate
        d: C-mol CO2 / C-mol substrate
    
    Returns:
        Dictionary with balance closure for each element
    """
    substrate = carbon_source.to_c_mole_basis()
    
    # Calculate e (H2O) from hydrogen balance
    # H balance: H_s + 3b = H_x × c + 2e
    H_substrate = substrate['H']
    H_biomass = biomass.H
    e = (H_substrate + 3*b - H_biomass * c) / 2.0
    
    # Carbon balance
    C_in = 1.0
    C_out = c + d
    C_balance = abs(C_in - C_out)
    
    # Hydrogen balance
    H_in = H_substrate + 3*b
    H_out = H_biomass * c + 2*e
    H_balance = abs(H_in - H_out)
    
    # Oxygen balance
    O_in = substrate['O'] + 2*a
    O_out = biomass.O * c + 2*d + e
    O_balance = abs(O_in - O_out)
    
    # Nitrogen balance
    N_in = substrate['N'] + b
    N_out = biomass.N * c
    N_balance = abs(N_in - N_out)
    
    # Electron balance
    gamma_s = carbon_source.degree_of_reduction
    gamma_x = biomass.degree_of_reduction
    e_in = gamma_s * 1.0
    e_out = gamma_x * c + 4*a  # O2 accepts 4 electrons per mol
    e_balance = abs(e_in - e_out)
    
    return {
        'C_balance': C_balance,
        'H_balance': H_balance,
        'O_balance': O_balance,
        'N_balance': N_balance,
        'electron_balance': e_balance,
        'all_balanced': (C_balance < 1e-6 and H_balance < 1e-6 and 
                        O_balance < 1e-6 and N_balance < 1e-6 and 
                        e_balance < 1e-6)
    }


def calculate_RQ(
    carbon_source: CarbonSource,
    biomass: BiomassComposition,
    Y_xs: float
) -> float:
    """
    Calculate respiratory quotient (RQ = CER/OUR) from stoichiometry.
    
    RQ = d/a = (C-mol CO2 / C-mol substrate) / (mol O2 / C-mol substrate)
    
    Args:
        carbon_source: Carbon source specification
        biomass: Biomass composition
        Y_xs: Biomass yield on substrate (g/g)
    
    Returns:
        RQ: Respiratory quotient (mol CO2 / mol O2)
    """
    a, b, c, d = calculate_stoichiometric_coefficients(
        carbon_source, biomass, Y_xs
    )
    
    if a <= 0:
        return 0.0
    
    RQ = d / a
    
    return RQ


# Example usage and validation
if __name__ == "__main__":
    from .base_models import GLUCOSE, STANDARD_BIOMASS
    
    print("=" * 60)
    print("STOICHIOMETRY VALIDATION")
    print("=" * 60)
    
    # Example: Glucose → Biomass
    Y_xs = 0.5  # g/g
    
    print(f"\nCarbon source: {GLUCOSE.name}")
    print(f"Biomass: {STANDARD_BIOMASS.name}")
    print(f"Y_x/S: {Y_xs} g/g")
    
    # Calculate yields
    Y_x_O2 = calculate_oxygen_yield(GLUCOSE, STANDARD_BIOMASS, Y_xs)
    Y_x_N = calculate_nitrogen_yield(STANDARD_BIOMASS, Y_xs, GLUCOSE)
    
    print(f"\nCalculated yields:")
    print(f"  Y_x/O2: {Y_x_O2:.3f} g biomass / g O2")
    print(f"  Y_x/N:  {Y_x_N:.3f} g biomass / g N")
    
    # Calculate stoichiometric coefficients
    a, b, c, d = calculate_stoichiometric_coefficients(
        GLUCOSE, STANDARD_BIOMASS, Y_xs
    )
    
    print(f"\nStoichiometric coefficients:")
    print(f"  a (O2):      {a:.4f} mol O2 / C-mol substrate")
    print(f"  b (NH3):     {b:.4f} mol NH3 / C-mol substrate")
    print(f"  c (biomass): {c:.4f} C-mol biomass / C-mol substrate")
    print(f"  d (CO2):     {d:.4f} C-mol CO2 / C-mol substrate")
    
    # Calculate RQ
    RQ = calculate_RQ(GLUCOSE, STANDARD_BIOMASS, Y_xs)
    print(f"\nRespiratory quotient (RQ): {RQ:.3f}")
    
    # Validate balance
    balance = validate_elemental_balance(GLUCOSE, STANDARD_BIOMASS, a, b, c, d)
    
    print(f"\nElemental balance validation:")
    print(f"  C balance error:  {balance['C_balance']:.2e}")
    print(f"  H balance error:  {balance['H_balance']:.2e}")
    print(f"  O balance error:  {balance['O_balance']:.2e}")
    print(f"  N balance error:  {balance['N_balance']:.2e}")
    print(f"  e- balance error: {balance['electron_balance']:.2e}")
    print(f"  All balanced: {balance['all_balanced']}")
    
    print("=" * 60)
