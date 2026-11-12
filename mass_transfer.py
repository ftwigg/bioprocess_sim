"""
Gas-liquid mass transfer models.

Implements:
- Temperature-dependent Henry's constants
- Dynamic kLa correlations
- Gas phase material balances
- Oxygen and CO2 transfer rates
"""

from abc import ABC, abstractmethod
import numpy as np

from .base_models import GasComposition, GasPhaseConditions


class HenryConstant:
    """
    Temperature-dependent Henry's constants for gas solubility.
    
    Uses van't Hoff equation:
    H(T) = H_ref × exp[-ΔH_sol/R × (1/T - 1/T_ref)]
    """
    
    # Reference values at 25°C (298.15 K)
    H_O2_ref = 1.28e-3  # mol/L/atm
    H_CO2_ref = 34.1e-3  # mol/L/atm
    H_N2_ref = 0.65e-3  # mol/L/atm
    
    # Enthalpy of dissolution (kJ/mol)
    dH_O2 = -13.0
    dH_CO2 = -19.4
    dH_N2 = -13.7
    
    T_ref = 298.15  # K
    R = 8.314  # J/mol/K
    
    @classmethod
    def get_H_O2(cls, T_celsius: float) -> float:
        """
        Calculate Henry's constant for O2 at temperature T.
        
        Args:
            T_celsius: Temperature (°C)
        
        Returns:
            H_O2 (mol/L/atm)
        """
        T = T_celsius + 273.15  # Convert to K
        
        # van't Hoff equation
        H = cls.H_O2_ref * np.exp(
            -cls.dH_O2 * 1000 / cls.R * (1/T - 1/cls.T_ref)
        )
        
        return H
    
    @classmethod
    def get_H_CO2(cls, T_celsius: float) -> float:
        """Calculate Henry's constant for CO2."""
        T = T_celsius + 273.15
        H = cls.H_CO2_ref * np.exp(
            -cls.dH_CO2 * 1000 / cls.R * (1/T - 1/cls.T_ref)
        )
        return H
    
    @classmethod
    def get_H_N2(cls, T_celsius: float) -> float:
        """Calculate Henry's constant for N2."""
        T = T_celsius + 273.15
        H = cls.H_N2_ref * np.exp(
            -cls.dH_N2 * 1000 / cls.R * (1/T - 1/cls.T_ref)
        )
        return H


class MassTransferModel(ABC):
    """Base class for mass transfer models."""
    
    @abstractmethod
    def get_kLa_O2(self, conditions: GasPhaseConditions) -> float:
        """Return kLa for O2 (1/h)."""
        pass
    
    @abstractmethod
    def get_kLa_CO2(self, conditions: GasPhaseConditions) -> float:
        """Return kLa for CO2 (1/h)."""
        pass
    
    def saturation_concentration_O2(self, P_O2: float, T: float) -> float:
        """
        Calculate O2 saturation concentration.
        
        Args:
            P_O2: Partial pressure of O2 (atm)
            T: Temperature (°C)
        
        Returns:
            C* for O2 (mmol/L)
        """
        H = HenryConstant.get_H_O2(T)  # mol/L/atm
        C_star = H * P_O2 * 1000  # Convert to mmol/L
        return C_star
    
    def saturation_concentration_CO2(self, P_CO2: float, T: float) -> float:
        """Calculate CO2 saturation concentration (mmol/L)."""
        H = HenryConstant.get_H_CO2(T)
        C_star = H * P_CO2 * 1000
        return C_star
    
    def oxygen_transfer_rate(self, C_L: float, P_O2: float, 
                            conditions: GasPhaseConditions) -> float:
        """
        Calculate oxygen transfer rate.
        
        OTR = kLa × (C* - C_L)
        
        Args:
            C_L: Dissolved O2 concentration (mmol/L)
            P_O2: Partial pressure of O2 (atm)
            conditions: Gas phase conditions
        
        Returns:
            OTR (mmol/L/h)
        """
        kLa = self.get_kLa_O2(conditions)
        C_star = self.saturation_concentration_O2(P_O2, conditions.T)
        OTR = kLa * (C_star - C_L)
        return OTR
    
    def co2_transfer_rate(self, C_L: float, P_CO2: float,
                         conditions: GasPhaseConditions) -> float:
        """
        Calculate CO2 transfer rate (from liquid to gas).
        
        CTR = kLa × (C_L - C*)
        
        Returns:
            CTR (mmol/L/h) - positive means leaving liquid
        """
        kLa = self.get_kLa_CO2(conditions)
        C_star = self.saturation_concentration_CO2(P_CO2, conditions.T)
        CTR = kLa * (C_L - C_star)
        return CTR
    
    def outlet_gas_composition(self, OUR: float, CER: float,
                               conditions: GasPhaseConditions) -> GasComposition:
        """
        Calculate outlet gas composition from mass balance.
        
        Args:
            OUR: Oxygen uptake rate (mmol/L/h)
            CER: CO2 evolution rate (mmol/L/h)
            conditions: Gas phase conditions
        
        Returns:
            Outlet gas composition
        """
        # Convert flow rate to molar basis
        P = conditions.P_total  # atm
        T_K = conditions.T + 273.15  # K
        R = 0.08206  # L·atm/(mol·K)
        
        # Molar flow rate (mol/h)
        F_in_molar = (P * conditions.Q_gas) / (R * T_K)
        
        # Inlet molar flows
        n_O2_in = F_in_molar * conditions.inlet_composition.O2
        n_CO2_in = F_in_molar * conditions.inlet_composition.CO2
        n_N2_in = F_in_molar * conditions.inlet_composition.N2
        
        # Transfer rates (mol/h)
        n_O2_transfer = (OUR / 1000) * conditions.V_liquid  # mmol/h → mol/h
        n_CO2_transfer = (CER / 1000) * conditions.V_liquid
        
        # Outlet molar flows
        n_O2_out = n_O2_in - n_O2_transfer  # O2 consumed
        n_CO2_out = n_CO2_in + n_CO2_transfer  # CO2 produced
        n_N2_out = n_N2_in  # Inert
        
        # Total outlet flow
        n_out_total = n_O2_out + n_CO2_out + n_N2_out
        
        # Outlet composition
        if n_out_total > 0:
            y_O2_out = max(0, n_O2_out / n_out_total)
            y_CO2_out = max(0, n_CO2_out / n_out_total)
            y_N2_out = max(0, n_N2_out / n_out_total)
            
            # Normalize
            total = y_O2_out + y_CO2_out + y_N2_out
            if total > 0:
                y_O2_out /= total
                y_CO2_out /= total
                y_N2_out /= total
        else:
            # Fallback to inlet
            y_O2_out = conditions.inlet_composition.O2
            y_CO2_out = conditions.inlet_composition.CO2
            y_N2_out = conditions.inlet_composition.N2
        
        return GasComposition(O2=y_O2_out, CO2=y_CO2_out, N2=y_N2_out)


class ConstantKLa(MassTransferModel):
    """Constant kLa (simplest model)."""
    
    def __init__(self, kLa_O2: float = 100.0, kLa_CO2: float = 80.0):
        """
        Args:
            kLa_O2: Oxygen mass transfer coefficient (1/h)
            kLa_CO2: CO2 mass transfer coefficient (1/h)
        """
        self.kLa_O2 = kLa_O2
        self.kLa_CO2 = kLa_CO2
    
    def get_kLa_O2(self, conditions: GasPhaseConditions) -> float:
        return self.kLa_O2
    
    def get_kLa_CO2(self, conditions: GasPhaseConditions) -> float:
        return self.kLa_CO2


class DynamicKLa(MassTransferModel):
    """
    Dynamic kLa based on operating conditions.
    
    Correlation: kLa = k × N^a × (Q/V)^b × exp(-k_X × X)
    
    Where:
        N: Agitation speed (rpm)
        Q/V: Specific aeration rate (vvm)
        X: Biomass concentration (g/L)
    """
    
    def __init__(self, 
                 k_O2: float = 0.00026,
                 a: float = 2.0,
                 b: float = 0.5,
                 k_X: float = 0.02,
                 kLa_ratio_CO2_O2: float = 0.8):
        """
        Args:
            k_O2: Correlation constant for O2
            a: Exponent for agitation speed
            b: Exponent for aeration rate
            k_X: Biomass correction factor (L/g)
            kLa_ratio_CO2_O2: Ratio of kLa_CO2 / kLa_O2
        """
        self.k_O2 = k_O2
        self.a = a
        self.b = b
        self.k_X = k_X
        self.ratio = kLa_ratio_CO2_O2
    
    def get_kLa_O2(self, conditions: GasPhaseConditions) -> float:
        # Get operating conditions
        N = conditions.N if conditions.N is not None else 300.0  # rpm
        vvm = conditions.Q_gas / conditions.V_liquid / 60.0  # Convert to vvm (1/min)
        X = conditions.X if conditions.X is not None else 0.0  # g/L
        
        # Base correlation
        kLa_base = self.k_O2 * (N ** self.a) * (vvm ** self.b)
        
        # Biomass correction
        kLa_corrected = kLa_base * np.exp(-self.k_X * X)
        
        return kLa_corrected
    
    def get_kLa_CO2(self, conditions: GasPhaseConditions) -> float:
        kLa_O2 = self.get_kLa_O2(conditions)
        return kLa_O2 * self.ratio


# Example usage and testing
if __name__ == "__main__":
    print("=" * 60)
    print("MASS TRANSFER VALIDATION")
    print("=" * 60)
    
    # Test Henry's constants at different temperatures
    print("\nHenry's constants:")
    for T in [20, 25, 30, 37, 40]:
        H_O2 = HenryConstant.get_H_O2(T)
        H_CO2 = HenryConstant.get_H_CO2(T)
        print(f"  T = {T}°C: H_O2 = {H_O2*1000:.3f} mmol/L/atm, H_CO2 = {H_CO2*1000:.3f} mmol/L/atm")
    
    # Test dynamic kLa
    print("\n" + "=" * 60)
    print("Dynamic kLa correlation")
    print("=" * 60)
    
    mass_transfer = DynamicKLa(k_O2=0.00026, a=2.0, b=0.5, k_X=0.02)
    
    inlet_gas = GasComposition(O2=0.21, CO2=0.0003, N2=0.7897)
    
    test_conditions = [
        (300, 60, 0, "Low agitation, low aeration, no biomass"),
        (600, 120, 0, "High agitation, high aeration, no biomass"),
        (600, 120, 20, "High agitation, high aeration, high biomass"),
    ]
    
    for N, Q_gas, X, description in test_conditions:
        conditions = GasPhaseConditions(
            inlet_composition=inlet_gas,
            Q_gas=Q_gas,
            V_liquid=2.0,
            T=37.0,
            N=N,
            X=X
        )
        
        kLa_O2 = mass_transfer.get_kLa_O2(conditions)
        kLa_CO2 = mass_transfer.get_kLa_CO2(conditions)
        
        print(f"\n{description}")
        print(f"  N = {N} rpm, Q_gas = {Q_gas} L/h, X = {X} g/L")
        print(f"  kLa_O2 = {kLa_O2:.1f} 1/h")
        print(f"  kLa_CO2 = {kLa_CO2:.1f} 1/h")
    
    # Test mass transfer rates
    print("\n" + "=" * 60)
    print("Mass transfer rates")
    print("=" * 60)
    
    conditions = GasPhaseConditions(
        inlet_composition=inlet_gas,
        Q_gas=120,
        V_liquid=2.0,
        T=37.0,
        N=600,
        X=10
    )
    
    # Dissolved concentrations
    DO = 0.05  # mmol/L (~25% saturation)
    DCO2 = 1.0  # mmol/L
    
    # Partial pressures (inlet)
    P_O2 = inlet_gas.O2 * 1.0  # atm
    P_CO2 = inlet_gas.CO2 * 1.0
    
    # Calculate saturation
    C_star_O2 = mass_transfer.saturation_concentration_O2(P_O2, conditions.T)
    C_star_CO2 = mass_transfer.saturation_concentration_CO2(P_CO2, conditions.T)
    
    print(f"\nConditions:")
    print(f"  DO = {DO:.3f} mmol/L")
    print(f"  C*_O2 = {C_star_O2:.3f} mmol/L ({DO/C_star_O2*100:.1f}% saturation)")
    print(f"  DCO2 = {DCO2:.3f} mmol/L")
    print(f"  C*_CO2 = {C_star_CO2:.3f} mmol/L")
    
    # Transfer rates
    OTR = mass_transfer.oxygen_transfer_rate(DO, P_O2, conditions)
    CTR = mass_transfer.co2_transfer_rate(DCO2, P_CO2, conditions)
    
    print(f"\nTransfer rates:")
    print(f"  OTR = {OTR:.2f} mmol/L/h")
    print(f"  CTR = {CTR:.2f} mmol/L/h")
    
    # Outlet gas composition
    OUR = 50.0  # mmol/L/h (example)
    CER = 54.0  # mmol/L/h
    
    outlet_gas = mass_transfer.outlet_gas_composition(OUR, CER, conditions)
    
    print(f"\nGas composition:")
    print(f"  Inlet:  O2 = {inlet_gas.O2*100:.2f}%, CO2 = {inlet_gas.CO2*100:.3f}%")
    print(f"  Outlet: O2 = {outlet_gas.O2*100:.2f}%, CO2 = {outlet_gas.CO2*100:.3f}%")
    
    print("=" * 60)
