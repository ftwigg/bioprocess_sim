# mass_balance.py
"""
Overall mass balance equations for batch reactor.
Handles liquid phase and gas phase balances.
"""

import numpy as np
from typing import Dict, Tuple
from base_models import ReactorState, ReactorConfig, GasComposition


class LiquidPhaseMassBalance:
    """
    Mass balance for liquid phase components in a batch reactor.
    
    For batch operation:
    - No liquid inlet/outlet
    - Volume changes only due to evaporation (assumed negligible with condenser)
    - Accumulation = Generation - Consumption
    """
    
    def __init__(self, config: ReactorConfig):
        self.config = config
    
    def biomass_balance(self, dX_dt: float) -> float:
        """
        Biomass balance: dX/dt = Î¼ * X
        
        Args:
            dX_dt: Biomass growth rate from kinetics (g/L/h)
            
        Returns:
            Rate of change of biomass concentration (g/L/h)
        """
        return dX_dt
    
    def substrate_balance(self, dS_dt: float) -> float:
        """
        Substrate balance: dS/dt = -qs * X
        
        Args:
            dS_dt: Substrate consumption rate from kinetics (g/L/h)
            
        Returns:
            Rate of change of substrate concentration (g/L/h)
        """
        return dS_dt  # Already negative from kinetics
    
    def product_balance(self, dP_dt: float) -> float:
        """
        Product balance: dP/dt = qp * X
        
        Args:
            dP_dt: Product formation rate from kinetics (g/L/h)
            
        Returns:
            Rate of change of product concentration (g/L/h)
        """
        return dP_dt
    
    def volume_balance(self, Q_in: float = 0.0, Q_out: float = 0.0, 
                       evap_rate: float = 0.0) -> float:
        """
        Volume balance for batch reactor.
        
        For true batch: dV/dt = 0
        For fed-batch: dV/dt = Q_in - Q_out - evap_rate
        
        Args:
            Q_in: Liquid feed rate (L/h)
            Q_out: Liquid removal rate (L/h)
            evap_rate: Evaporation rate (L/h)
            
        Returns:
            Rate of change of volume (L/h)
        """
        # For batch with condenser, all terms are zero
        return Q_in - Q_out - evap_rate


class GasPhaseMassBalance:
    """
    Mass balance for gas phase in batch reactor.
    
    Accounts for:
    - Inlet gas composition and flow rate
    - Oxygen consumption by cells
    - CO2 production by cells
    - Other gases (N2 as inert, potentially CH4, H2)
    - Mass transfer between gas and liquid phases
    """
    
    # Henry's constants at 37Â°C (mol/L/atm)
    # These are approximate and temperature-dependent
    H_O2 = 0.0013  # mol/L/atm
    H_CO2 = 0.034  # mol/L/atm
    H_N2 = 0.0007  # mol/L/atm
    
    def __init__(self, config: ReactorConfig):
        self.config = config
        self.inlet_gas = config.inlet_gas
        
        # Convert gas flow rate from VVM to L/h
        # VVM = volume gas / volume liquid / minute
        self.Q_gas = config.gas_flow_rate * config.V_working * 60  # L/h
    
    def saturation_concentration(self, partial_pressure: float, 
                                 henry_const: float, 
                                 T: float = 37.0) -> float:
        """
        Calculate saturation concentration using Henry's Law.
        
        C* = H * P
        
        Args:
            partial_pressure: Partial pressure of gas (atm)
            henry_const: Henry's constant (mol/L/atm) at reference T
            T: Temperature (Â°C)
            
        Returns:
            Saturation concentration (mmol/L)
        """
        # Temperature correction (van't Hoff equation)
        # Simplified: H(T) â‰ˆ H(T_ref) * exp[-Î”H/R * (1/T - 1/T_ref)]
        # For simplicity, using H at 37Â°C directly
        
        C_sat = henry_const * partial_pressure * 1000  # Convert to mmol/L
        return C_sat
    
    def oxygen_mass_transfer(self, C_L: float, P_O2: float, T: float = 37.0) -> float:
        """
        Calculate oxygen transfer rate from gas to liquid.
        
        OTR = kLa * (C* - C_L)
        
        Args:
            C_L: Dissolved oxygen concentration (mmol/L)
            P_O2: Oxygen partial pressure in gas phase (atm)
            T: Temperature (Â°C)
            
        Returns:
            Oxygen transfer rate (mmol/L/h)
        """
        C_sat = self.saturation_concentration(P_O2, self.H_O2, T)
        OTR = self.config.kLa_O2 * (C_sat - C_L)
        return OTR
    
    def co2_mass_transfer(self, C_L: float, P_CO2: float, T: float = 37.0) -> float:
        """
        Calculate CO2 transfer rate from liquid to gas.
        
        CTR = kLa * (C_L - C*)
        
        Args:
            C_L: Dissolved CO2 concentration (mmol/L)
            P_CO2: CO2 partial pressure in gas phase (atm)
            T: Temperature (Â°C)
            
        Returns:
            CO2 transfer rate (mmol/L/h) - positive means leaving liquid
        """
        C_sat = self.saturation_concentration(P_CO2, self.H_CO2, T)
        CTR = self.config.kLa_CO2 * (C_L - C_sat)
        return CTR
    
    def dissolved_oxygen_balance(self, OTR: float, OUR: float) -> float:
        """
        Balance for dissolved oxygen in liquid phase.
        
        dC_O2/dt = OTR - OUR
        
        Args:
            OTR: Oxygen transfer rate from gas (mmol/L/h)
            OUR: Oxygen uptake rate by cells (mmol/L/h)
            
        Returns:
            Rate of change of dissolved O2 (mmol/L/h)
        """
        return OTR - OUR
    
    def dissolved_co2_balance(self, CTR: float, CER: float) -> float:
        """
        Balance for dissolved CO2 in liquid phase.
        
        dC_CO2/dt = CER - CTR
        
        Args:
            CTR: CO2 transfer rate to gas (mmol/L/h)
            CER: CO2 evolution rate by cells (mmol/L/h)
            
        Returns:
            Rate of change of dissolved CO2 (mmol/L/h)
        """
        return CER - CTR
    
    def outlet_gas_composition(self, OUR: float, CER: float, V_liquid: float) -> GasComposition:
        """
        Calculate outlet gas composition based on cellular respiration.
        
        Material balance on gas phase:
        F_in * y_in - F_out * y_out + r_transfer * V_liquid = 0 (steady state)
        
        Args:
            OUR: Oxygen uptake rate (mmol/L/h)
            CER: CO2 evolution rate (mmol/L/h)
            V_liquid: Liquid volume (L)
            
        Returns:
            Outlet gas composition
        """
        # Convert OUR and CER to molar flow rates (mol/h)
        n_O2_consumed = OUR * V_liquid / 1000  # mol/h
        n_CO2_produced = CER * V_liquid / 1000  # mol/h
        
        # Inlet molar flow rates
        P_total = 1.0  # atm
        T_gas = 37.0 + 273.15  # K
        R = 0.08206  # LÂ·atm/(molÂ·K)
        
        # Total inlet molar flow (mol/h)
        n_in_total = (P_total * self.Q_gas) / (R * T_gas)
        
        n_O2_in = n_in_total * self.inlet_gas.O2
        n_CO2_in = n_in_total * self.inlet_gas.CO2
        n_N2_in = n_in_total * self.inlet_gas.N2
        
        # Outlet molar flows
        n_O2_out = max(0, n_O2_in - n_O2_consumed)
        n_CO2_out = n_CO2_in + n_CO2_produced
        n_N2_out = n_N2_in  # Inert
        
        # Total outlet flow
        n_out_total = n_O2_out + n_CO2_out + n_N2_out
        
        # Outlet composition
        if n_out_total > 0:
            y_O2_out = n_O2_out / n_out_total
            y_CO2_out = n_CO2_out / n_out_total
            y_N2_out = n_N2_out / n_out_total
        else:
            # Fallback to inlet
            y_O2_out = self.inlet_gas.O2
            y_CO2_out = self.inlet_gas.CO2
            y_N2_out = self.inlet_gas.N2
        
        return GasComposition(
            O2=y_O2_out,
            CO2=y_CO2_out,
            N2=y_N2_out,
            CH4=0.0,
            H2=0.0
        )


class ElementalMassBalance:
    """
    Elemental mass balance (C, H, N, O, etc.) to verify closure.
    
    Useful for checking simulation consistency and debugging.
    """
    
    def __init__(self, substrate_formula: Dict[str, float],
                 biomass_formula: Dict[str, float],
                 product_formula: Dict[str, float]):
        """
        Initialize with elemental formulas.
        
        Args:
            substrate_formula: Dict of element: count (e.g., {'C': 6, 'H': 12, 'O': 6} for glucose)
            biomass_formula: Elemental composition of biomass (e.g., CH1.8O0.5N0.2)
            product_formula: Elemental composition of product
        """
        self.substrate_formula = substrate_formula
        self.biomass_formula = biomass_formula
        self.product_formula = product_formula
    
    def carbon_balance(self, S_consumed: float, X_formed: float, 
                       P_formed: float, CO2_out: float,
                       MW_substrate: float, MW_biomass: float, 
                       MW_product: float) -> float:
        """
        Check carbon balance.
        
        C_in (substrate) = C_out (biomass + product + CO2)
        
        Returns:
            Carbon balance closure (should be ~0 for perfect closure)
        """
        # Carbon in substrate
        n_C_substrate = (S_consumed / MW_substrate) * self.substrate_formula.get('C', 0)
        
        # Carbon in biomass
        n_C_biomass = (X_formed / MW_biomass) * self.biomass_formula.get('C', 0)
        
        # Carbon in product
        n_C_product = (P_formed / MW_product) * self.product_formula.get('C', 0)
        
        # Carbon in CO2 (mol)
        n_C_CO2 = CO2_out
        
        # Balance
        C_in = n_C_substrate
        C_out = n_C_biomass + n_C_product + n_C_CO2
        
        return C_in - C_out
    
    def nitrogen_balance(self, S_consumed: float, X_formed: float,
                        P_formed: float, MW_substrate: float,
                        MW_biomass: float, MW_product: float) -> float:
        """Check nitrogen balance."""
        n_N_substrate = (S_consumed / MW_substrate) * self.substrate_formula.get('N', 0)
        n_N_biomass = (X_formed / MW_biomass) * self.biomass_formula.get('N', 0)
        n_N_product = (P_formed / MW_product) * self.product_formula.get('N', 0)
        
        N_in = n_N_substrate
        N_out = n_N_biomass + n_N_product
        
        return N_in - N_out
