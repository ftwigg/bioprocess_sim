# mass_balance.py
"""
Overall mass balance equations for batch and fed-batch reactors.
Handles liquid phase and gas phase balances with proper treatment of
variable volume, temperature-dependent properties, and pH effects.

Key improvements over basic implementation:
- Fed-batch support with dilution terms
- Temperature-dependent Henry's constants
- pH-dependent CO2 solubility
- Dynamic gas phase balance
- Proper coupling between mass transfer and gas composition
- Comprehensive elemental balance checking
"""

import numpy as np
from typing import Dict, Tuple, Optional, List, NamedTuple
from dataclasses import dataclass, field
from enum import Enum

from base_models import ReactorState, ReactorConfig, GasComposition


# =============================================================================
# CONSTANTS AND REFERENCE DATA
# =============================================================================

class GasProperties:
    """
    Reference properties for common gases in fermentation.
    
    Henry's constants follow: H(T) = H_ref * exp[-ΔH_sol/R * (1/T - 1/T_ref)]
    
    Units:
        H: mol/(L·atm) - concentration form of Henry's law
        ΔH_sol: kJ/mol - enthalpy of dissolution
        T_ref: K - reference temperature (typically 298.15 K = 25°C)
    """
    
    # Reference Henry's constants at 25°C (298.15 K) in mol/(L·atm)
    # These are "solubility" form: C = H * P
    H_REF = {
        'O2': 1.26e-3,      # mol/(L·atm)
        'CO2': 3.36e-2,     # mol/(L·atm) - for CO2(aq) only, not total dissolved
        'N2': 6.48e-4,      # mol/(L·atm)
        'CH4': 1.29e-3,     # mol/(L·atm)
        'H2': 7.8e-4,       # mol/(L·atm)
    }
    
    # Enthalpy of dissolution (kJ/mol) - negative means solubility decreases with T
    DELTA_H_SOL = {
        'O2': -12.1,
        'CO2': -19.4,
        'N2': -10.5,
        'CH4': -14.2,
        'H2': -4.2,
    }
    
    T_REF = 298.15  # K (25°C)
    R = 8.314e-3    # kJ/(mol·K)


@dataclass
class FeedStream:
    """
    Definition of a liquid feed stream for fed-batch operation.
    
    Attributes:
        name: Identifier for this feed stream
        flow_rate: Volumetric flow rate (L/h)
        concentrations: Dict of component name -> concentration (g/L)
        temperature: Feed temperature (°C)
    """
    name: str
    flow_rate: float = 0.0  # L/h
    concentrations: Dict[str, float] = field(default_factory=dict)
    temperature: float = 25.0  # °C
    
    def get_mass_flow(self, component: str) -> float:
        """Get mass flow rate of a component (g/h)."""
        return self.flow_rate * self.concentrations.get(component, 0.0)


@dataclass
class GasStreamState:
    """
    State of the gas phase at a given time.
    
    Attributes:
        composition: Mole fractions of each component
        total_flow: Total molar flow rate (mol/h)
        pressure: Total pressure (atm)
        temperature: Gas temperature (K)
    """
    composition: GasComposition
    total_flow: float  # mol/h
    pressure: float = 1.0  # atm
    temperature: float = 310.15  # K (37°C)
    
    def get_partial_pressure(self, component: str) -> float:
        """Get partial pressure of a component (atm)."""
        mole_fractions = {
            'O2': self.composition.O2,
            'CO2': self.composition.CO2,
            'N2': self.composition.N2,
            'CH4': self.composition.CH4,
            'H2': self.composition.H2,
        }
        return self.pressure * mole_fractions.get(component, 0.0)
    
    def get_molar_flow(self, component: str) -> float:
        """Get molar flow rate of a component (mol/h)."""
        mole_fractions = {
            'O2': self.composition.O2,
            'CO2': self.composition.CO2,
            'N2': self.composition.N2,
            'CH4': self.composition.CH4,
            'H2': self.composition.H2,
        }
        return self.total_flow * mole_fractions.get(component, 0.0)


# =============================================================================
# LIQUID PHASE MASS BALANCE
# =============================================================================

class LiquidPhaseMassBalance:
    """
    Mass balance for liquid phase components in batch or fed-batch reactor.
    
    Key equations (for fed-batch with variable volume):
    
    Total mass balance:
        d(V)/dt = Q_in - Q_out - Q_evap
    
    Component mass balance (on TOTAL MASS basis):
        d(V·C_i)/dt = Q_in·C_i,in - Q_out·C_i + r_i·V
    
    Which expands to concentration form:
        dC_i/dt = (Q_in/V)·(C_i,in - C_i) + r_i - C_i·(Q_evap/V) - C_i·(Q_out/V)
        
    Or equivalently:
        dC_i/dt = D_in·(C_i,in - C_i) + r_i - D_out·C_i - D_evap·C_i
    
    where D = Q/V is the dilution rate.
    
    For batch: Q_in = Q_out = 0, so dC_i/dt = r_i (concentration basis is valid)
    """
    
    def __init__(self, config: ReactorConfig, 
                 feed_streams: Optional[List[FeedStream]] = None):
        """
        Initialize liquid phase mass balance.
        
        Args:
            config: Reactor configuration
            feed_streams: Optional list of feed streams for fed-batch
        """
        self.config = config
        self.feed_streams = feed_streams or []
        
        # Track current volume (may change in fed-batch)
        self._current_volume = config.V_working
    
    @property
    def current_volume(self) -> float:
        """Current working volume (L)."""
        return self._current_volume
    
    @current_volume.setter
    def current_volume(self, value: float):
        """Set current volume with validation."""
        if value <= 0:
            raise ValueError("Volume must be positive")
        if value > self.config.V_reactor:
            raise ValueError(f"Volume {value} exceeds reactor capacity {self.config.V_reactor}")
        self._current_volume = value
    
    def get_total_feed_rate(self) -> float:
        """Get total volumetric feed rate from all streams (L/h)."""
        return sum(feed.flow_rate for feed in self.feed_streams)
    
    def get_feed_concentration(self, component: str) -> float:
        """
        Get flow-weighted average concentration of a component in combined feed.
        
        Args:
            component: Name of component ('X', 'S', 'P', etc.)
            
        Returns:
            Average feed concentration (g/L)
        """
        total_flow = self.get_total_feed_rate()
        if total_flow <= 0:
            return 0.0
        
        total_mass_flow = sum(
            feed.flow_rate * feed.concentrations.get(component, 0.0)
            for feed in self.feed_streams
        )
        return total_mass_flow / total_flow
    
    def volume_balance(self, Q_out: float = 0.0, 
                       evap_rate: float = 0.0) -> float:
        """
        Calculate rate of volume change.
        
        dV/dt = Σ Q_in,i - Q_out - Q_evap
        
        Args:
            Q_out: Liquid removal rate (L/h)
            evap_rate: Evaporation rate (L/h)
            
        Returns:
            Rate of change of volume (L/h)
        """
        Q_in_total = self.get_total_feed_rate()
        return Q_in_total - Q_out - evap_rate
    
    def dilution_rate(self, Q_out: float = 0.0) -> float:
        """
        Calculate dilution rate for fed-batch.
        
        D = Q_in / V
        
        For chemostat: D = Q_in/V = Q_out/V at steady state
        
        Args:
            Q_out: Liquid removal rate (L/h)
            
        Returns:
            Dilution rate (1/h)
        """
        Q_in_total = self.get_total_feed_rate()
        V = self.current_volume
        return Q_in_total / V if V > 0 else 0.0
    
    def component_balance(self, 
                          C: float, 
                          r: float, 
                          component: str,
                          Q_out: float = 0.0,
                          evap_rate: float = 0.0,
                          C_out: Optional[float] = None) -> float:
        """
        Calculate rate of change of concentration for any component.
        
        Full equation (concentration form):
        dC/dt = (Q_in/V)·(C_in - C) + r - (Q_out/V)·C - (Q_evap/V)·C
        
        Simplification for well-mixed reactor with same outlet conc:
        dC/dt = D_in·(C_in - C) + r - D_out·C - D_evap·C
        
        For batch (all Q = 0):
        dC/dt = r
        
        Args:
            C: Current concentration (g/L)
            r: Volumetric reaction rate (g/L/h) - positive for production
            component: Component name for getting feed concentration
            Q_out: Liquid removal rate (L/h)
            evap_rate: Evaporation rate (L/h)
            C_out: Outlet concentration if different from C (default: C)
            
        Returns:
            Rate of change of concentration (g/L/h)
        """
        V = self.current_volume
        if V <= 0:
            return 0.0
        
        # Get feed properties
        Q_in = self.get_total_feed_rate()
        C_in = self.get_feed_concentration(component)
        
        # Use current concentration for outlet if not specified
        if C_out is None:
            C_out = C
        
        # Calculate individual contributions
        feed_term = (Q_in / V) * (C_in - C)  # Dilution from feed
        reaction_term = r  # Biological/chemical reaction
        outlet_term = -(Q_out / V) * C_out  # Removal
        evap_term = -(evap_rate / V) * C  # Evaporation (concentrates)
        
        # Note: Evaporation of water concentrates non-volatile species
        # For volatile species (like ethanol), would need separate treatment
        
        return feed_term + reaction_term + outlet_term + evap_term
    
    def biomass_balance(self, X: float, r_X: float,
                        Q_out: float = 0.0,
                        evap_rate: float = 0.0) -> float:
        """
        Biomass balance with dilution effects.
        
        dX/dt = μX - D·X (for fed-batch with sterile feed)
        
        Args:
            X: Current biomass concentration (g/L)
            r_X: Biomass growth rate = μX (g/L/h)
            Q_out: Liquid removal rate (L/h)
            evap_rate: Evaporation rate (L/h)
            
        Returns:
            Rate of change of biomass concentration (g/L/h)
        """
        return self.component_balance(
            C=X, 
            r=r_X, 
            component='X',
            Q_out=Q_out,
            evap_rate=evap_rate
        )
    
    def substrate_balance(self, S: float, r_S: float,
                          Q_out: float = 0.0,
                          evap_rate: float = 0.0) -> float:
        """
        Substrate balance with feed addition (key for fed-batch!).
        
        dS/dt = D·(S_feed - S) - qS·X
        
        Args:
            S: Current substrate concentration (g/L)
            r_S: Substrate consumption rate (g/L/h) - typically NEGATIVE
            Q_out: Liquid removal rate (L/h)
            evap_rate: Evaporation rate (L/h)
            
        Returns:
            Rate of change of substrate concentration (g/L/h)
        """
        return self.component_balance(
            C=S, 
            r=r_S,  # Already negative from kinetics
            component='S',
            Q_out=Q_out,
            evap_rate=evap_rate
        )
    
    def product_balance(self, P: float, r_P: float,
                        Q_out: float = 0.0,
                        evap_rate: float = 0.0) -> float:
        """
        Product balance.
        
        dP/dt = qP·X - D·P (assuming no product in feed)
        
        Args:
            P: Current product concentration (g/L)
            r_P: Product formation rate (g/L/h)
            Q_out: Liquid removal rate (L/h)
            evap_rate: Evaporation rate (L/h)
            
        Returns:
            Rate of change of product concentration (g/L/h)
        """
        return self.component_balance(
            C=P, 
            r=r_P,
            component='P',
            Q_out=Q_out,
            evap_rate=evap_rate
        )
    
    def total_mass_derivative(self, C: float, V: float, dC_dt: float, dV_dt: float) -> float:
        """
        Calculate rate of change of total mass of a component.
        
        d(V·C)/dt = V·dC/dt + C·dV/dt
        
        Useful for verification that mass is conserved.
        
        Args:
            C: Concentration (g/L)
            V: Volume (L)
            dC_dt: Rate of concentration change (g/L/h)
            dV_dt: Rate of volume change (L/h)
            
        Returns:
            Rate of total mass change (g/h)
        """
        return V * dC_dt + C * dV_dt


# =============================================================================
# GAS PHASE MASS BALANCE
# =============================================================================

class GasPhaseMassBalance:
    """
    Mass balance for gas phase in batch/fed-batch reactor.
    
    Key improvements:
    1. Temperature-dependent Henry's constants
    2. pH-dependent CO2 solubility (accounts for carbonate equilibria)
    3. Dynamic gas phase balance (not steady-state assumption)
    4. Proper coupling between OTR/CTR and outlet gas composition
    
    Key equations:
    
    Mass transfer:
        OTR = kLa_O2 · (C*_O2 - C_O2)
        CTR = kLa_CO2 · (C_CO2 - C*_CO2)
    
    Dissolved gas balance:
        dC_O2/dt = OTR - OUR
        dC_CO2/dt = CER - CTR
    
    Gas phase balance (dynamic):
        d(n_i)/dt = F_in·y_i,in - F_out·y_i,out ± r_transfer·V_L
    
    Where:
        n_i = moles of component i in headspace
        F = molar flow rate
        y = mole fraction
        r_transfer = OTR or CTR (mass transfer rate)
    """
    
    def __init__(self, config: ReactorConfig,
                 headspace_volume: Optional[float] = None,
                 dynamic_gas_phase: bool = False):
        """
        Initialize gas phase mass balance.
        
        Args:
            config: Reactor configuration
            headspace_volume: Gas phase volume (L). If None, calculated from reactor geometry
            dynamic_gas_phase: If True, track gas phase dynamics; if False, assume steady state
        """
        self.config = config
        self.inlet_gas = config.inlet_gas
        self.dynamic_gas_phase = dynamic_gas_phase
        
        # Calculate headspace volume
        if headspace_volume is None:
            self.V_headspace = config.V_reactor - config.V_working
        else:
            self.V_headspace = headspace_volume
        
        # Convert gas flow rate from VVM to L/h
        # VVM = volume gas / volume liquid / minute
        self.Q_gas_inlet = config.gas_flow_rate * config.V_working * 60  # L/h
        
        # Current gas phase state (for dynamic mode)
        self._gas_state = None
    
    def henry_constant(self, gas: str, T: float) -> float:
        """
        Calculate temperature-corrected Henry's constant.
        
        H(T) = H_ref · exp[-ΔH_sol/R · (1/T - 1/T_ref)]
        
        Args:
            gas: Gas species ('O2', 'CO2', 'N2', etc.)
            T: Temperature (°C)
            
        Returns:
            Henry's constant (mol/L/atm)
        """
        T_K = T + 273.15
        
        H_ref = GasProperties.H_REF.get(gas, 1e-3)
        dH_sol = GasProperties.DELTA_H_SOL.get(gas, -10.0)
        T_ref = GasProperties.T_REF
        R = GasProperties.R
        
        # Van't Hoff temperature correction
        H = H_ref * np.exp(-dH_sol / R * (1/T_K - 1/T_ref))
        
        return H
    
    def co2_apparent_henry(self, T: float, pH: float) -> float:
        """
        Calculate apparent Henry's constant for CO2 including carbonate equilibria.
        
        CO2(g) ⇌ CO2(aq) + H2O ⇌ HCO3⁻ + H⁺ ⇌ CO3²⁻ + 2H⁺
        
        The apparent (total) solubility includes all dissolved forms:
        C_CO2,total = C_CO2(aq) + C_HCO3- + C_CO32-
        
        Using the "apparent" first dissociation constant (pKa1 ≈ 6.35) which 
        combines hydration and first dissociation:
        
        CO2(aq) + H2O ⇌ H⁺ + HCO3⁻   (apparent Ka1)
        
        The fraction existing as CO2(aq) is:
        α₀ = [H⁺]² / ([H⁺]² + Ka1·[H⁺] + Ka1·Ka2)
        
        Enhancement factor = 1/α₀
        
        At low pH (<5): H_apparent ≈ H_physical (enhancement ≈ 1)
        At pH 6.35 (pKa1): enhancement ≈ 2
        At pH 7: enhancement ≈ 5
        At pH 8: enhancement ≈ 50
        At pH 9: enhancement ≈ 500
        
        Args:
            T: Temperature (°C)
            pH: Liquid phase pH
            
        Returns:
            Apparent Henry's constant for total dissolved CO2 (mol/L/atm)
        """
        # Physical Henry's constant for CO2(aq)
        H_physical = self.henry_constant('CO2', T)
        
        # Apparent equilibrium constants
        # These are the commonly tabulated values that combine hydration with dissociation
        pKa1 = 6.35   # CO2(aq) + H2O ⇌ H+ + HCO3-
        pKa2 = 10.33  # HCO3- ⇌ H+ + CO32-
        
        # Temperature correction for pKa values
        # pKa1 decreases by ~0.005 per °C increase
        # pKa2 decreases by ~0.009 per °C increase
        pKa1_T = pKa1 - 0.005 * (T - 25)
        pKa2_T = pKa2 - 0.009 * (T - 25)
        
        Ka1 = 10**(-pKa1_T)
        Ka2 = 10**(-pKa2_T)
        
        # Hydrogen ion concentration
        H_plus = 10**(-pH)
        
        # Fraction of total dissolved carbonate existing as CO2(aq)
        # α₀ = [H⁺]² / ([H⁺]² + Ka1·[H⁺] + Ka1·Ka2)
        denominator = H_plus**2 + Ka1 * H_plus + Ka1 * Ka2
        alpha_0 = H_plus**2 / denominator
        
        # Enhancement factor = 1/α₀ (total solubility / CO2(aq) solubility)
        # Limit enhancement to reasonable range to avoid numerical issues at extreme pH
        enhancement = min(1.0 / alpha_0, 10000.0)
        
        H_apparent = H_physical * enhancement
        
        return H_apparent
    
    def saturation_concentration(self, gas: str, P_partial: float, 
                                  T: float, pH: float = 7.0) -> float:
        """
        Calculate saturation concentration using Henry's Law.
        
        C* = H(T) · P
        
        For CO2, uses apparent Henry's constant that includes carbonate equilibria.
        
        Args:
            gas: Gas species
            P_partial: Partial pressure (atm)
            T: Temperature (°C)
            pH: Liquid phase pH (only used for CO2)
            
        Returns:
            Saturation concentration (mmol/L)
        """
        if gas == 'CO2':
            H = self.co2_apparent_henry(T, pH)
        else:
            H = self.henry_constant(gas, T)
        
        # C* in mol/L, convert to mmol/L
        C_sat = H * P_partial * 1000
        
        return C_sat
    
    def kLa_correlation(self, gas: str, T: float, 
                        agitation: Optional[float] = None,
                        gas_flow: Optional[float] = None) -> float:
        """
        Calculate kLa with optional corrections for operating conditions.
        
        Base kLa from config, with corrections for:
        - Temperature: kLa(T) = kLa(T_ref) · 1.024^(T-T_ref) [typical correlation]
        - Different gases: kLa_i = kLa_O2 · (D_i/D_O2)^0.5 [penetration theory]
        
        Diffusivity ratios (approximate):
            D_CO2/D_O2 ≈ 0.89
            D_N2/D_O2 ≈ 0.93
        
        Args:
            gas: Gas species
            T: Temperature (°C)
            agitation: Agitation rate (rpm) - for future correlation
            gas_flow: Gas flow rate (VVM) - for future correlation
            
        Returns:
            kLa value (1/h)
        """
        # Base kLa from config
        if gas == 'O2':
            kLa_base = self.config.kLa_O2
        elif gas == 'CO2':
            kLa_base = self.config.kLa_CO2
        else:
            # Estimate from O2 using diffusivity ratio
            D_ratios = {'N2': 0.93, 'CH4': 0.80, 'H2': 1.5}
            kLa_base = self.config.kLa_O2 * np.sqrt(D_ratios.get(gas, 1.0))
        
        # Temperature correction
        T_ref = 37.0  # Reference temperature in config
        theta = 1.024  # Temperature coefficient
        kLa_T = kLa_base * (theta ** (T - T_ref))
        
        # Could add correlations for agitation and gas flow here
        # e.g., kLa ∝ (P/V)^a · (vs)^b where P/V is power input, vs is superficial gas velocity
        
        return kLa_T
    
    def oxygen_mass_transfer(self, C_L: float, P_O2: float, 
                              T: float, pH: float = 7.0) -> float:
        """
        Calculate oxygen transfer rate from gas to liquid.
        
        OTR = kLa · (C* - C_L)
        
        Args:
            C_L: Dissolved oxygen concentration (mmol/L)
            P_O2: Oxygen partial pressure in gas phase (atm)
            T: Temperature (°C)
            pH: Liquid pH (not used for O2, included for interface consistency)
            
        Returns:
            Oxygen transfer rate (mmol/L/h)
        """
        C_sat = self.saturation_concentration('O2', P_O2, T, pH)
        kLa = self.kLa_correlation('O2', T)
        OTR = kLa * (C_sat - C_L)
        return OTR
    
    def co2_mass_transfer(self, C_L: float, P_CO2: float,
                           T: float, pH: float = 7.0) -> float:
        """
        Calculate CO2 transfer rate from liquid to gas.
        
        CTR = kLa · (C_L - C*)
        
        Note: Uses apparent Henry's constant accounting for pH effects.
        
        Args:
            C_L: Total dissolved CO2 concentration (mmol/L)
            P_CO2: CO2 partial pressure in gas phase (atm)
            T: Temperature (°C)
            pH: Liquid pH (affects CO2 solubility!)
            
        Returns:
            CO2 transfer rate (mmol/L/h) - positive means leaving liquid
        """
        C_sat = self.saturation_concentration('CO2', P_CO2, T, pH)
        kLa = self.kLa_correlation('CO2', T)
        CTR = kLa * (C_L - C_sat)
        return CTR
    
    def dissolved_oxygen_balance(self, OTR: float, OUR: float,
                                  V: float, dV_dt: float = 0.0) -> float:
        """
        Balance for dissolved oxygen in liquid phase.
        
        For constant volume:
            dC_O2/dt = OTR - OUR
        
        For variable volume (fed-batch):
            d(V·C_O2)/dt = OTR·V - OUR·V
            dC_O2/dt = OTR - OUR - C_O2·(dV/dt)/V
        
        Args:
            OTR: Oxygen transfer rate from gas (mmol/L/h)
            OUR: Oxygen uptake rate by cells (mmol/L/h)
            V: Current liquid volume (L)
            dV_dt: Rate of volume change (L/h)
            
        Returns:
            Rate of change of dissolved O2 (mmol/L/h)
        """
        # Volume dilution effect (for fed-batch)
        # Note: OTR and OUR are already volumetric rates (per L)
        dilution_correction = 0.0  # Would need C_O2 to calculate
        
        return OTR - OUR  # Simplified for now
    
    def dissolved_co2_balance(self, CTR: float, CER: float,
                               V: float, dV_dt: float = 0.0) -> float:
        """
        Balance for dissolved CO2 in liquid phase.
        
        dC_CO2/dt = CER - CTR
        
        Args:
            CTR: CO2 transfer rate to gas (mmol/L/h)
            CER: CO2 evolution rate by cells (mmol/L/h)
            V: Current liquid volume (L)
            dV_dt: Rate of volume change (L/h)
            
        Returns:
            Rate of change of dissolved CO2 (mmol/L/h)
        """
        return CER - CTR
    
    def inlet_molar_flow(self, T_gas: float = 37.0, P_total: float = 1.0) -> float:
        """
        Calculate total inlet molar flow rate.
        
        Using ideal gas law: n/t = PQ/(RT)
        
        Args:
            T_gas: Gas temperature (°C)
            P_total: Total pressure (atm)
            
        Returns:
            Total molar flow rate (mol/h)
        """
        T_K = T_gas + 273.15
        R = 0.08206  # L·atm/(mol·K)
        
        n_dot = (P_total * self.Q_gas_inlet) / (R * T_K)
        return n_dot
    
    def gas_phase_balance(self, 
                          OTR: float, 
                          CTR: float, 
                          V_liquid: float,
                          T_gas: float = 37.0,
                          P_total: float = 1.0,
                          current_composition: Optional[GasComposition] = None
                          ) -> Tuple[GasComposition, Dict[str, float]]:
        """
        Calculate outlet gas composition using mass transfer rates (not OUR/CER).
        
        This properly couples the gas phase with the mass transfer rates,
        which may differ from biological rates during transients.
        
        Gas phase mole balance for each component:
            d(n_i)/dt = F_in·y_i,in - F_out·y_i,out ± OTR·V (for O2) ± CTR·V (for CO2)
        
        At quasi-steady state (small headspace):
            F_in·y_i,in - F_out·y_i,out ± r_transfer·V = 0
        
        Args:
            OTR: Oxygen transfer rate (mmol/L/h) - O2 going INTO liquid
            CTR: CO2 transfer rate (mmol/L/h) - CO2 leaving liquid
            V_liquid: Liquid volume (L)
            T_gas: Gas temperature (°C)
            P_total: Total pressure (atm)
            current_composition: Current gas composition (for dynamic mode)
            
        Returns:
            Tuple of (outlet composition, dict of molar flows)
        """
        # Inlet molar flows
        n_in_total = self.inlet_molar_flow(T_gas, P_total)
        
        n_O2_in = n_in_total * self.inlet_gas.O2
        n_CO2_in = n_in_total * self.inlet_gas.CO2
        n_N2_in = n_in_total * self.inlet_gas.N2
        n_CH4_in = n_in_total * self.inlet_gas.CH4
        n_H2_in = n_in_total * self.inlet_gas.H2
        
        # Mass transfer rates to/from liquid (convert mmol/L/h to mol/h)
        # OTR > 0 means O2 going from gas to liquid (consuming from gas)
        # CTR > 0 means CO2 going from liquid to gas (adding to gas)
        n_O2_transfer = OTR * V_liquid / 1000  # mol/h consumed from gas
        n_CO2_transfer = CTR * V_liquid / 1000  # mol/h added to gas
        
        # Outlet molar flows (steady state gas phase)
        n_O2_out = max(0, n_O2_in - n_O2_transfer)
        n_CO2_out = n_CO2_in + n_CO2_transfer
        n_N2_out = n_N2_in  # Inert
        n_CH4_out = n_CH4_in  # Inert (unless there's consumption)
        n_H2_out = n_H2_in  # Inert (unless there's consumption)
        
        # Total outlet flow
        n_out_total = n_O2_out + n_CO2_out + n_N2_out + n_CH4_out + n_H2_out
        
        # Outlet composition (mole fractions)
        if n_out_total > 0:
            y_O2_out = n_O2_out / n_out_total
            y_CO2_out = n_CO2_out / n_out_total
            y_N2_out = n_N2_out / n_out_total
            y_CH4_out = n_CH4_out / n_out_total
            y_H2_out = n_H2_out / n_out_total
        else:
            # Fallback to inlet
            return self.inlet_gas, {}
        
        outlet_composition = GasComposition(
            O2=y_O2_out,
            CO2=y_CO2_out,
            N2=y_N2_out,
            CH4=y_CH4_out,
            H2=y_H2_out
        )
        
        molar_flows = {
            'n_O2_in': n_O2_in,
            'n_O2_out': n_O2_out,
            'n_O2_transfer': n_O2_transfer,
            'n_CO2_in': n_CO2_in,
            'n_CO2_out': n_CO2_out,
            'n_CO2_transfer': n_CO2_transfer,
            'n_total_in': n_in_total,
            'n_total_out': n_out_total,
        }
        
        return outlet_composition, molar_flows
    
    def dynamic_headspace_balance(self,
                                   current_moles: Dict[str, float],
                                   OTR: float,
                                   CTR: float,
                                   V_liquid: float,
                                   T_gas: float = 37.0,
                                   P_total: float = 1.0) -> Dict[str, float]:
        """
        Calculate rate of change of moles in headspace (dynamic gas phase).
        
        d(n_i)/dt = F_in·y_i,in - F_out·y_i,out ± r_transfer·V
        
        For variable headspace composition with finite volume.
        
        Args:
            current_moles: Current moles in headspace {gas: n_moles}
            OTR: Oxygen transfer rate (mmol/L/h)
            CTR: CO2 transfer rate (mmol/L/h)
            V_liquid: Liquid volume (L)
            T_gas: Gas temperature (°C)
            P_total: Total pressure (atm)
            
        Returns:
            Dict of dn_i/dt for each component (mol/h)
        """
        # Inlet flows
        n_in_total = self.inlet_molar_flow(T_gas, P_total)
        
        # Current headspace composition (from current moles)
        n_total = sum(current_moles.values())
        if n_total <= 0:
            n_total = 1e-10  # Avoid division by zero
        
        y_current = {gas: n / n_total for gas, n in current_moles.items()}
        
        # Outlet flow rate (assuming pressure is maintained)
        # At constant pressure and temperature: F_out ≈ F_in + net moles added
        n_O2_transfer = OTR * V_liquid / 1000  # mol/h removed from gas
        n_CO2_transfer = CTR * V_liquid / 1000  # mol/h added to gas
        
        n_out_total = n_in_total - n_O2_transfer + n_CO2_transfer
        
        # Rate of change for each component
        dn_dt = {}
        
        # O2
        dn_dt['O2'] = (n_in_total * self.inlet_gas.O2 
                       - n_out_total * y_current.get('O2', 0.21)
                       - n_O2_transfer)
        
        # CO2
        dn_dt['CO2'] = (n_in_total * self.inlet_gas.CO2 
                        - n_out_total * y_current.get('CO2', 0.0003)
                        + n_CO2_transfer)
        
        # N2 (inert)
        dn_dt['N2'] = (n_in_total * self.inlet_gas.N2 
                       - n_out_total * y_current.get('N2', 0.79))
        
        # CH4 (if present)
        dn_dt['CH4'] = (n_in_total * self.inlet_gas.CH4 
                        - n_out_total * y_current.get('CH4', 0.0))
        
        # H2 (if present)
        dn_dt['H2'] = (n_in_total * self.inlet_gas.H2 
                       - n_out_total * y_current.get('H2', 0.0))
        
        return dn_dt
    
    # Keep backward-compatible method name
    def outlet_gas_composition(self, OUR: float, CER: float, V_liquid: float,
                               T: float = 37.0, pH: float = 7.0) -> GasComposition:
        """
        Calculate outlet gas composition (backward compatible interface).
        
        Note: This method uses biological rates (OUR/CER) directly, which 
        assumes mass transfer is fast relative to biological rates.
        For more accurate results, use gas_phase_balance() with OTR/CTR.
        
        Args:
            OUR: Oxygen uptake rate (mmol/L/h)
            CER: CO2 evolution rate (mmol/L/h)
            V_liquid: Liquid volume (L)
            T: Temperature (°C)
            pH: Liquid pH (for future use)
            
        Returns:
            Outlet gas composition
        """
        # At quasi-steady state for dissolved gases: OTR ≈ OUR, CTR ≈ CER
        outlet_comp, _ = self.gas_phase_balance(
            OTR=OUR,  # Assuming steady state: OTR = OUR
            CTR=CER,  # Assuming steady state: CTR = CER
            V_liquid=V_liquid,
            T_gas=T
        )
        return outlet_comp


# =============================================================================
# ELEMENTAL MASS BALANCE
# =============================================================================

@dataclass
class ElementalFormula:
    """
    Elemental composition of a compound.
    
    Example: Glucose C6H12O6
        ElementalFormula(C=6, H=12, O=6, N=0, S=0, P=0, MW=180.16)
    
    Example: Biomass CH1.8O0.5N0.2 (typical)
        ElementalFormula(C=1, H=1.8, O=0.5, N=0.2, S=0.01, P=0.02, MW=24.6)
    """
    C: float = 0.0
    H: float = 0.0
    O: float = 0.0
    N: float = 0.0
    S: float = 0.0
    P: float = 0.0
    MW: float = 1.0  # Molecular weight (g/mol)
    
    def to_dict(self) -> Dict[str, float]:
        return {'C': self.C, 'H': self.H, 'O': self.O, 
                'N': self.N, 'S': self.S, 'P': self.P}
    
    def degree_of_reduction(self) -> float:
        """
        Calculate degree of reduction per C-mol.
        
        γ = 4·C + H - 2·O - 3·N + 6·S + 5·P (per mole of compound)
        γ/C = (4·C + H - 2·O - 3·N + 6·S + 5·P) / C (per C-mol)
        
        Reference: γ_O2 = 0, γ_CO2 = 0, γ_H2O = 0
        """
        if self.C == 0:
            return 0.0
        
        gamma = 4*self.C + self.H - 2*self.O - 3*self.N + 6*self.S + 5*self.P
        return gamma / self.C


# Common formulas
COMMON_FORMULAS = {
    'glucose': ElementalFormula(C=6, H=12, O=6, N=0, S=0, P=0, MW=180.16),
    'biomass_ecoli': ElementalFormula(C=1, H=1.77, O=0.49, N=0.24, S=0.01, P=0.02, MW=24.6),
    'biomass_yeast': ElementalFormula(C=1, H=1.83, O=0.56, N=0.17, S=0.003, P=0.01, MW=25.0),
    'biomass_cho': ElementalFormula(C=1, H=1.8, O=0.5, N=0.2, S=0.005, P=0.01, MW=24.6),
    'lactate': ElementalFormula(C=3, H=6, O=3, N=0, S=0, P=0, MW=90.08),
    'acetate': ElementalFormula(C=2, H=4, O=2, N=0, S=0, P=0, MW=60.05),
    'ethanol': ElementalFormula(C=2, H=6, O=1, N=0, S=0, P=0, MW=46.07),
    'glycerol': ElementalFormula(C=3, H=8, O=3, N=0, S=0, P=0, MW=92.09),
    'ammonia': ElementalFormula(C=0, H=4, O=0, N=1, S=0, P=0, MW=18.04),
    'ammonium_sulfate': ElementalFormula(C=0, H=8, O=4, N=2, S=1, P=0, MW=132.14),
    'CO2': ElementalFormula(C=1, H=0, O=2, N=0, S=0, P=0, MW=44.01),
    'O2': ElementalFormula(C=0, H=0, O=2, N=0, S=0, P=0, MW=32.00),
    'H2O': ElementalFormula(C=0, H=2, O=1, N=0, S=0, P=0, MW=18.02),
}


class ElementalMassBalance:
    """
    Elemental mass balance (C, H, N, O, S, P) for verification and analysis.
    
    Key improvements:
    1. Separate nitrogen source tracking
    2. Rate-based balance checking (instantaneous, not just cumulative)
    3. Comprehensive element tracking
    4. Degree of reduction balance
    
    Applications:
    - Verify simulation consistency
    - Debugging mass balance closure
    - Calculate theoretical yields
    - Check stoichiometry
    """
    
    def __init__(self,
                 substrate: ElementalFormula,
                 biomass: ElementalFormula,
                 product: ElementalFormula,
                 nitrogen_source: Optional[ElementalFormula] = None):
        """
        Initialize elemental balance calculator.
        
        Args:
            substrate: Elemental composition of carbon/energy substrate
            biomass: Elemental composition of biomass
            product: Elemental composition of product
            nitrogen_source: Elemental composition of N source (e.g., NH4+)
        """
        self.substrate = substrate
        self.biomass = biomass
        self.product = product
        self.nitrogen_source = nitrogen_source or COMMON_FORMULAS['ammonia']
    
    def mass_to_moles(self, mass: float, formula: ElementalFormula) -> float:
        """Convert mass (g) to moles."""
        return mass / formula.MW if formula.MW > 0 else 0.0
    
    def element_rate(self, 
                     element: str,
                     dS_dt: float,
                     dX_dt: float,
                     dP_dt: float,
                     dN_source_dt: float = 0.0,
                     CO2_rate: float = 0.0,
                     O2_rate: float = 0.0) -> Dict[str, float]:
        """
        Calculate rate of element consumption/production.
        
        For carbon balance:
            d(C_total)/dt = -dS/dt·(C/MW)_S + dX/dt·(C/MW)_X + dP/dt·(C/MW)_P + dCO2/dt·(C/MW)_CO2
        
        Should equal zero for conservation.
        
        Args:
            element: Element symbol ('C', 'N', 'O', 'H', 'S', 'P')
            dS_dt: Substrate consumption rate (g/L/h) - negative for consumption
            dX_dt: Biomass growth rate (g/L/h)
            dP_dt: Product formation rate (g/L/h)
            dN_source_dt: Nitrogen source consumption rate (g/L/h)
            CO2_rate: CO2 production rate (mmol/L/h)
            O2_rate: O2 consumption rate (mmol/L/h)
            
        Returns:
            Dict with element flows in each stream (mol/L/h)
        """
        # Get element content per gram for each compound
        def element_per_gram(formula: ElementalFormula, elem: str) -> float:
            """Moles of element per gram of compound."""
            count = getattr(formula, elem, 0.0)
            return count / formula.MW if formula.MW > 0 else 0.0
        
        # Element flows (mol/L/h)
        # Negative dS_dt means consumption, so negate to get positive consumption
        substrate_flow = -dS_dt * element_per_gram(self.substrate, element)
        biomass_flow = dX_dt * element_per_gram(self.biomass, element)
        product_flow = dP_dt * element_per_gram(self.product, element)
        n_source_flow = -dN_source_dt * element_per_gram(self.nitrogen_source, element)
        
        # Gas flows (convert mmol to mol)
        co2_formula = COMMON_FORMULAS['CO2']
        o2_formula = COMMON_FORMULAS['O2']
        co2_flow = (CO2_rate / 1000) * getattr(co2_formula, element, 0.0)
        o2_flow = -(O2_rate / 1000) * getattr(o2_formula, element, 0.0)  # Negative = consumption
        
        # Water produced (estimate from H and O balance for complete balance)
        # This would require solving for H2O production rate
        
        flows = {
            'substrate_in': substrate_flow,
            'n_source_in': n_source_flow,
            'o2_in': -o2_flow,  # O2 consumed
            'biomass_out': biomass_flow,
            'product_out': product_flow,
            'co2_out': co2_flow,
        }
        
        # Balance (should be ~0)
        total_in = substrate_flow + n_source_flow + (-o2_flow)
        total_out = biomass_flow + product_flow + co2_flow
        flows['balance'] = total_in - total_out
        flows['closure'] = total_out / total_in if total_in != 0 else 1.0
        
        return flows
    
    def carbon_balance(self,
                       S_consumed: float,
                       X_formed: float,
                       P_formed: float,
                       CO2_produced: float) -> Dict[str, float]:
        """
        Check carbon balance on cumulative basis.
        
        C_in (substrate) = C_out (biomass + product + CO2)
        
        Args:
            S_consumed: Total substrate consumed (g/L or g)
            X_formed: Total biomass formed (g/L or g)
            P_formed: Total product formed (g/L or g)
            CO2_produced: Total CO2 produced (mol or mol/L)
            
        Returns:
            Dict with carbon in each stream (mol) and balance
        """
        # Carbon moles in each component
        n_C_substrate = self.mass_to_moles(S_consumed, self.substrate) * self.substrate.C
        n_C_biomass = self.mass_to_moles(X_formed, self.biomass) * self.biomass.C
        n_C_product = self.mass_to_moles(P_formed, self.product) * self.product.C
        n_C_CO2 = CO2_produced  # Already in moles
        
        C_in = n_C_substrate
        C_out = n_C_biomass + n_C_product + n_C_CO2
        
        return {
            'C_substrate': n_C_substrate,
            'C_biomass': n_C_biomass,
            'C_product': n_C_product,
            'C_CO2': n_C_CO2,
            'C_in': C_in,
            'C_out': C_out,
            'balance': C_in - C_out,
            'closure': C_out / C_in if C_in > 0 else 1.0
        }
    
    def nitrogen_balance(self,
                         S_consumed: float,
                         X_formed: float,
                         P_formed: float,
                         N_source_consumed: float = 0.0) -> Dict[str, float]:
        """
        Check nitrogen balance including separate N source.
        
        N_in (substrate + N_source) = N_out (biomass + product + NH3_lost)
        
        Args:
            S_consumed: Total substrate consumed (g/L or g)
            X_formed: Total biomass formed (g/L or g)
            P_formed: Total product formed (g/L or g)
            N_source_consumed: Nitrogen source consumed (g/L or g)
            
        Returns:
            Dict with nitrogen in each stream (mol) and balance
        """
        # Nitrogen moles in each component
        n_N_substrate = self.mass_to_moles(S_consumed, self.substrate) * self.substrate.N
        n_N_source = self.mass_to_moles(N_source_consumed, self.nitrogen_source) * self.nitrogen_source.N
        n_N_biomass = self.mass_to_moles(X_formed, self.biomass) * self.biomass.N
        n_N_product = self.mass_to_moles(P_formed, self.product) * self.product.N
        
        N_in = n_N_substrate + n_N_source
        N_out = n_N_biomass + n_N_product
        
        return {
            'N_substrate': n_N_substrate,
            'N_source': n_N_source,
            'N_biomass': n_N_biomass,
            'N_product': n_N_product,
            'N_in': N_in,
            'N_out': N_out,
            'balance': N_in - N_out,
            'closure': N_out / N_in if N_in > 0 else 1.0
        }
    
    def degree_of_reduction_balance(self,
                                     dS_dt: float,
                                     dX_dt: float,
                                     dP_dt: float,
                                     OUR: float) -> Dict[str, float]:
        """
        Check electron balance using degree of reduction.
        
        Electrons from substrate = Electrons to biomass + product + O2
        
        γ_S · dS/dt = γ_X · dX/dt + γ_P · dP/dt + 4 · O2_consumed
        
        Args:
            dS_dt: Substrate consumption rate (g/L/h) - negative
            dX_dt: Biomass growth rate (g/L/h)
            dP_dt: Product formation rate (g/L/h)
            OUR: Oxygen uptake rate (mmol/L/h)
            
        Returns:
            Dict with electron flows and balance
        """
        gamma_S = self.substrate.degree_of_reduction()
        gamma_X = self.biomass.degree_of_reduction()
        gamma_P = self.product.degree_of_reduction()
        gamma_O2 = 4.0  # Electrons per O2 molecule
        
        # Convert to C-mol basis rates
        dS_Cmol = -dS_dt / self.substrate.MW * self.substrate.C  # C-mol/L/h consumed
        dX_Cmol = dX_dt / self.biomass.MW * self.biomass.C  # C-mol/L/h formed
        dP_Cmol = dP_dt / self.product.MW * self.product.C if self.product.C > 0 else 0
        O2_mol = OUR / 1000  # mol/L/h
        
        # Electron flows
        e_substrate = gamma_S * dS_Cmol  # Electrons from substrate
        e_biomass = gamma_X * dX_Cmol    # Electrons to biomass
        e_product = gamma_P * dP_Cmol    # Electrons to product
        e_O2 = gamma_O2 * O2_mol         # Electrons to O2
        
        e_in = e_substrate
        e_out = e_biomass + e_product + e_O2
        
        return {
            'gamma_substrate': gamma_S,
            'gamma_biomass': gamma_X,
            'gamma_product': gamma_P,
            'e_substrate': e_substrate,
            'e_biomass': e_biomass,
            'e_product': e_product,
            'e_O2': e_O2,
            'e_in': e_in,
            'e_out': e_out,
            'balance': e_in - e_out,
            'closure': e_out / e_in if e_in > 0 else 1.0
        }
    
    def theoretical_yield_from_stoichiometry(self) -> Dict[str, float]:
        """
        Calculate theoretical yields based on elemental stoichiometry.
        
        For carbon-limited growth:
            Y_X/S,max = (γ_S / γ_X) · (MW_X / MW_S) · (C_S / C_X)
        
        Returns:
            Dict with theoretical yields
        """
        gamma_S = self.substrate.degree_of_reduction()
        gamma_X = self.biomass.degree_of_reduction()
        gamma_P = self.product.degree_of_reduction() if self.product.C > 0 else 0
        
        # Carbon-basis yields (C-mol product / C-mol substrate)
        # From electron balance: γ_S = Y_X/S·γ_X + (1-Y_X/S)·γ_CO2
        # Since γ_CO2 = 0: Y_X/S,max = γ_S / γ_X
        Y_X_S_Cmol = gamma_S / gamma_X if gamma_X > 0 else 0
        
        # Convert to mass basis
        # Y_X/S (g/g) = Y_X/S (C-mol/C-mol) · (MW_X / C_X) / (MW_S / C_S)
        Y_X_S_mass = Y_X_S_Cmol * (self.biomass.MW / self.biomass.C) / (self.substrate.MW / self.substrate.C) if self.substrate.C > 0 else 0
        
        # Product yield (if anaerobic)
        Y_P_S_Cmol = gamma_S / gamma_P if gamma_P > 0 else 0
        Y_P_S_mass = Y_P_S_Cmol * (self.product.MW / self.product.C) / (self.substrate.MW / self.substrate.C) if self.substrate.C > 0 and self.product.C > 0 else 0
        
        return {
            'Y_X_S_Cmol': Y_X_S_Cmol,
            'Y_X_S_mass': Y_X_S_mass,
            'Y_P_S_Cmol': Y_P_S_Cmol,
            'Y_P_S_mass': Y_P_S_mass,
            'gamma_substrate': gamma_S,
            'gamma_biomass': gamma_X,
            'gamma_product': gamma_P
        }


# =============================================================================
# CONVENIENCE FUNCTIONS
# =============================================================================

def create_liquid_balance_batch(config: ReactorConfig) -> LiquidPhaseMassBalance:
    """Create liquid phase balance for standard batch operation."""
    return LiquidPhaseMassBalance(config, feed_streams=[])


def create_liquid_balance_fedbatch(config: ReactorConfig,
                                    substrate_feed_conc: float,
                                    feed_rate: float = 0.0) -> LiquidPhaseMassBalance:
    """
    Create liquid phase balance for fed-batch with substrate feeding.
    
    Args:
        config: Reactor configuration
        substrate_feed_conc: Substrate concentration in feed (g/L)
        feed_rate: Initial feed rate (L/h)
    """
    feed = FeedStream(
        name='substrate_feed',
        flow_rate=feed_rate,
        concentrations={'S': substrate_feed_conc, 'X': 0.0, 'P': 0.0},
        temperature=config.T_set
    )
    return LiquidPhaseMassBalance(config, feed_streams=[feed])


def create_gas_balance_with_ph(config: ReactorConfig,
                                dynamic: bool = False) -> GasPhaseMassBalance:
    """
    Create gas phase balance with pH-dependent CO2 solubility.
    
    Args:
        config: Reactor configuration
        dynamic: Whether to use dynamic gas phase balance
    """
    return GasPhaseMassBalance(config, dynamic_gas_phase=dynamic)


def create_elemental_balance_glucose_ecoli() -> ElementalMassBalance:
    """Create elemental balance for E. coli on glucose."""
    return ElementalMassBalance(
        substrate=COMMON_FORMULAS['glucose'],
        biomass=COMMON_FORMULAS['biomass_ecoli'],
        product=COMMON_FORMULAS['lactate'],  # Or other product
        nitrogen_source=COMMON_FORMULAS['ammonia']
    )
