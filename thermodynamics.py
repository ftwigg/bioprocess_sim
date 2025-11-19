# thermodynamics.py
"""
Thermodynamic calculations for batch reactor.
Heat generation from metabolism, heat of combustion, and energy balances.
"""

import numpy as np
from typing import Dict, Optional


class MetabolicHeat:
    """
    Calculate heat generation from microbial metabolism.
    
    Heat is generated from:
    1. Substrate oxidation
    2. Biomass formation
    3. Product formation
    4. Maintenance energy
    
    Methods:
    - Direct calorimetry (measure heat)
    - Indirect calorimetry (from OUR and RQ)
    - Stoichiometric calculations (heat of combustion)
    """
    
    # Standard heats of combustion (kJ/g)
    # These are -ΔH_c values (positive = exothermic)
    HEAT_OF_COMBUSTION = {
        'glucose': 15.6,  # kJ/g
        'biomass': 20.0,  # kJ/g (typical for CH1.8O0.5N0.2)
        'ethanol': 29.7,  # kJ/g
        'acetate': 14.5,  # kJ/g
        'lactate': 15.0,  # kJ/g
        'glycerol': 18.0,  # kJ/g
    }
    
    # Oxycaloric equivalents (kJ/mol O2)
    # Heat released per mole of O2 consumed
    OXYCALORIC_EQUIVALENT = {
        'glucose': 468,  # kJ/mol O2 for complete oxidation
        'carbohydrate': 468,  # kJ/mol O2
        'lipid': 440,  # kJ/mol O2
        'protein': 430,  # kJ/mol O2
    }
    
    def __init__(self, substrate_type: str = 'glucose'):
        """
        Initialize with substrate type.
        
        Args:
            substrate_type: Type of substrate for heat calculations
        """
        self.substrate_type = substrate_type
        self.heat_combustion_substrate = self.HEAT_OF_COMBUSTION.get(substrate_type, 15.6)
        self.oxycaloric_equiv = self.OXYCALORIC_EQUIVALENT.get(substrate_type, 468)
    
    def heat_from_our(self, OUR: float, V: float) -> float:
        """
        Calculate heat generation from oxygen uptake rate (indirect calorimetry).
        
        Q = OUR * V * ΔH_ox
        
        Args:
            OUR: Oxygen uptake rate (mmol O2/L/h)
            V: Reactor volume (L)
            
        Returns:
            Heat generation rate (W = J/s)
        """
        # Convert OUR from mmol/L/h to mol/h
        OUR_mol_per_h = (OUR / 1000) * V
        
        # Heat generation (kJ/h)
        Q_kJ_per_h = OUR_mol_per_h * self.oxycaloric_equiv
        
        # Convert to W (J/s)
        Q_W = (Q_kJ_per_h * 1000) / 3600
        
        return Q_W
    
    def heat_from_substrate(self, qs: float, X: float, V: float) -> float:
        """
        Calculate heat generation from substrate consumption.
        
        Q = qs * X * V * (-ΔH_c)
        
        This assumes all consumed substrate energy not going to biomass
        is released as heat.
        
        Args:
            qs: Specific substrate consumption rate (g substrate/g biomass/h)
            X: Biomass concentration (g/L)
            V: Reactor volume (L)
            
        Returns:
            Heat generation rate (W)
        """
        # Substrate consumption rate (g/h)
        dS_dt = qs * X * V
        
        # Heat generation (kJ/h)
        Q_kJ_per_h = dS_dt * self.heat_combustion_substrate
        
        # Subtract energy that went into biomass
        # This is approximate - assumes Yx_s accounts for energetics
        # For more accurate: Q = (1 - Yx_s * ΔH_biomass/ΔH_substrate) * dS_dt * ΔH_substrate
        
        # Convert to W
        Q_W = (Q_kJ_per_h * 1000) / 3600
        
        return Q_W
    
    def heat_from_growth(self, mu: float, X: float, V: float, 
                        Yx_s: float, Yh_s: Optional[float] = None) -> float:
        """
        Calculate heat generation from growth (stoichiometric method).
        
        Q = (1 - Yh_s) * (1/Yx_s) * μ * X * V * ΔH_substrate
        
        where:
            Yh_s = energy yield = heat generated / substrate consumed
            Yx_s = biomass yield on substrate
        
        Args:
            mu: Specific growth rate (1/h)
            X: Biomass concentration (g/L)
            V: Reactor volume (L)
            Yx_s: Biomass yield on substrate
            Yh_s: Energy yield (optional)
            
        Returns:
            Heat generation rate (W)
        """
        # Growth rate (g/h)
        dX_dt = mu * X * V
        
        # Substrate consumed for growth (g/h)
        if Yx_s > 0:
            dS_dt = dX_dt / Yx_s
        else:
            return 0.0
        
        # Heat generation
        # Assume ~40% of substrate energy goes to biomass, 60% to heat
        if Yh_s is None:
            Yh_s = 0.6  # Typical value
        
        Q_kJ_per_h = Yh_s * dS_dt * self.heat_combustion_substrate
        
        # Convert to W
        Q_W = (Q_kJ_per_h * 1000) / 3600
        
        return Q_W
    
    def maintenance_heat(self, ms: float, X: float, V: float) -> float:
        """
        Calculate heat from maintenance metabolism.
        
        Q = ms * X * V * ΔH_substrate
        
        Args:
            ms: Maintenance coefficient (g substrate/g biomass/h)
            X: Biomass concentration (g/L)
            V: Reactor volume (L)
            
        Returns:
            Heat generation rate (W)
        """
        # Maintenance substrate consumption (g/h)
        dS_maint = ms * X * V
        
        # Heat from maintenance (all goes to heat)
        Q_kJ_per_h = dS_maint * self.heat_combustion_substrate
        
        # Convert to W
        Q_W = (Q_kJ_per_h * 1000) / 3600
        
        return Q_W
    
    def total_heat_generation(self, OUR: float, V: float, 
                            mu: Optional[float] = None,
                            X: Optional[float] = None,
                            ms: Optional[float] = None,
                            Yx_s: Optional[float] = None) -> float:
        """
        Calculate total heat generation rate.
        
        Primary method: Use OUR (most reliable)
        Backup method: Sum growth + maintenance
        
        Args:
            OUR: Oxygen uptake rate (mmol O2/L/h)
            V: Reactor volume (L)
            mu: Specific growth rate (1/h) - optional
            X: Biomass concentration (g/L) - optional
            ms: Maintenance coefficient - optional
            Yx_s: Biomass yield - optional
            
        Returns:
            Total heat generation rate (W)
        """
        # Primary: Use OUR
        Q_our = self.heat_from_our(OUR, V)
        
        # Verify with growth + maintenance if data available
        if all(x is not None for x in [mu, X, ms, Yx_s]):
            Q_growth = self.heat_from_growth(mu, X, V, Yx_s)
            Q_maint = self.maintenance_heat(ms, X, V)
            Q_stoich = Q_growth + Q_maint
            
            # Use OUR-based as primary, but could compare for validation
            return Q_our
        
        return Q_our


class HeatBalance:
    """
    Overall heat balance for the reactor.
    
    Heat generated - Heat removed - Heat accumulated = 0
    
    dT/dt = (Q_gen - Q_removed) / (m * Cp)
    
    where:
        Q_gen = metabolic heat generation
        Q_removed = cooling (jacket, coils) + losses
        m = total mass in reactor
        Cp = specific heat capacity
    """
    
    # Physical properties
    CP_WATER = 4.18  # kJ/(kg·K) = kJ/(L·K) for water
    RHO_WATER = 1.0  # kg/L
    
    def __init__(self, V_reactor: float, cooling_capacity: float = 1000.0):
        """
        Initialize heat balance.
        
        Args:
            V_reactor: Reactor volume (L)
            cooling_capacity: Maximum cooling capacity (W)
        """
        self.V_reactor = V_reactor
        self.cooling_capacity = cooling_capacity
        
        # Assume liquid properties similar to water
        self.mass = V_reactor * self.RHO_WATER  # kg
        self.Cp = self.CP_WATER  # kJ/(kg·K)
    
    def heat_removal(self, T: float, T_jacket: float, UA: float = 100.0) -> float:
        """
        Calculate heat removal rate.
        
        Q_removed = UA * (T - T_jacket)
        
        Args:
            T: Reactor temperature (°C)
            T_jacket: Cooling jacket temperature (°C)
            UA: Overall heat transfer coefficient × area (W/K)
            
        Returns:
            Heat removal rate (W)
        """
        Q_removed = UA * (T - T_jacket)
        
        # Cannot remove more than cooling capacity
        Q_removed = min(Q_removed, self.cooling_capacity)
        
        return Q_removed
    
    def temperature_change(self, Q_gen: float, Q_removed: float) -> float:
        """
        Calculate rate of temperature change.
        
        dT/dt = (Q_gen - Q_removed) / (m * Cp)
        
        Args:
            Q_gen: Heat generation rate (W)
            Q_removed: Heat removal rate (W)
            
        Returns:
            Rate of temperature change (°C/h)
        """
        # Convert W to kJ/h: W * 3.6
        Q_net_kJ_per_h = (Q_gen - Q_removed) * 3.6
        
        # Heat capacity (kJ/K)
        total_Cp = self.mass * self.Cp
        
        # Temperature change (K/h = °C/h)
        dT_dt = Q_net_kJ_per_h / total_Cp
        
        return dT_dt
    
    def required_cooling(self, Q_gen: float, T: float, T_set: float) -> float:
        """
        Calculate required cooling to maintain temperature.
        
        Args:
            Q_gen: Heat generation rate (W)
            T: Current temperature (°C)
            T_set: Set point temperature (°C)
            
        Returns:
            Required cooling rate (W)
        """
        # Simple proportional control
        # Q_removed = Q_gen + K * (T - T_set)
        K = 100.0  # Proportional gain (W/K)
        
        Q_required = Q_gen + K * (T - T_set)
        
        # Limit to cooling capacity
        Q_required = min(Q_required, self.cooling_capacity)
        Q_required = max(Q_required, 0.0)
        
        return Q_required


class EnthalpyBalance:
    """
    Enthalpy balance for verification and advanced calculations.
    
    Useful for:
    - Verifying energy conservation
    - Calculating theoretical maximum yields
    - Understanding energy distribution
    """
    
    def __init__(self, heat_combustion: Dict[str, float]):
        """
        Args:
            heat_combustion: Dictionary of heats of combustion (kJ/g)
                           for substrate, biomass, product
        """
        self.heat_combustion = heat_combustion
    
    def energy_balance(self, S_consumed: float, X_formed: float, 
                      P_formed: float, Q_measured: float) -> float:
        """
        Check energy balance closure.
        
        ΔH_substrate = ΔH_biomass + ΔH_product + Q_heat
        
        Args:
            S_consumed: Substrate consumed (g)
            X_formed: Biomass formed (g)
            P_formed: Product formed (g)
            Q_measured: Measured heat generation (kJ)
            
        Returns:
            Energy balance closure (kJ)
        """
        H_substrate = S_consumed * self.heat_combustion.get('substrate', 15.6)
        H_biomass = X_formed * self.heat_combustion.get('biomass', 20.0)
        H_product = P_formed * self.heat_combustion.get('product', 0.0)
        
        H_in = H_substrate
        H_out = H_biomass + H_product + Q_measured
        
        return H_in - H_out
    
    def theoretical_yield(self, substrate: str = 'glucose', 
                         product: str = 'biomass') -> float:
        """
        Calculate theoretical maximum yield based on energetics.
        
        Args:
            substrate: Substrate type
            product: Product type
            
        Returns:
            Theoretical maximum yield (g product / g substrate)
        """
        # Simplified: assumes all energy goes to product
        # Real yields are lower due to entropy and inefficiencies
        
        H_substrate = self.heat_combustion.get(substrate, 15.6)
        H_product = self.heat_combustion.get(product, 20.0)
        
        # Maximum yield if all energy conserved
        Y_max = H_substrate / H_product
        
        # Apply thermodynamic efficiency (~60-70% typical)
        Y_theoretical = Y_max * 0.65
        
        return Y_theoretical
