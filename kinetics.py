# kinetics.py
"""
Growth kinetics and substrate consumption based on Pirt model.
Does not include the Luedeking-Piret model extension for inclusion of high product formation rates.

Specific growth rate uses the Monod model for determining mu at any given [S] and assumes constant Ks and mu_max.
Commented out potential Haldane extension for inhibition models.

Product formation rate is based on 4 options: growth, non-growth, mixed, or no product (qp=0)

Oxygen consumption: this module hard-codes a maintenance-related oxygen consumption assumption based on glucose oxidation.

Handles specific growth rates, substrate utilization, and product formation.
"""

import numpy as np
from typing import Tuple, Optional
from base_models import CellParameters, ProductionModel


class PirtKinetics:
    """
    Implementation of the Pirt model for microbial growth.
    
    The Pirt model separates substrate consumption into:
    1. Growth-associated consumption
    2. Maintenance-associated consumption
    
    Key equation: dS/dt = -(1/Yx_s) * (dX/dt) - ms * X
    
    where:
        Yx_s = biomass yield on substrate (corrected for maintenance)
        ms = maintenance coefficient
    """
    
    def __init__(self, params: CellParameters):
        """
        Initialize with cell parameters.
        
        Args:
            params: CellParameters object containing kinetic constants
        """
        self.params = params
        
        # Calculate effective yield accounting for maintenance
        # At steady state: 1/Yx_s = 1/Yx_s_max + ms/mu
        self.Yx_s_max = params.Yx_s_max
        self.ms = params.ms
    
    def specific_growth_rate(self, S: float, inhibitor: Optional[float] = None) -> float:
        """
        Calculate specific growth rate using Monod kinetics.
        
        μ = μ_max * S / (Ks + S)
        
        Args:
            S: Substrate concentration (g/L)
            inhibitor: Optional inhibitor concentration for inhibition models
            
        Returns:
            Specific growth rate (1/h)
        """
        if S <= 0:
            return 0.0
        
        # Basic Monod kinetics
        mu = self.params.mu_max * S / (self.params.Ks + S)
        
        # Add substrate inhibition if present (Haldane model)
        # μ = μ_max * S / (Ks + S + S²/Ki)
        # This could be extended based on user needs
        
        return mu
    
    def effective_yield(self, mu: float) -> float:
        """
        Calculate effective biomass yield accounting for maintenance.
        
        From Pirt model:
        1/Yx_s = 1/Yx_s_max + ms/μ
        
        Therefore:
        Yx_s = Yx_s_max * μ / (μ + ms * Yx_s_max)
        
        Args:
            mu: Current specific growth rate (1/h)
            
        Returns:
            Effective biomass yield (g biomass / g substrate)
        """
        if mu <= 0:
            return 0.0
        
        # Avoid division by zero
        denominator = mu + self.ms * self.Yx_s_max
        if denominator <= 0:
            return self.Yx_s_max
        
        Yx_s = self.Yx_s_max * mu / denominator
        return Yx_s
    
    def substrate_consumption_rate(self, mu: float, X: float) -> float:
        """
        Calculate total substrate consumption rate (Pirt model).
        
        qs = qg + ms
        where:
            qg = μ / Yx_s_max (growth-associated)
            ms = maintenance coefficient
        
        Total: dS/dt = -(qs) * X = -(μ/Yx_s_max + ms) * X
        
        Args:
            mu: Specific growth rate (1/h)
            X: Biomass concentration (g/L)
            
        Returns:
            Substrate consumption rate (g/L/h)
        """
        if X <= 0:
            return 0.0
        
        # Growth-associated consumption
        q_growth = mu / self.Yx_s_max if self.Yx_s_max > 0 else 0.0
        
        # Total specific consumption rate
        qs = q_growth + self.ms
        
        return qs * X
    
    def product_formation_rate(self, mu: float, X: float) -> float:
        """
        Calculate product formation rate based on production model.
        
        Three models:
        1. Growth-associated: dP/dt = α * dX/dt = α * μ * X
        2. Non-growth-associated: dP/dt = β * X
        3. Mixed: dP/dt = α * μ * X + β * X
        
        Args:
            mu: Specific growth rate (1/h)
            X: Biomass concentration (g/L)
            
        Returns:
            Product formation rate (g/L/h)
        """
        if X <= 0:
            return 0.0
        
        model = self.params.production_model
        
        if model == ProductionModel.GROWTH_ASSOCIATED:
            # Product forms only during growth
            qp = self.params.alpha * mu * X
            
        elif model == ProductionModel.NON_GROWTH_ASSOCIATED:
            # Product forms independently of growth
            qp = self.params.beta * X
            
        elif model == ProductionModel.MIXED:
            # Combination of both
            qp = (self.params.alpha * mu + self.params.beta) * X
        
        else:
            qp = 0.0
        
        return qp
    
    def biomass_growth_rate(self, mu: float, X: float) -> float:
        """
        Calculate biomass growth rate.
        
        dX/dt = μ * X
        
        Args:
            mu: Specific growth rate (1/h)
            X: Biomass concentration (g/L)
            
        Returns:
            Biomass growth rate (g/L/h)
        """
        if X <= 0:
            return 0.0
        
        return mu * X


class SubstrateBalance:
    """
    Detailed substrate balance accounting for all consumption pathways.
    
    Total substrate consumption:
    dS/dt = -(dS/dt)_biomass - (dS/dt)_product - (dS/dt)_maintenance - (dS/dt)_other
    """
    
    def __init__(self, params: CellParameters):
        self.params = params
    
    def substrate_to_biomass(self, dX_dt: float) -> float:
        """
        Substrate consumed for biomass synthesis.
        
        Args:
            dX_dt: Biomass growth rate (g/L/h)
            
        Returns:
            Substrate consumption for biomass (g/L/h)
        """
        if self.params.Yx_s_max <= 0:
            return 0.0
        
        return dX_dt / self.params.Yx_s_max
    
    def substrate_to_product(self, dP_dt: float) -> float:
        """
        Substrate consumed for product formation.
        
        Args:
            dP_dt: Product formation rate (g/L/h)
            
        Returns:
            Substrate consumption for product (g/L/h)
        """
        if self.params.Yp_s <= 0:
            return 0.0
        
        return dP_dt / self.params.Yp_s
    
    def substrate_for_maintenance(self, X: float) -> float:
        """
        Substrate consumed for maintenance.
        
        Args:
            X: Biomass concentration (g/L)
            
        Returns:
            Substrate consumption for maintenance (g/L/h)
        """
        return self.params.ms * X
    
    def total_substrate_consumption(self, : float, dP_dt: float, X: float) -> float:
        """
        Total substrate consumption rate.
        
        Args:
            dX_dt: Biomass growth rate (g/L/h)
            dP_dt: Product formation rate (g/L/h)
            X: Biomass concentration (g/L)
            
        Returns:
            Total substrate consumption (g/L/h)
        """
        qs_biomass = self.substrate_to_biomass(dX_dt)
        qs_product = self.substrate_to_product(dP_dt)
        qs_maintenance = self.substrate_for_maintenance(X)
        
        return qs_biomass + qs_product + qs_maintenance


class RespiratoryQuotient:
    """
    Calculate oxygen consumption and CO2 production rates.
    
    Based on stoichiometry:
    - Oxygen consumed for biomass and product formation
    - CO2 produced based on respiratory quotient (RQ)
    """
    
    def __init__(self, params: CellParameters):
        self.params = params
    
    def oxygen_uptake_rate(self, dX_dt: float, dP_dt: float, X: float) -> float:
        """
        Calculate oxygen uptake rate (OUR).
        
        OUR = (dX/dt) / Yx_o2 + (dP/dt) / Yp_o2 + maintenance
        
        Args:
            dX_dt: Biomass growth rate (g/L/h)
            dP_dt: Product formation rate (g/L/h)
            X: Biomass concentration (g/L)
            
        Returns:
            Oxygen uptake rate (mmol O2/L/h)
        """
        OUR = 0.0
        
        # Biomass-associated oxygen consumption
        if self.params.Yx_o2 is not None and self.params.Yx_o2 > 0:
            OUR += dX_dt / self.params.Yx_o2
        
        # Product-associated oxygen consumption
        if self.params.Yp_o2 is not None and self.params.Yp_o2 > 0:
            OUR += dP_dt / self.params.Yp_o2
        
        # Maintenance oxygen consumption
        # Estimate from maintenance substrate and typical oxidation
        # For glucose: ~6 mol O2 / mol glucose (180 g/mol)
        # ~0.033 mol O2 / g glucose
        if self.params.ms > 0:
            maintenance_O2 = self.params.ms * X * 0.033  # Rough estimate
            OUR += maintenance_O2
        
        # Convert g O2/L/h to mmol O2/L/h (MW O2 = 32 g/mol)
        OUR_mmol = (OUR * 1000) / 32.0
        
        return OUR_mmol
    
    def co2_evolution_rate(self, OUR: float) -> float:
        """
        Calculate CO2 evolution rate (CER) from OUR and RQ.
        
        CER = RQ * OUR
        
        Args:
            OUR: Oxygen uptake rate (mmol O2/L/h)
            
        Returns:
            CO2 evolution rate (mmol CO2/L/h)
        """
        return self.params.RQ * OUR


def calculate_all_rates(params: CellParameters, S: float, X: float, P: float) -> dict:
    """
    Convenience function to calculate all kinetic rates at once.
    
    Args:
        params: Cell parameters
        S: Substrate concentration (g/L)
        X: Biomass concentration (g/L)
        P: Product concentration (g/L)
        
    Returns:
        Dictionary with all rates
    """
    kinetics = PirtKinetics(params)
    substrate_balance = SubstrateBalance(params)
    respiration = RespiratoryQuotient(params)
    
    # Calculate specific growth rate
    mu = kinetics.specific_growth_rate(S)
    
    # Calculate growth and production rates
    dX_dt = kinetics.biomass_growth_rate(mu, X)
    dP_dt = kinetics.product_formation_rate(mu, X)
    
    # Calculate substrate consumption
    dS_dt = -substrate_balance.total_substrate_consumption(dX_dt, dP_dt, X)
    
    # Calculate oxygen and CO2 rates
    OUR = respiration.oxygen_uptake_rate(dX_dt, dP_dt, X)
    CER = respiration.co2_evolution_rate(OUR)
    
    # Calculate effective yield
    Yx_s_eff = kinetics.effective_yield(mu)
    
    return {
        'mu': mu,
        'dX_dt': dX_dt,
        'dS_dt': dS_dt,
        'dP_dt': dP_dt,
        'OUR': OUR,
        'CER': CER,
        'Yx_s_eff': Yx_s_eff
    }
