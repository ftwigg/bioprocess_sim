# base_models.py
"""
Base data models and enumerations for microbial batch reactor simulation.
Defines core data structures with high cohesion.
"""

from dataclasses import dataclass, field
from typing import Dict, Optional, List
from enum import Enum
import numpy as np


class ProductionModel(Enum):
    """Types of product formation kinetics"""
    GROWTH_ASSOCIATED = "growth_associated"  # Product forms with growth
    NON_GROWTH_ASSOCIATED = "non_growth_associated"  # Product forms independently
    MIXED = "mixed"  # Combination of both


class GasComponent(Enum):
    """Common gases in fermentation"""
    O2 = "O2"
    CO2 = "CO2"
    N2 = "N2"
    CH4 = "CH4"
    H2 = "H2"


@dataclass
class GasComposition:
    """Gas phase composition (mole fractions)"""
    O2: float = 0.21  # Air composition by default
    CO2: float = 0.0003
    N2: float = 0.79
    CH4: float = 0.0
    H2: float = 0.0
    
    def __post_init__(self):
        """Validate that mole fractions sum to ~1.0"""
        total = self.O2 + self.CO2 + self.N2 + self.CH4 + self.H2
        if not (0.99 <= total <= 1.01):
            raise ValueError(f"Gas mole fractions must sum to 1.0, got {total}")
    
    def to_dict(self) -> Dict[str, float]:
        """Convert to dictionary"""
        return {
            'O2': self.O2,
            'CO2': self.CO2,
            'N2': self.N2,
            'CH4': self.CH4,
            'H2': self.H2
        }


@dataclass
class CellParameters:
    """Cellular kinetic and stoichiometric parameters (Pirt model basis)"""
    
    # Growth parameters
    mu_max: float  # Maximum specific growth rate (1/h)
    Ks: float  # Monod constant for substrate (g/L)
    
    # Yield coefficients
    Yx_s_max: float  # Maximum biomass yield on substrate (g biomass / g substrate)
    Yp_s: float = 0.0  # Product yield on substrate (g product / g substrate)
    
    # Maintenance
    ms: float = 0.0  # Maintenance coefficient (g substrate / g biomass / h)
    
    # Product formation kinetics
    production_model: ProductionModel = ProductionModel.GROWTH_ASSOCIATED
    alpha: float = 0.0  # Growth-associated product coefficient (g product / g biomass)
    beta: float = 0.0  # Non-growth-associated product coefficient (g product / g biomass / h)
    
    # Respiratory parameters
    Yx_o2: Optional[float] = None  # Biomass yield on oxygen (g biomass / g O2)
    Yp_o2: Optional[float] = None  # Product yield on oxygen (g product / g O2)
    RQ: float = 1.0  # Respiratory quotient (mol CO2 / mol O2)
    
    # Metabolite properties
    product_pKa: Optional[float] = None  # pKa if product is acidic/basic
    product_charge: int = 0  # Net charge of product at neutral pH
    
    def __post_init__(self):
        """Validate parameters"""
        if self.mu_max <= 0:
            raise ValueError("mu_max must be positive")
        if self.Ks <= 0:
            raise ValueError("Ks must be positive")
        if not (0 < self.Yx_s_max <= 1.0):
            raise ValueError("Yx_s_max must be between 0 and 1")


@dataclass
class ReactorState:
    """Current state of the batch reactor"""
    
    # Time
    time: float = 0.0  # hours
    
    # Liquid phase (concentrations in g/L unless noted)
    X: float = 0.1  # Biomass concentration
    S: float = 10.0  # Substrate concentration
    P: float = 0.0  # Product concentration
    
    # Other substrates/metabolites (extensible)
    metabolites: Dict[str, float] = field(default_factory=dict)
    
    # Volume
    V: float = 1.0  # Reactor volume (L)
    
    # Gas phase (partial pressures in atm or concentrations)
    dissolved_O2: float = 0.0  # Dissolved oxygen (mg/L or mM)
    dissolved_CO2: float = 0.0  # Dissolved CO2
    
    # Environmental
    T: float = 37.0  # Temperature (°C)
    pH: float = 7.0  # pH
    
    # Cumulative values
    Q_total: float = 0.0  # Cumulative heat generated (kJ)
    O2_consumed: float = 0.0  # Cumulative O2 consumed (mol)
    CO2_produced: float = 0.0  # Cumulative CO2 produced (mol)
    
    def to_dict(self) -> Dict[str, float]:
        """Convert state to dictionary for logging"""
        return {
            'time': self.time,
            'X': self.X,
            'S': self.S,
            'P': self.P,
            'V': self.V,
            'dissolved_O2': self.dissolved_O2,
            'dissolved_CO2': self.dissolved_CO2,
            'T': self.T,
            'pH': self.pH,
            'Q_total': self.Q_total,
            'O2_consumed': self.O2_consumed,
            'CO2_produced': self.CO2_produced,
            **self.metabolites
        }


@dataclass
class ReactorConfig:
    """Configuration for batch reactor operation"""
    
    # Reactor geometry
    V_reactor: float = 1.0  # Total reactor volume (L)
    V_working: float = 0.7  # Working volume (L)
    
    # Gas supply
    inlet_gas: GasComposition = field(default_factory=GasComposition)
    gas_flow_rate: float = 1.0  # VVM (volume/volume/min)
    
    # Mass transfer
    kLa_O2: float = 100.0  # Oxygen mass transfer coefficient (1/h)
    kLa_CO2: float = 80.0  # CO2 mass transfer coefficient (1/h)
    
    # Temperature control
    T_set: float = 37.0  # Set temperature (°C)
    cooling_capacity: float = 1000.0  # Cooling capacity (W)
    
    # Initial conditions
    X0: float = 0.1  # Initial biomass (g/L)
    S0: float = 10.0  # Initial substrate (g/L)
    P0: float = 0.0  # Initial product (g/L)
    pH0: float = 7.0  # Initial pH
    
    def __post_init__(self):
        """Validate configuration"""
        if self.V_working > self.V_reactor:
            raise ValueError("Working volume cannot exceed reactor volume")
        if self.kLa_O2 <= 0:
            raise ValueError("kLa_O2 must be positive")


@dataclass
class SimulationResults:
    """Results from a batch simulation"""
    
    time: np.ndarray  # Time points (h)
    X: np.ndarray  # Biomass trajectory (g/L)
    S: np.ndarray  # Substrate trajectory (g/L)
    P: np.ndarray  # Product trajectory (g/L)
    pH: np.ndarray  # pH trajectory
    mu: np.ndarray  # Specific growth rate trajectory (1/h)
    OUR: np.ndarray  # Oxygen uptake rate (mmol/L/h)
    CER: np.ndarray  # CO2 evolution rate (mmol/L/h)
    Q_gen: np.ndarray  # Heat generation rate (W)
    
    # Additional tracked variables
    extra_vars: Dict[str, np.ndarray] = field(default_factory=dict)
    
    def get_summary(self) -> Dict[str, float]:
        """Get summary statistics"""
        return {
            'final_biomass': self.X[-1],
            'final_substrate': self.S[-1],
            'final_product': self.P[-1],
            'max_growth_rate': np.max(self.mu),
            'avg_OUR': np.mean(self.OUR),
            'total_heat': np.trapz(self.Q_gen, self.time) * 3600 / 1000,  # kJ
            'biomass_productivity': self.X[-1] / self.time[-1],  # g/L/h
            'product_productivity': self.P[-1] / self.time[-1]  # g/L/h
        }
