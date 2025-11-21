"""
Base data models for fed-batch bioprocess simulator.

This module defines all core data structures used throughout the simulator.
"""

from dataclasses import dataclass, field
from typing import Dict, Optional
import numpy as np


@dataclass
class CarbonSource:
    """
    Generic carbon source specification with elemental composition.
    
    Attributes:
        name: Carbon source name (e.g., 'glucose', 'glycerol', 'acetate')
        formula: Molecular formula (e.g., 'C6H12O6')
        MW: Molecular weight (g/mol)
        C: Number of carbon atoms
        H: Number of hydrogen atoms
        O: Number of oxygen atoms
        N: Number of nitrogen atoms (usually 0)
        degree_of_reduction: γ, available electrons per C-mol (calculated if None)
        delta_H_combustion: Heat of combustion (kJ/mol), for heat calculations
    """
    name: str
    formula: str
    MW: float
    C: int
    H: int
    O: int
    N: int = 0
    degree_of_reduction: Optional[float] = None
    delta_H_combustion: Optional[float] = None
    
    def __post_init__(self):
        """Calculate degree of reduction if not provided."""
        if self.degree_of_reduction is None:
            self.degree_of_reduction = self.calculate_degree_of_reduction()
    
    def calculate_degree_of_reduction(self) -> float:
        """
        Calculate degree of reduction (γ).
        
        γ = (4×C + H - 2×O - 3×N) / C
        
        Returns:
            Degree of reduction (electrons per C-mol)
        """
        if self.C == 0:
            return 0.0
        gamma = (4*self.C + self.H - 2*self.O - 3*self.N) / self.C
        return gamma
    
    def to_c_mole_basis(self) -> Dict[str, float]:
        """
        Convert composition to C-mole basis (per mole of carbon).
        
        Returns:
            Dictionary with composition per C-mol
        """
        if self.C == 0:
            raise ValueError("Cannot convert to C-mole basis: no carbon atoms")
        
        return {
            'C': 1.0,
            'H': self.H / self.C,
            'O': self.O / self.C,
            'N': self.N / self.C,
            'MW_per_Cmol': self.MW / self.C
        }


@dataclass
class BiomassComposition:
    """
    Biomass elemental composition.
    
    Standard biomass formula: CH1.8O0.5N0.2 (per C-mol)
    
    Attributes:
        name: Biomass type name
        formula: Molecular formula per C-mol
        C: Carbon atoms per C-mol (always 1.0)
        H: Hydrogen atoms per C-mol
        O: Oxygen atoms per C-mol
        N: Nitrogen atoms per C-mol
        MW_per_Cmol: Molecular weight per C-mol (g/C-mol), calculated if None
        degree_of_reduction: γ, calculated if None
    """
    name: str = 'standard_biomass'
    formula: str = 'CH1.8O0.5N0.2'
    C: float = 1.0
    H: float = 1.8
    O: float = 0.5
    N: float = 0.2
    MW_per_Cmol: Optional[float] = None
    degree_of_reduction: Optional[float] = None
    
    def __post_init__(self):
        """Calculate MW and γ if not provided."""
        if self.MW_per_Cmol is None:
            self.MW_per_Cmol = 12*self.C + 1*self.H + 16*self.O + 14*self.N
        
        if self.degree_of_reduction is None:
            self.degree_of_reduction = (4*self.C + self.H - 2*self.O - 3*self.N) / self.C


@dataclass
class SubstrateParameters:
    """
    Parameters for a single substrate (carbon, nitrogen, etc.).
    
    Attributes:
        name: Substrate name (e.g., 'carbon', 'nitrogen')
        Ks: Monod half-saturation constant (g/L)
        Ki: Inhibition constant for Haldane model (g/L), None for no inhibition
        Y_xs: Biomass yield on substrate (g biomass / g substrate)
        ms: Maintenance coefficient (g substrate / g biomass / h)
    """
    name: str
    Ks: float
    Ki: Optional[float] = None  # None means no inhibition (Monod)
    Y_xs: float = 0.5
    ms: float = 0.0


@dataclass
class GasComposition:
    """
    Gas phase composition (mole fractions).
    
    Attributes:
        O2: Oxygen mole fraction (0-1)
        CO2: Carbon dioxide mole fraction (0-1)
        N2: Nitrogen mole fraction (0-1)
    """
    O2: float = 0.21      # Air: 21% O2
    CO2: float = 0.0003   # Air: 0.03% CO2
    N2: float = 0.7897    # Air: 78.97% N2
    
    def __post_init__(self):
        """Validate mole fractions sum to 1."""
        total = self.O2 + self.CO2 + self.N2
        if not (0.99 <= total <= 1.01):
            raise ValueError(f"Mole fractions must sum to 1, got {total}")


@dataclass
class ReactorState:
    """
    Current state of the fed-batch reactor.
    
    Attributes:
        time: Current time (h)
        X: Biomass concentration (g/L)
        S_carbon: Carbon substrate concentration (g/L)
        S_nitrogen: Nitrogen substrate concentration (g/L)
        P: Product concentration (g/L)
        DO: Dissolved oxygen (mmol/L)
        DCO2: Dissolved CO2 (mmol/L)
        V: Reactor volume (L)
        T: Temperature (°C)
        N: Agitation speed (rpm)
        pH: pH value
        OUR: Oxygen uptake rate (mmol/L/h), calculated
    """
    time: float = 0.0
    X: float = 0.0
    S_carbon: float = 0.0
    S_nitrogen: float = 0.0
    P: float = 0.0
    DO: float = 0.0
    DCO2: float = 0.0
    V: float = 1.0
    T: float = 37.0
    N: float = 300.0
    pH: float = 7.0
    OUR: float = 0.0  # Calculated during simulation


@dataclass
class FeedComposition:
    """
    Composition of feed stream.
    
    Attributes:
        S_carbon: Carbon substrate concentration in feed (g/L)
        S_nitrogen: Nitrogen substrate concentration in feed (g/L)
        temperature: Feed temperature (°C)
        pH: Feed pH
    """
    S_carbon: float = 0.0
    S_nitrogen: float = 0.0
    temperature: float = 25.0
    pH: float = 7.0


@dataclass
class GasPhaseConditions:
    """
    Current gas phase conditions for mass transfer calculations.
    
    Attributes:
        inlet_composition: Inlet gas composition
        P_total: Total pressure (atm)
        Q_gas: Gas flow rate (L/h)
        V_liquid: Liquid volume (L)
        T: Temperature (°C)
        N: Agitation speed (rpm), optional
        X: Biomass concentration (g/L), optional
    """
    inlet_composition: GasComposition
    P_total: float = 1.0
    Q_gas: float = 60.0
    V_liquid: float = 1.0
    T: float = 37.0
    N: Optional[float] = None
    X: Optional[float] = None


@dataclass
class ThermodynamicProperties:
    """
    Physical properties for heat balance calculations.
    
    Attributes:
        rho: Liquid density (kg/m³)
        Cp: Specific heat capacity (kJ/kg/°C)
        delta_H_ox: Heat of oxidation per mole O2 (kJ/mol O2), negative value
        lambda_vap: Latent heat of vaporization (kJ/kg)
    """
    rho: float = 1000.0        # Water-like
    Cp: float = 4.0            # Slightly less than pure water
    delta_H_ox: float = -460.0 # Universal for aerobic metabolism
    lambda_vap: float = 2400.0


@dataclass
class HeatTransferConfig:
    """
    Heat transfer configuration for reactor.
    
    Attributes:
        U: Overall heat transfer coefficient (W/m²/°C)
        D_tank: Tank diameter (m)
        D_impeller: Impeller diameter (m), optional
        N_p: Power number (dimensionless)
        T_feed: Feed temperature (°C)
        rho_feed: Feed density (kg/L)
        Cp_feed: Feed specific heat (kJ/kg/°C)
        k_evap: Evaporation coefficient (0 with condenser)
        RH_inlet: Inlet gas relative humidity (0-1)
    """
    U: float = 300.0
    D_tank: float = 0.3
    D_impeller: Optional[float] = None
    N_p: float = 5.0
    T_feed: float = 25.0
    rho_feed: float = 1.2
    Cp_feed: float = 3.8
    k_evap: float = 0.0
    RH_inlet: float = 0.5


@dataclass
class ReactorConfiguration:
    """
    Complete reactor configuration with predefined parameters.
    
    Attributes:
        name: Configuration name
        reactor_type: Type of reactor ('STR', 'airlift', 'bubble_column')
        V_reactor: Total reactor volume (L)
        V_working_max: Maximum working volume (L)
        D_tank: Tank diameter (m)
        D_impeller: Impeller diameter (m), optional
        N_min: Minimum agitation speed (rpm), optional
        N_max: Maximum agitation speed (rpm), optional
        Q_gas_min: Minimum aeration rate (vvm)
        Q_gas_max: Maximum aeration rate (vvm)
        k_O2: kLa correlation constant
        a: kLa exponent for agitation
        b: kLa exponent for aeration
        k_X: Biomass correction factor (L/g)
        U: Heat transfer coefficient (W/m²/°C)
        inlet_gas: Inlet gas composition
        T_setpoint: Temperature setpoint (°C)
    """
    name: str
    reactor_type: str
    V_reactor: float
    V_working_max: float
    D_tank: float
    D_impeller: Optional[float] = None
    N_min: Optional[float] = None
    N_max: Optional[float] = None
    Q_gas_min: float = 0.5
    Q_gas_max: float = 2.0
    k_O2: float = 0.00026
    a: float = 2.0
    b: float = 0.5
    k_X: float = 0.02
    U: float = 300.0
    inlet_gas: GasComposition = field(default_factory=GasComposition)
    T_setpoint: float = 37.0


@dataclass
class SimulationResults:
    """
    Complete simulation results.
    
    All arrays have the same length (number of time points).
    """
    time: np.ndarray
    X: np.ndarray
    S_carbon: np.ndarray
    S_nitrogen: np.ndarray
    P: np.ndarray
    DO: np.ndarray
    DCO2: np.ndarray
    V: np.ndarray
    T: np.ndarray
    N: np.ndarray
    mu: np.ndarray
    OUR: np.ndarray
    CER: np.ndarray
    OTR: np.ndarray
    CTR: np.ndarray
    Q_gas: np.ndarray
    kLa: np.ndarray
    pH: np.ndarray
    T_jacket: np.ndarray
    F_feed: np.ndarray
    D: np.ndarray
    
    def get_summary(self) -> str:
        """Generate summary statistics of simulation."""
        summary = []
        summary.append("=" * 60)
        summary.append("SIMULATION RESULTS SUMMARY")
        summary.append("=" * 60)
        summary.append(f"Simulation time: {self.time[0]:.2f} to {self.time[-1]:.2f} h")
        summary.append(f"Number of time points: {len(self.time)}")
        summary.append("")
        summary.append("Final values:")
        summary.append(f"  Biomass (X):        {self.X[-1]:.2f} g/L")
        summary.append(f"  Carbon substrate:   {self.S_carbon[-1]:.2f} g/L")
        summary.append(f"  Nitrogen substrate: {self.S_nitrogen[-1]:.2f} g/L")
        summary.append(f"  Product (P):        {self.P[-1]:.2f} g/L")
        summary.append(f"  Volume (V):         {self.V[-1]:.2f} L")
        summary.append(f"  Temperature (T):    {self.T[-1]:.2f} °C")
        summary.append(f"  DO:                 {self.DO[-1]:.3f} mmol/L")
        summary.append(f"  pH:                 {self.pH[-1]:.2f}")
        summary.append("")
        summary.append("Maximum values:")
        summary.append(f"  Max biomass:        {np.max(self.X):.2f} g/L")
        summary.append(f"  Max growth rate:    {np.max(self.mu):.3f} 1/h")
        summary.append(f"  Max OUR:            {np.max(self.OUR):.2f} mmol/L/h")
        summary.append(f"  Max agitation:      {np.max(self.N):.0f} rpm")
        summary.append("=" * 60)
        
        return "\n".join(summary)


# Predefined carbon sources
GLUCOSE = CarbonSource(
    name='glucose',
    formula='C6H12O6',
    MW=180.0,
    C=6, H=12, O=6,
    delta_H_combustion=-2803.0
)

GLYCEROL = CarbonSource(
    name='glycerol',
    formula='C3H8O3',
    MW=92.09,
    C=3, H=8, O=3,
    delta_H_combustion=-1654.0
)

ACETATE = CarbonSource(
    name='acetate',
    formula='C2H4O2',
    MW=60.05,
    C=2, H=4, O=2,
    delta_H_combustion=-875.0
)

METHANOL = CarbonSource(
    name='methanol',
    formula='CH4O',
    MW=32.04,
    C=1, H=4, O=1,
    delta_H_combustion=-726.0
)

# Predefined biomass compositions
STANDARD_BIOMASS = BiomassComposition(
    name='standard_biomass',
    formula='CH1.8O0.5N0.2'
)

ECOLI_BIOMASS = BiomassComposition(
    name='E. coli',
    formula='CH1.77O0.49N0.24',
    H=1.77, O=0.49, N=0.24
)

YEAST_BIOMASS = BiomassComposition(
    name='S. cerevisiae',
    formula='CH1.83O0.55N0.17',
    H=1.83, O=0.55, N=0.17
)

# Predefined reactor configurations
LAB_STR_5L = ReactorConfiguration(
    name='Lab STR 5L',
    reactor_type='STR',
    V_reactor=5.0,
    V_working_max=3.5,
    D_tank=0.15,
    D_impeller=0.05,
    N_min=200,
    N_max=1000,
    Q_gas_min=0.5,
    Q_gas_max=2.0,
    k_O2=0.00026,
    a=2.0,
    b=0.5,
    k_X=0.02,
    U=300.0
)

PILOT_STR_100L = ReactorConfiguration(
    name='Pilot STR 100L',
    reactor_type='STR',
    V_reactor=100.0,
    V_working_max=70.0,
    D_tank=0.30,
    D_impeller=0.10,
    N_min=150,
    N_max=600,
    Q_gas_min=0.5,
    Q_gas_max=1.5,
    k_O2=0.00026,
    a=2.0,
    b=0.5,
    k_X=0.02,
    U=300.0
)

PRODUCTION_STR_10000L = ReactorConfiguration(
    name='Production STR 10000L',
    reactor_type='STR',
    V_reactor=10000.0,
    V_working_max=7000.0,
    D_tank=1.5,
    D_impeller=0.5,
    N_min=60,
    N_max=200,
    Q_gas_min=0.3,
    Q_gas_max=1.0,
    k_O2=0.00026,
    a=2.0,
    b=0.5,
    k_X=0.02,
    U=250.0
)
