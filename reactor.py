# reactor.py
"""
Main batch reactor class that integrates all components.
Provides high-level interface for simulation.
"""

import numpy as np
from typing import Dict, Optional, Callable
from dataclasses import replace

from base_models import (
    CellParameters, ReactorConfig, ReactorState, 
    ProductionModel, GasComposition
)
from kinetics import PirtKinetics, calculate_all_rates
from mass_balance import LiquidPhaseMassBalance, GasPhaseMassBalance
from thermodynamics import MetabolicHeat, HeatBalance
from chemistry import pHCalculator, BufferSystem


class BatchReactor:
    """
    Batch reactor for microbial fermentation.
    
    Integrates:
    - Cell kinetics (Pirt model)
    - Mass balances (liquid and gas phases)
    - Heat generation and removal
    - pH and chemical equilibria
    
    Main equation system:
    dX/dt = μ * X
    dS/dt = -(μ/Yx_s + ms) * X
    dP/dt = (α*μ + β) * X
    dCO2/dt = OTR - OUR
    pH = f(metabolites, buffers)
    T = f(Q_gen, Q_removed)
    """
    
    def __init__(self, 
                 cell_params: CellParameters,
                 config: ReactorConfig,
                 initial_state: Optional[ReactorState] = None):
        """
        Initialize batch reactor.
        
        Args:
            cell_params: Cellular kinetic parameters
            config: Reactor configuration
            initial_state: Initial reactor state (uses defaults if None)
        """
        self.cell_params = cell_params
        self.config = config
        
        # Initialize state
        if initial_state is None:
            self.state = ReactorState(
                X=config.X0,
                S=config.S0,
                P=config.P0,
                pH=config.pH0,
                V=config.V_working,
                T=config.T_set
            )
        else:
            self.state = initial_state
        
        # Initialize subsystems
        self.kinetics = PirtKinetics(cell_params)
        self.liquid_balance = LiquidPhaseMassBalance(config)
        self.gas_balance = GasPhaseMassBalance(config)
        self.heat_gen = MetabolicHeat('glucose')  # Can parameterize
        self.heat_balance = HeatBalance(config.V_working, config.cooling_capacity)
        self.ph_calc = pHCalculator(buffer_capacity=0.05)
        self.buffer = BufferSystem('phosphate', concentration=0.05)
        
        # History for tracking
        self.history: Dict[str, list] = {
            'time': [],
            'X': [],
            'S': [],
            'P': [],
            'pH': [],
            'mu': [],
            'OUR': [],
            'CER': [],
            'Q_gen': [],
            'T': []
        }
    
    def update_state(self, t: float, y: np.ndarray):
        """
        Update reactor state from ODE solution vector.
        
        Args:
            t: Current time (h)
            y: State vector [X, S, P, dissolved_O2, dissolved_CO2, T]
        """
        self.state.time = t
        self.state.X = max(0, y[0])  # Biomass
        self.state.S = max(0, y[1])  # Substrate
        self.state.P = max(0, y[2])  # Product
        self.state.dissolved_O2 = max(0, y[3])  # Dissolved O2
        self.state.dissolved_CO2 = max(0, y[4])  # Dissolved CO2
        self.state.T = y[5]  # Temperature
        
        # Update pH based on product concentration
        if self.state.P > 0:
            # Assume product is an organic acid (e.g., lactate)
            self.state.pH = self.ph_calc.organic_acid_pH(
                self.state.P,
                'lactate',  # Can parameterize
                self.buffer.concentration,
                self.config.pH0
            )
    
    def derivatives(self, t: float, y: np.ndarray) -> np.ndarray:
        """
        Calculate derivatives for ODE solver.
        
        State vector y = [X, S, P, DO, DCO2, T]
        
        Args:
            t: Current time (h)
            y: State vector
            
        Returns:
            Derivative vector dy/dt
        """
        # Unpack state
        X = max(0, y[0])
        S = max(0, y[1])
        P = max(0, y[2])
        DO = max(0, y[3])  # Dissolved O2 (mmol/L)
        DCO2 = max(0, y[4])  # Dissolved CO2 (mmol/L)
        T = y[5]
        
        # Calculate all kinetic rates
        rates = calculate_all_rates(self.cell_params, S, X, P)
        
        mu = rates['mu']
        dX_dt = rates['dX_dt']
        dS_dt = rates['dS_dt']
        dP_dt = rates['dP_dt']
        OUR = rates['OUR']
        CER = rates['CER']
        
        # Gas transfer
        # Assume inlet gas composition for partial pressures
        P_O2 = self.config.inlet_gas.O2 * 1.0  # atm
        P_CO2 = self.config.inlet_gas.CO2 * 1.0  # atm
        
        OTR = self.gas_balance.oxygen_mass_transfer(DO, P_O2, T)
        CTR = self.gas_balance.co2_mass_transfer(DCO2, P_CO2, T)
        
        # Dissolved gas balances
        dDO_dt = self.gas_balance.dissolved_oxygen_balance(OTR, OUR)
        dDCO2_dt = self.gas_balance.dissolved_co2_balance(CTR, CER)
        
        # Heat generation and removal
        Q_gen = self.heat_gen.heat_from_our(OUR, self.state.V)
        Q_removed = self.heat_balance.heat_removal(T, self.config.T_set, UA=100.0)
        
        # Temperature change
        dT_dt = self.heat_balance.temperature_change(Q_gen, Q_removed)
        
        # Assemble derivative vector
        dydt = np.array([
            dX_dt,      # Biomass
            dS_dt,      # Substrate
            dP_dt,      # Product
            dDO_dt,     # Dissolved O2
            dDCO2_dt,   # Dissolved CO2
            dT_dt       # Temperature
        ])
        
        # Store additional info for history
        self._store_rates(t, mu, OUR, CER, Q_gen)
        
        return dydt
    
    def _store_rates(self, t: float, mu: float, OUR: float, 
                    CER: float, Q_gen: float):
        """Store calculated rates for history tracking."""
        # This is called during ODE solving
        # Store in temporary buffer and consolidate later
        pass
    
    def get_state_vector(self) -> np.ndarray:
        """Get current state as vector for ODE solver."""
        return np.array([
            self.state.X,
            self.state.S,
            self.state.P,
            self.state.dissolved_O2,
            self.state.dissolved_CO2,
            self.state.T
        ])
    
    def record_state(self):
        """Record current state to history."""
        # Calculate current rates for recording
        rates = calculate_all_rates(
            self.cell_params, 
            self.state.S, 
            self.state.X, 
            self.state.P
        )
        
        Q_gen = self.heat_gen.heat_from_our(rates['OUR'], self.state.V)
        
        self.history['time'].append(self.state.time)
        self.history['X'].append(self.state.X)
        self.history['S'].append(self.state.S)
        self.history['P'].append(self.state.P)
        self.history['pH'].append(self.state.pH)
        self.history['mu'].append(rates['mu'])
        self.history['OUR'].append(rates['OUR'])
        self.history['CER'].append(rates['CER'])
        self.history['Q_gen'].append(Q_gen)
        self.history['T'].append(self.state.T)
    
    def get_history_arrays(self) -> Dict[str, np.ndarray]:
        """Convert history to numpy arrays."""
        return {key: np.array(val) for key, val in self.history.items()}
    
    def check_limits(self) -> Dict[str, bool]:
        """
        Check if reactor is within operational limits.
        
        Returns:
            Dictionary of limit checks
        """
        checks = {
            'substrate_depleted': self.state.S < 0.01,
            'biomass_low': self.state.X < 0.001,
            'temperature_high': self.state.T > self.config.T_set + 5.0,
            'temperature_low': self.state.T < self.config.T_set - 5.0,
            'ph_low': self.state.pH < 4.0,
            'ph_high': self.state.pH > 9.0,
            'oxygen_limited': self.state.dissolved_O2 < 0.1  # mmol/L
        }
        
        return checks
    
    def is_finished(self) -> bool:
        """
        Determine if batch is finished.
        
        Typical end conditions:
        - Substrate depleted
        - Growth stopped (mu ~ 0)
        - Time limit reached
        """
        limits = self.check_limits()
        
        # End if substrate depleted or biomass not growing
        if limits['substrate_depleted']:
            return True
        
        # Calculate current growth rate
        mu = self.kinetics.specific_growth_rate(self.state.S)
        if mu < 0.001:  # Essentially stopped
            return True
        
        return False
    
    def get_summary(self) -> Dict:
        """
        Get summary of current reactor state.
        
        Returns:
            Dictionary with key metrics
        """
        rates = calculate_all_rates(
            self.cell_params,
            self.state.S,
            self.state.X,
            self.state.P
        )
        
        # Calculate yields
        S_consumed = self.config.S0 - self.state.S
        if S_consumed > 0:
            Yx_s_actual = (self.state.X - self.config.X0) / S_consumed
            Yp_s_actual = self.state.P / S_consumed
        else:
            Yx_s_actual = 0.0
            Yp_s_actual = 0.0
        
        return {
            'time': self.state.time,
            'biomass': self.state.X,
            'substrate': self.state.S,
            'product': self.state.P,
            'pH': self.state.pH,
            'temperature': self.state.T,
            'growth_rate': rates['mu'],
            'OUR': rates['OUR'],
            'CER': rates['CER'],
            'Yx_s_actual': Yx_s_actual,
            'Yp_s_actual': Yp_s_actual,
            'Yx_s_theoretical': self.cell_params.Yx_s_max,
            'substrate_consumed': S_consumed,
            'biomass_formed': self.state.X - self.config.X0,
            'product_formed': self.state.P
        }
    
    def print_summary(self):
        """Print formatted summary of reactor state."""
        summary = self.get_summary()
        
        print("\n" + "="*60)
        print(f"Batch Reactor Status - Time: {summary['time']:.2f} h")
        print("="*60)
        print(f"\nConcentrations:")
        print(f"  Biomass (X):    {summary['biomass']:.3f} g/L")
        print(f"  Substrate (S):  {summary['substrate']:.3f} g/L")
        print(f"  Product (P):    {summary['product']:.3f} g/L")
        
        print(f"\nEnvironmental:")
        print(f"  pH:             {summary['pH']:.2f}")
        print(f"  Temperature:    {summary['temperature']:.1f} °C")
        
        print(f"\nKinetics:")
        print(f"  Growth rate (μ): {summary['growth_rate']:.4f} 1/h")
        print(f"  OUR:            {summary['OUR']:.2f} mmol/L/h")
        print(f"  CER:            {summary['CER']:.2f} mmol/L/h")
        
        print(f"\nYields:")
        print(f"  Yx/s (actual):   {summary['Yx_s_actual']:.3f} g/g")
        print(f"  Yx/s (max):      {summary['Yx_s_theoretical']:.3f} g/g")
        print(f"  Yp/s (actual):   {summary['Yp_s_actual']:.3f} g/g")
        
        print(f"\nCumulative:")
        print(f"  Substrate used:  {summary['substrate_consumed']:.2f} g/L")
        print(f"  Biomass formed:  {summary['biomass_formed']:.3f} g/L")
        print(f"  Product formed:  {summary['product_formed']:.3f} g/L")
        print("="*60 + "\n")


def create_example_reactor() -> BatchReactor:
    """
    Create an example reactor with typical E. coli parameters.
    
    Returns:
        Configured BatchReactor instance
    """
    # Typical E. coli parameters on glucose
    cell_params = CellParameters(
        mu_max=0.7,  # 1/h
        Ks=0.1,  # g/L
        Yx_s_max=0.5,  # g/g
        Yp_s=0.0,  # No product initially
        ms=0.03,  # g/g/h maintenance
        production_model=ProductionModel.GROWTH_ASSOCIATED,
        alpha=0.0,
        beta=0.0,
        RQ=1.0
    )
    
    # Reactor configuration
    config = ReactorConfig(
        V_reactor=2.0,  # L
        V_working=1.5,  # L
        kLa_O2=100.0,  # 1/h
        kLa_CO2=80.0,  # 1/h
        T_set=37.0,  # °C
        X0=0.1,  # g/L
        S0=20.0,  # g/L
        P0=0.0,  # g/L
        pH0=7.0
    )
    
    reactor = BatchReactor(cell_params, config)
    
    return reactor
