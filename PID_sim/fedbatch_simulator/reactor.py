"""
Fed-batch bioreactor simulator - main reactor class.

Integrates all subsystems:
- Growth kinetics
- Mass balances
- Mass transfer
- Heat balance
- Feed control
- DO cascade control
- pH modeling
"""

import numpy as np
from typing import Optional, Dict

from .base_models import (
    ReactorConfiguration,
    ReactorState,
    GasComposition,
    GasPhaseConditions,
    ThermodynamicProperties,
    HeatTransferConfig,
)
from .kinetics import GrowthKinetics, CellParameters
from .mass_transfer import MassTransferModel, DynamicKLa
from .thermodynamics import HeatBalance, TemperatureController
from .feed_strategies import FeedStrategy
from .control import DOControlStrategy
from .chemistry import pHModel, SimplifiedpHModel


class FedBatchReactor:
    """
    Fed-batch bioreactor simulator.
    
    Integrates:
    - Multiple substrate kinetics (carbon, nitrogen, oxygen)
    - Fed-batch mass balances with dilution
    - Dynamic kLa and DO cascade control
    - Heat generation and temperature control
    - pH modeling (simplified)
    
    State vector (9 ODEs):
    y = [X, S_carbon, S_nitrogen, P, DO, DCO2, V, T, N]
    """
    
    def __init__(self,
                 cell_params: CellParameters,
                 reactor_config: ReactorConfiguration,
                 initial_state: ReactorState,
                 feed_strategy: FeedStrategy,
                 do_control: DOControlStrategy,
                 temp_controller: Optional[TemperatureController] = None,
                 mass_transfer_model: Optional[MassTransferModel] = None,
                 pH_model: Optional[pHModel] = None):
        """
        Initialize fed-batch reactor.
        
        Args:
            cell_params: Cellular kinetic parameters
            reactor_config: Reactor configuration
            initial_state: Initial reactor state
            feed_strategy: Feed control strategy
            do_control: DO control strategy
            temp_controller: Temperature controller (None for fixed jacket temp)
            mass_transfer_model: Mass transfer model (None for default dynamic kLa)
            pH_model: pH model (None for default simplified model)
        """
        # Store configuration
        self.cell_params = cell_params
        self.config = reactor_config
        self.state = initial_state
        self.feed_strategy = feed_strategy
        self.do_control = do_control
        self.temp_controller = temp_controller
        
        # Initialize subsystems
        self.kinetics = GrowthKinetics(cell_params)
        
        if mass_transfer_model is None:
            # Default: dynamic kLa
            self.mass_transfer = DynamicKLa(
                k_O2=reactor_config.k_O2,
                a=reactor_config.a,
                b=reactor_config.b,
                k_X=reactor_config.k_X
            )
        else:
            self.mass_transfer = mass_transfer_model
        
        self.heat_balance = HeatBalance(
            thermo_props=ThermodynamicProperties(),
            heat_config=HeatTransferConfig(
                U=reactor_config.U,
                D_tank=reactor_config.D_tank,
                D_impeller=reactor_config.D_impeller,
                N_p=5.0,
                T_feed=feed_strategy.feed_composition.temperature
            ),
            temp_controller=temp_controller
        )
        
        if pH_model is None:
            self.pH_model = SimplifiedpHModel(
                pH0=initial_state.pH,
                k_acid=0.0  # Neutral product by default
            )
        else:
            self.pH_model = pH_model
        
        # Gas phase
        self.inlet_gas = reactor_config.inlet_gas
        self.P_total = 1.0  # atm
        
        # Temperature control
        self.T_jacket_fixed = reactor_config.T_setpoint
        
        # History tracking
        self.history = {
            'time': [],
            'X': [], 'S_carbon': [], 'S_nitrogen': [], 'P': [],
            'DO': [], 'DCO2': [], 'V': [], 'T': [], 'N': [],
            'mu': [], 'OUR': [], 'CER': [], 'OTR': [], 'CTR': [],
            'Q_gas': [], 'kLa': [], 'pH': [], 'T_jacket': [],
            'F_feed': [], 'D': []
        }
    
    def get_state_vector(self) -> np.ndarray:
        """Get current state as vector for ODE solver."""
        return np.array([
            self.state.X,
            self.state.S_carbon,
            self.state.S_nitrogen,
            self.state.P,
            self.state.DO,
            self.state.DCO2,
            self.state.V,
            self.state.T,
            self.state.N
        ])
    
    def update_state(self, t: float, y: np.ndarray):
        """Update reactor state from ODE solution."""
        self.state.time = t
        self.state.X = max(0, y[0])
        self.state.S_carbon = max(0, y[1])
        self.state.S_nitrogen = max(0, y[2])
        self.state.P = max(0, y[3])
        self.state.DO = max(0, y[4])
        self.state.DCO2 = max(0, y[5])
        self.state.V = max(0, y[6])
        self.state.T = y[7]
        self.state.N = max(0, y[8])
        
        # Calculate pH
        self.state.pH = self.pH_model.calculate_pH(self.state.P)
    
    def derivatives(self, t: float, y: np.ndarray) -> np.ndarray:
        """
        Calculate dy/dt for the fed-batch reactor.
        
        State vector:
        y = [X, S_carbon, S_nitrogen, P, DO, DCO2, V, T, N]
        
        Returns:
        dydt = [dX/dt, dS_c/dt, dS_n/dt, dP/dt, dDO/dt, dDCO2/dt, dV/dt, dT/dt, dN/dt]
        """
        
        # ========== 1. UNPACK STATE ==========
        X = max(0, y[0])
        S_carbon = max(0, y[1])
        S_nitrogen = max(0, y[2])
        P = max(0, y[3])
        DO = max(0, y[4])
        DCO2 = max(0, y[5])
        V = max(0.1, y[6])
        T = y[7]
        N = max(0, y[8])
        
        # Create current state object
        current_state = ReactorState(
            time=t, X=X, S_carbon=S_carbon, S_nitrogen=S_nitrogen,
            P=P, DO=DO, DCO2=DCO2, V=V, T=T, N=N
        )
        
        # ========== 2. FEED RATE ==========
        F_in = self.feed_strategy.get_feed_rate(t, current_state)
        feed_comp = self.feed_strategy.get_composition(t)
        
        # Dilution rate
        D = F_in / V if V > 0 else 0.0
        
        # ========== 3. GROWTH KINETICS ==========
        # Multiple substrate limitation
        substrates = {
            'carbon': S_carbon,
            'nitrogen': S_nitrogen
        }
        
        mu = self.kinetics.specific_growth_rate(substrates, DO)
        
        # Biomass growth
        dX_dt = self.kinetics.biomass_growth_rate(mu, X)
        
        # Substrate consumption rates
        substrate_rates = self.kinetics.substrate_consumption_rates(mu, X)
        dS_carbon_rxn = substrate_rates['carbon']  # Negative (consumption)
        dS_nitrogen_rxn = substrate_rates['nitrogen']
        
        # Product formation
        dP_dt_rxn = self.kinetics.product_formation_rate(mu, X)
        
        # Oxygen uptake and CO2 evolution
        OUR = self.kinetics.oxygen_uptake_rate(mu, X)
        CER = self.kinetics.co2_evolution_rate(OUR)
        
        # ========== 4. DO CASCADE CONTROL ==========
        # Update state with current OUR for control
        current_state.OUR = OUR
        
        control_result = self.do_control.calculate_operating_conditions(
            DO=DO, OUR=OUR, state=current_state
        )
        
        N_target = control_result['N']
        Q_gas = control_result['Q_gas']
        
        # Agitation dynamics (first-order lag, 10 second time constant)
        tau_N = 10.0 / 3600.0  # Convert to hours
        dN_dt = (N_target - N) / tau_N
        
        # ========== 5. GAS-LIQUID MASS TRANSFER ==========
        # Gas phase conditions
        gas_conditions = GasPhaseConditions(
            inlet_composition=self.inlet_gas,
            P_total=self.P_total,
            Q_gas=Q_gas,
            V_liquid=V,
            T=T,
            N=N,
            X=X
        )
        
        # Outlet gas composition (for average partial pressure)
        outlet_gas = self.mass_transfer.outlet_gas_composition(
            OUR, CER, gas_conditions
        )
        
        # Average partial pressures
        y_O2_avg = (self.inlet_gas.O2 + outlet_gas.O2) / 2
        y_CO2_avg = (self.inlet_gas.CO2 + outlet_gas.CO2) / 2
        P_O2 = y_O2_avg * self.P_total
        P_CO2 = y_CO2_avg * self.P_total
        
        # Mass transfer rates
        OTR = self.mass_transfer.oxygen_transfer_rate(DO, P_O2, gas_conditions)
        CTR = self.mass_transfer.co2_transfer_rate(DCO2, P_CO2, gas_conditions)
        
        # ========== 6. MASS BALANCES ==========
        # Biomass (no feed)
        dX_dt_total = dX_dt - D * X
        
        # Carbon substrate
        dS_carbon_dt = D * (feed_comp.S_carbon - S_carbon) + dS_carbon_rxn
        
        # Nitrogen substrate
        dS_nitrogen_dt = D * (feed_comp.S_nitrogen - S_nitrogen) + dS_nitrogen_rxn
        
        # Product (no feed)
        dP_dt = dP_dt_rxn - D * P
        
        # Dissolved oxygen
        dDO_dt = OTR - OUR - D * DO
        
        # Dissolved CO2
        dDCO2_dt = CER - CTR - D * DCO2
        
        # Volume
        dV_dt = F_in
        
        # ========== 7. HEAT BALANCE ==========
        # Temperature control
        if self.temp_controller is not None:
            T_jacket = self.temp_controller.get_jacket_temperature(t, T)
        else:
            T_jacket = self.T_jacket_fixed
        
        # Heat balance
        dT_dt = self.heat_balance.temperature_derivative(
            t=t, T=T, OUR=OUR, N=N, F_in=F_in, Q_gas=Q_gas,
            V=V, T_jacket=T_jacket
        )
        
        # ========== 8. RECORD STEP ==========
        self._record_step(t, current_state, mu, OUR, CER, OTR, CTR,
                         Q_gas, T_jacket, control_result, F_in, D)
        
        # ========== 9. ASSEMBLE DERIVATIVE VECTOR ==========
        dydt = np.array([
            dX_dt_total,      # Biomass
            dS_carbon_dt,     # Carbon substrate
            dS_nitrogen_dt,   # Nitrogen substrate
            dP_dt,            # Product
            dDO_dt,           # Dissolved O2
            dDCO2_dt,         # Dissolved CO2
            dV_dt,            # Volume
            dT_dt,            # Temperature
            dN_dt             # Agitation speed
        ])
        
        return dydt
    
    def _record_step(self, t, state, mu, OUR, CER, OTR, CTR,
                    Q_gas, T_jacket, control_result, F_in, D):
        """Record current step to history."""
        self.history['time'].append(t)
        self.history['X'].append(state.X)
        self.history['S_carbon'].append(state.S_carbon)
        self.history['S_nitrogen'].append(state.S_nitrogen)
        self.history['P'].append(state.P)
        self.history['DO'].append(state.DO)
        self.history['DCO2'].append(state.DCO2)
        self.history['V'].append(state.V)
        self.history['T'].append(state.T)
        self.history['N'].append(state.N)
        self.history['mu'].append(mu)
        self.history['OUR'].append(OUR)
        self.history['CER'].append(CER)
        self.history['OTR'].append(OTR)
        self.history['CTR'].append(CTR)
        self.history['Q_gas'].append(Q_gas)
        self.history['kLa'].append(control_result.get('kLa', 0))
        self.history['pH'].append(self.pH_model.calculate_pH(state.P))
        self.history['T_jacket'].append(T_jacket)
        self.history['F_feed'].append(F_in)
        self.history['D'].append(D)
    
    def check_termination(self, t: float, y: np.ndarray) -> bool:
        """Check if simulation should terminate."""
        X, S_c, S_n, P, DO, DCO2, V, T, N = y
        
        # Volume limit
        if V >= self.config.V_working_max:
            print(f"Termination at t={t:.2f}h: Volume limit reached ({V:.2f} L)")
            return True
        
        # Substrate depletion
        if S_c < 0.01 and S_n < 0.01:
            print(f"Termination at t={t:.2f}h: Substrates depleted")
            return True
        
        # Temperature out of range
        if T < 10 or T > 60:
            print(f"Termination at t={t:.2f}h: Temperature out of range ({T:.1f}°C)")
            return True
        
        return False
