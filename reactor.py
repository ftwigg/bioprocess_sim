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
from dataclasses import replace

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
    
    State vector (12 ODEs):
    y = [X, S_carbon, S_nitrogen, P, DO, DCO2, V, T, N, I_DO, I_T, I_F]
    
    The last three are the integral states of the DO, temperature and feed
    controllers. Keeping them in the state vector makes the right-hand side
    a pure function of (t, y): the solver may evaluate it in any order, any
    number of times, and get the same answer.
    """
    
    STATE_NAMES = ['X', 'S_carbon', 'S_nitrogen', 'P', 'DO', 'DCO2',
                   'V', 'T', 'N', 'I_DO', 'I_T', 'I_F']
    
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
        # Work on a copy so the caller's initial_state is left as given
        self.state = replace(initial_state)
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
        
        # History of the last simulation (filled by FedBatchSimulator from
        # the accepted solver steps)
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
            self.state.N,
            self.state.I_DO,
            self.state.I_T,
            self.state.I_F
        ])
    
    def update_state(self, t: float, y: np.ndarray):
        """Update reactor state from ODE solution."""
        self.state.time = t
        self.state.X = y[0]
        self.state.S_carbon = y[1]
        self.state.S_nitrogen = y[2]
        self.state.P = y[3]
        self.state.DO = y[4]
        self.state.DCO2 = y[5]
        self.state.V = y[6]
        self.state.T = y[7]
        self.state.N = y[8]
        self.state.I_DO = y[9]
        self.state.I_T = y[10]
        self.state.I_F = y[11]
        
        # Calculate pH
        self.state.pH = self.pH_model.calculate_pH(max(0.0, self.state.P))
    
    def _evaluate(self, t: float, y: np.ndarray):
        """
        Evaluate the model at (t, y).
        
        Pure function: no state is stored or modified, so it can be called by
        the solver during integration and again afterwards to recover the
        algebraic outputs on the accepted solution.
        
        Returns:
            (dydt, outputs) where outputs is a dictionary of the algebraic
            quantities (rates, controller actions, transfer rates).
        """
        # ========== 1. UNPACK STATE ==========
        # Concentrations are floored at zero only to shield the rate laws
        # from solver round-off (values of order -1e-12). The balances below
        # cannot drive any of them negative: uptake, oxygen consumption and
        # CO2 stripping all vanish as their concentration goes to zero.
        X = max(0, y[0])
        S_carbon = max(0, y[1])
        S_nitrogen = max(0, y[2])
        P = max(0, y[3])
        DO = max(0, y[4])
        DCO2 = max(0, y[5])
        V = max(0.1, y[6])
        T = y[7]
        N = max(0, y[8])
        I_DO, I_T, I_F = y[9], y[10], y[11]
        
        # Create current state object
        current_state = ReactorState(
            time=t, X=X, S_carbon=S_carbon, S_nitrogen=S_nitrogen,
            P=P, DO=DO, DCO2=DCO2, V=V, T=T, N=N,
            I_DO=I_DO, I_T=I_T, I_F=I_F
        )
        
        # ========== 2. FEED RATE ==========
        F_in = self.feed_strategy.get_feed_rate(t, current_state)
        feed_comp = self.feed_strategy.get_composition(t)
        dI_F_dt = self.feed_strategy.integral_derivative(t, current_state)
        
        # Dilution rate
        D = F_in / V if V > 0 else 0.0
        
        # ========== 3. GROWTH KINETICS ==========
        # Multiple substrate limitation
        substrates = {
            'carbon': S_carbon,
            'nitrogen': S_nitrogen
        }
        
        rates = self.kinetics.rates(substrates, DO, X)
        mu = rates['mu']
        
        # Net biomass growth (negative during starvation: endogenous decay)
        dX_dt = rates['dX_dt']
        
        # Substrate consumption rates (negative = consumption)
        dS_carbon_rxn = rates['substrate_rates']['carbon']
        dS_nitrogen_rxn = rates['substrate_rates'].get('nitrogen', 0.0)
        
        # Product formation
        dP_dt_rxn = rates['dP_dt']
        
        # Oxygen uptake and CO2 evolution (electron and carbon balances)
        OUR = rates['OUR']
        CER = rates['CER']
        
        # ========== 4. DO CONTROL ==========
        # Update state with current OUR for control
        current_state.OUR = OUR
        
        control_result = self.do_control.calculate_operating_conditions(
            DO=DO, OUR=OUR, state=current_state
        )
        
        N_target = control_result['N']
        Q_gas = control_result['Q_gas']
        dI_DO_dt = control_result.get('dI_dt', 0.0)
        
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
        kLa = self.mass_transfer.get_kLa_O2(gas_conditions)
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
            T_jacket = self.temp_controller.get_jacket_temperature(t, T, I_T)
            dI_T_dt = self.temp_controller.integral_derivative(T, I_T)
        else:
            T_jacket = self.T_jacket_fixed
            dI_T_dt = 0.0
        
        # Heat balance
        dT_dt = self.heat_balance.temperature_derivative(
            t=t, T=T, OUR=OUR, N=N, F_in=F_in, Q_gas=Q_gas,
            V=V, T_jacket=T_jacket
        )
        
        # ========== 8. ASSEMBLE DERIVATIVE VECTOR ==========
        dydt = np.array([
            dX_dt_total,      # Biomass
            dS_carbon_dt,     # Carbon substrate
            dS_nitrogen_dt,   # Nitrogen substrate
            dP_dt,            # Product
            dDO_dt,           # Dissolved O2
            dDCO2_dt,         # Dissolved CO2
            dV_dt,            # Volume
            dT_dt,            # Temperature
            dN_dt,            # Agitation speed
            dI_DO_dt,         # DO controller integral
            dI_T_dt,          # Temperature controller integral
            dI_F_dt           # Feed controller integral
        ])
        
        outputs = {
            'mu': mu, 'OUR': OUR, 'CER': CER, 'OTR': OTR, 'CTR': CTR,
            'Q_gas': Q_gas, 'kLa': kLa, 'pH': self.pH_model.calculate_pH(P),
            'T_jacket': T_jacket, 'F_feed': F_in, 'D': D,
            'N_target': N_target,
            'oxygen_limited': bool(control_result.get('oxygen_limited', False)),
            'state': current_state, 'control_result': control_result,
        }
        
        return dydt, outputs
    
    def derivatives(self, t: float, y: np.ndarray) -> np.ndarray:
        """
        Calculate dy/dt for the fed-batch reactor.
        
        State vector:
        y = [X, S_carbon, S_nitrogen, P, DO, DCO2, V, T, N, I_DO, I_T, I_F]
        """
        return self._evaluate(t, y)[0]
    
    def outputs(self, t: float, y: np.ndarray) -> Dict:
        """Algebraic outputs (rates, controller actions) at (t, y)."""
        return self._evaluate(t, y)[1]
    
    def termination_events(self):
        """
        Terminal events for the ODE solver.
        
        Each is a continuous function of the state that crosses zero at the
        limit, so the solver can locate the crossing exactly.
        
        Returns:
            List of (description, event_function) tuples
        """
        V_max = self.config.V_working_max
        
        def volume_limit(t, y):
            return V_max - y[6]
        
        def temperature_low(t, y):
            return y[7] - 10.0
        
        def temperature_high(t, y):
            return 60.0 - y[7]
        
        events = [
            (f"Volume limit reached ({V_max:.2f} L)", volume_limit),
            ("Temperature below 10°C", temperature_low),
            ("Temperature above 60°C", temperature_high),
        ]
        for _, fn in events:
            fn.terminal = True
            fn.direction = -1
        return events
    
    def check_termination(self, t: float, y: np.ndarray) -> bool:
        """Check if the state is outside the operating limits."""
        return any(fn(t, y) <= 0 for _, fn in self.termination_events())
