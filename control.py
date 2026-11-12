"""
Control strategies for fed-batch operation.

Implements:
- PI DO cascade with real dynamics (agitation → aeration, split range)
- Fixed agitation and aeration (DO left to the feed, or uncontrolled)
- Idealised algebraic cascade (DO never leaves the setpoint)
- Perfect DO control (OTR = OUR assumption)

Controllers are stateless: any integral they need is carried in the ODE
state vector (ReactorState.I_DO) and they return its derivative.
"""

from abc import ABC, abstractmethod
from typing import Dict
import numpy as np

from .base_models import ReactorState
from .mass_transfer import DynamicKLa


class DOControlStrategy(ABC):
    """Base class for DO control strategies."""
    
    @abstractmethod
    def calculate_operating_conditions(self, DO: float, OUR: float, 
                                       state: ReactorState) -> Dict:
        """
        Calculate N and Q_gas to maintain DO.
        
        Returns:
            Dictionary with 'N', 'Q_gas', 'kLa', 'oxygen_limited' and,
            for controllers with integral action, 'dI_dt' (the derivative
            of ReactorState.I_DO).
        """
        pass


def _saturation_air(T: float) -> float:
    """Oxygen saturation with air at 1 atm (mmol/L)."""
    from .mass_transfer import HenryConstant
    return HenryConstant.get_H_O2(T) * 0.21 * 1000.0


class PICascade(DOControlStrategy):
    """
    PI DO cascade with split-range actuation: agitation first, then aeration.
    
    The controller output u (0-1) is
    
        u = K_c × (e + I/τ_i),   e = (DO_setpoint − DO) / DO_saturation
    
    and is mapped onto the actuators in sequence:
    
        0 ≤ u ≤ u_split : N from N_min to N_max, Q_gas at minimum
        u_split < u ≤ 1 : N at N_max, Q_gas from minimum to maximum
    
    Unlike SimplifiedCascade this controller reacts with finite gain and
    integral time, so DO overshoots when demand changes quickly: it spikes
    when the substrate runs out and sags when a feed starts, and it crashes
    once both actuators are saturated.
    
    The integral I is an ODE state (ReactorState.I_DO); anti-windup is by
    back-calculation, which keeps the equations continuous for the solver.
    """
    
    def __init__(self,
                 DO_setpoint: float,
                 N_min: float,
                 N_max: float,
                 Q_gas_min: float,
                 Q_gas_max: float,
                 kLa_correlation: DynamicKLa,
                 K_c: float = 1.0,
                 tau_i: float = 0.1,
                 tau_t: float = None,
                 u_split: float = 0.6):
        """
        Args:
            DO_setpoint: Target DO (mmol/L)
            N_min: Minimum agitation speed (rpm)
            N_max: Maximum agitation speed (rpm)
            Q_gas_min: Minimum gas flow rate (L/h)
            Q_gas_max: Maximum gas flow rate (L/h)
            kLa_correlation: Dynamic kLa model (used to report kLa)
            K_c: Controller gain (output span per unit of DO saturation)
            tau_i: Integral time (h); 0.1 h = 6 min
            tau_t: Anti-windup tracking time (h), defaults to tau_i
            u_split: Fraction of the output span assigned to agitation
        """
        self.DO_sp = DO_setpoint
        self.N_min = N_min
        self.N_max = N_max
        self.Q_min = Q_gas_min
        self.Q_max = Q_gas_max
        self.kLa_model = kLa_correlation
        self.K_c = K_c
        self.tau_i = tau_i
        self.tau_t = tau_i if tau_t is None else tau_t
        self.u_split = u_split
    
    def calculate_operating_conditions(self, DO: float, OUR: float,
                                       state: ReactorState) -> Dict:
        """Calculate N, Q_gas and the integral derivative."""
        C_sat = _saturation_air(state.T)
        error = (self.DO_sp - DO) / C_sat          # positive when DO is low
        
        u_raw = self.K_c * (error + state.I_DO / self.tau_i)
        u = max(0.0, min(u_raw, 1.0))
        
        if u <= self.u_split:
            N = self.N_min + (self.N_max - self.N_min) * u / self.u_split
            Q_gas = self.Q_min
        else:
            N = self.N_max
            Q_gas = self.Q_min + (self.Q_max - self.Q_min) * (u - self.u_split) / (1.0 - self.u_split)
        
        # Back-calculation anti-windup: bleed the integral while saturated
        dI_dt = error + (u - u_raw) * self.tau_i / (self.K_c * self.tau_t)
        
        vvm = Q_gas / state.V / 60.0
        kLa = (self.kLa_model.k_O2 * (N ** self.kLa_model.a) * (vvm ** self.kLa_model.b)
               * np.exp(-self.kLa_model.k_X * state.X))
        
        return {
            'N': N,
            'Q_gas': Q_gas,
            'kLa': kLa,
            'oxygen_limited': u >= 1.0,
            'dI_dt': dI_dt,
            'u': u,
        }


class FixedAerationAgitation(DOControlStrategy):
    """
    Agitation and aeration held at fixed values.
    
    Use this when DO is to be controlled by the feed alone (DO-stat), which
    avoids two controllers fighting over the same measurement, or to see the
    uncontrolled DO profile of a batch.
    """
    
    def __init__(self, N: float, Q_gas: float, kLa_correlation: DynamicKLa = None):
        """
        Args:
            N: Agitation speed (rpm)
            Q_gas: Gas flow rate (L/h)
            kLa_correlation: Dynamic kLa model (optional, used to report kLa)
        """
        self.N = N
        self.Q_gas = Q_gas
        self.kLa_model = kLa_correlation
    
    def calculate_operating_conditions(self, DO: float, OUR: float,
                                       state: ReactorState) -> Dict:
        kLa = None
        if self.kLa_model is not None:
            vvm = self.Q_gas / state.V / 60.0
            kLa = (self.kLa_model.k_O2 * (self.N ** self.kLa_model.a) * (vvm ** self.kLa_model.b)
                   * np.exp(-self.kLa_model.k_X * state.X))
        return {'N': self.N, 'Q_gas': self.Q_gas, 'kLa': kLa, 'oxygen_limited': False}


class SimplifiedCascade(DOControlStrategy):
    """
    Idealised DO cascade control (algebraic).
    
    This inverts the kLa correlation at every instant, so DO sits exactly on
    the setpoint whenever the actuators are within range. It is useful as a
    best-case reference, but it cannot show the DO spike at substrate
    exhaustion or any other controller transient. Use PICascade for that.
    
    Control logic:
    1. Calculate required kLa to maintain DO setpoint
    2. Try to achieve with agitation (N) at minimum aeration
    3. If N hits max, increase aeration (Q_gas)
    4. If both hit max, oxygen limitation occurs
    """
    
    def __init__(self,
                 DO_setpoint: float,
                 N_min: float,
                 N_max: float,
                 Q_gas_min: float,
                 Q_gas_max: float,
                 kLa_correlation: DynamicKLa):
        """
        Args:
            DO_setpoint: Target DO (mmol/L)
            N_min: Minimum agitation speed (rpm)
            N_max: Maximum agitation speed (rpm)
            Q_gas_min: Minimum gas flow rate (L/h)
            Q_gas_max: Maximum gas flow rate (L/h)
            kLa_correlation: Dynamic kLa model
        """
        self.DO_sp = DO_setpoint
        self.N_min = N_min
        self.N_max = N_max
        self.Q_min = Q_gas_min
        self.Q_max = Q_gas_max
        self.kLa_model = kLa_correlation
    
    def calculate_operating_conditions(self, DO: float, OUR: float,
                                       state: ReactorState) -> Dict:
        """Calculate N and Q_gas to maintain DO setpoint."""
        
        # Saturation concentration (assuming air at 1 atm)
        P_O2 = 0.21 * 1.0  # atm
        from .mass_transfer import HenryConstant
        H_O2 = HenryConstant.get_H_O2(state.T)
        C_star = H_O2 * P_O2 * 1000  # mmol/L
        
        # Required kLa based on CURRENT DO (feedback control)
        # At steady state: OTR = kLa × (C* - DO_current) = OUR
        # This gives the kLa needed to balance current OUR at current DO
        if C_star > DO and OUR > 0:
            kLa_base = OUR / (C_star - DO)
        else:
            # Cannot maintain DO (too low driving force or no consumption)
            kLa_base = 100.0  # High value
        
        # Add proportional feedback to drive DO toward setpoint
        # If DO < setpoint, increase kLa; if DO > setpoint, decrease kLa
        error = self.DO_sp - DO  # Positive when DO is low
        K_p = 20.0  # Proportional gain (tuning parameter)
        
        # Feedback correction (percentage adjustment)
        feedback_correction = 1.0 + K_p * error / max(self.DO_sp, 0.01)
        feedback_correction = max(0.5, min(feedback_correction, 2.0))  # Limit to ±100%
        
        kLa_req = kLa_base * feedback_correction
        
        # Extract correlation parameters
        k = self.kLa_model.k_O2
        a = self.kLa_model.a
        b = self.kLa_model.b
        k_X = self.kLa_model.k_X
        X = state.X
        V = state.V
        
        # Biomass correction factor
        biomass_factor = np.exp(-k_X * X)
        
        # Try minimum aeration first
        vvm_min = self.Q_min / V / 60.0  # Convert to vvm (1/min)
        
        # Solve for N: kLa_req = k × N^a × vvm_min^b × biomass_factor
        if vvm_min > 0 and biomass_factor > 0:
            N_req = ((kLa_req / (k * (vvm_min ** b) * biomass_factor)) ** (1/a))
        else:
            N_req = self.N_max
        
        if N_req <= self.N_max:
            # Can control with agitation alone
            N = max(self.N_min, min(N_req, self.N_max))
            Q_gas = self.Q_min
        else:
            # Need to increase aeration
            N = self.N_max
            
            # Solve for vvm: kLa_req = k × N_max^a × vvm^b × biomass_factor
            if N > 0 and biomass_factor > 0:
                vvm_req = ((kLa_req / (k * (N ** a) * biomass_factor)) ** (1/b))
                Q_gas_req = vvm_req * V * 60.0  # Convert to L/h
                Q_gas = max(self.Q_min, min(Q_gas_req, self.Q_max))
            else:
                Q_gas = self.Q_max
        
        # Calculate actual kLa achieved
        vvm_actual = Q_gas / V / 60.0
        kLa_actual = k * (N ** a) * (vvm_actual ** b) * biomass_factor
        
        # Check if oxygen limited
        oxygen_limited = (Q_gas >= self.Q_max * 0.99 and N >= self.N_max * 0.99)
        
        return {
            'N': N,
            'Q_gas': Q_gas,
            'kLa': kLa_actual,
            'oxygen_limited': oxygen_limited
        }


class PerfectDOControl(DOControlStrategy):
    """
    Perfect DO control: Assume OTR = OUR.
    
    DO is maintained at setpoint without tracking N or Q_gas.
    Useful for comparison or when oxygen transfer is not limiting.
    """
    
    def __init__(self, DO_setpoint: float):
        """
        Args:
            DO_setpoint: Target DO (mmol/L)
        """
        self.DO_sp = DO_setpoint
    
    def calculate_operating_conditions(self, DO: float, OUR: float,
                                       state: ReactorState) -> Dict:
        """Return perfect control (OTR = OUR)."""
        
        return {
            'N': 300.0,  # Nominal value
            'Q_gas': 60.0,  # Nominal value
            'kLa': None,  # Not calculated
            'OTR': OUR,  # Perfect match
            'oxygen_limited': False
        }


# Example usage and testing
if __name__ == "__main__":
    from .base_models import GasComposition, GasPhaseConditions
    
    print("=" * 60)
    print("DO CASCADE CONTROL VALIDATION")
    print("=" * 60)
    
    # Create dynamic kLa model
    kLa_model = DynamicKLa(k_O2=0.00026, a=2.0, b=0.5, k_X=0.02)
    
    # Create DO cascade controller
    do_control = SimplifiedCascade(
        DO_setpoint=0.06,  # mmol/L (~30% saturation)
        N_min=200,
        N_max=1000,
        Q_gas_min=30,  # L/h (for 2L reactor, ~0.5 vvm)
        Q_gas_max=240,  # L/h (~2 vvm)
        kLa_correlation=kLa_model
    )
    
    # Test different OUR scenarios
    print(f"\nDO setpoint: {do_control.DO_sp} mmol/L")
    print(f"Agitation limits: {do_control.N_min}-{do_control.N_max} rpm")
    print(f"Aeration limits: {do_control.Q_min}-{do_control.Q_max} L/h")
    
    test_cases = [
        (0, 5.0, 2.0, "Low OUR, low biomass"),
        (0, 20.0, 2.0, "Medium OUR, low biomass"),
        (0, 50.0, 2.0, "High OUR, low biomass"),
        (0, 100.0, 2.0, "Very high OUR, low biomass"),
        (0, 50.0, 20.0, "High OUR, high biomass"),
        (0, 150.0, 20.0, "Very high OUR, high biomass (O2 limited)"),
    ]
    
    print("\n" + "=" * 60)
    print("Test scenarios:")
    print("=" * 60)
    
    for t, OUR, X, description in test_cases:
        state = ReactorState(
            time=t,
            X=X,
            V=2.0,
            T=37.0,
            DO=0.06
        )
        
        result = do_control.calculate_operating_conditions(
            DO=state.DO,
            OUR=OUR,
            state=state
        )
        
        print(f"\n{description}")
        print(f"  OUR = {OUR:.1f} mmol/L/h, X = {X:.1f} g/L")
        print(f"  → N = {result['N']:.0f} rpm, Q_gas = {result['Q_gas']:.1f} L/h")
        print(f"  → kLa = {result['kLa']:.1f} 1/h")
        if result['oxygen_limited']:
            print(f"  ⚠️  OXYGEN LIMITED!")
    
    print("\n" + "=" * 60)
