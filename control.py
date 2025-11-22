"""
Control strategies for fed-batch operation.

Implements:
- DO cascade control (agitation → aeration)
- Simplified algebraic cascade
- Perfect DO control (OTR = OUR assumption)
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
            Dictionary with 'N', 'Q_gas', 'kLa', 'oxygen_limited'
        """
        pass


class SimplifiedCascade(DOControlStrategy):
    """
    Simplified DO cascade control (algebraic).
    
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
        
        # Required kLa to maintain DO setpoint
        # OTR = kLa × (C* - DO_sp) = OUR
        if C_star > self.DO_sp and OUR > 0:
            kLa_req = OUR / (C_star - self.DO_sp)
        else:
            # Cannot maintain DO (too low driving force)
            kLa_req = 1000.0  # Very high
        
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
