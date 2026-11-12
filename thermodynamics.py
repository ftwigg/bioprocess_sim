"""
Thermodynamics and heat balance for fed-batch reactors.

Implements:
- Metabolic heat generation from OUR
- Agitation heat from power input
- Feed heat from temperature difference
- Cooling via jacket
- Temperature PID control
"""

import numpy as np
from typing import Optional

from .base_models import ThermodynamicProperties, HeatTransferConfig


class TemperatureController:
    """
    PI controller for reactor temperature via jacket temperature.
    
    The integral is carried in the ODE state vector (ReactorState.I_T) and
    passed in, so the controller itself holds no memory and gives the same
    answer however the solver orders its calls. Anti-windup is by
    back-calculation.
    """
    
    def __init__(self, 
                 T_setpoint: float,
                 K_p: float,
                 K_i: float = 0.0,
                 K_d: float = 0.0,
                 T_jacket_min: float = 4.0,
                 T_jacket_max: float = 80.0):
        """
        Args:
            T_setpoint: Target temperature (°C)
            K_p: Proportional gain (°C jacket per °C error)
            K_i: Integral gain (°C jacket per °C·h)
            K_d: Derivative gain. Accepted for compatibility but not applied:
                 derivative action on a noise-free simulated signal adds
                 nothing and would make the equations implicit.
            T_jacket_min: Minimum jacket temperature (°C)
            T_jacket_max: Maximum jacket temperature (°C)
        """
        self.T_sp = T_setpoint
        self.K_p = K_p
        self.K_i = K_i
        self.K_d = K_d
        self.T_jacket_min = T_jacket_min
        self.T_jacket_max = T_jacket_max
    
    def _unsaturated(self, T: float, integral: float) -> float:
        error = T - self.T_sp
        # Inverse action: higher T → lower T_jacket
        return self.T_sp - (self.K_p * error + self.K_i * integral)
    
    def get_jacket_temperature(self, t: float, T: float, integral: float = 0.0) -> float:
        """
        Calculate required jacket temperature.
        
        Args:
            t: Current time (h)
            T: Current reactor temperature (°C)
            integral: Integral of (T − T_setpoint) over time (°C·h)
        
        Returns:
            T_jacket (°C)
        """
        T_jacket = self._unsaturated(T, integral)
        return max(self.T_jacket_min, min(T_jacket, self.T_jacket_max))
    
    def integral_derivative(self, T: float, integral: float = 0.0) -> float:
        """
        Derivative of the integral state, with back-calculation anti-windup.
        
        Returns:
            dI/dt (°C)
        """
        if self.K_i == 0.0:
            return 0.0
        raw = self._unsaturated(T, integral)
        sat = max(self.T_jacket_min, min(raw, self.T_jacket_max))
        tau_t = abs(self.K_p / self.K_i) if self.K_p != 0.0 else 0.1
        # raw falls as the integral grows, so a positive (raw − sat) must bleed it up
        return (T - self.T_sp) + (raw - sat) / (self.K_i * tau_t)


class HeatBalance:
    """Complete heat balance for fed-batch reactor."""
    
    def __init__(self,
                 thermo_props: ThermodynamicProperties,
                 heat_config: HeatTransferConfig,
                 temp_controller: Optional[TemperatureController] = None):
        """
        Args:
            thermo_props: Thermodynamic properties
            heat_config: Heat transfer configuration
            temp_controller: Temperature controller, None for fixed jacket temp
        """
        self.props = thermo_props
        self.config = heat_config
        self.temp_controller = temp_controller
    
    def metabolic_heat(self, OUR: float, V: float) -> float:
        """
        Calculate metabolic heat generation from OUR.
        
        Q_metabolic = -ΔH_ox × OUR × V
        
        Args:
            OUR: Oxygen uptake rate (mmol/L/h)
            V: Reactor volume (L)
        
        Returns:
            Q_metabolic (W)
        """
        # Convert OUR to mol/h
        OUR_mol_per_h = (OUR / 1000) * V
        
        # Heat generation (kJ/h)
        # delta_H_ox is negative, so negate to get positive heat
        Q_kJ_per_h = -self.props.delta_H_ox * OUR_mol_per_h
        
        # Convert to W (J/s)
        Q_W = Q_kJ_per_h * 1000 / 3600
        
        return Q_W
    
    def agitation_heat(self, N: float, V: float) -> float:
        """
        Calculate heat from agitation.
        
        Args:
            N: Agitation speed (rpm)
            V: Reactor volume (L)
        
        Returns:
            Q_agitation (W)
        """
        if self.config.D_impeller is not None:
            # Full correlation: P = N_p × ρ × N³ × D⁵
            N_per_s = N / 60.0  # rpm → 1/s
            P = (self.config.N_p * self.props.rho * 
                 (N_per_s ** 3) * (self.config.D_impeller ** 5))
        else:
            # Simplified correlation: P = k × N³ × V
            k_power = 1e-9  # W·min³/rpm³/L
            P = k_power * (N ** 3) * V
        
        return P
    
    def feed_heat(self, F_in: float, T: float) -> float:
        """
        Calculate heat from feed addition.
        
        Q_feed = F_in × ρ_feed × Cp_feed × (T_feed - T)
        
        Args:
            F_in: Feed rate (L/h)
            T: Reactor temperature (°C)
        
        Returns:
            Q_feed (W), negative if feed is cooler than reactor
        """
        # Heat flow (kJ/h)
        Q_kJ_per_h = (F_in * self.config.rho_feed * self.config.Cp_feed * 
                      (self.config.T_feed - T))
        
        # Convert to W
        Q_W = Q_kJ_per_h * 1000 / 3600
        
        return Q_W
    
    def cooling_heat(self, T: float, T_jacket: float, V: float) -> float:
        """
        Calculate heat removal via cooling jacket.
        
        Q_cooling = U × A × (T - T_jacket)
        
        For cylindrical reactor: A = 4V/D_tank
        
        Args:
            T: Reactor temperature (°C)
            T_jacket: Jacket temperature (°C)
            V: Reactor volume (L)
        
        Returns:
            Q_cooling (W), positive when removing heat
        """
        # Heat transfer area (m²)
        V_m3 = V / 1000.0  # L → m³
        A = 4 * V_m3 / self.config.D_tank  # m²
        
        # Heat transfer (W)
        Q = self.config.U * A * (T - T_jacket)
        
        return Q
    
    def evaporation_heat(self, Q_gas: float) -> float:
        """
        Calculate heat removed by evaporation.
        
        Q_evaporation = λ_vap × F_evap × ρ_water
        
        Args:
            Q_gas: Gas flow rate (L/h)
        
        Returns:
            Q_evaporation (W)
        """
        if self.config.k_evap == 0:
            return 0.0
        
        # Evaporation rate (L/h)
        F_evap = self.config.k_evap * Q_gas * (1 - self.config.RH_inlet)
        
        # Heat removal (kJ/h)
        Q_kJ_per_h = self.props.lambda_vap * F_evap * 1.0  # ρ_water = 1 kg/L
        
        # Convert to W
        Q_W = Q_kJ_per_h * 1000 / 3600
        
        return Q_W
    
    def temperature_derivative(self, 
                              t: float,
                              T: float,
                              OUR: float,
                              N: float,
                              F_in: float,
                              Q_gas: float,
                              V: float,
                              T_jacket: Optional[float] = None) -> float:
        """
        Calculate dT/dt from complete heat balance.
        
        dT/dt = Q_net / (ρ × V × Cp)
        
        Args:
            t: Time (h)
            T: Reactor temperature (°C)
            OUR: Oxygen uptake rate (mmol/L/h)
            N: Agitation speed (rpm)
            F_in: Feed rate (L/h)
            Q_gas: Gas flow rate (L/h)
            V: Reactor volume (L)
            T_jacket: Jacket temperature (°C), or None for controlled
        
        Returns:
            dT/dt (°C/h)
        """
        # Heat generation
        Q_metabolic = self.metabolic_heat(OUR, V)
        Q_agitation = self.agitation_heat(N, V)
        Q_feed = self.feed_heat(F_in, T)
        
        # Heat removal
        if T_jacket is None and self.temp_controller is not None:
            T_jacket = self.temp_controller.get_jacket_temperature(t, T)
        elif T_jacket is None:
            T_jacket = 25.0  # Default
        
        Q_cooling = self.cooling_heat(T, T_jacket, V)
        Q_evaporation = self.evaporation_heat(Q_gas)
        
        # Net heat (W)
        Q_net = Q_metabolic + Q_agitation + Q_feed - Q_cooling - Q_evaporation
        
        # Temperature change
        # mass = ρ × V (kg), ρ in kg/m³, V in L
        mass = self.props.rho * V / 1000.0  # kg
        
        # dT/dt = Q / (m × Cp)
        # Q in W (J/s), Cp in kJ/kg/°C
        dT_dt = Q_net / (mass * self.props.Cp * 1000)  # °C/s
        dT_dt = dT_dt * 3600  # Convert to °C/h
        
        return dT_dt


# Example usage and testing
if __name__ == "__main__":
    print("=" * 60)
    print("THERMODYNAMICS VALIDATION")
    print("=" * 60)
    
    # Create thermodynamic properties
    thermo_props = ThermodynamicProperties(
        rho=1000.0,
        Cp=4.0,
        delta_H_ox=-460.0,
        lambda_vap=2400.0
    )
    
    # Create heat transfer configuration
    heat_config = HeatTransferConfig(
        U=300.0,
        D_tank=0.3,
        D_impeller=0.05,
        N_p=5.0,
        T_feed=25.0,
        rho_feed=1.2,
        Cp_feed=3.8,
        k_evap=0.0  # No evaporation (condenser)
    )
    
    # Create temperature controller
    temp_controller = TemperatureController(
        T_setpoint=37.0,
        K_p=10.0,
        K_i=0.5,
        T_jacket_min=4.0,
        T_jacket_max=80.0
    )
    
    # Create heat balance
    heat_balance = HeatBalance(thermo_props, heat_config, temp_controller)
    
    # Test conditions
    t = 10.0  # h
    T = 37.5  # °C (slightly above setpoint)
    OUR = 50.0  # mmol/L/h
    N = 600  # rpm
    F_in = 0.1  # L/h
    Q_gas = 120  # L/h
    V = 2.5  # L
    
    print(f"\nTest conditions:")
    print(f"  Time: {t} h")
    print(f"  Temperature: {T} °C")
    print(f"  OUR: {OUR} mmol/L/h")
    print(f"  Agitation: {N} rpm")
    print(f"  Feed rate: {F_in} L/h")
    print(f"  Gas flow: {Q_gas} L/h")
    print(f"  Volume: {V} L")
    
    # Calculate individual heat terms
    Q_metabolic = heat_balance.metabolic_heat(OUR, V)
    Q_agitation = heat_balance.agitation_heat(N, V)
    Q_feed = heat_balance.feed_heat(F_in, T)
    
    T_jacket = temp_controller.get_jacket_temperature(t, T)
    Q_cooling = heat_balance.cooling_heat(T, T_jacket, V)
    Q_evaporation = heat_balance.evaporation_heat(Q_gas)
    
    print(f"\nHeat generation:")
    print(f"  Metabolic: {Q_metabolic:.2f} W")
    print(f"  Agitation: {Q_agitation:.2f} W")
    print(f"  Feed: {Q_feed:.2f} W")
    print(f"  Total generation: {Q_metabolic + Q_agitation + Q_feed:.2f} W")
    
    print(f"\nHeat removal:")
    print(f"  Jacket (T_jacket = {T_jacket:.1f}°C): {Q_cooling:.2f} W")
    print(f"  Evaporation: {Q_evaporation:.2f} W")
    print(f"  Total removal: {Q_cooling + Q_evaporation:.2f} W")
    
    # Calculate temperature derivative
    dT_dt = heat_balance.temperature_derivative(t, T, OUR, N, F_in, Q_gas, V)
    
    print(f"\nNet heat balance:")
    print(f"  Q_net: {Q_metabolic + Q_agitation + Q_feed - Q_cooling - Q_evaporation:.2f} W")
    print(f"  dT/dt: {dT_dt:.4f} °C/h")
    
    # Test temperature controller response
    print(f"\n" + "=" * 60)
    print("TEMPERATURE CONTROLLER TEST")
    print("=" * 60)
    
    test_temps = [36.0, 36.5, 37.0, 37.5, 38.0, 39.0]
    
    print(f"\nSetpoint: {temp_controller.T_sp}°C")
    print(f"Controller: K_p={temp_controller.K_p}, K_i={temp_controller.K_i}")
    
    for T_test in test_temps:
        T_j = temp_controller.get_jacket_temperature(t, T_test)
        error = T_test - temp_controller.T_sp
        print(f"  T = {T_test:.1f}°C (error = {error:+.1f}°C) → T_jacket = {T_j:.1f}°C")
    
    print("=" * 60)
