"""
Chemistry models for pH calculation.

Implements:
- Simplified empirical pH model (default)
- Extensible for mechanistic pH model (future)
"""

from abc import ABC, abstractmethod


class pHModel(ABC):
    """Base class for pH models."""
    
    @abstractmethod
    def calculate_pH(self, P: float) -> float:
        """
        Calculate pH from product concentration.
        
        Args:
            P: Product concentration (g/L)
        
        Returns:
            pH
        """
        pass


class SimplifiedpHModel(pHModel):
    """
    Simplified empirical pH model.
    
    pH = pH0 - k_acid × P
    
    Assumes product is an organic acid (or neutral if k_acid = 0).
    """
    
    def __init__(self, pH0: float = 7.0, k_acid: float = 0.0):
        """
        Args:
            pH0: Initial pH
            k_acid: Empirical constant (pH drop per g/L product)
                   0 for neutral product
                   >0 for acidic product (e.g., 0.1 for lactate)
        """
        self.pH0 = pH0
        self.k_acid = k_acid
    
    def calculate_pH(self, P: float) -> float:
        """Calculate pH from product concentration."""
        pH = self.pH0 - self.k_acid * P
        
        # Limit to reasonable range
        pH = max(4.0, min(pH, 9.0))
        
        return pH


# Placeholder for future mechanistic pH model
class MechanisticpHModel(pHModel):
    """
    Mechanistic pH model with charge balance (future implementation).
    
    Would include:
    - CO2/HCO3- equilibrium
    - NH3/NH4+ equilibrium
    - Organic acid equilibrium
    - Buffer equilibrium
    - Charge balance solution
    """
    
    def __init__(self):
        raise NotImplementedError(
            "Mechanistic pH model not yet implemented. "
            "Use SimplifiedpHModel for now."
        )
    
    def calculate_pH(self, P: float) -> float:
        raise NotImplementedError


# Example usage
if __name__ == "__main__":
    print("=" * 60)
    print("pH MODEL VALIDATION")
    print("=" * 60)
    
    # Test 1: Neutral product
    print("\n1. Neutral product (k_acid = 0)")
    print("-" * 60)
    
    pH_model_neutral = SimplifiedpHModel(pH0=7.0, k_acid=0.0)
    
    for P in [0, 5, 10, 20, 50]:
        pH = pH_model_neutral.calculate_pH(P)
        print(f"  P = {P:2.0f} g/L → pH = {pH:.2f}")
    
    # Test 2: Acidic product (lactate-like)
    print("\n2. Acidic product (k_acid = 0.1)")
    print("-" * 60)
    
    pH_model_acidic = SimplifiedpHModel(pH0=7.0, k_acid=0.1)
    
    for P in [0, 5, 10, 20, 30, 50]:
        pH = pH_model_acidic.calculate_pH(P)
        print(f"  P = {P:2.0f} g/L → pH = {pH:.2f}")
    
    # Test 3: Strong acidic product
    print("\n3. Strong acidic product (k_acid = 0.3)")
    print("-" * 60)
    
    pH_model_strong = SimplifiedpHModel(pH0=7.0, k_acid=0.3)
    
    for P in [0, 5, 10, 15, 20]:
        pH = pH_model_strong.calculate_pH(P)
        print(f"  P = {P:2.0f} g/L → pH = {pH:.2f}")
    
    print("\n" + "=" * 60)
    print("Note: pH is limited to range [4.0, 9.0]")
    print("=" * 60)
