# chemistry.py
"""
Chemical equilibria and pH calculations for batch reactor.
Handles weak acid/base equilibria, buffer systems, and pH changes.
"""

import numpy as np
from typing import Dict, Optional, Tuple
from scipy.optimize import fsolve


class pHCalculator:
    """
    Calculate pH based on metabolite concentrations and buffer systems.
    
    Considers:
    - Weak acid/base equilibria
    - Buffer systems (phosphate, carbonate)
    - CO2 dissolution and carbonic acid
    - Organic acid production (lactate, acetate, etc.)
    """
    
    # pKa values at 37°C (approximate)
    PKA_VALUES = {
        'acetate': 4.76,
        'lactate': 3.86,
        'formate': 3.75,
        'phosphate1': 2.15,  # H3PO4 <-> H2PO4-
        'phosphate2': 7.20,  # H2PO4- <-> HPO42-
        'phosphate3': 12.35,  # HPO42- <-> PO43-
        'carbonate1': 6.35,  # H2CO3 <-> HCO3-
        'carbonate2': 10.33,  # HCO3- <-> CO32-
        'ammonia': 9.25,  # NH4+ <-> NH3
    }
    
    def __init__(self, buffer_capacity: float = 0.01):
        """
        Initialize pH calculator.
        
        Args:
            buffer_capacity: Buffer capacity of medium (M)
        """
        self.buffer_capacity = buffer_capacity
    
    def henderson_hasselbalch(self, pKa: float, acid: float, base: float) -> float:
        """
        Calculate pH using Henderson-Hasselbalch equation.
        
        pH = pKa + log10([base] / [acid])
        
        Args:
            pKa: pKa of the acid
            acid: Concentration of protonated form (M)
            base: Concentration of deprotonated form (M)
            
        Returns:
            pH
        """
        if acid <= 0 or base <= 0:
            return 7.0  # Neutral default
        
        pH = pKa + np.log10(base / acid)
        return pH
    
    def weak_acid_pH(self, Ca: float, pKa: float) -> float:
        """
        Calculate pH of a weak acid solution.
        
        For HA <-> H+ + A-:
        pH = 0.5 * (pKa - log10(Ca))
        
        Args:
            Ca: Total acid concentration (M)
            pKa: pKa of the acid
            
        Returns:
            pH
        """
        if Ca <= 0:
            return 7.0
        
        # Approximate for weak acid
        pH = 0.5 * (pKa - np.log10(Ca))
        
        # Clamp to reasonable range
        pH = np.clip(pH, 2.0, 12.0)
        
        return pH
    
    def buffer_pH(self, buffer_conc: float, acid_conc: float, 
                  pKa: float, pH_initial: float = 7.0) -> float:
        """
        Calculate pH with buffer system and added acid.
        
        Args:
            buffer_conc: Total buffer concentration (M)
            acid_conc: Concentration of added acid (M)
            pKa: pKa of buffer
            pH_initial: Initial pH
            
        Returns:
            Final pH
        """
        # Use Henderson-Hasselbalch with buffer equilibrium
        # [HA] + [A-] = buffer_conc
        # [H+] added = acid_conc
        
        # Initial buffer speciation
        ratio_initial = 10 ** (pH_initial - pKa)
        A_initial = buffer_conc * ratio_initial / (1 + ratio_initial)
        HA_initial = buffer_conc - A_initial
        
        # After acid addition
        A_final = max(0, A_initial - acid_conc)
        HA_final = HA_initial + acid_conc
        
        if A_final <= 0 or HA_final <= 0:
            # Buffer capacity exceeded
            return self.weak_acid_pH(acid_conc, pKa)
        
        pH_final = self.henderson_hasselbalch(pKa, HA_final, A_final)
        
        return pH_final
    
    def co2_pH_effect(self, CO2_dissolved: float, alkalinity: float = 0.002) -> float:
        """
        Calculate pH effect of dissolved CO2.
        
        CO2 + H2O <-> H2CO3 <-> H+ + HCO3-
        
        Args:
            CO2_dissolved: Dissolved CO2 concentration (mM)
            alkalinity: Total alkalinity (M)
            
        Returns:
            pH
        """
        # Convert to M
        CO2_M = CO2_dissolved / 1000
        
        # Simplified calculation
        # pH ≈ pKa1 + log10([HCO3-] / [H2CO3])
        
        # Assume [H2CO3] ≈ [CO2_dissolved]
        # [HCO3-] from alkalinity
        HCO3 = alkalinity
        H2CO3 = CO2_M
        
        if H2CO3 <= 0:
            return 7.4  # Default physiological pH
        
        pKa_carbonate = self.PKA_VALUES['carbonate1']
        pH = self.henderson_hasselbalch(pKa_carbonate, H2CO3, HCO3)
        
        return pH
    
    def organic_acid_pH(self, acid_conc: float, acid_type: str,
                       buffer_conc: float, pH_initial: float = 7.0) -> float:
        """
        Calculate pH after organic acid production.
        
        Args:
            acid_conc: Organic acid concentration (g/L)
            acid_type: Type of acid ('acetate', 'lactate', etc.)
            buffer_conc: Buffer concentration (M)
            pH_initial: Initial pH
            
        Returns:
            Final pH
        """
        # Convert g/L to M (need molecular weight)
        MW = {'acetate': 60.05, 'lactate': 90.08, 'formate': 46.03}
        
        if acid_type not in MW:
            return pH_initial
        
        acid_M = acid_conc / MW[acid_type]
        
        # Get pKa
        pKa = self.PKA_VALUES.get(acid_type, 4.0)
        
        # Calculate new pH with buffer
        pH_final = self.buffer_pH(buffer_conc, acid_M, pKa, pH_initial)
        
        return pH_final
    
    def calculate_pH_from_metabolites(self, 
                                     metabolites: Dict[str, float],
                                     buffer_system: Optional[Dict[str, float]] = None,
                                     pH_initial: float = 7.0) -> float:
        """
        Calculate pH from multiple metabolites and buffers.
        
        Args:
            metabolites: Dict of metabolite name: concentration (g/L)
            buffer_system: Dict of buffer components (M)
            pH_initial: Initial pH
            
        Returns:
            Calculated pH
        """
        if buffer_system is None:
            buffer_system = {'phosphate': self.buffer_capacity}
        
        # Start with initial pH
        pH = pH_initial
        
        # Account for each metabolite
        for metabolite, conc in metabolites.items():
            if metabolite in self.PKA_VALUES:
                buffer_conc = buffer_system.get('phosphate', self.buffer_capacity)
                pH = self.organic_acid_pH(conc, metabolite, buffer_conc, pH)
        
        return pH


class BufferSystem:
    """
    Model buffer systems in fermentation media.
    
    Common buffers:
    - Phosphate (most common, pH 6-8)
    - HEPES (pH 6.8-8.2)
    - Carbonate/bicarbonate (pH 9-11)
    - Tris (pH 7-9)
    """
    
    def __init__(self, buffer_type: str = 'phosphate', 
                 concentration: float = 0.05):
        """
        Initialize buffer system.
        
        Args:
            buffer_type: Type of buffer
            concentration: Total buffer concentration (M)
        """
        self.buffer_type = buffer_type
        self.concentration = concentration
        
        # Get pKa based on buffer type
        self.pKa = self._get_pKa(buffer_type)
    
    def _get_pKa(self, buffer_type: str) -> float:
        """Get pKa for buffer type."""
        pKa_map = {
            'phosphate': 7.20,  # Second pKa (most useful range)
            'hepes': 7.50,
            'tris': 8.06,
            'carbonate': 10.33,
            'mes': 6.15,
        }
        return pKa_map.get(buffer_type, 7.0)
    
    def buffer_capacity(self, pH: float) -> float:
        """
        Calculate buffer capacity at given pH.
        
        β = 2.303 * C * Ka * [H+] / (Ka + [H+])²
        
        Args:
            pH: Current pH
            
        Returns:
            Buffer capacity (mol/L per pH unit)
        """
        H = 10 ** (-pH)
        Ka = 10 ** (-self.pKa)
        
        # Van Slyke equation
        beta = 2.303 * self.concentration * Ka * H / ((Ka + H) ** 2)
        
        return beta
    
    def pH_after_acid_addition(self, pH_initial: float, 
                               acid_added: float) -> float:
        """
        Calculate pH after acid addition.
        
        Args:
            pH_initial: Initial pH
            acid_added: Moles of acid added per liter (M)
            
        Returns:
            Final pH
        """
        # Use buffer capacity
        beta = self.buffer_capacity(pH_initial)
        
        if beta > 0:
            delta_pH = -acid_added / beta
        else:
            delta_pH = -acid_added  # No buffering
        
        pH_final = pH_initial + delta_pH
        
        # Clamp to reasonable range
        pH_final = np.clip(pH_final, 2.0, 12.0)
        
        return pH_final


class IonicStrength:
    """
    Calculate ionic strength and activity coefficients.
    
    Important for:
    - Accurate pH calculations
    - pKa adjustments
    - Solubility predictions
    """
    
    def __init__(self):
        pass
    
    def calculate_ionic_strength(self, ions: Dict[str, Tuple[float, int]]) -> float:
        """
        Calculate ionic strength.
        
        I = 0.5 * Σ(ci * zi²)
        
        Args:
            ions: Dict of ion_name: (concentration_M, charge)
            
        Returns:
            Ionic strength (M)
        """
        I = 0.0
        for ion, (conc, charge) in ions.items():
            I += conc * (charge ** 2)
        
        I *= 0.5
        return I
    
    def activity_coefficient(self, charge: int, ionic_strength: float) -> float:
        """
        Calculate activity coefficient using Davies equation.
        
        log(γ) = -A * z² * (√I / (1 + √I) - 0.3 * I)
        
        Args:
            charge: Ion charge
            ionic_strength: Ionic strength (M)
            
        Returns:
            Activity coefficient
        """
        A = 0.509  # At 25°C
        
        sqrt_I = np.sqrt(ionic_strength)
        
        log_gamma = -A * (charge ** 2) * (sqrt_I / (1 + sqrt_I) - 0.3 * ionic_strength)
        
        gamma = 10 ** log_gamma
        
        return gamma
    
    def adjust_pKa(self, pKa_ideal: float, ionic_strength: float,
                   charge_acid: int = 0, charge_base: int = -1) -> float:
        """
        Adjust pKa for ionic strength.
        
        Args:
            pKa_ideal: pKa at infinite dilution
            ionic_strength: Ionic strength (M)
            charge_acid: Charge of acid form
            charge_base: Charge of base form
            
        Returns:
            Adjusted pKa
        """
        gamma_acid = self.activity_coefficient(charge_acid, ionic_strength)
        gamma_base = self.activity_coefficient(charge_base, ionic_strength)
        
        pKa_adjusted = pKa_ideal + np.log10(gamma_acid / gamma_base)
        
        return pKa_adjusted
