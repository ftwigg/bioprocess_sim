"""
Fed-Batch Bioprocess Simulator

A first-principles simulator for fed-batch fermentation processes.
"""

__version__ = "1.0.0"

# Import key classes and functions for easy access
from .base_models import (
    CarbonSource,
    BiomassComposition,
    SubstrateParameters,
    GasComposition,
    ReactorState,
    FeedComposition,
    ReactorConfiguration,
    SimulationResults,
    # Predefined instances
    GLUCOSE,
    GLYCEROL,
    ACETATE,
    METHANOL,
    STANDARD_BIOMASS,
    ECOLI_BIOMASS,
    YEAST_BIOMASS,
    LAB_STR_5L,
    PILOT_STR_100L,
    PRODUCTION_STR_10000L,
)

from .stoichiometry import (
    calculate_oxygen_yield,
    calculate_nitrogen_yield,
    calculate_stoichiometric_coefficients,
    calculate_RQ,
    validate_elemental_balance,
)

__all__ = [
    # Data models
    'CarbonSource',
    'BiomassComposition',
    'SubstrateParameters',
    'GasComposition',
    'ReactorState',
    'FeedComposition',
    'ReactorConfiguration',
    'SimulationResults',
    # Predefined carbon sources
    'GLUCOSE',
    'GLYCEROL',
    'ACETATE',
    'METHANOL',
    # Predefined biomass
    'STANDARD_BIOMASS',
    'ECOLI_BIOMASS',
    'YEAST_BIOMASS',
    # Predefined reactors
    'LAB_STR_5L',
    'PILOT_STR_100L',
    'PRODUCTION_STR_10000L',
    # Stoichiometry functions
    'calculate_oxygen_yield',
    'calculate_nitrogen_yield',
    'calculate_stoichiometric_coefficients',
    'calculate_RQ',
    'validate_elemental_balance',
]
