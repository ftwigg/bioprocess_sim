#!/usr/bin/env python
"""
test_installation.py

Quick verification script to test if batch reactor simulation is properly installed.
Run this to verify all modules work before starting simulations.
"""

import sys

def test_imports():
    """Test if all required modules can be imported"""
    print("="*60)
    print("Testing Module Imports...")
    print("="*60)
    
    try:
        import numpy as np
        print(f"✓ NumPy {np.__version__}")
    except ImportError as e:
        print(f"✗ NumPy failed: {e}")
        return False
    
    try:
        import scipy
        print(f"✓ SciPy {scipy.__version__}")
    except ImportError as e:
        print(f"✗ SciPy failed: {e}")
        return False
    
    try:
        import matplotlib
        print(f"✓ Matplotlib {matplotlib.__version__}")
    except ImportError as e:
        print(f"✗ Matplotlib failed: {e}")
        return False
    
    try:
        import pandas as pd
        print(f"✓ Pandas {pd.__version__}")
    except ImportError as e:
        print(f"✗ Pandas failed: {e}")
        return False
    
    try:
        from base_models import CellParameters, ReactorConfig, ProductionModel
        print("✓ base_models")
    except ImportError as e:
        print(f"✗ base_models failed: {e}")
        return False
    
    try:
        from kinetics import PirtKinetics
        print("✓ kinetics")
    except ImportError as e:
        print(f"✗ kinetics failed: {e}")
        return False
    
    try:
        from mass_balance import LiquidPhaseMassBalance
        print("✓ mass_balance")
    except ImportError as e:
        print(f"✗ mass_balance failed: {e}")
        return False
    
    try:
        from thermodynamics import MetabolicHeat
        print("✓ thermodynamics")
    except ImportError as e:
        print(f"✗ thermodynamics failed: {e}")
        return False
    
    try:
        from chemistry import pHCalculator
        print("✓ chemistry")
    except ImportError as e:
        print(f"✗ chemistry failed: {e}")
        return False
    
    try:
        from reactor import BatchReactor
        print("✓ reactor")
    except ImportError as e:
        print(f"✗ reactor failed: {e}")
        return False
    
    try:
        from simulator import BatchSimulator, run_quick_simulation
        print("✓ simulator")
    except ImportError as e:
        print(f"✗ simulator failed: {e}")
        return False
    
    try:
        from utils import plot_batch_results, calculate_yields
        print("✓ utils")
    except ImportError as e:
        print(f"✗ utils failed: {e}")
        return False
    
    print("\n✅ All modules imported successfully!\n")
    return True


def test_basic_simulation():
    """Test if a basic simulation runs"""
    print("="*60)
    print("Testing Basic Simulation...")
    print("="*60)
    
    try:
        from simulator import run_quick_simulation
        import numpy as np
        
        print("Running 1-hour test simulation...")
        results = run_quick_simulation(
            mu_max=0.7,
            Ks=0.1,
            Yx_s=0.5,
            ms=0.03,
            S0=20.0,
            X0=0.1,
            t_end=1.0  # Just 1 hour
        )
        
        # Check results
        assert len(results.time) > 0, "No time points"
        assert len(results.X) > 0, "No biomass data"
        assert results.X[-1] > results.X[0], "Biomass should increase"
        assert results.S[-1] < results.S[0], "Substrate should decrease"
        
        print(f"\n✓ Simulation completed successfully!")
        print(f"  Duration: {results.time[-1]:.2f} h")
        print(f"  Initial biomass: {results.X[0]:.4f} g/L")
        print(f"  Final biomass: {results.X[-1]:.4f} g/L")
        print(f"  Biomass increase: {(results.X[-1] - results.X[0]):.4f} g/L")
        print(f"  Substrate consumed: {(results.S[0] - results.S[-1]):.4f} g/L")
        print(f"  Max growth rate: {np.max(results.mu):.4f} 1/h")
        
        return True
        
    except Exception as e:
        print(f"\n✗ Simulation failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_visualization():
    """Test if plotting works"""
    print("\n" + "="*60)
    print("Testing Visualization...")
    print("="*60)
    
    try:
        import matplotlib.pyplot as plt
        from simulator import run_quick_simulation
        from utils import calculate_yields
        
        # Run quick simulation
        results = run_quick_simulation(
            mu_max=0.7,
            S0=20.0,
            X0=0.1,
            t_end=1.0
        )
        
        # Test plotting (don't show, just verify it doesn't crash)
        fig, ax = plt.subplots(1, 1, figsize=(8, 6))
        ax.plot(results.time, results.X)
        ax.set_xlabel('Time (h)')
        ax.set_ylabel('Biomass (g/L)')
        ax.set_title('Test Plot')
        plt.close(fig)
        
        # Test utility functions
        yields = calculate_yields(results, X0=0.1, S0=20.0)
        
        print("✓ Matplotlib works!")
        print("✓ Utility functions work!")
        print(f"  Calculated Yx/s: {yields['Yx_s']:.4f}")
        
        return True
        
    except Exception as e:
        print(f"✗ Visualization test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_parameter_creation():
    """Test if parameter objects can be created"""
    print("\n" + "="*60)
    print("Testing Parameter Creation...")
    print("="*60)
    
    try:
        from base_models import CellParameters, ReactorConfig, ProductionModel
        
        # Create cell parameters
        params = CellParameters(
            mu_max=0.7,
            Ks=0.1,
            Yx_s_max=0.5,
            ms=0.03,
            production_model=ProductionModel.GROWTH_ASSOCIATED,
            Yp_s=0.0
        )
        print("✓ CellParameters created")
        
        # Create reactor config
        config = ReactorConfig(
            V_working=1.0,
            X0=0.1,
            S0=20.0,
            P0=0.0
        )
        print("✓ ReactorConfig created")
        
        # Create reactor
        from reactor import BatchReactor
        reactor = BatchReactor(params, config)
        print("✓ BatchReactor created")
        
        return True
        
    except Exception as e:
        print(f"✗ Parameter creation failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Run all tests"""
    print("\n" + "#"*60)
    print("# BATCH REACTOR SIMULATION - INSTALLATION TEST")
    print("#"*60 + "\n")
    
    all_passed = True
    
    # Test 1: Imports
    if not test_imports():
        all_passed = False
        print("\n❌ Import test FAILED!")
        print("   Make sure all .py files are present and requirements are installed.")
        return False
    
    # Test 2: Parameter creation
    if not test_parameter_creation():
        all_passed = False
        print("\n❌ Parameter creation test FAILED!")
        return False
    
    # Test 3: Basic simulation
    if not test_basic_simulation():
        all_passed = False
        print("\n❌ Simulation test FAILED!")
        return False
    
    # Test 4: Visualization
    if not test_visualization():
        all_passed = False
        print("\n❌ Visualization test FAILED!")
        return False
    
    # Summary
    print("\n" + "="*60)
    if all_passed:
        print("🎉 ALL TESTS PASSED! 🎉")
        print("="*60)
        print("\n✅ Your batch reactor simulation is ready to use!")
        print("\nNext steps:")
        print("  1. Open batch_reactor_examples.ipynb or batch_reactor_examples_colab.ipynb")
        print("  2. Run the example simulations")
        print("  3. Modify parameters for your organism")
        print("  4. Start exploring!\n")
        return True
    else:
        print("❌ SOME TESTS FAILED")
        print("="*60)
        print("\nPlease check:")
        print("  • All .py files are present in the current directory")
        print("  • Requirements are installed: pip install -r requirements.txt")
        print("  • Python version is 3.8 or higher")
        print("\nSee COLAB_SETUP.md for troubleshooting.\n")
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
