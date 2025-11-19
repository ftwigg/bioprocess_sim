# Google Colab Setup Guide

## 📦 Running Batch Reactor Simulation in Google Colab

### Method 1: Upload Files Directly (Recommended for Quick Start)

1. **Open Google Colab**: Go to https://colab.research.google.com/

2. **Upload the Colab notebook**:
   - Upload `batch_reactor_examples_colab.ipynb`
   - Or create a new notebook and copy the cells

3. **Run the setup cell** (first code cell):
   - It will prompt you to upload files
   - Upload these 9 files:
     * `base_models.py`
     * `kinetics.py`
     * `mass_balance.py`
     * `thermodynamics.py`
     * `chemistry.py`
     * `reactor.py`
     * `simulator.py`
     * `utils.py`
     * `requirements.txt`

4. **Install dependencies**:
   ```python
   !pip install -r requirements.txt -q
   ```

5. **Run the import cell** to verify everything works

6. **Start simulating!**

---

### Method 2: Clone from GitHub (Recommended for Repeated Use)

1. **Upload files to GitHub**:
   ```bash
   # In your local terminal
   git init
   git add *.py *.txt *.ipynb *.md
   git commit -m "Initial commit: Batch reactor simulation"
   git remote add origin https://github.com/YOUR_USERNAME/batch-reactor-sim.git
   git push -u origin main
   ```

2. **In Google Colab**:
   ```python
   !git clone https://github.com/YOUR_USERNAME/batch-reactor-sim.git
   %cd batch-reactor-sim
   !pip install -r requirements.txt -q
   ```

3. **Import and run**:
   ```python
   from simulator import run_quick_simulation
   from utils import plot_batch_results
   
   results = run_quick_simulation(mu_max=0.7, S0=20.0, t_end=24.0)
   plot_batch_results(results)
   ```

---

### Method 3: Google Drive Integration

1. **Upload files to Google Drive**:
   - Create a folder: `My Drive/batch_reactor_sim/`
   - Upload all `.py` files and `requirements.txt`

2. **In Google Colab**:
   ```python
   from google.colab import drive
   drive.mount('/content/drive')
   
   import sys
   sys.path.append('/content/drive/My Drive/batch_reactor_sim/')
   
   !pip install -r '/content/drive/My Drive/batch_reactor_sim/requirements.txt' -q
   ```

3. **Import and use**:
   ```python
   from simulator import run_quick_simulation
   results = run_quick_simulation(mu_max=0.7, S0=20.0, t_end=24.0)
   ```

---

## 📋 Required Files

Make sure you have all these files:

### Core Modules (Required):
- ✅ `base_models.py` (6.9 KB)
- ✅ `kinetics.py` (10.4 KB)
- ✅ `mass_balance.py` (10.4 KB)
- ✅ `thermodynamics.py` (12.2 KB)
- ✅ `chemistry.py` (11.0 KB)
- ✅ `reactor.py` (12.3 KB)
- ✅ `simulator.py` (11.9 KB)
- ✅ `utils.py` (13.0 KB)

### Configuration:
- ✅ `requirements.txt` (0.4 KB)

### Documentation (Optional):
- 📄 `README_batch_sim.md`
- 📄 `QUICKSTART.md`
- 📄 `ARCHITECTURE.md`

### Notebooks:
- 📓 `batch_reactor_examples.ipynb` (full version)
- 📓 `batch_reactor_examples_colab.ipynb` (Colab-optimized)

**Total size: ~88 KB** (very lightweight!)

---

## 🧪 Quick Test in Colab

After setup, verify everything works:

```python
# Test 1: Import all modules
from base_models import CellParameters, ReactorConfig
from reactor import BatchReactor
from simulator import BatchSimulator
print("✓ All imports successful")

# Test 2: Run quick simulation
from simulator import run_quick_simulation
results = run_quick_simulation(mu_max=0.7, S0=20.0, t_end=1.0)
print(f"✓ Simulation works! Final biomass: {results.X[-1]:.2f} g/L")

# Test 3: Visualization
from utils import plot_batch_results
import matplotlib.pyplot as plt
plot_batch_results(results)
print("✓ Plotting works!")
```

Expected output:
```
✓ All imports successful
✓ Simulation works! Final biomass: 0.14 g/L
✓ Plotting works!
```

---

## 💡 Tips for Google Colab

### 1. **Memory Management**
- Colab provides ~12 GB RAM (free tier)
- Each simulation uses <50 MB
- Can run 100+ simulations without issues
- Clear outputs if needed: `Runtime` → `Clear all outputs`

### 2. **Session Persistence**
- Files uploaded directly are lost when session ends
- Use Google Drive or GitHub for permanent storage
- Reconnect time: ~30 seconds to reinstall requirements

### 3. **Speed**
- Typical simulation time: 0.5-2 seconds
- E. coli 24h batch: ~1 second
- CHO 168h batch: ~2 seconds
- Colab is fast enough for interactive use!

### 4. **Saving Results**
```python
from utils import export_to_csv
export_to_csv(results, 'my_results.csv')

# Download in Colab
from google.colab import files
files.download('my_results.csv')
```

### 5. **GPU/TPU Not Needed**
- This is CPU-based numerical integration
- Standard runtime is sufficient
- No need for GPU acceleration

---

## 🔧 Troubleshooting

### Problem: "Module not found"
**Solution**: Make sure all `.py` files are uploaded/accessible
```python
import os
print(os.listdir('.'))  # Check what files are present
```

### Problem: "No module named 'scipy'"
**Solution**: Install requirements
```python
!pip install numpy scipy matplotlib pandas -q
```

### Problem: "AttributeError" or "ImportError"
**Solution**: Restart runtime and re-run setup
```python
# In Colab: Runtime → Restart runtime
```

### Problem: "Negative concentrations"
**Solution**: Check your parameters are reasonable
```python
# Ensure:
# - mu_max > 0
# - Ks > 0
# - Yx_s between 0 and 1
# - S0 > 0
```

### Problem: Integration fails
**Solution**: Try different solver or adjust tolerances
```python
simulator.simulate(
    t_end=24.0,
    method='BDF',  # Try 'LSODA', 'RK45', or 'BDF'
    rtol=1e-8,     # Tighter tolerance
    atol=1e-10
)
```

---

## 📊 Example Colab Workflow

```python
# 1. Setup (run once per session)
!pip install -r requirements.txt -q

# 2. Import
from simulator import run_quick_simulation
from utils import plot_batch_results, print_summary_table

# 3. Define organism
results = run_quick_simulation(
    mu_max=0.7,      # Max growth rate
    Ks=0.1,          # Substrate affinity
    Yx_s=0.5,        # Biomass yield
    ms=0.03,         # Maintenance
    S0=20.0,         # Initial substrate
    X0=0.1,          # Initial biomass
    t_end=24.0       # Duration
)

# 4. Analyze
print_summary_table(results, X0=0.1, S0=20.0)

# 5. Visualize
plot_batch_results(results)

# 6. Export
from utils import export_to_csv
export_to_csv(results, 'results.csv')

# 7. Download
from google.colab import files
files.download('results.csv')
```

---

## 🎯 Common Use Cases

### 1. Parameter Optimization
```python
mu_values = [0.3, 0.5, 0.7, 0.9]
for mu in mu_values:
    results = run_quick_simulation(mu_max=mu, S0=20.0, t_end=24.0)
    print(f"μ_max={mu}: Final X={results.X[-1]:.2f} g/L")
```

### 2. Medium Comparison
```python
substrates = [10, 20, 40, 80]
for S0 in substrates:
    results = run_quick_simulation(S0=S0, t_end=24.0)
    print(f"S0={S0}: Final X={results.X[-1]:.2f} g/L")
```

### 3. Temperature Screening
```python
# Adjust mu_max with Arrhenius
temps = [25, 30, 37, 42]
mu_ref = 0.7
for T in temps:
    mu_T = mu_ref * np.exp(5000 * (1/310 - 1/(T+273)))
    results = run_quick_simulation(mu_max=mu_T, t_end=24.0)
    print(f"T={T}°C: Final X={results.X[-1]:.2f} g/L")
```

---

## 🔗 Useful Links

- **Colab Documentation**: https://colab.research.google.com/
- **NumPy**: https://numpy.org/
- **SciPy**: https://scipy.org/
- **Matplotlib**: https://matplotlib.org/

---

## ✅ Checklist

Before starting your simulation in Colab:

- [ ] All `.py` files uploaded
- [ ] `requirements.txt` uploaded
- [ ] Dependencies installed (`!pip install -r requirements.txt -q`)
- [ ] Imports working
- [ ] Test simulation runs
- [ ] Visualization displays

**You're ready to simulate!** 🎉

---

## 📧 Need Help?

If you encounter issues:

1. Check this guide's troubleshooting section
2. Verify all files are present
3. Restart runtime and try again
4. Review the source code comments
5. Check the documentation files

**Happy simulating in the cloud!** ☁️🧫
