"""
Example: Fed-batch on glycerol instead of glucose.

Demonstrates:
- The same model with a different carbon source: only the CarbonSource
  object and the kinetic constants change
- Oxygen yield, oxygen maintenance and RQ recalculated from stoichiometry
- A more reduced substrate needs more oxygen per gram, so the feed cap set
  by the oxygen-transfer ceiling is lower than for glucose

What to look for in the output:
- RQ of about 0.75 while growing on glycerol (above 1 on glucose)
- RQ of about 0.95 while starving, whatever the substrate was: the cells
  are then oxidising their own biomass
"""

import numpy as np

import _bootstrap  # noqa: F401  (makes the package importable from a fresh clone)
from fedbatch_simulator import *
from fedbatch_simulator.kinetics import CellParameters
from fedbatch_simulator.mass_transfer import DynamicKLa
from fedbatch_simulator.thermodynamics import TemperatureController
from fedbatch_simulator.feed_strategies import ConstantFeed, ExponentialFeed, PiecewiseFeed
from fedbatch_simulator.control import PICascade
from fedbatch_simulator.reactor import FedBatchReactor
from fedbatch_simulator.simulator import FedBatchSimulator

DO_SAT_AIR = 0.2187  # mmol/L at 37°C, 1 atm air


def main():
    print("=" * 70)
    print("GLYCEROL FED-BATCH SIMULATION")
    print("=" * 70)
    
    # ========== 1. DEFINE CARBON SOURCE ==========
    glycerol = GLYCEROL
    print("\nCarbon source comparison:")
    print(f"  {'':22s} {'glucose':>10} {'glycerol':>10}")
    print(f"  {'Formula':22s} {GLUCOSE.formula:>10} {glycerol.formula:>10}")
    print(f"  {'Degree of reduction':22s} {GLUCOSE.degree_of_reduction:10.2f} {glycerol.degree_of_reduction:10.2f}")
    for label, fn in [('Y_x/O2 (g/g)', calculate_oxygen_yield), ('RQ of growth', calculate_RQ)]:
        print(f"  {label:22s} {fn(GLUCOSE, STANDARD_BIOMASS, 0.5):10.3f} {fn(glycerol, STANDARD_BIOMASS, 0.5):10.3f}")
    
    # ========== 2. DEFINE CELL PARAMETERS ==========
    # E. coli-like defaults; slightly slower and lower affinity on glycerol
    carbon_params = SubstrateParameters(
        name='carbon',
        Ks=0.15,  # g/L
        Ki=None,  # No inhibition (Monod)
        Y_xs=0.5,  # g biomass / g glycerol (true yield)
        ms=0.03   # g glycerol / g biomass / h
    )
    
    nitrogen_params = SubstrateParameters(
        name='nitrogen',
        Ks=0.05,  # g/L
        Ki=None,
        Y_xs=10.0,  # g biomass / g nitrogen
        ms=0.001
    )
    
    cell_params = CellParameters(
        mu_max=0.4,  # 1/h
        carbon_source=glycerol,
        biomass_composition=STANDARD_BIOMASS,
        substrates={
            'carbon': carbon_params,
            'nitrogen': nitrogen_params
        },
        Y_xs=0.5,
        Ks_O2=0.003,  # mmol/L
        alpha=0.0,  # No product
        beta=0.0
    )
    
    print(f"\nCell parameters:")
    print(f"  μ_max: {cell_params.mu_max} 1/h")
    print(f"  Y_x/S: {cell_params.Y_xs} g/g")
    print(f"  Y_x/O2: {cell_params.Y_x_O2:.3f} g/g (from the electron balance)")
    print(f"  m_O2: {cell_params.m_O2:.4f} g/g/h (from m_S)")
    print(f"  RQ of growth: {cell_params.RQ:.3f} (from the carbon balance)")
    
    # ========== 3. REACTOR CONFIGURATION ==========
    reactor_config = LAB_STR_5L
    print(f"\nReactor: {reactor_config.name}")
    print(f"  Volume: {reactor_config.V_reactor} L (working: {reactor_config.V_working_max} L)")
    print(f"  Agitation: {reactor_config.N_min}-{reactor_config.N_max} rpm")
    
    # ========== 4. INITIAL CONDITIONS ==========
    initial_state = ReactorState(
        time=0.0,
        X=0.5,  # g/L
        S_carbon=10.0,  # g/L (batch phase glycerol)
        S_nitrogen=2.0,  # g/L
        P=0.0,
        DO=0.2,  # mmol/L (~90% saturation)
        DCO2=0.5,  # mmol/L
        V=2.0,  # L (starting volume)
        T=37.0,  # °C
        N=300,  # rpm (initial agitation)
        pH=7.0
    )
    
    print(f"\nInitial conditions:")
    print(f"  X: {initial_state.X} g/L")
    print(f"  S_carbon: {initial_state.S_carbon} g/L")
    print(f"  V: {initial_state.V} L")
    print(f"  T: {initial_state.T}°C")
    
    # ========== 5. FEED STRATEGY ==========
    feed_composition = FeedComposition(
        S_carbon=600.0,  # g/L concentrated glycerol
        S_nitrogen=50.0,  # g/L
        temperature=25.0,  # °C
        pH=5.0
    )
    
    # The batch glycerol runs out at about 6.8 h, leaving roughly
    # X0 + 0.48 × S0 of biomass. The feed starts shortly after, as it would
    # on a DO-spike trigger.
    t_feed_start = 7.0   # h
    t_feed_stop = 22.0   # h
    mu_set = 0.10        # 1/h
    X_batch_end = initial_state.X + 0.48 * initial_state.S_carbon
    
    # F0 supplies exactly what that biomass needs to grow at mu_set
    F0 = ExponentialFeed.initial_rate(
        mu_set=mu_set, X=X_batch_end, V=initial_state.V,
        S_feed=feed_composition.S_carbon,
        Y_xs=carbon_params.Y_xs, ms=carbon_params.ms)
    
    # The cap comes from oxygen, not from the pump. Glycerol carries more
    # electrons per gram than glucose, so the same oxygen supply covers less
    # of it: about 3 g/h here against 4.5 g/h for glucose.
    glycerol_cap = 3.0    # g glycerol/h
    F_max = glycerol_cap / feed_composition.S_carbon
    
    batch_feed = ConstantFeed(feed_composition, F_constant=0.0)
    
    exp_feed = ExponentialFeed(
        feed_composition=feed_composition,
        F0=F0,
        mu_set=mu_set,
        F_max=F_max
    )
    exp_feed.t_start = t_feed_start
    
    feed_strategy = PiecewiseFeed([
        (0.0, batch_feed),
        (t_feed_start, exp_feed),
        (t_feed_stop, batch_feed)
    ])
    
    print(f"\nFeed strategy:")
    print(f"  0-{t_feed_start} h: Batch (F = 0)")
    print(f"  {t_feed_start}-{t_feed_stop} h: Exponential feed, then constant at the cap")
    print(f"    F0: {F0*1000:.2f} mL/h")
    print(f"    μ_set: {exp_feed.mu_set} 1/h")
    print(f"    F_max: {F_max*1000:.2f} mL/h ({glycerol_cap} g glycerol/h)")
    print(f"  {t_feed_stop}+ h: Feed off (starvation)")
    print(f"  Feed composition: {feed_composition.S_carbon} g/L glycerol")
    
    # ========== 6. DO CASCADE CONTROL ==========
    do_control = PICascade(
        DO_setpoint=0.06,  # mmol/L (~27% saturation)
        N_min=reactor_config.N_min,
        N_max=reactor_config.N_max,
        Q_gas_min=reactor_config.Q_gas_min * initial_state.V * 60,  # L/h
        Q_gas_max=reactor_config.Q_gas_max * initial_state.V * 60,
        kLa_correlation=DynamicKLa(
            k_O2=reactor_config.k_O2,
            a=reactor_config.a,
            b=reactor_config.b,
            k_X=reactor_config.k_X
        )
    )
    
    print(f"\nDO control (PI cascade):")
    print(f"  Setpoint: {do_control.DO_sp} mmol/L ({do_control.DO_sp/DO_SAT_AIR*100:.0f}% saturation)")
    print(f"  Cascade: Agitation → Aeration")
    print(f"  Tuning: K_c = {do_control.K_c}, τ_i = {do_control.tau_i*60:.0f} min")
    
    # ========== 7. TEMPERATURE CONTROL ==========
    temp_controller = TemperatureController(
        T_setpoint=37.0,  # °C
        K_p=10.0,
        K_i=0.5,
        T_jacket_min=4.0,
        T_jacket_max=80.0
    )
    
    print(f"\nTemperature control:")
    print(f"  Setpoint: {temp_controller.T_sp}°C")
    print(f"  PI gains: K_p={temp_controller.K_p}, K_i={temp_controller.K_i}")
    
    # ========== 8. CREATE REACTOR ==========
    print(f"\n" + "=" * 70)
    print("Creating reactor...")
    
    reactor = FedBatchReactor(
        cell_params=cell_params,
        reactor_config=reactor_config,
        initial_state=initial_state,
        feed_strategy=feed_strategy,
        do_control=do_control,
        temp_controller=temp_controller
    )
    
    # ========== 9. RUN SIMULATION ==========
    print("=" * 70)
    
    simulator = FedBatchSimulator(reactor)
    results = simulator.simulate(t_end=26.0, max_step=0.1)
    
    # ========== 10. ANALYZE RESULTS ==========
    print("\n" + "=" * 70)
    print(results.get_summary())
    print("=" * 70)
    
    t = results.time
    DO_pct = results.DO / DO_SAT_AIR * 100
    feeding = (t > t_feed_start + 0.5) & (t < t_feed_stop)
    
    print("\nTime course:")
    print(f"  {'t (h)':>6} {'X (g/L)':>8} {'S (g/L)':>8} {'DO (%)':>7} {'N (rpm)':>8} "
          f"{'OUR':>6} {'RQ':>5} {'mu (1/h)':>9} {'F (mL/h)':>9}")
    for t_show in [2, 4, 6.9, 7.5, 9, 12, 15, 18, 21.9, 22.5, 24, 26]:
        i = np.argmin(np.abs(t - t_show))
        print(f"  {t[i]:6.1f} {results.X[i]:8.2f} {round(results.S_carbon[i], 3) + 0.0:8.3f} {DO_pct[i]:7.1f} "
              f"{results.N[i]:8.0f} {results.OUR[i]:6.1f} {results.CER[i]/results.OUR[i]:5.2f} "
              f"{results.mu[i]:9.3f} {results.F_feed[i]*1000:9.2f}")
    
    print("\nProcess performance:")
    
    # Productivity (at peak biomass)
    i_peak = np.argmax(results.X * results.V)
    total_biomass = results.X[i_peak] * results.V[i_peak]
    productivity = results.X[i_peak] / t[i_peak]
    
    print(f"  Peak biomass: {total_biomass:.1f} g ({results.X[i_peak]:.1f} g/L) at {t[i_peak]:.1f} h")
    print(f"  Volumetric productivity to peak: {productivity:.2f} g/L/h")
    
    # Yield
    fed_substrate = float(np.sum(0.5 * (results.F_feed[1:] + results.F_feed[:-1]) * np.diff(t))
                          * feed_composition.S_carbon)
    total_substrate = (initial_state.S_carbon * initial_state.V + fed_substrate
                       - results.S_carbon[-1] * results.V[-1])
    biomass_formed = results.X[-1] * results.V[-1] - initial_state.X * initial_state.V
    
    print(f"  Total substrate consumed: {total_substrate:.1f} g")
    print(f"  Overall yield at end of run: {biomass_formed/total_substrate:.3f} g biomass / g substrate")
    
    print("\nSignatures:")
    after_batch = (t > t_feed_start - 1.0) & (t < t_feed_start + 0.3)
    after_stop = t > t_feed_stop
    print(f"  DO spike at batch glycerol exhaustion: {DO_pct[after_batch].max():.0f}% of saturation")
    print(f"  DO spike at feed stop:                 {DO_pct[after_stop].max():.0f}% of saturation")
    print(f"  Glycerol during feeding:               {results.S_carbon[feeding].max()*1000:.0f} mg/L at most")
    print(f"  DO during feeding:                     {DO_pct[feeding].min():.0f}-{DO_pct[feeding].max():.0f}% of saturation")
    print(f"  Peak OUR:                              {results.OUR.max():.0f} mmol/L/h")
    growing = results.mu > 0.05
    starving = results.mu < 0
    print(f"  RQ while growing:                      {np.mean(results.CER[growing]/results.OUR[growing]):.3f}"
          f" (stoichiometric for growth alone: {cell_params.RQ:.3f})")
    print(f"  RQ while starving:                     {np.mean(results.CER[starving]/results.OUR[starving]):.3f}")
    
    # Oxygen limitation check
    if np.any(results.DO[feeding] < 0.01):
        t_limited = t[feeding][results.DO[feeding] < 0.01][0]
        print(f"  ⚠️  Oxygen limitation during feeding, first at t = {t_limited:.1f} h")
    
    print()
    print(format_balances(check_balances(results, cell_params, feed_strategy)))
    
    print("\n" + "=" * 70)
    print("Simulation complete!")
    print("=" * 70)
    
    return results


if __name__ == "__main__":
    results = main()
