"""
Example: DO-stat fed-batch simulation.

Demonstrates:
- Agitation and aeration held at maximum for the whole run
- Batch glucose chosen so that it runs out just as oxygen demand reaches
  what the reactor can transfer
- DO-stat feeding: after the batch, the feed pump alone holds DO at setpoint

Why fixed agitation: if a DO cascade and a DO-stat feed both act on the same
DO measurement they fight each other. Fixing the oxygen supply at its maximum
and letting the feed set the oxygen demand removes the conflict and runs the
reactor at its full transfer capacity.

What to look for in the output:
- Batch: DO falls from ~90% as biomass grows and reaches the setpoint region
  right at glucose exhaustion, then jumps to ~95%
- DO-stat: the feed pulls DO back to the setpoint and holds it there
- The feed rate declines slowly, because kLa falls as biomass and volume rise
"""

import numpy as np

import _bootstrap  # noqa: F401  (makes the package importable from a fresh clone)
from fedbatch_simulator import *
from fedbatch_simulator.kinetics import CellParameters
from fedbatch_simulator.mass_transfer import DynamicKLa
from fedbatch_simulator.thermodynamics import TemperatureController
from fedbatch_simulator.feed_strategies import ConstantFeed, DOStatFeed, PiecewiseFeed
from fedbatch_simulator.control import FixedAerationAgitation
from fedbatch_simulator.reactor import FedBatchReactor
from fedbatch_simulator.simulator import FedBatchSimulator

DO_SAT_AIR = 0.2187  # mmol/L at 37°C, 1 atm air


def main():
    print("=" * 70)
    print("DO-STAT FED-BATCH SIMULATION")
    print("=" * 70)
    print("\nStrategy: oxygen supply fixed at maximum, feed rate controlled by DO")
    print("  - DO above setpoint → the culture can take more substrate → feed more")
    print("  - DO below setpoint → the culture is overfed → feed less")
    
    # ========== CELL PARAMETERS ==========
    carbon_params = SubstrateParameters(
        name='carbon',
        Ks=0.1,
        Y_xs=0.5,
        ms=0.03
    )
    
    nitrogen_params = SubstrateParameters(
        name='nitrogen',
        Ks=0.05,
        Y_xs=10.0,
        ms=0.001
    )
    
    cell_params = CellParameters(
        mu_max=0.5,
        carbon_source=GLUCOSE,
        biomass_composition=STANDARD_BIOMASS,
        substrates={
            'carbon': carbon_params,
            'nitrogen': nitrogen_params
        },
        Y_xs=0.5,
        Ks_O2=0.003
    )
    
    print(f"\nCell parameters:")
    print(f"  μ_max: {cell_params.mu_max} 1/h")
    print(f"  Y_x/S: {cell_params.Y_xs} g/g")
    print(f"  Y_x/O2: {cell_params.Y_x_O2:.3f} g/g")
    
    # ========== REACTOR AND OXYGEN SUPPLY ==========
    reactor_config = LAB_STR_5L
    V0 = 2.0  # L
    
    N_fixed = reactor_config.N_max                       # rpm
    Q_gas_fixed = reactor_config.Q_gas_max * V0 * 60     # L/h
    
    do_control = FixedAerationAgitation(
        N=N_fixed,
        Q_gas=Q_gas_fixed,
        kLa_correlation=DynamicKLa(
            k_O2=reactor_config.k_O2,
            a=reactor_config.a,
            b=reactor_config.b,
            k_X=reactor_config.k_X
        )
    )
    
    print(f"\nOxygen supply (fixed):")
    print(f"  Agitation: {N_fixed:.0f} rpm, aeration: {Q_gas_fixed:.0f} L/h")
    
    # ========== INITIAL CONDITIONS ==========
    # 10 g/L of batch glucose gives about 5.3 g/L of biomass, whose oxygen
    # demand at μ_max (~50 mmol/L/h) is what this reactor transfers at the
    # DO setpoint. The batch therefore ends at the oxygen-transfer limit.
    initial_state = ReactorState(
        time=0.0,
        X=0.5,
        S_carbon=10.0,
        S_nitrogen=2.0,
        P=0.0,
        DO=0.2,
        DCO2=0.5,
        V=V0,
        T=37.0,
        N=N_fixed,
        pH=7.0
    )
    
    print(f"\nInitial conditions:")
    print(f"  X: {initial_state.X} g/L")
    print(f"  S_carbon: {initial_state.S_carbon} g/L (sized to end the batch at the oxygen limit)")
    print(f"  V: {initial_state.V} L")
    
    # ========== FEED STRATEGY: DO-STAT ==========
    feed_composition = FeedComposition(
        S_carbon=500.0,
        S_nitrogen=50.0,
        temperature=25.0,
        pH=5.0
    )
    
    t_dostat_start = 5.6  # h, shortly after the batch glucose runs out (~5.3 h)
    
    batch_feed = ConstantFeed(feed_composition, F_constant=0.0)
    
    do_stat_feed = DOStatFeed(
        feed_composition=feed_composition,
        DO_setpoint=0.06,  # mmol/L (~27% saturation)
        K_p=0.1,           # L/h per mmol/L; positive: high DO → more feed
        K_i=1.0,           # L/h per mmol/L per h
        F_min=0.0,
        F_max=0.03,        # L/h (15 g glucose/h)
        F_base=0.0
    )
    
    feed_strategy = PiecewiseFeed([
        (0.0, batch_feed),
        (t_dostat_start, do_stat_feed)
    ])
    
    print(f"\nFeed strategy:")
    print(f"  0-{t_dostat_start} h: Batch (F = 0)")
    print(f"  {t_dostat_start}+ h: DO-stat")
    print(f"    DO setpoint: {do_stat_feed.DO_setpoint} mmol/L "
          f"({do_stat_feed.DO_setpoint/DO_SAT_AIR*100:.0f}% saturation)")
    print(f"    K_p: {do_stat_feed.K_p}, K_i: {do_stat_feed.K_i}")
    print(f"    F range: {do_stat_feed.F_min*1000:.0f}-{do_stat_feed.F_max*1000:.0f} mL/h")
    
    # ========== TEMPERATURE CONTROL ==========
    temp_controller = TemperatureController(
        T_setpoint=37.0,
        K_p=10.0,
        K_i=0.5
    )
    
    # ========== CREATE AND RUN REACTOR ==========
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
    
    print("=" * 70)
    
    simulator = FedBatchSimulator(reactor)
    results = simulator.simulate(t_end=24.0, max_step=0.1)
    
    # ========== ANALYZE RESULTS ==========
    print("\n" + "=" * 70)
    print(results.get_summary())
    print("=" * 70)
    
    t = results.time
    DO_pct = results.DO / DO_SAT_AIR * 100
    
    print("\nTime course:")
    print(f"  {'t (h)':>6} {'X (g/L)':>8} {'S (g/L)':>8} {'DO (%)':>7} {'OUR':>6} "
          f"{'kLa (1/h)':>10} {'mu (1/h)':>9} {'F (mL/h)':>9}")
    for t_show in [1, 3, 5, 5.4, 5.7, 6, 7, 9, 12, 16, 20, 24]:
        i = np.argmin(np.abs(t - t_show))
        print(f"  {t[i]:6.1f} {results.X[i]:8.2f} {round(results.S_carbon[i], 3) + 0.0:8.3f} {DO_pct[i]:7.1f} "
              f"{results.OUR[i]:6.1f} {results.kLa[i]:10.0f} {results.mu[i]:9.3f} {results.F_feed[i]*1000:9.2f}")
    
    # DO-stat performance, once the controller has settled
    settled = t >= t_dostat_start + 1.0
    batch = t < t_dostat_start
    
    print("\nBatch phase:")
    print(f"  Lowest DO, at glucose exhaustion: {DO_pct[batch].min():.0f}% of saturation")
    print(f"  DO spike after exhaustion:        {DO_pct[batch].max():.0f}% of saturation")
    
    print(f"\nDO-stat performance (t > {t_dostat_start + 1.0:.1f} h):")
    print(f"  DO range: {DO_pct[settled].min():.1f}-{DO_pct[settled].max():.1f}% of saturation "
          f"(setpoint {do_stat_feed.DO_setpoint/DO_SAT_AIR*100:.1f}%)")
    print(f"  Feed rate range: {results.F_feed[settled].min()*1000:.1f}-{results.F_feed[settled].max()*1000:.1f} mL/h")
    print(f"  Glucose: {results.S_carbon[settled].max()*1000:.0f} mg/L at most")
    
    # Final performance
    final_X = results.X[-1]
    final_V = results.V[-1]
    final_t = results.time[-1]
    total_biomass = final_X * final_V
    
    print(f"\nProcess performance:")
    print(f"  Total biomass: {total_biomass:.1f} g ({final_X:.1f} g/L)")
    print(f"  Volumetric productivity: {final_X / final_t:.2f} g/L/h")
    
    print()
    print(format_balances(check_balances(results, cell_params, feed_strategy)))
    
    print("\n" + "=" * 70)
    print("DO-stat simulation complete!")
    print("=" * 70)
    
    return results


if __name__ == "__main__":
    results = main()
