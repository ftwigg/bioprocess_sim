"""
Physical-sanity tests for the fed-batch simulator.

These check behaviour, not implementation: nothing goes negative, carbon and
electrons are conserved, the results are states the model actually passed
through, and the run shows the signatures a real fermentation shows.

Run with pytest, or directly:  python tests/test_physical_sanity.py
"""
import contextlib
import importlib
import io
import pathlib
import re
import sys
import warnings

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'examples'))
import _bootstrap  # noqa: E402,F401

from fedbatch_simulator import *  # noqa: E402,F401,F403
from fedbatch_simulator.kinetics import CellParameters, GrowthKinetics  # noqa: E402
from fedbatch_simulator.mass_transfer import DynamicKLa  # noqa: E402
from fedbatch_simulator.thermodynamics import TemperatureController  # noqa: E402
from fedbatch_simulator.feed_strategies import (  # noqa: E402
    ConstantFeed, ExponentialFeed, DOStatFeed, PiecewiseFeed)
from fedbatch_simulator.control import (  # noqa: E402
    PICascade, SimplifiedCascade, FixedAerationAgitation)
from fedbatch_simulator.reactor import FedBatchReactor  # noqa: E402
from fedbatch_simulator.simulator import FedBatchSimulator  # noqa: E402

DO_SAT = 0.2187     # mmol/L, air at 37°C
DO_SP = 0.06        # mmol/L
TINY = 1e-6         # solver round-off allowance on a concentration
CFG = LAB_STR_5L
V0 = 2.0


# --------------------------------------------------------------------------
# Scenario builders
# --------------------------------------------------------------------------

def make_cell(carbon=GLUCOSE, mu_max=0.5, Ks=0.1, Y=0.5, ms=0.03):
    return CellParameters(
        mu_max=mu_max, carbon_source=carbon, biomass_composition=STANDARD_BIOMASS,
        substrates={'carbon': SubstrateParameters(name='carbon', Ks=Ks, Y_xs=Y, ms=ms),
                    'nitrogen': SubstrateParameters(name='nitrogen', Ks=0.05, Y_xs=10.0, ms=0.001)},
        Y_xs=Y)


def kla_model():
    return DynamicKLa(k_O2=CFG.k_O2, a=CFG.a, b=CFG.b, k_X=CFG.k_X)


def cascade(cls=PICascade):
    return cls(DO_setpoint=DO_SP, N_min=CFG.N_min, N_max=CFG.N_max,
               Q_gas_min=CFG.Q_gas_min * V0 * 60, Q_gas_max=CFG.Q_gas_max * V0 * 60,
               kLa_correlation=kla_model())


def sized_feed(t_start=5.5, t_stop=22.0, mu_set=0.12, cap_gph=4.5, S_feed=500.0):
    fc = FeedComposition(S_carbon=S_feed, S_nitrogen=50.0, temperature=25.0)
    F0 = ExponentialFeed.initial_rate(mu_set, 5.3, V0, S_feed, 0.5, 0.03)
    ef = ExponentialFeed(fc, F0=F0, mu_set=mu_set, F_max=cap_gph / S_feed)
    ef.t_start = t_start
    segments = [(0.0, ConstantFeed(fc, 0.0)), (t_start, ef)]
    if t_stop is not None:
        segments.append((t_stop, ConstantFeed(fc, 0.0)))
    return PiecewiseFeed(segments)


def overfeed():
    """The feed profile the examples originally shipped with."""
    fc = FeedComposition(S_carbon=500.0, S_nitrogen=50.0, temperature=25.0)
    ef = ExponentialFeed(fc, F0=0.01, mu_set=0.2, F_max=0.5)
    ef.t_start = 10.0
    return PiecewiseFeed([(0.0, ConstantFeed(fc, 0.0)), (10.0, ef)])


def simulate(cell, feed, do_control, t_end, S0=10.0, N0=300.0, **kw):
    state = ReactorState(X=0.5, S_carbon=S0, S_nitrogen=2.0, DO=0.2, DCO2=0.5,
                         V=V0, T=37.0, N=N0, pH=7.0)
    reactor = FedBatchReactor(
        cell_params=cell, reactor_config=CFG, initial_state=state,
        feed_strategy=feed, do_control=do_control,
        temp_controller=TemperatureController(T_setpoint=37.0, K_p=10.0, K_i=0.5))
    with contextlib.redirect_stdout(io.StringIO()):
        results = FedBatchSimulator(reactor).simulate(t_end=t_end, **kw)
    return results, reactor


_cache = {}


def reference_run():
    """Batch → sized exponential feed → feed off at 22 h, PI cascade."""
    if 'ref' not in _cache:
        cell, feed = make_cell(), sized_feed()
        results, reactor = simulate(cell, feed, cascade(), 26.0)
        _cache['ref'] = (results, reactor, cell, feed)
    return _cache['ref']


def assert_non_negative(results):
    for name in ['X', 'S_carbon', 'S_nitrogen', 'P', 'DO', 'DCO2']:
        lowest = float(getattr(results, name).min())
        assert lowest > -TINY, f"{name} went negative: {lowest:.3e}"


# --------------------------------------------------------------------------
# Conservation and non-negativity
# --------------------------------------------------------------------------

def test_no_state_goes_negative_through_starvation():
    results, *_ = reference_run()
    assert_non_negative(results)


def test_carbon_and_electrons_are_conserved():
    results, _, cell, feed = reference_run()
    report = check_balances(results, cell, feed)
    assert abs(report['carbon_closure_pct'] - 100.0) < 0.1, report
    assert abs(report['electron_closure_pct'] - 100.0) < 0.1, report


def test_overfeeding_stays_physical():
    """Feed far beyond the oxygen ceiling: glucose piles up and DO collapses,
    but nothing goes negative and the balances still close."""
    cell, feed = make_cell(), overfeed()
    results, _ = simulate(cell, feed, cascade(), 24.0)
    assert_non_negative(results)
    report = check_balances(results, cell, feed)
    assert abs(report['carbon_closure_pct'] - 100.0) < 0.1, report
    assert abs(report['electron_closure_pct'] - 100.0) < 0.1, report
    late = results.time > 18.0
    assert results.S_carbon[-1] > 20.0, "glucose should accumulate when overfed"
    assert results.DO[late].max() / DO_SAT < 0.05, "DO should be near zero when overfed"
    assert np.allclose(results.OUR[late], results.OTR[late], rtol=0.02), \
        "uptake should be pinned to the transfer rate at the kLa ceiling"


def test_glycerol_run_conserves_and_has_lower_rq():
    cell = make_cell(carbon=GLYCEROL, mu_max=0.4, Ks=0.15)
    feed = sized_feed(t_start=7.0, mu_set=0.10, cap_gph=3.0, S_feed=600.0)
    results, _ = simulate(cell, feed, cascade(), 26.0)
    assert_non_negative(results)
    report = check_balances(results, cell, feed)
    assert abs(report['carbon_closure_pct'] - 100.0) < 0.1, report
    assert abs(report['electron_closure_pct'] - 100.0) < 0.1, report
    growing = results.mu > 0.05
    rq = results.CER[growing] / results.OUR[growing]
    assert 0.70 < rq.mean() < 0.80, rq.mean()


# --------------------------------------------------------------------------
# Results are what the solver accepted
# --------------------------------------------------------------------------

def test_results_are_accepted_steps():
    results, reactor, *_ = reference_run()
    assert np.all(np.diff(results.time) >= 0), "time must not run backwards"
    assert results.N.max() <= CFG.N_max * 1.001, results.N.max()
    assert results.N.min() >= CFG.N_min * 0.999 or results.N[0] == 300.0
    assert len(reactor.history['time']) == len(results.time)


def test_right_hand_side_is_stateless():
    """Evaluating the model elsewhere must not change what it returns here."""
    _, reactor, *_ = reference_run()
    y = reactor.get_state_vector()
    first = reactor.derivatives(3.0, y)
    for t_other in [25.0, 0.1, 12.0, 3.0001]:
        reactor.derivatives(t_other, y * 1.01)
    again = reactor.derivatives(3.0, y)
    assert np.array_equal(first, again)


def test_answer_does_not_depend_on_solver():
    cell, feed = make_cell(), sized_feed()
    finals, T_max = {}, {}
    for method in ['BDF', 'LSODA', 'Radau']:
        results, _ = simulate(cell, feed, cascade(), 26.0, method=method)
        finals[method] = results.X[-1] * results.V[-1]
        T_max[method] = results.T.max()
    reference = finals['BDF']
    for method, value in finals.items():
        assert abs(value - reference) / reference < 0.005, finals
        assert T_max[method] < 37.5, T_max


def test_volume_limit_is_located_exactly():
    fc = FeedComposition(S_carbon=500.0, S_nitrogen=50.0)
    results, _ = simulate(make_cell(), ConstantFeed(fc, 0.5), cascade(), 10.0)
    assert abs(results.V[-1] - CFG.V_working_max) < 1e-6
    assert abs(results.time[-1] - 3.0) < 1e-4


# --------------------------------------------------------------------------
# Fermentation signatures
# --------------------------------------------------------------------------

def test_do_spikes_at_substrate_exhaustion_and_recovers():
    results, *_ = reference_run()
    t, do_pct = results.time, results.DO / DO_SAT * 100
    setpoint = DO_SP / DO_SAT * 100
    t_exhausted = t[np.argmax(results.S_carbon < 0.01)]
    window = (t >= t_exhausted) & (t <= t_exhausted + 0.25)
    assert do_pct[window].max() > setpoint + 20.0, do_pct[window].max()
    before = (t > 1.0) & (t < t_exhausted - 0.3)
    assert np.all(np.abs(do_pct[before] - setpoint) < 5.0), "DO should track setpoint while growing"
    # and again when the feed stops
    after_stop = (t > 22.0) & (t < 22.3)
    assert do_pct[after_stop].max() > setpoint + 15.0
    settled = t > 23.5
    assert np.all(np.abs(do_pct[settled] - setpoint) < 2.0)


def test_ideal_cascade_shows_no_spike():
    """SimplifiedCascade is the idealised reference: DO never leaves setpoint."""
    fc = FeedComposition(S_carbon=500.0, S_nitrogen=50.0)
    results, _ = simulate(make_cell(), ConstantFeed(fc, 0.0), cascade(SimplifiedCascade), 8.0)
    late = results.time > 1.0
    assert np.ptp(results.DO[late]) / DO_SAT * 100 < 3.0


def test_feed_phase_is_carbon_limited_with_oxygen_in_hand():
    results, *_ = reference_run()
    t = results.time
    feeding = (t > 6.0) & (t < 22.0)
    assert results.S_carbon[feeding].max() < 0.2, results.S_carbon[feeding].max()
    assert results.DO[feeding].min() / DO_SAT > 0.15
    exponential = (t > 6.5) & (t < 9.0)
    assert np.allclose(results.mu[exponential], 0.12, atol=0.005), results.mu[exponential]


def test_biomass_decays_slowly_when_starved():
    results, _, cell, _ = reference_run()
    starved = results.time > 23.0
    expected = -cell.substrates['carbon'].Y_xs * cell.substrates['carbon'].ms
    assert np.allclose(results.mu[starved], expected, atol=1e-3), results.mu[starved]
    assert results.X[-1] < results.X[results.time > 22.0][0]
    assert np.all(results.OUR[starved] > 0), "endogenous respiration should continue"


def test_respiration_tracks_growth_with_rq_near_one_on_glucose():
    results, *_ = reference_run()
    growing = results.mu > 0.05
    rq = results.CER[growing] / results.OUR[growing]
    assert 1.0 < rq.min() and rq.max() < 1.1, (rq.min(), rq.max())
    batch = (results.time > 0.5) & (results.time < 4.5)
    specific = results.OUR[batch] / results.X[batch]
    assert 8.0 < specific.mean() < 20.0, "mmol O2/g/h at near-max growth"
    corr = np.corrcoef(results.OUR[batch], results.mu[batch] * results.X[batch])[0, 1]
    assert corr > 0.999


def test_metabolic_heat_is_460_kj_per_mol_oxygen():
    results, reactor, *_ = reference_run()
    i = np.argmax(results.OUR)
    watts = reactor.heat_balance.metabolic_heat(results.OUR[i], results.V[i])
    mol_per_h = results.OUR[i] * results.V[i] / 1000.0
    assert abs(watts * 3.6 / mol_per_h - 460.0) < 1e-6
    assert results.T.max() < 37.5 and results.T.min() > 36.5


def test_do_stat_holds_setpoint_with_feed_alone():
    fc = FeedComposition(S_carbon=500.0, S_nitrogen=50.0, temperature=25.0)
    do_stat = DOStatFeed(fc, DO_setpoint=DO_SP, K_p=0.1, K_i=1.0, F_min=0.0, F_max=0.03, F_base=0.0)
    feed = PiecewiseFeed([(0.0, ConstantFeed(fc, 0.0)), (5.6, do_stat)])
    cell = make_cell()
    supply = FixedAerationAgitation(N=CFG.N_max, Q_gas=CFG.Q_gas_max * V0 * 60, kLa_correlation=kla_model())
    results, _ = simulate(cell, feed, supply, 24.0, N0=CFG.N_max)
    assert_non_negative(results)
    t, do_pct = results.time, results.DO / DO_SAT * 100
    batch = t < 5.6
    assert do_pct[batch].max() > 90.0, "uncontrolled DO should jump at exhaustion"
    settled = t > 6.6
    assert np.all(np.abs(do_pct[settled] - DO_SP / DO_SAT * 100) < 1.0)
    assert results.S_carbon[settled].max() < 0.3
    report = check_balances(results, cell, feed)
    assert abs(report['carbon_closure_pct'] - 100.0) < 0.1, report


# --------------------------------------------------------------------------
# Kinetics
# --------------------------------------------------------------------------

def test_kinetics_limits():
    cell = make_cell()
    kin = GrowthKinetics(cell)
    unlimited = kin.specific_rates({'carbon': 1e6, 'nitrogen': 1e6}, 1e6)
    assert abs(unlimited['mu'] - cell.mu_max) < 1e-3
    starved = kin.specific_rates({'carbon': 0.0, 'nitrogen': 2.0}, 0.2)
    assert starved['q_substrate']['carbon'] == 0.0
    assert starved['mu'] < 0 and starved['q_O2'] > 0 and starved['q_CO2'] > 0
    anoxic = kin.specific_rates({'carbon': 10.0, 'nitrogen': 2.0}, 0.0)
    assert anoxic['q_O2'] == 0.0 and anoxic['q_substrate']['carbon'] == 0.0


def test_conflicting_oxygen_parameters_warn():
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always')
        cell = CellParameters(
            mu_max=0.5, carbon_source=GLUCOSE, biomass_composition=STANDARD_BIOMASS,
            substrates={'carbon': SubstrateParameters(name='carbon', Ks=0.1, Y_xs=0.5, ms=0.03)},
            Y_xs=0.5, m_O2=0.01)
    assert any('m_O2' in str(w.message) for w in caught)
    assert abs(cell.m_O2 - 0.032) < 1e-6


def test_impossible_yield_is_rejected():
    try:
        make_cell(Y=0.9)
    except ValueError:
        return
    raise AssertionError("a yield above the electron-balance limit should be rejected")


# --------------------------------------------------------------------------
# Shipped examples and documentation
# --------------------------------------------------------------------------

def _run_example(name):
    module = importlib.import_module(name)
    with contextlib.redirect_stdout(io.StringIO()):
        return module.main()


def test_shipped_examples_run_and_stay_carbon_limited():
    for name, feed_window in [('exponential_fedbatch', (6.0, 22.0)),
                              ('glycerol_fedbatch', (7.5, 22.0)),
                              ('do_stat_fedbatch', (6.6, 24.0))]:
        results = _run_example(name)
        assert_non_negative(results)
        feeding = (results.time > feed_window[0]) & (results.time < feed_window[1])
        assert results.S_carbon[feeding].max() < 0.3, (name, results.S_carbon[feeding].max())
        assert results.DO[feeding].min() / DO_SAT > 0.15, (name, results.DO[feeding].min())
        assert results.X.max() > 10.0, name


def test_quickstart_minimal_example_runs():
    text = (ROOT / 'QUICKSTART.md').read_text()
    section = text.split('## Complete Minimal Example', 1)[1]
    code = re.search(r"```python\n(.*?)```", section, re.S).group(1)
    namespace = {}
    with contextlib.redirect_stdout(io.StringIO()):
        exec(code, namespace)
    assert_non_negative(namespace['results'])
    assert namespace['results'].X[-1] > 10.0


if __name__ == '__main__':
    tests = [(name, fn) for name, fn in sorted(globals().items())
             if name.startswith('test_') and callable(fn)]
    failed = 0
    for name, fn in tests:
        try:
            fn()
            print(f"PASS  {name}")
        except Exception as exc:  # noqa: BLE001
            failed += 1
            print(f"FAIL  {name}: {type(exc).__name__}: {exc}")
    print(f"\n{len(tests) - failed} passed, {failed} failed")
    sys.exit(1 if failed else 0)
