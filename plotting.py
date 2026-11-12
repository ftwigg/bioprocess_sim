"""
Plots for fed-batch simulation results.

Requires matplotlib, which the simulator itself does not need; import this
module explicitly:

    from fedbatch_simulator.plotting import plot_fedbatch_results, plot_comparison

Each panel shows one quantity on its own axis, so nothing shares a y-scale
with a different unit.
"""

from typing import Dict, Optional
import numpy as np
import matplotlib.pyplot as plt

from .base_models import SimulationResults
from .mass_transfer import HenryConstant

# Colours: series hues in a fixed order, neutral ink for text and guides
SERIES = ['#2a78d6', '#eb6834', '#1baf7a']
INK = '#0b0b0b'
INK_MUTED = '#52514e'
GRID = '#e4e3df'
SURFACE = '#fcfcfb'


def do_saturation(T: float = 37.0) -> float:
    """Dissolved oxygen in equilibrium with air at 1 atm (mmol/L)."""
    return HenryConstant.get_H_O2(T) * 0.21 * 1000.0


def _style(ax, ylabel: str):
    ax.set_facecolor(SURFACE)
    ax.set_title(ylabel, loc='left', fontsize=10, color=INK, pad=6)
    ax.grid(True, color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    for side in ('top', 'right'):
        ax.spines[side].set_visible(False)
    for side in ('left', 'bottom'):
        ax.spines[side].set_color(GRID)
    ax.tick_params(colors=INK_MUTED, length=0, labelsize=9)


def _marks(ax, marks: Optional[Dict[str, float]], label: bool):
    if not marks:
        return
    for name, t_mark in marks.items():
        ax.axvline(t_mark, color=INK_MUTED, linewidth=0.8, linestyle=(0, (2, 3)))
        if label:
            ax.annotate(name, xy=(t_mark, 1.0), xycoords=('data', 'axes fraction'),
                        xytext=(3, -3), textcoords='offset points', rotation=90,
                        va='top', ha='left', fontsize=8, color=INK_MUTED)


def _legend(ax):
    legend = ax.legend(frameon=False, fontsize=9, loc='best', handlelength=1.6)
    for text in legend.get_texts():
        text.set_color(INK_MUTED)


def _rq(results: SimulationResults) -> np.ndarray:
    """CER/OUR, blanked where respiration is too small for the ratio to mean much."""
    OUR = np.asarray(results.OUR, dtype=float)
    rq = np.full_like(OUR, np.nan)
    ok = OUR > 0.01 * np.nanmax(OUR)
    rq[ok] = results.CER[ok] / OUR[ok]
    return rq


def plot_fedbatch_results(results: SimulationResults,
                          title: str = "Fed-batch simulation",
                          DO_setpoint: Optional[float] = None,
                          marks: Optional[Dict[str, float]] = None,
                          figsize=(15, 11.5)):
    """
    Twelve-panel overview of one run.
    
    Args:
        results: Output of FedBatchSimulator.simulate
        title: Figure title
        DO_setpoint: DO setpoint (mmol/L), drawn as a reference line if given
        marks: Named times to mark on every panel, e.g. {'feed on': 5.5}
        figsize: Figure size in inches
    
    Returns:
        The matplotlib Figure
    """
    r = results
    t = r.time
    C_sat = do_saturation(float(np.mean(r.T)))
    blue, orange = SERIES[0], SERIES[1]
    
    fig, axes = plt.subplots(4, 3, figsize=figsize, sharex=True, facecolor=SURFACE)
    ax = axes.ravel()
    
    def line(a, y, color=blue, **kw):
        a.plot(t, y, color=color, linewidth=2, solid_capstyle='round', **kw)
    
    # Row 1: what the culture is doing
    _style(ax[0], 'Biomass, g/L')
    line(ax[0], r.X)
    
    _style(ax[1], 'Carbon substrate, g/L (log scale; below 1 mg/L not shown)')
    S = np.where(r.S_carbon >= 1e-3, r.S_carbon, np.nan)
    line(ax[1], S)
    ax[1].set_yscale('log')
    ax[1].set_ylim(bottom=1e-3)
    
    _style(ax[2], 'Specific growth rate, 1/h')
    line(ax[2], r.mu)
    ax[2].axhline(0.0, color=INK_MUTED, linewidth=0.8)
    
    # Row 2: oxygen
    _style(ax[3], 'Dissolved oxygen, % of air saturation')
    line(ax[3], r.DO / C_sat * 100.0)
    ax[3].set_ylim(-3, 103)
    if DO_setpoint is not None:
        sp = DO_setpoint / C_sat * 100.0
        ax[3].axhline(sp, color=INK_MUTED, linewidth=0.8, linestyle=(0, (4, 3)))
        ax[3].annotate(f'setpoint {sp:.0f}%', xy=(1.0, sp), xycoords=('axes fraction', 'data'),
                       xytext=(-3, 4), textcoords='offset points', ha='right',
                       fontsize=8, color=INK_MUTED)
    
    _style(ax[4], 'Oxygen uptake and CO2 evolution, mmol/L/h')
    line(ax[4], r.OUR, color=blue, label='OUR')
    line(ax[4], r.CER, color=orange, label='CER')
    _legend(ax[4])
    
    _style(ax[5], 'Respiratory quotient, CER/OUR')
    line(ax[5], _rq(r))
    
    # Row 3: feeding and transfer
    _style(ax[6], 'Feed rate, mL/h')
    line(ax[6], r.F_feed * 1000.0)
    
    _style(ax[7], 'Volume, L')
    line(ax[7], r.V)
    
    _style(ax[8], 'kLa, 1/h')
    line(ax[8], r.kLa)
    
    # Row 4: actuators and temperature
    _style(ax[9], 'Agitation, rpm')
    line(ax[9], r.N)
    
    _style(ax[10], 'Gas flow, L/h')
    line(ax[10], r.Q_gas)
    
    _style(ax[11], 'Temperature, °C')
    line(ax[11], r.T, color=blue, label='Broth')
    line(ax[11], r.T_jacket, color=orange, label='Jacket')
    _legend(ax[11])
    
    for i, a in enumerate(ax):
        _marks(a, marks, label=(i < 3))
        if i >= 9:
            a.set_xlabel('Time, h', color=INK_MUTED, fontsize=9)
    
    fig.suptitle(title, x=0.01, ha='left', fontsize=14, color=INK)
    fig.tight_layout(rect=(0, 0, 1, 0.97), h_pad=1.4, w_pad=1.6)
    return fig


def plot_comparison(runs: Dict[str, SimulationResults],
                    title: str = "Comparison",
                    figsize=(15, 7.5)):
    """
    Overlay up to three runs on six shared panels.
    
    Args:
        runs: {label: SimulationResults}; colours follow the order given
        title: Figure title
        figsize: Figure size in inches
    
    Returns:
        The matplotlib Figure
    """
    if len(runs) > len(SERIES):
        raise ValueError(f"plot_comparison shows at most {len(SERIES)} runs; got {len(runs)}")
    
    panels = [
        ('Biomass, g/L', lambda r: r.X),
        ('Carbon substrate, g/L (log scale; below 1 mg/L not shown)',
         lambda r: np.where(r.S_carbon >= 1e-3, r.S_carbon, np.nan)),
        ('Dissolved oxygen, % of air saturation',
         lambda r: r.DO / do_saturation(float(np.mean(r.T))) * 100.0),
        ('Feed rate, mL/h', lambda r: r.F_feed * 1000.0),
        ('Oxygen uptake rate, mmol/L/h', lambda r: r.OUR),
        ('Respiratory quotient, CER/OUR', _rq),
    ]
    
    fig, axes = plt.subplots(2, 3, figsize=figsize, sharex=True, facecolor=SURFACE)
    for a, (label, value) in zip(axes.ravel(), panels):
        _style(a, label)
        for color, (name, r) in zip(SERIES, runs.items()):
            a.plot(r.time, value(r), color=color, linewidth=2, solid_capstyle='round', label=name)
        if label.startswith('Carbon substrate'):
            a.set_yscale('log')
            a.set_ylim(bottom=1e-3)
        _legend(a)
    for a in axes[-1]:
        a.set_xlabel('Time, h', color=INK_MUTED, fontsize=9)
    
    fig.suptitle(title, x=0.01, ha='left', fontsize=14, color=INK)
    fig.tight_layout(rect=(0, 0, 1, 0.96), h_pad=1.4, w_pad=1.6)
    return fig
