"""
Feed strategies for fed-batch operation.

Implements:
- Constant feed
- Exponential feed
- DO-stat feed
- Piecewise feed (batch → fed-batch transitions)
"""

from abc import ABC, abstractmethod
from typing import List, Tuple
import numpy as np

from .base_models import FeedComposition, ReactorState


class FeedStrategy(ABC):
    """Base class for feed strategies."""
    
    def __init__(self, feed_composition: FeedComposition):
        """
        Args:
            feed_composition: Composition of feed stream
        """
        self.feed_composition = feed_composition
    
    @abstractmethod
    def get_feed_rate(self, t: float, state: ReactorState) -> float:
        """
        Calculate feed rate.
        
        Args:
            t: Current time (h)
            state: Current reactor state
        
        Returns:
            Feed rate (L/h)
        """
        pass
    
    def get_composition(self, t: float) -> FeedComposition:
        """
        Get feed composition (can be time-varying in future).
        
        Args:
            t: Current time (h)
        
        Returns:
            Feed composition
        """
        return self.feed_composition


class ConstantFeed(FeedStrategy):
    """Constant feed rate."""
    
    def __init__(self, feed_composition: FeedComposition, F_constant: float):
        """
        Args:
            feed_composition: Feed composition
            F_constant: Constant feed rate (L/h)
        """
        super().__init__(feed_composition)
        self.F_constant = F_constant
    
    def get_feed_rate(self, t: float, state: ReactorState) -> float:
        return self.F_constant


class ExponentialFeed(FeedStrategy):
    """
    Exponential feed for constant specific growth rate.
    
    F(t) = F0 × exp(μ_set × t)
    
    This maintains constant μ if substrate is non-limiting.
    """
    
    def __init__(self,
                 feed_composition: FeedComposition,
                 F0: float,
                 mu_set: float,
                 F_max: float = float('inf')):
        """
        Args:
            feed_composition: Feed composition
            F0: Initial feed rate (L/h)
            mu_set: Setpoint specific growth rate (1/h)
            F_max: Maximum feed rate (L/h)
        """
        super().__init__(feed_composition)
        self.F0 = F0
        self.mu_set = mu_set
        self.F_max = F_max
        self.t_start = 0.0  # Can be set when feed starts
    
    def get_feed_rate(self, t: float, state: ReactorState) -> float:
        t_elapsed = t - self.t_start
        F = self.F0 * np.exp(self.mu_set * t_elapsed)
        F = min(F, self.F_max)
        return F


class DOStatFeed(FeedStrategy):
    """
    DO-stat feed: Adjust feed rate to maintain DO setpoint.
    
    Uses PI control:
    F = F_base + K_p × (DO - DO_setpoint) + K_i × integral(error)
    
    Set K_i = 0 for proportional-only control.
    """
    
    def __init__(self,
                 feed_composition: FeedComposition,
                 DO_setpoint: float,
                 K_p: float,
                 K_i: float = 0.0,
                 F_min: float = 0.0,
                 F_max: float = 1.0):
        """
        Args:
            feed_composition: Feed composition
            DO_setpoint: Target DO (mmol/L)
            K_p: Proportional gain (L/h per mmol/L)
            K_i: Integral gain (L/h per mmol/L per hour)
            F_min: Minimum feed rate (L/h)
            F_max: Maximum feed rate (L/h)
        """
        super().__init__(feed_composition)
        self.DO_setpoint = DO_setpoint
        self.K_p = K_p
        self.K_i = K_i
        self.F_min = F_min
        self.F_max = F_max
        self.F_base = (F_min + F_max) / 2  # Start at midpoint
        
        # Integral state
        self.integral = 0.0
        self.last_t = None
        self.last_error = 0.0
    
    def get_feed_rate(self, t: float, state: ReactorState) -> float:
        # Calculate error
        error = state.DO - self.DO_setpoint
        
        # Update integral (trapezoidal rule)
        if self.last_t is not None and self.K_i != 0.0:
            dt = t - self.last_t
            if dt > 0:
                # Trapezoidal integration
                self.integral += 0.5 * (error + self.last_error) * dt
                
                # Anti-windup: Reset integral if saturated
                F_test = self.F_base + self.K_p * error + self.K_i * self.integral
                if F_test > self.F_max or F_test < self.F_min:
                    # Don't accumulate integral when saturated
                    self.integral -= 0.5 * (error + self.last_error) * dt
        
        # PI control
        F = self.F_base + self.K_p * error + self.K_i * self.integral
        
        # Saturate
        F = max(self.F_min, min(F, self.F_max))
        
        # Update state for next call
        self.last_t = t
        self.last_error = error
        
        return F


class PiecewiseFeed(FeedStrategy):
    """
    Piecewise feed strategy for transitions (e.g., batch → fed-batch).
    
    Switches between different feed strategies at specified times.
    """
    
    def __init__(self, segments: List[Tuple[float, FeedStrategy]]):
        """
        Args:
            segments: List of (time, strategy) tuples
                     e.g., [(0, batch_feed), (10, exp_feed)]
        """
        if not segments:
            raise ValueError("Must provide at least one segment")
        
        # Sort by time
        self.segments = sorted(segments, key=lambda x: x[0])
        
        # Use composition from first strategy
        super().__init__(self.segments[0][1].feed_composition)
    
    def get_feed_rate(self, t: float, state: ReactorState) -> float:
        # Find active segment
        active_strategy = self.segments[0][1]
        
        for t_switch, strategy in self.segments:
            if t >= t_switch:
                active_strategy = strategy
            else:
                break
        
        return active_strategy.get_feed_rate(t, state)
    
    def get_composition(self, t: float) -> FeedComposition:
        # Find active segment
        active_strategy = self.segments[0][1]
        
        for t_switch, strategy in self.segments:
            if t >= t_switch:
                active_strategy = strategy
            else:
                break
        
        return active_strategy.get_composition(t)


# Example usage and testing
if __name__ == "__main__":
    print("=" * 60)
    print("FEED STRATEGIES VALIDATION")
    print("=" * 60)
    
    # Create feed composition
    feed_comp = FeedComposition(
        S_carbon=500.0,  # g/L concentrated glucose
        S_nitrogen=50.0,  # g/L
        temperature=25.0,
        pH=5.0
    )
    
    # Create dummy state
    state = ReactorState(
        time=0,
        X=5.0,
        S_carbon=1.0,
        S_nitrogen=0.5,
        DO=0.05,  # mmol/L
        V=2.0
    )
    
    # Test 1: Constant feed
    print("\n1. Constant Feed")
    print("-" * 60)
    const_feed = ConstantFeed(feed_comp, F_constant=0.1)
    
    for t in [0, 5, 10, 20]:
        F = const_feed.get_feed_rate(t, state)
        print(f"  t = {t:2.0f} h: F = {F:.3f} L/h")
    
    # Test 2: Exponential feed
    print("\n2. Exponential Feed")
    print("-" * 60)
    exp_feed = ExponentialFeed(feed_comp, F0=0.01, mu_set=0.2, F_max=0.5)
    
    for t in [0, 5, 10, 15, 20]:
        F = exp_feed.get_feed_rate(t, state)
        print(f"  t = {t:2.0f} h: F = {F:.3f} L/h")
    
    # Test 3: DO-stat feed
    print("\n3. DO-stat Feed")
    print("-" * 60)
    do_stat_feed = DOStatFeed(
        feed_comp,
        DO_setpoint=0.06,
        K_p=-1.0,  # Negative: higher DO → lower feed
        F_min=0.0,
        F_max=0.5
    )
    
    DO_values = [0.10, 0.08, 0.06, 0.04, 0.02]
    for DO in DO_values:
        state.DO = DO
        F = do_stat_feed.get_feed_rate(10.0, state)
        print(f"  DO = {DO:.2f} mmol/L: F = {F:.3f} L/h")
    
    # Test 4: Piecewise feed (batch → exponential)
    print("\n4. Piecewise Feed (Batch → Exponential)")
    print("-" * 60)
    
    batch_feed = ConstantFeed(feed_comp, F_constant=0.0)
    exp_feed2 = ExponentialFeed(feed_comp, F0=0.01, mu_set=0.2, F_max=0.5)
    exp_feed2.t_start = 10.0  # Start exponential at t=10
    
    piecewise_feed = PiecewiseFeed([
        (0.0, batch_feed),
        (10.0, exp_feed2)
    ])
    
    for t in [0, 5, 10, 12, 15, 20]:
        F = piecewise_feed.get_feed_rate(t, state)
        phase = "Batch" if t < 10 else "Fed-batch"
        print(f"  t = {t:2.0f} h ({phase:10s}): F = {F:.3f} L/h")
    
    print("=" * 60)
