import math
from functools import cached_property
from typing import Dict, List


class HarmonicOscillators:
    """Class for generating time series data for a harmonic oscillator,

    $$
    \mathbf F = - k\mathbf x
    $$



    :param length: Length of the pendulum.
    :param gravity: Acceleration due to gravity.
    """

    def __init__(self, length: float, gravity: float = 9.81) -> None:
        self.length = length
        self.gravity = gravity

    @cached_property
    def period(self) -> float:
        """Calculate the period of the pendulum."""
        return 2 * math.pi * math.sqrt(self.length / self.gravity)

    def __call__(
        self, num_periods: int, num_samples_per_period: int, initial_angle: float = 0.1
    ) -> Dict[str, List[float]]:
        """Generate time series data for the pendulum.

        Returns a list of floats representing the angle
        of the pendulum at each time step.

        :param num_periods: Number of periods to generate.
        :param num_samples_per_period: Number of samples per period.
        :param initial_angle: Initial angle of the pendulum.
        """
        time_step = self.period / num_samples_per_period
        steps = []
        time_series = []
        for i in range(num_periods * num_samples_per_period):
            t = i * time_step
            # angle = self.length * math.cos(2 * math.pi * t / self.period)
            angle = initial_angle * math.cos(2 * math.pi * t / self.period)
            steps.append(t)
            time_series.append(angle)

        return {"t": steps, "theta": time_series}
