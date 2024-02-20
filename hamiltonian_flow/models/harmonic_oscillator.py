from functools import cached_property
from typing import Dict, List, Union

import numpy as np
import pandas as pd
from pydantic import BaseModel, computed_field


class HarmonicOscillatorParams(BaseModel):
    """The params for the harmonic oscillator"""

    omega: float
    amplitude: float = 1.0
    phi: float = 0.0

    @computed_field  # type: ignore[misc]
    @cached_property
    def period(self) -> float:
        return 2 * np.pi / self.omega

    @computed_field  # type: ignore[misc]
    @cached_property
    def frequency(self) -> float:
        return 1 / self.period


class HarmonicOscillators:
    r"""Generate time series data for a harmonic oscillator.

    !!! example "A Simple Harmonic Oscillator"

        In a one dimensional world, a mass $m$, driven by a force $F=-kx$, is described as

        $$
        \begin{align}
        F &= - k x \\
        F &= m a
        \end{align}
        $$

        The mass behaves like a simple harmonic oscillator.

    In general, the solution to a simple harmonic oscillator is

    $$
    x(t) = A \cos(\omega t + \phi),
    $$

    where $\omega$ is the angular frequency, $\phi$ is the initial phase, and $A$ is the amplitude.

    :param params: all the params that defines the harmonic oscillator.
    """

    def __init__(self, params: Dict[str, float]):
        self.params = HarmonicOscillatorParams.model_validate(params)

    @cached_property
    def model_params(self) -> Dict[str, float]:
        """model params defined as a dictionary."""
        return self.params.model_dump()

    def _x(self, t: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
        return self.params.amplitude * np.cos(self.params.omega * t + self.params.phi)

    def __call__(
        self, num_periods: int, num_samples_per_period: int
    ) -> Dict[str, List[float]]:
        """Generate time series data for the harmonic oscillator.

        Returns a list of floats representing the displacement at each time step.

        :param num_periods: Number of periods to generate.
        :param num_samples_per_period: Number of samples per period.
        """
        time_step = self.params.period / num_samples_per_period
        time_steps = np.arange(0, num_periods * num_samples_per_period) * time_step

        return pd.DataFrame({"t": time_steps, "x": self._x(time_steps)})
