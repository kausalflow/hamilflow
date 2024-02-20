from functools import cached_property
from typing import Dict, List, Union

import numpy as np
import pandas as pd
from pydantic import BaseModel, computed_field


class HarmonicOscillatorParams(BaseModel):
    """The params for the harmonic oscillator

    :param omega: angular frequency of the harmonic oscillator
    :param amplitude: the amplitude of the oscillation
    :param phi: initial phase
    """

    omega: float
    amplitude: float = 1.0
    phi: float = 0.0

    @computed_field  # type: ignore[misc]
    @cached_property
    def period(self) -> float:
        """period of the oscillator"""
        return 2 * np.pi / self.omega

    @computed_field  # type: ignore[misc]
    @cached_property
    def frequency(self) -> float:
        """frequency of the oscillator"""
        return 1 / self.period


class HarmonicOscillator:
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
        self, n_periods: int, n_samples_per_period: int
    ) -> Dict[str, List[float]]:
        """Generate time series data for the harmonic oscillator.

        Returns a list of floats representing the displacement at each time step.

        :param n_periods: Number of periods to generate.
        :param n_samples_per_period: Number of samples per period.
        """
        time_step = self.params.period / n_samples_per_period
        time_steps = np.arange(0, n_periods * n_samples_per_period) * time_step

        return pd.DataFrame({"t": time_steps, "x": self._x(time_steps)})
