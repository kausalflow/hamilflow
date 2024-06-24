from functools import cached_property
from typing import Any, Sequence

import numpy as np
import pandas as pd
from numpy.typing import ArrayLike
from pandas.api.types import is_scalar
from scipy.fft import ifft

from ..d0.complex_harmonic_oscillator import SimpleComplexHarmonicOscillator


class HarmonicOscillatorsChain:
    r"""Generate time series data for a coupled harmonic oscillator chain
    with periodic boundary condition.
    """

    def __init__(
        self,
        omega: float,
        half_initial_conditions: Sequence[dict[str, complex | float | int]],
        odd_dof: bool,
    ) -> None:
        self.omega = omega
        self.dof = len(half_initial_conditions) + odd_dof
        self.travelling_waves = [
            SimpleComplexHarmonicOscillator(
                dict(omega=2 * omega * np.sin(np.pi * k / self.dof)),
                dict(x0=ic["y0"], phi=ic["phi"]),
            )
            for k, ic in enumerate(half_initial_conditions)
        ]
        if odd_dof:
            duplicate = self.travelling_waves[-1:0:-1]
        else:
            if self.travelling_waves[-1].initial_condition.x0.imag != 0:
                raise ValueError("The amplitude of wave number N // 2 must be real")
            duplicate = self.travelling_waves[-2:0:-1]
        self.travelling_waves += [
            SimpleComplexHarmonicOscillator(
                dict(omega=sho.system.omega),
                dict(x0=(ic := sho.initial_condition).x0.conjugate(), phi=ic.phi),
            )
            for sho in duplicate
        ]

    @cached_property
    def definition(self) -> dict[str, Any]:
        """model params and initial conditions defined as a dictionary."""
        return dict(
            omega=self.omega,
            dof=self.dof,
            travelling_waves=[tw.definition for tw in self.travelling_waves],
        )

    def _x(self, t: ArrayLike) -> tuple[ArrayLike, ArrayLike]:
        if is_scalar(t):
            t = np.asarray([t])
        travelling_waves = np.asarray([tw._x(t) for tw in self.travelling_waves])
        original_xs = np.real(ifft(travelling_waves, axis=0, norm="ortho"))

        return original_xs, travelling_waves

    def __call__(self, t: ArrayLike) -> pd.DataFrame:
        """Generate time series data for the harmonic oscillator.

        Returns a list of floats representing the displacement at each time step.

        :param n_periods: Number of periods to generate.
        :param n_samples_per_period: Number of samples per period.
        """
        if is_scalar(t):
            t = np.asarray([t])
        original_xs, travelling_waves = self._x(t)
        data = {
            f"{name}{i}": values
            for name, xs in zip(("x", "y"), (original_xs, travelling_waves))
            for i, values in enumerate(xs)
        }

        return pd.DataFrame(data, index=t)
