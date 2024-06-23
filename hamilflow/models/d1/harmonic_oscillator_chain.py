from functools import cached_property
from typing import Any, Sequence

import numpy as np
import pandas as pd
from numpy.typing import ArrayLike
from pandas.api.types import is_scalar
from scipy.fft import ifft

from ..harmonic_oscillator import SimpleHarmonicOscillator


class HarmonicOscillatorsChain:
    r"""Generate time series data for a
    [simple harmonic oscillator](https://en.wikipedia.org/wiki/Harmonic_oscillator).


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


    To use this generator,

    ```python
    params = {"omega": omega}

    ho = SimpleHarmonicOscillator(params=params)

    df = ho(n_periods=1, n_samples_per_period=10)
    ```

    `df` will be a pandas dataframe with two columns: `t` and `x`.
    """

    def __init__(
        self, omega: float, initial_conditions: Sequence[dict[str, float]]
    ) -> None:
        self.omega = omega
        self.dof = len(initial_conditions)
        self.travelling_waves = [
            SimpleHarmonicOscillator(
                dict(omega=2 * omega * np.sin(np.pi * k / self.dof)),
                dict(x0=ic["y0"], phi=ic["phi"]),
            )
            for k, ic in enumerate(initial_conditions)
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
        r"""Solution to simple harmonic oscillators:

        $$
        x(t) = x_0 \cos(\omega t + \phi).
        $$
        """
        if is_scalar(t):
            t = np.asarray([t])
        travelling_waves = np.asarray([tw._x(t) for tw in self.travelling_waves])
        # FIXME this is imaginary
        original_xs = ifft(travelling_waves.T, norm="ortho").T

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
