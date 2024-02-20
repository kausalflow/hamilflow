from functools import cached_property
from typing import Dict, List, Literal, Optional, Union

import numpy as np
import pandas as pd
from pydantic import BaseModel, computed_field


class HarmonicOscillatorSystem(BaseModel):
    """The params for the harmonic oscillator

    :param omega: angular frequency of the harmonic oscillator
    :param amplitude: the amplitude of the oscillation
    :param phi: initial phase
    """

    omega: float
    zeta: float = 0.0

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

    @computed_field  # type: ignore[misc]
    @cached_property
    def type(
        self,
    ) -> Literal["simple", "under_damped", "critical_damped", "over_damped"]:
        """which type of harmonic oscillators"""
        if self.zeta == 0:
            return "simple"
        elif self.zeta < 1:
            return "under_damped"
        elif self.zeta == 1:
            return "critical_damped"
        else:
            return "over_damped"


class HarmonicOscillatorIC(BaseModel):
    """The initial condition for a harmonic oscillator

    :param x0: the initial displacement
    :param v0: the initial velocity
    """

    x0: float = 1.0
    v0: float = 0.0


class HarmonicOscillator:
    r"""Generate time series data for a [harmonic oscillator](https://en.wikipedia.org/wiki/Harmonic_oscillator).

    The equation for a general un-driven harmonic oscillator is[^wiki_ho]

    $$
    \frac{\mathrm d x^2}{\mathrm d t^2} + 2\zeta \omega \frac{\mathrm d x}{\mathrm dt} + \omega^2 x = 0,
    $$

    where $x$ is the displacement, $\omega$ is the angular frequency of an undamped oscillator ($\zeta=0$),
    and $\zeta$ is the damping ratio.

    [^wiki_ho]: Contributors to Wikimedia projects. Harmonic oscillator.
                In: Wikipedia [Internet]. 18 Feb 2024 [cited 20 Feb 2024].
                Available: https://en.wikipedia.org/wiki/Harmonic_oscillator#Damped_harmonic_oscillator

    The solution to the above harmonic oscillator is

    $$
    x(t) = \left( x_0 \cos(\Omega t) + \frac{\zeta \omega x_0 + v_0}{\Omega} \sin(\Omega t) \right)
        e^{-\zeta \omega t},
    $$

    where

    $$
    \Omega = \omega\sqrt{ 1 - \zeta^2}.
    $$


    !!! example "A Simple Harmonic Oscillator ($\zeta=0$)"

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

    ho = HarmonicOscillator(params=params)

    df = ho(n_periods=1, n_samples_per_period=10)
    ```

    `df` will be a pandas dataframe with two columns: `t` and `x`.

    :param params: all the params that defines the harmonic oscillator.
    """

    def __init__(
        self,
        system: Dict[str, float],
        initial_condition: Optional[Dict[str, float]] = {},
    ):
        self.system = HarmonicOscillatorSystem.model_validate(system)
        self.initial_condition = HarmonicOscillatorIC.model_validate(initial_condition)

    @cached_property
    def definition(self) -> Dict[str, float]:
        """model params and initial conditions defined as a dictionary."""
        return {
            "system": self.system.model_dump(),
            "initial_condition": self.initial_condition.model_dump(),
        }

    def _x_simple(self, t: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
        r"""Solution to simple harmonic oscillators:

        $$
        x(t) = x_0 \cos(\omega t).
        $$
        """
        return self.initial_condition.x0 * np.cos(self.system.omega * t)

    def _x_under_damped(self, t: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
        r"""Solution to under damped harmonic oscillators:

        $$
        x(t) = \left( x_0 \cos(\Omega t) + \frac{\zeta \omega x_0 + v_0}{\Omega} \sin(\Omega t) \right)
        e^{-\zeta \omega t},
        $$

        where

        $$
        \Omega = \omega\sqrt{ 1 - \zeta^2}.
        $$
        """
        omega_damp = self.system.omega * np.sqrt(1 - self.system.zeta)
        return (
            self.initial_condition.x0 * np.cos(omega_damp * t)
            + (
                self.system.zeta * self.system.omega * self.initial_condition.x0
                + self.initial_condition.v0
            )
            / omega_damp
            * np.sin(omega_damp * t)
        ) * np.exp(-self.system.zeta * self.system.omega * t)

    def _x_over_damped(self, t: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
        r"""Solution to under over harmonic oscillators:

        $$
        x(t) = \left( x_0 \cosh(\Gamma t) + \frac{\zeta \omega x_0 + v_0}{\Gamma} \sinh(\Gamma t) \right)
        e^{-\zeta \omega t},
        $$

        where

        $$
        \Gamma = \omega\sqrt{ \zeta^2 - 1 }.
        $$
        """
        gamma_damp = self.system.omega * np.sqrt(self.system.zeta - 1)

        return (
            self.initial_condition.x0 * np.cosh(gamma_damp * t)
            + (
                self.system.zeta * self.system.omega * self.initial_condition.x0
                + self.initial_condition.v0
            )
            / gamma_damp
            * np.sinh(gamma_damp * t)
        ) * np.exp(-self.system.zeta * self.system.omega * t)

    def __call__(self, n_periods: int, n_samples_per_period: int) -> pd.DataFrame:
        """Generate time series data for the harmonic oscillator.

        Returns a list of floats representing the displacement at each time step.

        :param n_periods: Number of periods to generate.
        :param n_samples_per_period: Number of samples per period.
        """
        time_delta = self.system.period / n_samples_per_period
        time_steps = np.arange(0, n_periods * n_samples_per_period) * time_delta

        if self.system.type == "simple":
            data = self._x_simple(time_steps)
        elif self.system.type == "under_damped":
            data = self._x_under_damped(time_steps)
        elif self.system.type == "over_damped":
            data = self._x_over_damped(time_steps)
        elif self.system.type == "critical_damped":
            data = self._x_under_damped(time_steps)
        else:
            raise ValueError(f"system type is not defined: {self.system.type}")

        return pd.DataFrame({"t": time_steps, "x": data})
