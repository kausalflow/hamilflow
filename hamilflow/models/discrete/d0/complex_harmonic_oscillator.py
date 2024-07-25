from functools import cached_property
from typing import Mapping, Sequence

import numpy as np
import pandas as pd
from numpy.typing import ArrayLike
from pydantic import BaseModel, Field

from ...harmonic_oscillator import HarmonicOscillatorSystem


class ComplexSimpleHarmonicOscillatorIC(BaseModel):
    """The initial condition for a complex harmonic oscillator

    :cvar x0: the initial displacements
    :cvar phi: initial phases
    """

    x0: tuple[float | int, float | int] = Field()
    phi: tuple[float | int, float | int] = Field(default=(0, 0))


class ComplexSimpleHarmonicOscillator:
    r"""Generate time series data for a complex simple harmonic oscillator.

    :param system: all the params that defines the complex harmonic oscillator.
    :param initial_condition: the initial condition of the complex harmonic oscillator.
    """

    def __init__(
        self,
        system: Mapping[str, float | int],
        initial_condition: Mapping[str, tuple[float | int, float | int]],
    ) -> None:
        self.system = HarmonicOscillatorSystem.model_validate(system)
        self.initial_condition = ComplexSimpleHarmonicOscillatorIC.model_validate(
            initial_condition
        )
        if self.system.type != "simple":
            raise ValueError(
                f"System is not a Simple Harmonic Oscillator: {self.system}"
            )

    @cached_property
    def definition(
        self,
    ) -> dict[str, dict[str, float | int | tuple[float | int, float | int]]]:
        """model params and initial conditions defined as a dictionary."""

        return dict(
            system=self.system.model_dump(),
            initial_condition=self.initial_condition.model_dump(),
        )

    def _z(self, t: float | int | Sequence[float | int]) -> ArrayLike:
        r"""Solution to complex simple harmonic oscillators:

        $$
        x(t) = x_+ \exp(-\mathbb{i} (\omega t + \phi_+)) + x_- \exp(+\mathbb{i} (\omega t + \phi_-)).
        $$
        """
        omega = self.system.omega
        x0, phi = self.initial_condition.x0, self.initial_condition.phi
        phases = -omega * t - phi[0], omega * t + phi[1]
        return x0[0] * np.exp(1j * phases[0]) + x0[1] * np.exp(1j * phases[1])

    def __call__(self, t: float | int | Sequence[float | int]) -> pd.DataFrame:
        """Generate time series data for the harmonic oscillator.

        Returns a list of floats representing the displacement at each time.

        :param t: time(s).
        """
        if not isinstance(t, Sequence):
            t = np.array([t], copy=False)
        data = self._z(t)

        return pd.DataFrame({"t": t, "z": data})
