from abc import ABC, abstractmethod
from dataclasses import asdict, dataclass, field
from functools import cached_property

import numpy as np
import pandas as pd
from numpy.typing import ArrayLike

from ..harmonic_oscillator import HarmonicOscillatorSystem


@dataclass
class ComplexHarmonicOscillatorIC:
    """The initial condition for a harmonic oscillator

    :cvar x0: the initial displacement
    :cvar v0: the initial velocity
    :cvar phi: initial phase
    """

    x0: complex | float | int = field(default=complex(1.0, 0.0))
    phi: float | int = field(default=0.0)


class SimpleComplexHarmonicOscillator:
    r"""Base class to generate time series data
    for a [harmonic oscillator](https://en.wikipedia.org/wiki/Harmonic_oscillator).

    :param system: all the params that defines the harmonic oscillator.
    :param initial_condition: the initial condition of the harmonic oscillator.
    """

    def __init__(
        self,
        system: dict[str, float],
        initial_condition: dict[str, complex | float | int] | None = None,
    ) -> None:
        self.system = HarmonicOscillatorSystem.model_validate(system)
        self.initial_condition = ComplexHarmonicOscillatorIC(**(initial_condition or {}))  # type: ignore [arg-type]

    @cached_property
    def definition(self) -> dict[str, dict[str, complex | float | int]]:
        """model params and initial conditions defined as a dictionary."""
        return {
            "system": self.system.model_dump(),
            "initial_condition": asdict(self.initial_condition),
        }

    def _x(self, t: ArrayLike) -> ArrayLike:
        r"""Solution to simple harmonic oscillators:

        $$
        x(t) = x_0 \cos(\omega t + \phi).
        $$
        """
        return self.initial_condition.x0 * np.cos(
            self.system.omega * t + self.initial_condition.phi
        )
