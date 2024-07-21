from dataclasses import asdict, dataclass, field
from functools import cached_property
from typing import Mapping

import numpy as np
from numpy.typing import ArrayLike
from pydantic import BaseModel, Field, computed_field


class ComplexHarmonicOscillatorSystem(BaseModel):
    """The params for the complex harmonic oscillator

    :cvar omega: angular frequency of the harmonic oscillator
    :cvar real: whether use the real solution
    """

    omega: float
    real: bool = Field(default=False)

    @computed_field  # type: ignore[misc]
    @cached_property
    def period(self) -> float:
        """period of the oscillator"""
        try:
            return 2 * np.pi / self.omega
        except ZeroDivisionError:
            return np.inf

    @computed_field  # type: ignore[misc]
    @cached_property
    def frequency(self) -> float:
        """frequency of the oscillator"""
        return 1 / self.period


@dataclass
class ComplexHarmonicOscillatorIC:
    """The initial condition for a complex harmonic oscillator

    :cvar x0: the initial complex displacement
    :cvar phi: initial phase
    """

    x0: complex | float | int = field(default=complex(1.0, 0.0))
    phi: float | int = field(default=0.0)


class SimpleComplexHarmonicOscillator:
    r"""Base class to generate time series data
    for a complex harmonic oscillator.

    :param system: all the params that defines the complex harmonic oscillator.
    :param initial_condition: the initial condition of the complex harmonic oscillator.
    """

    def __init__(
        self,
        system: Mapping[str, float],
        initial_condition: Mapping[str, complex | float | int] | None = None,
    ) -> None:
        self.system = ComplexHarmonicOscillatorSystem.model_validate(system)
        self.initial_condition = ComplexHarmonicOscillatorIC(**(initial_condition or {}))  # type: ignore [arg-type]

    @cached_property
    def definition(self) -> Mapping[str, Mapping[str, complex | float | int]]:
        """model params and initial conditions defined as a dictionary."""
        return {
            "system": self.system.model_dump(),
            "initial_condition": asdict(self.initial_condition),
        }

    def _z(self, t: ArrayLike) -> ArrayLike:
        r"""Solution to the complex simple harmonic oscillator:

        $$
        x(t) = x_0 \exp(-\mathbb{i} * (\omega t + \phi))\,,
        $$
        or
        $$
        x(t) = x_0 \cos(\omega t + \phi)\,.
        $$
        """

        def f(phase: ArrayLike) -> ArrayLike:
            return np.cos(phase) if self.system.real else np.exp(-1j * phase)

        return (ic := self.initial_condition).x0 * f(self.system.omega * t + ic.phi)
