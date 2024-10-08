"""Main module for a free particle."""

from collections.abc import Mapping, Sequence
from functools import cached_property
from typing import cast

import numpy as np
import pandas as pd
from numpy import typing as npt
from pydantic import BaseModel, Field, model_validator

try:
    from typing import Self
except ImportError:
    from typing_extensions import Self


class FreeParticleIC(BaseModel):
    """The initial condition for a free particle.

    :cvar x0: the initial displacement
    :cvar v0: the initial velocity
    """

    x0: float | Sequence[float] = Field()
    v0: float | Sequence[float] = Field()

    @model_validator(mode="after")
    def _check_dimensions_match(self) -> Self:
        if (x0_seq := isinstance(self.x0, Sequence)) != isinstance(self.v0, Sequence):
            msg = "x0 and v0 need both to be scalars or Sequences"
            raise TypeError(msg)
        elif x0_seq and len(cast(Sequence, self.x0)) != len(cast(Sequence, self.v0)):
            msg = "Sequences x0 and v0 need to have the same length"
            raise ValueError(msg)

        return self


class FreeParticle:
    r"""Base class to generate time series data for a free particle.

    :param initial_condition: the initial condition of the free particle.
    """

    def __init__(
        self,
        initial_condition: Mapping[str, float | Sequence[float]],
    ) -> None:
        self.initial_condition = FreeParticleIC.model_validate(initial_condition)

    @cached_property
    def definition(self) -> dict[str, dict[str, float | list[float]]]:
        """Model params and initial conditions defined as a dictionary."""
        return {"initial_condition": self.initial_condition.model_dump()}

    def _x(self, t: "Sequence[float] |  npt.ArrayLike") -> "npt.NDArray[np.float64]":
        t = np.asarray(t)
        v0 = np.asarray(self.initial_condition.v0)
        x0 = np.asarray(self.initial_condition.x0)
        return np.outer(t, v0) + x0

    def __call__(self, t: "Sequence[float] |  npt.ArrayLike") -> pd.DataFrame:
        """Generate time series data for the free particle.

        :param t: time(s).
        """
        data = self._x(t)
        columns = [f"x{i+1}" for i in range(data.shape[1])]

        return pd.DataFrame(data, columns=columns).assign(t=t).sort_index(axis=1)
