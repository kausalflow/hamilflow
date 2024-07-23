from functools import cached_property
from typing import Mapping, Sequence, cast

import numpy as np
import pandas as pd
from pydantic import BaseModel, Field, model_validator

try:
    from typing import Self
except ImportError:
    from typing_extensions import Self


class FreeParticleIC(BaseModel):
    """The initial condition for a free particle

    :cvar x0: the initial displacement
    :cvar v0: the initial velocity
    """

    x0: float | int | Sequence[float | int] = Field()
    v0: float | int | Sequence[float | int] = Field()

    @model_validator(mode="after")
    def check_dimensions_match(self) -> Self:
        assert (
            len(self.x0) == len(cast(Sequence, self.v0))
            if isinstance(self.x0, Sequence)
            else not isinstance(self.v0, Sequence)
        )
        return self


class FreeParticle:
    r"""Base class to generate time series data for a free particle.

    :param initial_condition: the initial condition of the free particle.
    """

    def __init__(
        self, initial_condition: Mapping[str, float | int | Sequence[float | int]]
    ) -> None:
        self.initial_condition = FreeParticleIC.model_validate(initial_condition)

    @cached_property
    def definition(self) -> dict[str, dict[str, int | float | Sequence[int | float]]]:
        """model params and initial conditions defined as a dictionary."""
        return dict(initial_condition=self.initial_condition.model_dump())

    def _x(self, t: float | int | Sequence[float | int]) -> np.ndarray:
        return np.outer(t, self.initial_condition.v0) + self.initial_condition.x0

    def __call__(self, t: float | int | Sequence[float | int]) -> pd.DataFrame:
        """Generate time series data for the free particle.

        :param t: time(s).
        """

        data = self._x(t)
        columns = (f"x{i+1}" for i in range(data.shape[1]))

        return pd.DataFrame(data, columns=columns).assign(t=t).sort_index(axis=1)
