import math
from functools import cached_property
from typing import Any, Dict

import numpy as np
import pandas as pd
from numpy.typing import ArrayLike
from pydantic import BaseModel, Field, computed_field
from scipy.special import ellipj, ellipk


class PendulumSystem(BaseModel):
    omega: float = Field(default=2 * math.pi, gt=0)


class PendulumIC(BaseModel):
    theta0: float = Field(ge=-math.pi / 2, le=math.pi / 2)

    @computed_field  # type: ignore[misc]
    @cached_property
    def k(self) -> float:
        return math.sin(self.theta0 / 2)


class Pendulum(BaseModel):
    system: PendulumSystem  # | float | Dict[str, float]
    initial_condition: PendulumIC  # | float | Dict[str, float]

    # def model_post_init(self, __context: Any) -> None:
    #     if not isinstance(self.system, PendulumSystem):
    #         self.system = PendulumSystem(self.system)
    #     if not isinstance(self.initial_condition, PendulumIC):
    #         self.initial_condition = PendulumIC(self.initial_condition)

    #     return super().model_post_init(__context)

    @computed_field  # type: ignore[misc]
    @cached_property
    def k(self) -> float:
        return self.initial_condition.k

    @computed_field  # type: ignore[misc]
    @cached_property
    def m(self) -> float:
        return self.k**2

    @computed_field  # type: ignore[misc]
    @cached_property
    def period(self) -> float:
        return ellipk(self.m)

    def theta(self, t: ArrayLike) -> ArrayLike:
        u = self.system.omega * t
        _, cn, dn, _ = ellipj(u, self.m)

        return 2 * np.arcsin(cn / dn * self.k)

    def __call__(self, n_periods: int, n_samples_per_period: int) -> pd.DataFrame:
        time_delta = self.period / n_samples_per_period
        time_steps = np.arange(0, n_periods * n_samples_per_period) * time_delta
        thetas = self.theta(time_steps)

        return pd.DataFrame({"t": time_steps, "x": thetas})
