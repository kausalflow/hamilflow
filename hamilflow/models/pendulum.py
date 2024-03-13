import math

import pandas as pd
from pydantic import BaseModel, field_validator


class PendulumSystem(BaseModel):
    omega: float = 2 * math.pi

    @field_validator("omega")
    @classmethod
    def check_omega_positive(cls, o: float) -> float:
        if o <= 0:
            raise ValueError(f"omega should be positive but is {o}")

        return o


class PendulumIC(BaseModel):
    theta0: float

    @field_validator("theta0")
    @classmethod
    def check_theta_range(cls, t: float) -> float:
        if abs(t) > math.pi / 2:
            raise ValueError(f"theta0 should be between -pi/2 and pi/2 but is {t}")

        return t


class Pendulum(BaseModel):
    system: PendulumSystem
    initial_condition: PendulumIC

    def __call__(self) -> pd.DataFrame:
        return pd.DataFrame({"t": [0], "x": self.initial_condition.theta0})
