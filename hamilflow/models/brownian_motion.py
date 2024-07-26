from functools import cached_property
from typing import Mapping

import numpy as np
import pandas as pd
import scipy as sp
from pydantic import BaseModel, Field, computed_field, field_validator


class BrownianMotionSystem(BaseModel):
    r"""Definition of the Brownian Motion system

    For consistency, we always use
    $\mathbf x$ for displacement, and
    $t$ for steps. The model we are using is

    $$
    \begin{align}
    \mathbf x(t + \mathrm dt) &= \mathbf x(t) +
    \mathcal{N}(\mu=0, \sigma=\sigma \sqrt{\mathrm d t})
    \end{align}
    $$

    References:

    1. Brownian motion and random walks. [cited 13 Mar 2024].
        Available: https://web.mit.edu/8.334/www/grades/projects/projects17/OscarMickelin/brownian.html
    2. Contributors to Wikimedia projects. Brownian motion.
        In: Wikipedia [Internet]. 22 Jan 2024 [cited 13 Mar 2024].
        Available: https://en.wikipedia.org/wiki/Brownian_motion

    :cvar sigma: base standard deviation
        to be used to compute the variance
    :cvar delta_t: time granunality of the motion
    """

    sigma: float = Field(ge=0.0)
    delta_t: float = Field(ge=0.0, default=1.0)

    @computed_field  # type: ignore[misc]
    @cached_property
    def gaussian_scale(self) -> float:
        """The scale (standard deviation) of the Gaussian term
        in Brownian motion
        """
        return self.sigma**2 * self.delta_t


class BrownianMotionIC(BaseModel):
    """The initial condition for a Brownian motion

    :cvar x0: initial displacement of the particle,
        the diminsion of this initial condition determines
        the dimension of the model too.
    """

    x0: float | int | list[float | int] = Field(default=1.0)

    @field_validator("x0")
    @classmethod
    def check_x0_types(cls, v: float | int | list[float]) -> np.ndarray:
        if not isinstance(v, (float, int, list)):
            raise ValueError(f"Value of x0 should be int/float/list of int/float: {v=}")

        return np.asarray(v)


class BrownianMotion:
    r"""Brownian motion describes motion of small particles
    with stochastic forces applied to them.
    The math of Brownian motion can be modeled
    with Wiener process.

    For consistency, we always use
    $\mathbf x$ for displacement, and
    $t$ for steps. The model we are using is

    $$
    \begin{align}
    \mathbf x(t + \mathrm dt) &= \mathbf x(t) +
    \mathcal{N}(\mu=0, \sigma=\sigma \sqrt{\mathrm d t})
    \end{align}
    $$

    References:

    1. Brownian motion and random walks. [cited 13 Mar 2024].
        Available: https://web.mit.edu/8.334/www/grades/projects/projects17/OscarMickelin/brownian.html
    2. Contributors to Wikimedia projects. Brownian motion.
        In: Wikipedia [Internet]. 22 Jan 2024 [cited 13 Mar 2024].
        Available: https://en.wikipedia.org/wiki/Brownian_motion


    !!! example "1D Brownian Motion"

        The dimsion of our Brownian motion is specified by
        the dimension of the initial condition.

        To simulate a 1D Browian motion, we define the system and initial condition:

        ```python
        system = {
            "sigma": 1,
            "delta_t": 1,
        }

        initial_condition = {
            "x0": 0
        }
        ```

        The Brownian motion can be simulated using

        ```python
        bm = BrownianMotion(system=system, initial_condition=initial_condition)

        bm(n_steps=100)
        ```

    !!! example "2D Brownian Motion"

        To simulate a 2D Browian motion,

        ```python
        system = {
            "sigma": 1,
            "delta_t": 1,
        }

        initial_condition = {
            "x0": [0, 0]
        }

        bm = BrownianMotion(system=system, initial_condition=initial_condition)

        bm(n_steps=100)
        ```

    :param system: the Brownian motion system definition
    :param initial_condition: the initial condition for the simulation
    """

    def __init__(
        self,
        system: Mapping[str, float],
        initial_condition: Mapping[str, float] | None = None,
    ):
        initial_condition = initial_condition or {}
        self.system = BrownianMotionSystem.model_validate(system)
        self.initial_condition = BrownianMotionIC.model_validate(initial_condition)

    @property
    def dim(self) -> int:
        """Dimension of the Brownian motion"""
        return self.initial_condition.x0.size

    @property
    def _axis_names(self) -> list[str]:
        return [f"y_{i}" for i in range(self.dim)]

    def _trajectory(self, n_new_steps: int, seed: int) -> np.ndarray:
        """The trajectory of the particle.

        We first compute the delta displacement in each step.
        With the displacement at each step, we perform a cumsum
        including the initial coordinate to get the displacement at each step.

        :param n_new_steps: number of new steps to simulate, excluding the initial step.
        :param seed: seed for the random generator.
        """
        step_history = sp.stats.norm.rvs(
            size=(n_new_steps, self.dim) if self.dim > 1 else n_new_steps,
            scale=self.system.gaussian_scale,
            random_state=np.random.RandomState(seed=seed),
        )

        step_history = np.concatenate(
            (np.expand_dims(self.initial_condition.x0, axis=0), step_history)
        )

        trajectory = np.cumsum(step_history, axis=0)

        return trajectory

    def __call__(self, n_steps: int, seed: int = 42) -> pd.DataFrame:
        """Simulate the coordinates of the particle

        :param n_steps: total number of steps to be simulated, including the inital step.
        :param seed: random generator seed for the stochastic process.
            Use it to reproduce results.
        """
        trajectory = self._trajectory(n_new_steps=n_steps - 1, seed=seed)

        df = pd.DataFrame(trajectory, columns=self._axis_names)

        df["t"] = np.arange(0, n_steps) * self.system.delta_t

        return df
