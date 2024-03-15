from functools import cached_property
from typing import Dict, List, Optional, Union

import numpy as np
import pandas as pd
from pydantic import BaseModel, Field, computed_field, field_validator


class BrownianMotionSystem(BaseModel):
    """Definition of the Brownian Motion system

    For consistency, we always use
    $\mathbf x$ for displacement, and
    $t$ for steps. The model we are using is

    $$
    \begin{align}
    \mathbf x(t + \mathrm dt) &= \mathrm x(t) +
    \mathcal{N}(\mu=0, \sigma=\sigma \sqrt{\mathrm d t})
    \end{align}
    $$

    References:

    1. Brownian motion and random walks. [cited 13 Mar 2024].
        Available: https://web.mit.edu/8.334/www/grades/projects/projects17/OscarMickelin/brownian.html
    2. Contributors to Wikimedia projects. Brownian motion.
        In: Wikipedia [Internet]. 22 Jan 2024 [cited 13 Mar 2024].
        Available: https://en.wikipedia.org/wiki/Brownian_motion

    :param sigma: base standard deviation
        to be used to compute the variance
    :param delta_t: time granunality of the motion
    """

    sigma: float
    delta_t: float = 1

    @computed_field  # type: ignore[misc]
    @cached_property
    def gaussian_scale(self) -> float:
        """The scale (standard deviation) of the Gaussian term
        in Brownian motion
        """
        return self.sigma**2 * self.delta_t

    @field_validator("sigma", "delta_t")
    @classmethod
    def check_sigma_delta_t_non_negative(cls, v: float) -> float:
        if v < 0:
            raise ValueError(f"Value of sigma and delta_t should be positive: {v=}")

        return v


class BrownianMotionIC(BaseModel):
    """The initial condition for a Brownian motion

    :param x0: initial displacement of the particle,
        the diminsion of this initial condition determines
        the dimension of the model too.
    """

    x0: Union[float, int, List[Union[float, int]]] = 1.0

    @field_validator("x0")
    @classmethod
    def check_x0_types(cls, v: Union[float, int, list]) -> float:
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
    \mathbf x(t + \mathrm dt) &= \mathrm x(t) +
    \mathcal{N}(\mu=0, \sigma=\sigma \sqrt{\mathrm d t})
    \end{align}
    $$

    References:

    1. Brownian motion and random walks. [cited 13 Mar 2024].
        Available: https://web.mit.edu/8.334/www/grades/projects/projects17/OscarMickelin/brownian.html
    2. Contributors to Wikimedia projects. Brownian motion.
        In: Wikipedia [Internet]. 22 Jan 2024 [cited 13 Mar 2024].
        Available: https://en.wikipedia.org/wiki/Brownian_motion
    """

    def __init__(
        self,
        system: Dict[str, float],
        initial_condition: Optional[Dict[str, float]] = {},
    ):
        self.system = BrownianMotionSystem.model_validate(system)
        self.initial_condition = BrownianMotionIC.model_validate(initial_condition)

    def __call__(self, n_steps: float) -> pd.DataFrame:
        self.initial_condition
