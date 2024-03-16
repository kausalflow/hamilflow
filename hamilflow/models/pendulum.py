import math
from functools import cached_property

import numpy as np
import pandas as pd
from numpy.typing import ArrayLike
from pydantic import BaseModel, Field, computed_field
from scipy.special import ellipj, ellipk


class PendulumSystem(BaseModel):
    r"""The params for the pendulum

    :param omega0 (float): $\ometa_0$, frequency parameter in the action
    """

    omega0: float = Field(default=2 * math.pi, gt=0)


class PendulumIC(BaseModel):
    r"""The initial condition for a pendulum

    :param theta0 (float): $\theta_0$, the initial angular displacement
    """

    theta0: float = Field(ge=-math.pi / 2, le=math.pi / 2)

    @computed_field  # type: ignore[misc]
    @cached_property
    def k(self) -> float:
        r"""$k\coloneqq\sin(\theta_0 / 2)$"""
        return math.sin(self.theta0 / 2)


class Pendulum(BaseModel):
    r"""Generate time series data for a pendulum.

    # Lagrangian action
    The Lagrangian action for a pendulum is
    $$
    S\[\theta(t)\] = \int_{0}^{t_0} \mathbb{d}t
    \left\\{\frac{1}{2} I \dot\theta^2 + U \cos\theta \right\\} \eqqcolon
    \int_{0}^{t_0} \mathbb{d}t\,L_\text{P}(\theta, \dot\theta)\,,
    $$
    where $\theta$ is the angular displacement from the vertical to the
    pendulum; $I$ is an _inertial parameter_, $U$ is a potential parameter;
    $L_\text{P}$ is the Lagrangian.

    This setup contains both the single and the physical pendula. For a single
    pendulum,
    $$
    I = m l^2\,,\qquad U = mgl\,,
    $$
    where $m$ is the mass of the pendulum, $l$ is the length of the rod or cord,
    and $g$ is the gravitational acceleration.

    # Integral of motion
    $\mathbb{\delta}S / \mathbb{\delta}{t} = 0$
    $$
    \dot\theta\frac{\partial L_\text{P}}{\partial \dot\theta} - L_\text{P}
    \equiv E = U \cos\theta_0
    $$

    $$
    \left(\frac{\mathbb{d}t}{\mathbb{d}\theta}\right)^2 = \frac{1}{2\omega_0^2}
    \frac{1}{\cos\theta - \cos\theta_0}
    $$
    $\omega_0 \coloneqq \sqrt{\frac{U}{I}}$

    ## Coordinate transformation
    $$
    \sin u \coloneqq \frac{\sin\frac{\theta}{2}}{\sin\frac{\theta_0}{2}}
    $$

    $$
    \left(\frac{\mathbb{d}t}{\mathbb{d}u}\right)^2 = \frac{1}{\omega_0^2}
    \frac{1}{1-k^2\sin^2 u}
    $$
    """

    system: PendulumSystem  # | float | Dict[str, float]
    initial_condition: PendulumIC  # | float | Dict[str, float]

    # def model_post_init(self, __context: Any) -> None:
    #     if not isinstance(self.system, PendulumSystem):
    #         self.system = PendulumSystem(self.system)
    #     if not isinstance(self.initial_condition, PendulumIC):
    #         self.initial_condition = PendulumIC(self.initial_condition)

    #     return super().model_post_init(__context)

    @property
    def omega0(self) -> float:
        return self.system.omega0

    @property
    def _k(self) -> float:
        return self.initial_condition.k

    @property
    def _math_m(self) -> float:
        return self._k**2

    @computed_field  # type: ignore[misc]
    @cached_property
    def period(self) -> float:
        """$4K(k) / \omega_0$"""
        return 4 * ellipk(self._math_m) / self.omega0

    def _math_u(self, t: ArrayLike) -> np.ndarray[float]:
        return self.omega0 * np.asarray(t)

    def u(self, t: ArrayLike) -> np.ndarray:
        _, _, _, ph = ellipj(self._math_u(t) + ellipk(self._math_m), self._math_m)

        return ph

    def theta(self, t: ArrayLike) -> np.ndarray:
        _, cn, dn, _ = ellipj(self._math_u(t), self._math_m)

        return 2 * np.arcsin(cn / dn * self._k)

    def __call__(self, n_periods: int, n_samples_per_period: int) -> pd.DataFrame:
        time_delta = self.period / n_samples_per_period
        time_steps = np.arange(0, n_periods * n_samples_per_period) * time_delta
        thetas = self.theta(time_steps)

        return pd.DataFrame({"t": time_steps, "x": thetas})
