import math
from functools import cached_property

import numpy as np
import pandas as pd
from numpy.typing import ArrayLike
from pydantic import BaseModel, Field, computed_field
from scipy.special import ellipj, ellipk


class PendulumSystem(BaseModel):
    r"""The params for the pendulum.

    :param omega0: $\omega_0 \coloneqq \sqrt{\frac{U}{I}} > 0$, frequency
    parameter
    """

    omega0: float = Field(gt=0.0, frozen=True)


class PendulumIC(BaseModel):
    r"""The initial condition for a pendulum.

    :param theta0: $-\frac{\pi}{2} \le \theta_0 \le \frac{\pi}{2}$, the
    initial angle
    """

    theta0: float = Field(ge=-math.pi / 2, le=math.pi / 2, frozen=True)

    @computed_field  # type: ignore[misc]
    @cached_property
    def k(self) -> float:
        r"""A convenient number for elliptic functions.

        :return: $\sin\frac{\theta_0}{2}$
        """
        return math.sin(self.theta0 / 2)


class Pendulum:
    r"""Generate time series data for a pendulum.

    We describe a generic pendulum system by the Lagrangian action
    $$
    S_L\[\theta\] = I \int_{t_0}^{t_1} \mathbb{d}t
    \left\\{\frac{1}{2} \dot\theta^2 + \omega_0^2 \cos\theta \right\\}\,,
    $$
    where $\theta$ is the _angle_ from the vertical to the pendulum;
    $I$ is the _inertia parameter_ introduced for dimensional reasons,
    and $\omega_0$ the _frequency parameter_.

    Details are collected in the tutorial.
    """

    def __init__(
        self,
        system: int | float | dict[str, int | float],
        initial_condition: int | float | dict[str, int | float],
    ) -> None:
        if isinstance(system, (float, int)):
            system = {"omega0": system}
        if isinstance(initial_condition, (float, int)):
            initial_condition = {"theta0": initial_condition}
        self.system = PendulumSystem.model_validate(system)
        self.initial_condition = PendulumIC.model_validate(initial_condition)

    @cached_property
    def definition(self) -> dict[str, float]:
        """Model params and initial conditions defined as a dictionary."""
        return dict(
            system=self.system.model_dump(),
            initial_condition=self.initial_condition.model_dump(),
        )

    @property
    def omega0(self) -> float:
        return self.system.omega0

    @property
    def _k(self) -> float:
        return self.initial_condition.k

    @property
    def _math_m(self) -> float:
        return self._k**2

    @cached_property
    def freq(self) -> float:
        r"""Frequency.

        :return: $\frac{\pi}{2K(k^2)}\omega_0$, where
        $K(m)$ is [Legendre's complete elliptic integral of the first kind](https://dlmf.nist.gov/19.2#E8)
        """
        return math.pi * self.omega0 / (2 * ellipk(self._math_m))

    @cached_property
    def period(self) -> float:
        r"""Period.

        :return: $\frac{4K(k^2)}{\omega_0}$, where
        $K(m)$ is [Legendre's complete elliptic integral of the first kind](https://dlmf.nist.gov/19.2#E8)
        """
        return 4 * ellipk(self._math_m) / self.omega0

    def _math_u(self, t: ArrayLike) -> np.ndarray[float]:
        return self.omega0 * np.asarray(t)

    def u(self, t: ArrayLike) -> np.ndarray[float]:
        r"""The convenient generalised coordinate $u$,
        $\sin u \coloneqq \frac{\sin\frac{\theta}{2}}{\sin\frac{\theta_0}{2}}$.

        :param t: time
        :return: $u(t) = \mathrm{am}\!\big(\omega_0 t + K(k^2), k^2\big)$, where
        $\mathrm{am}(x, k)$ is [Jacobi's amplitude function](https://dlmf.nist.gov/22.16#E1),
        $K(m)$ is [Legendre's complete elliptic integral of the first kind](https://dlmf.nist.gov/19.2#E8)
        """
        _, _, _, ph = ellipj(self._math_u(t) + ellipk(self._math_m), self._math_m)

        return ph

    def theta(self, t: ArrayLike) -> np.ndarray[float]:
        r"""Angle $\theta$.

        :param t: time
        :return: $\theta(t) = 2\arcsin\!\big(k\cdot\mathrm{cd}(\omega_0 t, k^2)\big)$, where
        $\mathrm{cd}(z, k)$ is a [Jacobian elliptic function](https://dlmf.nist.gov/22.2#E8)
        """
        _, cn, dn, _ = ellipj(self._math_u(t), self._math_m)

        return 2 * np.arcsin(cn / dn * self._k)

    def __call__(self, n_periods: int, n_samples_per_period: int) -> pd.DataFrame:
        time_delta = self.period / n_samples_per_period
        time_steps = np.arange(0, n_periods * n_samples_per_period) * time_delta

        thetas = self.theta(time_steps)
        us = self.u(time_steps)

        return pd.DataFrame(dict(t=time_steps, x=thetas, u=us))
