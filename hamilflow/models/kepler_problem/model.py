import math
from functools import cached_property
from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd
from pydantic import BaseModel, Field, field_validator

from ...maths.trigonometrics import acos_with_shift
from .dynamics import (
    tau_of_u_elliptic,
    tau_of_u_hyperbolic,
    tau_of_u_parabolic
    
)
from .numerics import u_of_tau

if TYPE_CHECKING:
    from typing import Collection, Mapping

    from numpy import typing as npt


class Kepler2DSystem(BaseModel):
    r"""Definition of the Kepler problem

    Potential:

    $$
    V(r) = - \frac{\alpha}{r}.
    $$

    For reference, if an object is orbiting our Sun, the constant $\alpha = G M_{\odot} ~ 1.327×10^20 m^3/s^2$ in SI,
    which is also called 1 TCB, or 1 solar mass parameter. For computational stability, we recommend using
    TCB as the unit instead of the large SI values.

    !!! note "Units"

        When specifying the parameters of the system, be ware of the consistency of the units.

    :cvar alpha: the proportional constant of the potential energy.
    :cvar mass: the mass of the orbiting object
    """

    # TODO add repulsive alpha < 0
    alpha: float = Field(gt=0, default=1.0)
    mass: float = Field(gt=0, default=1.0)


class Kepler2DIoM(BaseModel):
    """The integrals of motion for a Kepler problem

    :cvar ene: the energy
    :cvar angular_mom: the angular momentum
    """

    # :cvar t_0: the time at which the radial position is closest to 0
    # :cvar phi_0: the angle at which the radial position is closest to 0

    ene: float = Field()
    angular_mom: float = Field()

    # TODO process angular momentum = 0
    @field_validator("angular_mom")
    @classmethod
    def angular_mom_non_zero(cls, v: Any) -> float:
        if v == 0:
            raise NotImplementedError("Only non-zero angular momenta are supported")
        return v


class Kepler2D:
    r"""Kepler problem in two dimensional space.

    :param system: the Kepler problem system definition
    :param initial_condition: the initial condition for the simulation
    """

    def __init__(
        self,
        system: "Mapping[str, float]",
        integrals_of_motion: "Mapping[str, float]",
    ) -> None:
        self.system = Kepler2DSystem.model_validate(system)
        self.integrals_of_motion = Kepler2DIoM.model_validate(integrals_of_motion)

        if self.ene < self.minimal_ene:
            raise ValueError(
                f"Energy {self.ene} less than minimally allowed {self.minimal_ene}"
            )

        if 0 <= self.ecc < 1:
            self.tau_of_u = tau_of_u_elliptic
        elif self.ecc == 1:
            self.tau_of_u = tau_of_u_parabolic
        elif self.ecc > 1:
            self.tau_of_u = tau_of_u_hyperbolic
        else:
            raise RuntimeError

    @property
    def mass(self) -> float:
        return self.system.mass

    @property
    def alpha(self) -> float:
        return self.system.alpha

    @property
    def ene(self) -> float:
        return self.integrals_of_motion.ene

    @property
    def angular_mom(self) -> float:
        return self.integrals_of_motion.angular_mom

    @cached_property
    def minimal_ene(self) -> float:
        return -self.mass * self.alpha**2 / (2 * self.angular_mom**2)

    @cached_property
    def period(self) -> float:
        if self.ene >= 0:
            msg = f"Only energy < 0 gives a bounded motion where the system has a period, got {self.ene}"
            raise TypeError(msg)
        return math.pi * self.alpha * math.sqrt(self.mass / 2 / self.ene**3)

    # FIXME is it called parameter in English?
    @cached_property
    def parameter(self) -> float:
        return self.angular_mom**2 / self.mass * self.alpha

    @cached_property
    def ecc(self) -> float:
        return math.sqrt(
            1 + 2 * self.ene * self.angular_mom**2 / self.mass * self.alpha**2
        )

    @cached_property
    def period_in_tau(self) -> float:
        if self.ecc >= 1:
            msg = (
                f"Only systems with 0 <= eccentricity < 1 have a period, got {self.ecc}"
            )
            raise TypeError(msg)
        return 2 * math.pi / (1 - self.ecc**2) ** 1.5

    def tau(self, t: "Collection[float] | npt.ArrayLike") -> "npt.ArrayLike":
        return np.array(t, copy=False) * np.sqrt(
            self.mass * self.alpha**2 / self.angular_mom**3
        )

    def u_of_tau(self, tau: "Collection[float] | npt.ArrayLike") -> "npt.ArrayLike":
        tau = np.array(tau, copy=False)
        return (
            np.zeros(tau.shape, dtype=np.float32)
            if self.ecc == 0
            else u_of_tau(self.ecc, tau)
        )

    def r_of_u(self, u: "Collection[float] | npt.ArrayLike") -> "npt.ArrayLike":
        return (np.array(u, copy=False) + 1) / self.parameter

    def phi_of_u_tau(
        self,
        u: "Collection[float] | npt.ArrayLike",
        tau: "Collection[float] | npt.ArrayLike",
    ) -> "npt.ArrayLike":
        u = np.array(u, copy=False)
        tau = np.array(tau, copy=False)
        if self.ecc == 0:
            return np.zeros(u.shape)
        else:
            return acos_with_shift(u / self.ecc, tau / self.period_in_tau)

    def __call__(self, t: "Collection[float] | npt.ArrayLike") -> pd.DataFrame:
        tau = self.tau(t)
        u = self.u_of_tau(tau)
        r = self.r_of_u(u)
        phi = self.phi_of_u_tau(u, tau)

        return pd.DataFrame(dict(t=t, tau=tau, u=u, r=r, phi=phi))
