import math
from functools import cached_property, partial
from typing import Collection, Mapping

import numpy as np
import numpy.typing as npt
import pandas as pd
import scipy as sp
from numba import njit
from pydantic import BaseModel, Field

from hamilflow.models.kepler_problem.math import (
    tau_of_u_root_elliptic,
    tau_of_u_root_hyperbolic,
    tau_of_u_root_parabolic,
    u_of_tau_by_inverse,
)


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
    alpha: float = Field(gt=0, default=1)
    mass: float = Field(gt=0, default=1)


class Kepler2DIM(BaseModel):
    """The integrals of motion for a Kepler problem

    :cvar ene: the energy
    :cvar angular_mom: the angular momentum
    """

    # :cvar t_0: the time at which the radial position is closest to 0
    # :cvar phi_0: the angle at which the radial position is closest to 0

    ene: float = Field()
    angular_mom: float = Field()


class Kepler2D:
    r"""Kepler problem in two dimensional space.

    :param system: the Kepler problem system definition
    :param initial_condition: the initial condition for the simulation
    """

    def __init__(
        self,
        system: Mapping[str, float],
        initial_condition: Mapping[str, float],
    ) -> None:
        self.system = Kepler2DSystem.model_validate(system)
        self.integrals_of_motion = Kepler2DIM.model_validate(initial_condition)

        if self.ene < self.minimal_ene:
            raise ValueError(
                f"Energy {self.ene} less than minimally allowed {self.minimal_ene}"
            )

        if 0 <= self.eccentricity < 1:
            self.tau_of_u_root = tau_of_u_root_elliptic
        elif self.eccentricity == 1:
            self.tau_of_u_root = tau_of_u_root_parabolic
        elif self.eccentricity > 1:
            self.tau_of_u_root = tau_of_u_root_hyperbolic
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

    # FIXME is it called parameter in English?
    @cached_property
    def parameter(self) -> float:
        return self.angular_mom**2 / self.mass * self.alpha

    @cached_property
    def eccentricity(self) -> float:
        return math.sqrt(
            1 + 2 * self.ene * self.angular_mom**2 / self.mass * self.alpha**2
        )

    def tau(
        self, t: "Collection[float] | npt.ArrayLike[float]"
    ) -> "npt.ArrayLike[float]":
        return np.array(t, copy=False) * self.mass * self.alpha**2 / self.angular_mom**3

    def u_of_tau(
        self, tau: "Collection[float] | npt.ArrayLike[float]"
    ) -> "npt.ArrayLike[float]":
        return np.array(
            [
                u_of_tau_by_inverse(self.tau_of_u_root, self.eccentricity, ta)
                for ta in tau
            ]
        )

    def r_of_u(
        self, u: "Collection[float] | npt.ArrayLike[float]"
    ) -> "npt.ArrayLike[float]":
        return (np.array(u, copy=False) + 1) / self.parameter

    def phi(
        self, t: "Collection[float] | npt.ArrayLike[float]"
    ) -> "npt.ArrayLike[float]":
        pass

    def __call__(self, t: "Collection[float] | npt.ArrayLike[float]") -> pd.DataFrame:
        tau = self.tau(t)
        u = self.u_of_tau(tau)
        r = self.r_of_u(u)
        phi = self.phi(t)

        return pd.DataFrame(dict(t=t, tau=tau, u=u, r=r, phi=phi))
