import math
from functools import cached_property, partial
from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd
from pydantic import BaseModel, Field, field_validator

from ...maths.trigonometrics import acos_with_shift
from .dynamics import tau_of_u_elliptic, tau_of_u_hyperbolic, tau_of_u_parabolic
from .numerics import u_of_tau

if TYPE_CHECKING:
    from typing import Collection, Mapping

    from numpy import typing as npt
    from typing_extensions import Self


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
    :cvar t0: the time at which the radial position is closest to 0, default to 0
    :cvar phi0: the angle at which the radial position is closest to 0, default to 0
    """

    ene: float = Field()
    angular_mom: float = Field()
    t0: float = Field(default=0)
    phi0: float = Field(ge=0, lt=2 * math.pi, default=0)

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

        integrals_of_motion = dict(integrals_of_motion)
        ene = integrals_of_motion["ene"]
        minimal_ene = Kepler2D.minimal_ene(integrals_of_motion["angular_mom"], system)
        if ene < minimal_ene:
            msg = f"Energy {ene} less than minimally allowed {minimal_ene}"
            raise ValueError(msg)

        self.integrals_of_motion = Kepler2DIoM.model_validate(integrals_of_motion)

        if 0 <= self.ecc < 1:
            self.tau_of_u = partial(tau_of_u_elliptic, self.ecc)
        elif self.ecc == 1:
            self.tau_of_u = partial(tau_of_u_parabolic, self.ecc)
        elif self.ecc > 1:
            self.tau_of_u = partial(tau_of_u_hyperbolic, self.ecc)
        else:
            raise RuntimeError

    @classmethod
    def from_geometry(
        cls, system: "Mapping[str, float]", geometries: "Mapping[str, bool | float]"
    ) -> "Self":
        mass, alpha = system["mass"], system["alpha"]
        positive_angular_mom = bool(geometries["positive_angular_mom"])
        ecc, parameter = map(lambda k: float(geometries[k]), ["ecc", "parameter"])
        abs_angular_mom = math.sqrt(mass * parameter * alpha)
        # abs_minimal_ene = alpha / 2 / parameter: numerically unstable
        abs_minimal_ene = mass * alpha**2 / 2 / abs_angular_mom**2
        ene = (ecc**2 - 1) * abs_minimal_ene
        iom = dict(
            ene=ene,
            angular_mom=abs_angular_mom if positive_angular_mom else -abs_angular_mom,
        )
        return cls(system, iom)

    @staticmethod
    def minimal_ene(
        angular_mom: float,
        system: "Mapping[str, float]",
    ) -> float:
        mass, alpha = system["mass"], system["alpha"]
        return -mass * alpha**2 / (2 * angular_mom**2)

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

    @property
    def t0(self) -> float:
        return self.integrals_of_motion.t0

    @property
    def phi0(self) -> float:
        return self.integrals_of_motion.phi0

    @cached_property
    def period(self) -> float:
        if self.ene >= 0:
            msg = f"Only energy < 0 gives a bounded motion where the system has a period, got {self.ene}"
            raise TypeError(msg)
        return math.pi * self.alpha * math.sqrt(-self.mass / 2 / self.ene**3)

    # FIXME is it called parameter in English?
    @cached_property
    def parameter(self) -> float:
        return self.angular_mom**2 / self.mass / self.alpha

    @cached_property
    def ecc(self) -> float:
        return math.sqrt(
            1 + 2 * self.ene * self.angular_mom**2 / self.mass / self.alpha**2
        )

    @cached_property
    def period_in_tau(self) -> float:
        if self.ecc >= 1:
            raise TypeError(
                f"Only systems with 0 <= eccentricity < 1 have a period, got {self.ecc}"
            )
        return 2 * math.pi / (1 - self.ecc**2) ** 1.5

    @property
    def t_to_tau_factor(self) -> float:
        return abs(self.mass * self.alpha**2 / self.angular_mom**3)

    def tau(self, t: "Collection[float] | npt.ArrayLike") -> "npt.ArrayLike":
        return (np.array(t, copy=False) - self.t0) * self.t_to_tau_factor

    def u_of_tau(self, tau: "Collection[float] | npt.ArrayLike") -> "npt.ArrayLike":
        tau = np.array(tau, copy=False)
        if self.ecc == 0:
            return np.zeros(tau.shape)
        else:
            if self.ecc < 1:
                # FIXME this is not right: tau % self.period_in_tau; 2 * period - tau
                tau = tau % (self.period_in_tau / 2)
            return u_of_tau(self.ecc, np.abs(tau))

    def r_of_u(self, u: "Collection[float] | npt.ArrayLike") -> "npt.ArrayLike":
        return self.parameter / (np.array(u, copy=False) + 1)

    def phi_of_u_tau(
        self,
        u: "Collection[float] | npt.ArrayLike",
        tau: "Collection[float] | npt.ArrayLike",
    ) -> "npt.ArrayLike":
        u, tau = np.array(u, copy=False), np.array(tau, copy=False)
        if self.ecc == 0:
            phi = 2 * math.pi * tau / self.period_in_tau
        else:
            if self.ecc < 1:
                shift = tau / self.period_in_tau
            else:
                shift = np.where(tau >= 0, 0, -np.pi)
            phi = acos_with_shift(u / self.ecc, shift)  # type: ignore [assignment]
        return phi + self.phi0

    def __call__(self, t: "Collection[float] | npt.ArrayLike") -> pd.DataFrame:
        tau = self.tau(t)
        u = self.u_of_tau(tau)
        r = self.r_of_u(u)
        phi = self.phi_of_u_tau(u, tau)

        return pd.DataFrame(dict(t=t, tau=tau, u=u, r=r, phi=phi))
