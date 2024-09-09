"""Main module for Kepler problem."""

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
    r"""Definition of the Kepler problem.

    Potential:

    $$
    V(r) = - \frac{\alpha}{r}.
    $$

    For reference, if an object is orbiting our Sun, the constant
    $\alpha = G M_{\odot} ~ 1.327×10^{20} \mathrm{m}^3/\mathrm{s}^2$ in SI,
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


class Kepler2DFI(BaseModel):
    r"""The first integrals for a Kepler problem.

    :cvar ene: the energy $E$
    :cvar angular_mom: the angular momentum $l$
    :cvar t0: the time $t_0$ at which the radial position is closest to 0, defaults to 0
    :cvar phi0: the angle $\phi_0$ at which the radial position is closest to 0, defaults to 0
    """

    ene: float = Field()
    angular_mom: float = Field()
    t0: float = Field(default=0)
    phi0: float = Field(ge=0, lt=2 * math.pi, default=0)

    # TODO process angular momentum = 0
    @field_validator("angular_mom")
    @classmethod
    def _angular_mom_non_zero(cls, v: Any) -> float:
        if v == 0:
            raise NotImplementedError("Only non-zero angular momenta are supported")
        return v


class Kepler2D:
    """Kepler problem in two dimensional space.

    :param system: the Kepler problem system specification
    :param first_integrals: the first integrals for the system.
    """

    def __init__(
        self,
        system: "Mapping[str, float]",
        first_integrals: "Mapping[str, float]",
    ) -> None:
        self.system = Kepler2DSystem.model_validate(system)

        first_integrals = dict(first_integrals)
        ene = first_integrals["ene"]
        minimal_ene = Kepler2D.minimal_ene(first_integrals["angular_mom"], system)
        if ene < minimal_ene:
            msg = f"Energy {ene} less than minimally allowed {minimal_ene}"
            raise ValueError(msg)

        self.first_integrals = Kepler2DFI.model_validate(first_integrals)

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
        cls,
        system: "Mapping[str, float]",
        geometries: "Mapping[str, bool | float]",
    ) -> "Self":
        r"""Alternative initialiser from system and geometry specifications.

        Given the eccentricity $e$ and the conic section parameter $p$,
        $$l = \pm \sqrt{mp}\,,\quad E = (e^2-1) \left|E_\text{min}\right|\,.$$

        :param system: the Kepler problem system specification
        :param geometries: geometric specifications
            `positive_angular_mom`: whether the angular momentum is positive
            `ecc`: eccentricity of the conic section
            `parameter`: parameter of the conic section
        """
        mass, alpha = system["mass"], system["alpha"]
        positive_angular_mom = bool(geometries["positive_angular_mom"])
        ecc, parameter = (float(geometries[k]) for k in ["ecc", "parameter"])
        abs_angular_mom = math.sqrt(mass * parameter * alpha)
        # abs_minimal_ene = alpha / 2 / parameter: numerically unstable
        abs_minimal_ene = abs(cls.minimal_ene(abs_angular_mom, system))
        ene = (ecc**2 - 1) * abs_minimal_ene
        fi = {
            "ene": ene,
            "angular_mom": (
                abs_angular_mom if positive_angular_mom else -abs_angular_mom
            ),
        }
        return cls(system, fi)

    @staticmethod
    def minimal_ene(
        angular_mom: float,
        system: "Mapping[str, float]",
    ) -> float:
        r"""Minimal possible energy from the system specification and an angular momentum.

        $$ E_\text{min} = -\frac{m\alpha^2}{2l^2}\,. $$

        :param angular_mom: angular momentum
        :param system: system specification
        :return: minimal possible energy
        """
        mass, alpha = system["mass"], system["alpha"]
        return -mass * alpha**2 / (2 * angular_mom**2)

    @property
    def mass(self) -> float:
        """Mass $m$ from the system specification."""
        return self.system.mass

    @property
    def alpha(self) -> float:
        r"""Alpha $\alpha$ from the system specification."""
        return self.system.alpha

    @property
    def ene(self) -> float:
        """Energy $E$ of the Kepler problem."""
        return self.first_integrals.ene

    @property
    def angular_mom(self) -> float:
        """Angular momentum $l$ of the Kepler problem."""
        return self.first_integrals.angular_mom

    @property
    def t0(self) -> float:
        r"""t0 $t_0$ of the Kepler problem."""
        return self.first_integrals.t0

    @property
    def phi0(self) -> float:
        r"""phi0 $\phi_0$ of the Kepler problem."""
        return self.first_integrals.phi0

    @cached_property
    def period(self) -> float:
        r"""Period $T$ of the Kepler problem.

        For $E < 0$,
        $$ T = \pi \alpha \sqrt{-\frac{m}{2E^3}}\,. $$
        """
        if self.ene >= 0:
            msg = f"Only systems with energy < 0 have a period, got {self.ene}"
            raise TypeError(msg)
        return math.pi * self.alpha * math.sqrt(-self.mass / 2 / self.ene**3)

    # FIXME is it called parameter in English?
    @cached_property
    def parameter(self) -> float:
        r"""Conic section parameter of the Kepler problem.

        $$ p = \frac{l^2}{\alpha m}\,. $$
        """
        return self.angular_mom**2 / self.mass / self.alpha

    @cached_property
    def ecc(self) -> float:
        r"""Conic section eccentricity of the Kepler problem.

        $$ e = \sqrt{1 + \frac{2El}{\alpha^2 m}}\,. $$
        """
        return math.sqrt(
            1 + 2 * self.ene * self.angular_mom**2 / self.mass / self.alpha**2,
        )

    @cached_property
    def period_in_tau(self) -> float:
        r"""Period in the scaled time tau.

        $$ T_\tau = \frac{2\pi}{(1-e^2)^\frac{3}{2}}\,. $$
        """
        if self.ecc >= 1:
            raise TypeError(
                f"Only systems with 0 <= eccentricity < 1 have a period, got {self.ecc}",
            )
        return 2 * math.pi / (1 - self.ecc**2) ** 1.5

    @property
    def t_to_tau_factor(self) -> float:
        r"""Scale factor from t to tau.

        $$ \tau = \frac{\alpha^2 m}{|l|^3} (t-t_0)\,. $$
        """
        return abs(self.mass * self.alpha**2 / self.angular_mom**3)

    def tau(self, t: "Collection[float] | npt.ArrayLike") -> "npt.ArrayLike":
        r"""Give the scaled time tau from t.

        $$ \tau = \frac{\alpha^2 m}{|l|^3} (t-t_0)\,. $$
        """
        return (np.asarray(t) - self.t0) * self.t_to_tau_factor

    def u_of_tau(self, tau: "Collection[float] | npt.ArrayLike") -> "npt.ArrayLike":
        """Give the convenient radial inverse u from tau."""
        tau = np.asarray(tau)
        if self.ecc == 0:
            return np.zeros(tau.shape)
        else:
            if self.ecc < 1:
                p = self.period_in_tau
                r = tau % p
                tau = np.where(r <= p / 2, r, p - r)
            else:
                tau = np.abs(tau)
            return u_of_tau(self.ecc, tau)  # type: ignore [arg-type]

    def r_of_u(self, u: "Collection[float] | npt.ArrayLike") -> "npt.ArrayLike":
        r"""Give the radial r from u.

        $$ r = \frac{p}{u+1}\,. $$
        """
        return self.parameter / (np.asarray(u) + 1)

    def phi_of_u_tau(
        self,
        u: "Collection[float] | npt.ArrayLike",
        tau: "Collection[float] | npt.ArrayLike",
    ) -> "npt.ArrayLike":
        r"""Give the angular phi from u and tau.

        For $e = 0$,
        $$ \phi - \phi_0 = 2\pi \frac{\tau}{T_\tau}\,; $$
        For $e > 0$,
        $$ \cos(\phi - \phi_0) = \frac{u}{e}\,. $$
        """
        u, tau = np.asarray(u), np.asarray(tau)
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
        """Give a DataFrame of tau, u, r and phi from t."""
        tau = self.tau(t)
        u = self.u_of_tau(tau)
        r = self.r_of_u(u)
        phi = self.phi_of_u_tau(u, tau)

        return pd.DataFrame({"t": t, "tau": tau, "u": u, "r": r, "phi": phi})
