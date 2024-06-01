from functools import cached_property
from typing import Dict, Optional

import numpy as np
import numpy.typing as npt
import pandas as pd
import scipy as sp
from pydantic import BaseModel, Field


class CentralField2DSystem(BaseModel):
    r"""Definition of the central field system

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

    alpha: float = Field(gt=0, default=1)
    mass: float = Field(gt=0, default=1)


class CentralField2DIC(BaseModel):
    """The initial condition for a Brownian motion

    :cvar r_0: the initial radial coordinate
    :cvar phi_0: the initial phase
    :cvar drdt_0: the initial radial velocity
    :cvar dphidt_0: the initial phase velocity
    """

    r_0: float = Field(gt=0, default=1.0)
    phi_0: float = Field(ge=0, default=0.0)
    drdt_0: float = 1.0
    dphidt_0: float = 0.0


class CentralField2D:
    r"""Central field motion in two dimensional space.

    !!! info "Inverse Sqare Law"

        We only consider central field of the form $-\frac{\alpha}{r}$.

    :param system: the Central field motion system definition
    :param initial_condition: the initial condition for the simulation
    """

    def __init__(
        self,
        system: Dict[str, float],
        initial_condition: Optional[Dict[str, float]] = {},
        rtol: float = 1e-6,
        atol: float = 1e-6,
    ):
        self.system = CentralField2DSystem.model_validate(system)
        self.initial_condition = CentralField2DIC.model_validate(initial_condition)
        self.rtol = rtol
        self.atol = atol

    @cached_property
    def _angular_momentum(self) -> float:
        """computes the angular momentum of the motion. Since the angular momentum is
        conserved, it doesn't change through time.
        """
        return (
            self.system.mass
            * self.initial_condition.r_0**2
            * self.initial_condition.dphidt_0
        )

    @cached_property
    def _energy(self) -> float:
        """computes the total energy"""
        drdt_0 = self.initial_condition.drdt_0
        dphidt_0 = self.initial_condition.dphidt_0
        r_0 = self.initial_condition.r_0

        potential_energy = self._potential(r_0)

        return (
            0.5 * self.system.mass * (drdt_0**2 + r_0**2 * dphidt_0**2)
            + potential_energy
        )

    def _potential(self, r: npt.ArrayLike) -> npt.ArrayLike:
        return -1 * self.system.alpha / r

    def drdt(self, t: npt.ArrayLike, r: npt.ArrayLike) -> npt.ArrayLike:
        return np.sqrt(
            2 / self.system.mass * (self._energy - self._potential(r))
            - self._angular_momentum**2 / self.system.mass**2 / r**2
        )

    def r(self, t: npt.ArrayLike) -> npt.ArrayLike:
        t_span = t.min(), t.max()
        sol = sp.integrate.solve_ivp(
            self.drdt,
            t_span=t_span,
            y0=[self.initial_condition.r_0],
            t_eval=t,
            rtol=self.rtol,
            atol=self.atol,
        )

        return sol.y[0]

    def phi(self, t: npt.ArrayLike, r: npt.ArrayLike) -> npt.ArrayLike:
        return (
            self.initial_condition.phi_0
            + self._angular_momentum / self.system.mass / r**2 * t
        )

    def __call__(self, t: npt.ArrayLike) -> npt.ArrayLike:
        r = self.r(t)
        phi = self.phi(t, r)

        return pd.DataFrame(dict(t=t, r=r, phi=phi))
