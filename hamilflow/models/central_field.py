from functools import cached_property
from typing import Dict, Optional

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
    :cvar v_r_0: the initial radial velocity
    :cvar v_phi_0: the initial phase velocity
    """

    r_0: float = Field(gt=0, default=1.0)
    phi_0: float = Field(ge=0, default=0.0)
    v_r_0: float = 1.0
    v_phi_0: float = 1.0


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
    ):
        self.system = CentralField2DSystem.model_validate(system)
        self.initial_condition = CentralField2DIC.model_validate(initial_condition)

    @cached_property
    def _angular_momentum(self) -> float:
        """computes the angular momentum of the motion. Since the angular momentum is
        conserved, it doesn't change through time.
        """
        return (
            self.system.mass
            * self.initial_condition.r_0**2
            * self.initial_condition.v_phi_0
        )

    @cached_property
    def _energy(self) -> float:
        """computes the total energy"""
        v_r_0 = self.initial_condition.v_r_0
        v_phi_0 = self.initial_condition.v_phi_0
        r_0 = self.initial_condition.r_0

        potential_energy = -1 * self.system.alpha / r_0

        return (
            0.5 * self.system.mass * (v_r_0**2 + r_0**2 * v_phi_0**2) + potential_energy
        )
