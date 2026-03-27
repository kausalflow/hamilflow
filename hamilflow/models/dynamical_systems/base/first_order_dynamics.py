"""Base classes for first order dynamical systems.

In this module, we introduce the base classes for
first order dynamical systems, and the corresponding
abstract classes for system and initial condition.
"""

from abc import ABC, abstractmethod
from functools import cached_property
from typing import Any

import numpy as np
import pandas as pd
from numpy import typing as npt
from pydantic import BaseModel, Field

from hamilflow.models.utils.typing import TypeTime


class FirstOrderSystem(BaseModel, ABC):
    """The base params for a first order dynamical system.

    Create your own system by inheriting from this class
    and adding your own params.

    ```python
    class MyCustomSystem(FirstOrderSystem):
        # your fields here
        omega: float
    ```
    """


class FirstOrderIC(BaseModel, ABC):
    """The base initial condition for a first order dynamical system.

    :cvar x0: the initial state of the system
    :cvar t0: the initial time (default to 0.0)
    """

    x0: float | list[float] = Field(...)
    t0: float = Field(default=0.0)


class FirstOrderDynamicsBase(ABC):
    """Base class to generate time series data for a first order dynamical system.

    :param system: all the params that defines the system.
    :param initial_condition: the initial condition of the system.
    """

    def __init__(
        self,
        system: FirstOrderSystem,
        initial_condition: FirstOrderIC,
    ) -> None:

        self.system = system
        self.ic = initial_condition

    @cached_property
    def definition(self) -> dict[str, dict[str, Any]]:
        """Model params and initial conditions defined as a dictionary.

        :return: dictionary containing system and initial condition parameters.
        """
        return {
            "system": self.system.model_dump(),
            "initial_condition": self.ic.model_dump(),
        }

    @abstractmethod
    def derivatives(
        self,
        state: npt.NDArray[np.float64],
        t: float,
    ) -> npt.NDArray[np.float64]:
        """Return the derivative of the state with respect to time.

        :param state: the current state of the system.
        :param t: the current time.
        :return: the derivative of the state with respect to time.
        """

    def step(
        self,
        state: npt.NDArray[np.float64],
        t: float,
        dt: float,
    ) -> tuple[npt.NDArray[np.float64], float]:
        """Advances the system by one time step (dt) using
        4th-Order Runge-Kutta (RK4) integration.

        :param state: the current state of the system.
        :param t: the current time.
        :param dt: the time step size.
        :return: the new state of the system and the new time.
        """
        k1 = self.derivatives(state, t)
        k2 = self.derivatives(state + 0.5 * dt * k1, t + 0.5 * dt)
        k3 = self.derivatives(state + 0.5 * dt * k2, t + 0.5 * dt)
        k4 = self.derivatives(state + dt * k3, t + dt)

        new_state = state + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)
        new_t = t + dt

        return new_state, new_t

    def __call__(self, t: TypeTime) -> pd.DataFrame:
        """Generate time series data for the dynamical system.

        :param t: sequence of time steps.
        :return: values of the variables including time `t` and state `x`.
        """
        t_arr = np.asarray(t)

        current_state = np.asarray(self.ic.x0, dtype=np.float64)
        current_t = self.ic.t0

        effective_t = [current_t]
        history = [current_state]
        for target_t in t_arr:
            dt = target_t - current_t
            if dt > 0:
                current_state, current_t = self.step(current_state, current_t, dt)
                effective_t.append(current_t)
                history.append(current_state)

        data = np.asarray(history)

        if data.ndim == 1:
            columns = ["x"]
        else:
            columns = [f"x{i+1}" for i in range(data.shape[1])]

        return (
            pd.DataFrame(data, columns=columns).assign(t=effective_t).sort_index(axis=1)
        )
