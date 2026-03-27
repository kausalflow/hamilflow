import numpy as np
from numpy import typing as npt
from pydantic import Field

from hamilflow.models.dynamical_systems.base.first_order_dynamics import (
    FirstOrderDynamicsBase,
    FirstOrderIC,
    FirstOrderSystem,
)


class ExponentialDecaySystem(FirstOrderSystem):
    k: float = Field(default=1.0)


class ExpontialDecayIC(FirstOrderIC):
    x0: float | list[float] = Field(...)
    t0: float = Field(default=0.0)


class ExponentialDecayModel(FirstOrderDynamicsBase):
    def derivatives(
        self,
        state: npt.NDArray[np.float64],
        t: float,
    ) -> npt.NDArray[np.float64]:
        # dy/dt = -k * y
        return -self.system.k * state  # type: ignore[attr-defined]


def test_first_order_dynamics_instantiation():
    model = ExponentialDecayModel(
        system=ExponentialDecaySystem(k=2.0),
        initial_condition=ExpontialDecayIC(x0=5.0, t0=0.0),
    )
    assert model.system.k == 2.0
    assert model.ic.x0 == 5.0
    assert model.ic.t0 == 0.0

    definition = model.definition
    assert definition["system"]["k"] == 2.0
    assert definition["initial_condition"]["x0"] == 5.0


def test_first_order_dynamics_step():
    model = ExponentialDecayModel(
        system=ExponentialDecaySystem(k=1.0),
        initial_condition=ExpontialDecayIC(x0=1.0),
    )
    state = np.array([1.0], dtype=np.float64)
    dt = 0.1
    t = 0.0

    new_state, new_t = model.step(state, t, dt)
    assert new_t == 0.1

    # For dx/dt = -x, RK4 step starting at x=1.0 with dt=0.1
    # RK4 match to 4th order Taylor expansion
    expected = 1.0 - 0.1 + (0.1**2) / 2.0 - (0.1**3) / 6.0 + (0.1**4) / 24.0
    np.testing.assert_allclose(new_state, expected, rtol=1e-5)


def test_first_order_dynamics_call_scalar_x0():
    model = ExponentialDecayModel(
        system=ExponentialDecaySystem(k=1.0),
        initial_condition=ExpontialDecayIC(x0=3.0, t0=0.0),
    )

    t_arr = np.linspace(0.0, 1.0, 101)
    df = model(t_arr)

    assert list(df.columns) == ["t", "x"]
    np.testing.assert_allclose(df["t"].values, t_arr)

    # Check values against analytical: x = 3 * e^{-t}
    expected_x = 3.0 * np.exp(-1.0 * t_arr)
    np.testing.assert_allclose(df["x"].values, expected_x, rtol=1e-4)


def test_first_order_dynamics_call_array_x0():
    model = ExponentialDecayModel(
        system=ExponentialDecaySystem(k=2.0),
        initial_condition=ExpontialDecayIC(x0=[2.0, 5.0], t0=0.0),
    )

    t_arr = np.linspace(0.0, 0.5, 101)
    df = model(t_arr)

    assert list(df.columns) == ["t", "x1", "x2"]

    # Check values against analytical: x = x0 * e^{-2t}
    expected_x1 = 2.0 * np.exp(-2.0 * t_arr)
    expected_x2 = 5.0 * np.exp(-2.0 * t_arr)

    np.testing.assert_allclose(df["x1"].values, expected_x1, rtol=1e-4)
    np.testing.assert_allclose(df["x2"].values, expected_x2, rtol=1e-4)
