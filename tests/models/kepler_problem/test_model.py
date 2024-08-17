import math
from typing import Mapping

import pydantic
import pytest

from hamilflow.models.kepler_problem import Kepler2DIoM, Kepler2DSystem


@pytest.fixture(params=[(1.0, 1.0), (1.0, 2.0), (2.0, 1.0), (2.0, 2.0)])
def system_kwargs(request: pytest.FixtureRequest) -> dict[str, float]:
    return dict(zip(("alpha", "mass"), request.param))


@pytest.fixture(params=[1.0, 2.0])
def parameter(request: pytest.FixtureRequest) -> float:
    return request.param


# 5 / 7, 12 / 11, 257 / 13 makes u_of_tau from newton fail
@pytest.fixture(params=[0.0, 0.1, 0.3, 0.7, 0.9, 1.0, 1.1, 2.0, 11.0, 101.0])
def ecc(request: pytest.FixtureRequest) -> float:
    return request.param


@pytest.fixture(params=[-1, 1])
def iom_kwargs(
    request: pytest.FixtureRequest,
    system_kwargs: Mapping[str, float],
    ecc: float,
    parameter: float,
) -> dict[str, float]:
    alpha, mass = system_kwargs["alpha"], system_kwargs["mass"]
    return dict(
        ene=(ecc**2 - 1) * alpha / 2 / parameter,
        angular_mom=request.param * math.sqrt(mass * parameter * alpha),
    )


# @pytest.fixture
# def t():
#     return np.linspace(0, 10, 101)


class TestKepler2DSystem:
    def test_init(self, system_kwargs: Mapping[str, float]) -> None:
        Kepler2DSystem(**system_kwargs)

    @pytest.mark.parametrize(("alpha", "mass"), [(-1, 1), (1, -1), (0, 1), (1, 0)])
    def test_raise(self, alpha: int, mass: int) -> None:
        with pytest.raises(pydantic.ValidationError):
            Kepler2DSystem(alpha=alpha, mass=mass)


class TestKepler2DIoM:
    def test_init(self, iom_kwargs: Mapping[str, float]) -> None:
        Kepler2DIoM(**iom_kwargs)

    def test_raise(self) -> None:
        with pytest.raises(
            NotImplementedError, match="Only non-zero angular momenta are supported"
        ):
            Kepler2DIoM(ene=1, angular_mom=0)


class TestKepler2D:
    pass


# def test_central_field_2d_conserved(
#     central_field_2d_ic_params, central_field_2d_system_params
# ):
#     cf = Kepler2D(
#         system=central_field_2d_system_params,
#         initial_condition=central_field_2d_ic_params,
#     )

#     assert cf._energy == 0.0
#     assert cf._angular_momentum == 1.0


# def test_central_field_2d_orbit(
#     central_field_2d_ic_params, central_field_2d_system_params, t
# ):

#     cf = Kepler2D(
#         system=central_field_2d_system_params,
#         initial_condition=central_field_2d_ic_params,
#     )

#     cf(t)
