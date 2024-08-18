import math
from typing import TYPE_CHECKING

import pydantic
import pytest

from hamilflow.models.kepler_problem import Kepler2DIoM, Kepler2DSystem
from hamilflow.models.kepler_problem.model import Kepler2D

if TYPE_CHECKING:
    from typing import Mapping


@pytest.fixture(params=[(1.0, 1.0), (1.0, 2.0), (2.0, 1.0), (2.0, 2.0)])
def system_kwargs(request: pytest.FixtureRequest) -> dict[str, float]:
    return dict(zip(("alpha", "mass"), request.param))


@pytest.fixture(params=[1.0, 2.0])
def parameter(request: pytest.FixtureRequest) -> float:
    return request.param


@pytest.fixture(params=[False, True])
def positive_angular_mom(request: pytest.FixtureRequest) -> bool:
    return request.param


@pytest.fixture()
def kepler_system(system_kwargs: "Mapping[str, float]") -> Kepler2DSystem:
    return Kepler2DSystem(**system_kwargs)


@pytest.fixture()
def kepler_iom(
    positive_angular_mom: bool,
    ecc: float,
    parameter: float,
    kepler_system: Kepler2DSystem,
) -> Kepler2DIoM:
    return Kepler2DIoM.from_geometry(
        positive_angular_mom, ecc, parameter, kepler_system
    )


# @pytest.fixture
# def t():
#     return np.linspace(0, 10, 101)


class TestKepler2DSystem:
    def test_init(self, system_kwargs: "Mapping[str, float]") -> None:
        Kepler2DSystem(**system_kwargs)

    @pytest.mark.parametrize(("alpha", "mass"), [(-1, 1), (1, -1), (0, 1), (1, 0)])
    def test_raise(self, alpha: int, mass: int) -> None:
        with pytest.raises(pydantic.ValidationError):
            Kepler2DSystem(alpha=alpha, mass=mass)


class TestKepler2DIoM:
    def test_from_geometry(
        self,
        positive_angular_mom: bool,
        ecc: float,
        parameter: float,
        kepler_system: Kepler2DSystem,
    ) -> None:
        assert Kepler2DIoM.from_geometry(
            positive_angular_mom, ecc, parameter, kepler_system
        )

    def test_raise(self) -> None:
        with pytest.raises(
            NotImplementedError, match="Only non-zero angular momenta are supported"
        ):
            Kepler2DIoM(ene=1, angular_mom=0)


class TestKepler2D:
    @pytest.mark.parametrize("ecc", [0.0])
    def test_minimal_ene(
        self,
        system_kwargs: "Mapping[str, float]",
        kepler_system: Kepler2DSystem,
        kepler_iom: Kepler2DIoM,
    ) -> None:
        kep = Kepler2D(system_kwargs, kepler_iom.model_dump())
        assert kep.ene == kepler_iom.minimal_ene(kepler_iom.angular_mom, kepler_system)
        with pytest.raises(ValueError):
            Kepler2D(system_kwargs, dict(ene=kep.ene - 1, angular_mom=kep.angular_mom))

    def test_period(self, system_kwargs: "Mapping[str, float]") -> None:
        kep = Kepler2D


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
