import math
from typing import TYPE_CHECKING

import pydantic
import pytest
from numpy.testing import assert_approx_equal

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


class Test2DSystem:
    def test_init(self, system_kwargs: "Mapping[str, float]") -> None:
        Kepler2DSystem(**system_kwargs)

    @pytest.mark.parametrize(("alpha", "mass"), [(-1, 1), (1, -1), (0, 1), (1, 0)])
    def test_raise(self, alpha: int, mass: int) -> None:
        with pytest.raises(pydantic.ValidationError):
            Kepler2DSystem(alpha=alpha, mass=mass)


class Test2DIoM:
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
        match = "Only non-zero angular momenta are supported"
        with pytest.raises(NotImplementedError, match=match):
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

    def test_period_from_u(
        self, ecc: float, system_kwargs: "Mapping[str, float]", kepler_iom: Kepler2DIoM
    ) -> None:
        kep = Kepler2D(system_kwargs, kepler_iom.model_dump())
        if ecc >= 0 and ecc < 1:
            assert_approx_equal(kep.period_in_tau, kep.period * kep.t_to_tau_factor)
            if ecc > 0:
                period_in_tau = 2 * (kep.tau_of_u(-kep.ecc) - kep.tau_of_u(kep.ecc))
                assert_approx_equal(period_in_tau, kep.period_in_tau)
        elif ecc >= 1:
            match = "Only energy < 0 gives a bounded motion where the system has a period, got"
            with pytest.raises(TypeError, match=match):
                _ = kep.period
        else:
            raise ValueError(f"Expect ecc >= 0, got {ecc}")

    def test_call(self, kepler_system: Kepler2DSystem, kepler_iom: Kepler2DIoM) -> None:
        pass
