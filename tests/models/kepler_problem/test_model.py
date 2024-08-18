import math
from typing import TYPE_CHECKING, ClassVar

import numpy as np
import pydantic
import pytest
from numpy.testing import (
    assert_almost_equal,
    assert_array_almost_equal,
    assert_array_equal,
    assert_equal,
)

from hamilflow.models.kepler_problem import Kepler2DIoM, Kepler2DSystem
from hamilflow.models.kepler_problem.model import Kepler2D

if TYPE_CHECKING:
    from typing import Collection, Mapping


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
def geometries(
    positive_angular_mom: bool, ecc: float, parameter: float
) -> dict[str, bool | float]:
    return dict(positive_angular_mom=positive_angular_mom, ecc=ecc, parameter=parameter)


class Test2DSystem:
    def test_init(self, system_kwargs: "Mapping[str, float]") -> None:
        Kepler2DSystem(**system_kwargs)

    @pytest.mark.parametrize(("alpha", "mass"), [(-1, 1), (1, -1), (0, 1), (1, 0)])
    def test_raise(self, alpha: int, mass: int) -> None:
        with pytest.raises(pydantic.ValidationError):
            Kepler2DSystem(alpha=alpha, mass=mass)


class Test2DIoM:

    def test_raise(self) -> None:
        match = "Only non-zero angular momenta are supported"
        with pytest.raises(NotImplementedError, match=match):
            Kepler2DIoM(ene=1, angular_mom=0)


class TestKepler2D:
    def test_from_geometry(
        self,
        system_kwargs: "Mapping[str, float]",
        geometries: "Mapping[str, bool | float]",
    ) -> None:
        kep = Kepler2D.from_geometry(system_kwargs, geometries)
        assert_almost_equal(kep.ecc, geometries["ecc"])
        assert_almost_equal(kep.parameter, geometries["parameter"])

    @pytest.mark.parametrize("ecc", [0.0])
    def test_minimal_ene(
        self,
        system_kwargs: "Mapping[str, float]",
        geometries: "Mapping[str, bool | float]",
    ) -> None:
        kep = Kepler2D.from_geometry(system_kwargs, geometries)
        assert_equal(kep.ene, Kepler2D.minimal_ene(kep.angular_mom, system_kwargs))
        with pytest.raises(ValueError):
            Kepler2D(system_kwargs, dict(ene=kep.ene - 1, angular_mom=kep.angular_mom))

    def test_period_from_u(
        self,
        ecc: float,
        system_kwargs: "Mapping[str, float]",
        geometries: "Mapping[str, bool | float]",
    ) -> None:
        kep = Kepler2D.from_geometry(system_kwargs, geometries)
        if ecc >= 0 and ecc < 1:
            assert_almost_equal(kep.period_in_tau, kep.period * kep.t_to_tau_factor)
            if ecc > 0:
                period_in_tau = 2 * (kep.tau_of_u(-kep.ecc) - kep.tau_of_u(kep.ecc))
                assert_equal(period_in_tau, kep.period_in_tau)
        elif ecc >= 1:
            match = "Only systems with energy < 0 have a period, got "
            with pytest.raises(TypeError, match=match):
                _ = kep.period
        else:
            raise ValueError(f"Expect ecc >= 0, got {ecc}")

    def test_phi_of_u_tau_naive(
        self,
        ecc: float,
        system_kwargs: "Mapping[str, float]",
        geometries: "Mapping[str, bool | float]",
    ) -> None:
        kep = Kepler2D.from_geometry(system_kwargs, geometries)
        ecc = kep.ecc  # numeric instability
        if ecc >= 0 and ecc < 1:
            us, taus = (ecc, max(-1, -ecc)), (0, kep.period_in_tau / 2)
        else:
            us, taus = (ecc,), (0,)  # type: ignore [assignment]
        for u, tau, phi in zip(us, taus, (0, math.pi)):
            assert_equal(kep.phi_of_u_tau(u, tau), phi)
            assert_array_equal(kep.phi_of_u_tau([u], [tau]), [phi])

    _some_numbers: ClassVar[list[float]] = [x / 2 - 5 for x in range(20)]

    @pytest.fixture(params=[_some_numbers[0], _some_numbers])
    def t(self, request: pytest.FixtureRequest) -> float | list[float]:
        return request.param

    def test_r_and_phi(
        self,
        t: "float | Collection[float]",
        system_kwargs: "Mapping[str, float]",
        geometries: "Mapping[str, bool | float]",
    ) -> None:
        kep = Kepler2D.from_geometry(system_kwargs, geometries)
        tau = kep.tau(t)
        u = kep.u_of_tau(tau)
        r, phi = kep.r_of_u(u), kep.phi_of_u_tau(u, tau)
        assert_array_almost_equal(
            np.array(r, copy=False),
            kep.parameter / (1 + kep.ecc * np.cos(phi)),
            err_msg=f"{kep.ecc}, {u}, {tau}",
        )
