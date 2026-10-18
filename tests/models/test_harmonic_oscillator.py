"""Tests for the harmonic oscillator main module."""

from typing import TYPE_CHECKING

import numpy as np
import pandas as pd
import pytest
from numpy.testing import assert_array_equal
from scipy.integrate import solve_ivp

from hamilflow.models.harmonic_oscillator import (
    ComplexSimpleHarmonicOscillator,
    ComplexSimpleHarmonicOscillatorIC,
    DampedHarmonicOscillator,
    HarmonicOscillatorSystem,
    SimpleHarmonicOscillator,
)

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence


@pytest.fixture
def omega() -> int:
    """Give omega."""
    return 1


@pytest.mark.parametrize("zeta", [-0.5, -2.0])
def test_system_damping_zeta(omega: int, zeta: float) -> None:
    """Test raises from HarmonicOscillatorSystem upon illegal zeta."""
    m = r"\d+ validation error for HarmonicOscillatorSystem\nzeta\n "
    with pytest.raises(
        ValueError,
        match=m,
    ):
        HarmonicOscillatorSystem(omega=omega, zeta=zeta)


@pytest.mark.parametrize("zeta", [-2.0, -0.5, 0.5, 1.0])
def test_oscillator_damping_zeta(omega: int, zeta: float) -> None:
    """Test raises from SimpleHarmonicOscillator upon illegal zeta."""
    if zeta < 0:
        m = r"\d+ validation error for HarmonicOscillatorSystem\nzeta\n"
    else:
        m = r"System is not a Simple Harmonic Oscillator: omega="
    with pytest.raises(ValueError, match=m):
        SimpleHarmonicOscillator(system={"omega": omega, "zeta": zeta})


@pytest.mark.parametrize(
    ("omega", "expected"),
    [
        (
            0.5,
            [
                {"t": 0.0, "x": 1.0},
                {"t": 1.2566370614359172, "x": 0.8090169943749475},
                {"t": 2.5132741228718345, "x": 0.30901699437494745},
                {"t": 3.7699111843077517, "x": -0.30901699437494734},
                {"t": 5.026548245743669, "x": -0.8090169943749473},
                {"t": 6.283185307179586, "x": -1.0},
                {"t": 7.5398223686155035, "x": -0.8090169943749475},
                {"t": 8.79645943005142, "x": -0.30901699437494756},
                {"t": 10.053096491487338, "x": 0.30901699437494723},
                {"t": 11.309733552923255, "x": 0.8090169943749473},
            ],
        ),
    ],
)
def test_simple_harmonic_oscillator(
    omega: float,
    expected: "Mapping[str, float]",
) -> None:
    """Test SimpleHarmonicOscillator from periods and the number of samples, comparing with caculations of the author."""
    ho = SimpleHarmonicOscillator(system={"omega": omega})

    df = ho(n_periods=1, n_samples_per_period=10)

    pd.testing.assert_frame_equal(df, pd.DataFrame(expected))


@pytest.mark.parametrize("omega", [0.5, 2.0])
@pytest.mark.parametrize("zeta", [0.2, 0.5, 1.0, 1 + 1e-9, 1.5, 3.0, 50.0])
@pytest.mark.parametrize(("x0", "v0"), [(1.0, 0.0), (0.0, 1.0), (-0.5, 2.0)])
def test_damped_harmonic_oscillator_solves_ode(
    omega: float,
    zeta: float,
    x0: float,
    v0: float,
) -> None:
    """Test DampedHarmonicOscillator against a numerical solution of its equation of motion."""
    ho = DampedHarmonicOscillator(
        system={"omega": omega, "zeta": zeta},
        initial_condition={"x0": x0, "v0": v0},
    )
    t = np.linspace(0, 3 * ho.system.period, 31)

    sol = solve_ivp(
        lambda _, y: [y[1], -2 * zeta * omega * y[1] - omega**2 * y[0]],
        (t[0], t[-1]),
        [x0, v0],
        t_eval=t,
        method="LSODA",
        rtol=1e-10,
        atol=1e-12,
    )

    np.testing.assert_allclose(np.asarray(ho._x(t)), sol.y[0], atol=1e-8)


class TestComplexHarmonicOscillatorIC:
    """Tests for the class ComplexHarmonicOscillatorIC."""

    @pytest.mark.parametrize("kwargs", [{"x0": (1, 2), "phi": (2, 3)}, {"x0": (1, 2)}])
    def test_ic(self, kwargs: "Mapping[str, tuple[int, int]]") -> None:
        """Test initialising ComplexSimpleHarmonicOscillatorIC."""
        assert ComplexSimpleHarmonicOscillatorIC(**kwargs)


class TestComplexHarmonicOscillator:
    """Tests for the class ComplexHarmonicOscillator."""

    def test_complex(self) -> None:
        """Test initialising ComplexSimpleHarmonicOscillator."""
        assert ComplexSimpleHarmonicOscillator(
            {"omega": 3},
            {"x0": (1, 2), "phi": (2, 3)},
        )

    @pytest.mark.parametrize("zeta", [0.5, 1.0, 1.5])
    def test_raise(self, zeta: float) -> None:
        """Test raises from ComplexSimpleHarmonicOscillator upon illegal zeta."""
        m = r"System is not a Simple Harmonic Oscillator: omega="
        with pytest.raises(
            ValueError,
            match=m,
        ):
            ComplexSimpleHarmonicOscillator(
                {"omega": 3, "zeta": zeta},
                {"x0": (2, 3), "phi": (3, 4)},
            )

    @pytest.fixture(params=(1, (1,), [1, 2], np.array([2, 3, 5, 7, 11])))
    def times(self, request: pytest.FixtureRequest) -> "int | Sequence[int]":
        """Give scalar time, Sequences of and numpy array of times."""
        return request.param

    @pytest.mark.parametrize("omega", [3, 5])
    @pytest.mark.parametrize("x0", [2, 4])
    @pytest.mark.parametrize("phi", [1, 6])
    def test_degenerate_real(
        self,
        omega: int,
        x0: int,
        phi: int,
        times: "int | Sequence[int]",
    ) -> None:
        """Test the degenerate case where ComplexSimpleHarmonicOscillator reduces to SimpleHarmonicOscillator."""
        csho = ComplexSimpleHarmonicOscillator(
            {"omega": omega},
            {"x0": (x0, x0), "phi": (phi, phi)},
        )
        sho = SimpleHarmonicOscillator({"omega": omega}, {"x0": 2 * x0, "phi": phi})
        z = np.asarray(csho._z(times))
        x = sho._x(times)

        assert np.all(z.imag == 0.0)
        assert_array_equal(z.real, x)
