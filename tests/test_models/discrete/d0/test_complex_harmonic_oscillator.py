import numpy as np
import pytest
from numpy.testing import assert_array_equal

from hamilflow.models.discrete.d0.complex_harmonic_oscillator import (
    ComplexSimpleHarmonicOscillator,
    ComplexSimpleHarmonicOscillatorIC,
)
from hamilflow.models.harmonic_oscillator import SimpleHarmonicOscillator


class TestComplexHarmonicOscillatorIC:
    def test_ic(self) -> None:
        assert ComplexSimpleHarmonicOscillatorIC(x0=(1, 2), phi=(2, 3))
        assert ComplexSimpleHarmonicOscillatorIC(x0=(1, 2))


class TestComplexHarmonicOscillator:
    def test_complex(self) -> None:
        ComplexSimpleHarmonicOscillator(dict(omega=3), dict(x0=(1, 2), phi=(2, 3)))

    @pytest.mark.parametrize("zeta", [0.5, 1.0, 1.5])
    def test_raise(self, zeta: float) -> None:
        with pytest.raises(ValueError):
            ComplexSimpleHarmonicOscillator(
                dict(omega=3, zeta=zeta), dict(x0=(2, 3), phi=(3, 4))
            )

    @pytest.mark.parametrize("omega", [3, 5])
    @pytest.mark.parametrize("x0", [2, 4])
    @pytest.mark.parametrize("phi", [1, 6])
    def test_degenerate_real(self, omega: int, x0: int, phi: int) -> None:
        csho = ComplexSimpleHarmonicOscillator(
            dict(omega=omega), dict(x0=(x0, x0), phi=(phi, phi))
        )
        sho = SimpleHarmonicOscillator(dict(omega=omega), dict(x0=2 * x0, phi=phi))
        t = np.array([2, 3, 5, 7, 11], copy=False)
        z = csho._z(t)
        x = sho._x(t)

        assert np.all(z.imag == 0.0)
        assert_array_equal(z.real, x)
