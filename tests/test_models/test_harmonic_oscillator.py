import pandas as pd
import pytest

from hamiltonian_flow.models.harmonic_oscillator import HarmonicOscillator


@pytest.fixture
def omega():
    return 0.5


@pytest.mark.parametrize("omega", [0.5])
@pytest.mark.parametrize(
    "expected",
    [
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
        ]
    ],
)
def test_harmonic_oscillator(omega, expected):
    ho = HarmonicOscillator(params={"omega": omega})

    df = ho(n_periods=1, n_samples_per_period=10)

    pd.testing.assert_frame_equal(df, pd.DataFrame(expected))
