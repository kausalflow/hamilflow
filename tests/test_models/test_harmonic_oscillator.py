import pandas as pd
import pytest
from pydantic import ValidationError

from hamilflow.models.harmonic_oscillator import (
    DampedHarmonicOscillator,
    HarmonicOscillatorSystem,
    SimpleHarmonicOscillator,
)


@pytest.mark.parametrize("zeta", [(-0.5), (-2.0)])
def test_harmonic_oscillator_system_damping_zeta(zeta):
    with pytest.raises(ValueError):
        HarmonicOscillatorSystem(omega=1, zeta=zeta)

    with pytest.raises(ValueError):
        SimpleHarmonicOscillator(system={"omega": 1, "zeta": zeta})


@pytest.mark.parametrize("zeta", [(0.5), (1.0)])
def test_simple_harmonic_oscillator_instantiation(zeta):
    with pytest.raises(ValueError):
        SimpleHarmonicOscillator(system={"omega": 1, "zeta": zeta})


@pytest.mark.parametrize(
    "omega,expected",
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
        )
    ],
)
def test_simple_harmonic_oscillator(omega, expected):
    ho = SimpleHarmonicOscillator(system={"omega": omega})

    df = ho(n_periods=1, n_samples_per_period=10)

    pd.testing.assert_frame_equal(df, pd.DataFrame(expected))


@pytest.mark.parametrize(
    "omega,zeta,expected",
    [
        (
            0.5,
            0.5,
            [
                {"t": 0.0, "x": 1.0},
                {"t": 1.2566370614359172, "x": 0.8814807646720196},
                {"t": 2.5132741228718345, "x": 0.6291724461162962},
                {"t": 3.7699111843077517, "x": 0.35960997861868876},
                {"t": 5.026548245743669, "x": 0.1386644177513349},
                {"t": 6.283185307179586, "x": -0.008951254750058093},
                {"t": 7.5398223686155035, "x": -0.08578239268073515},
                {"t": 8.79645943005142, "x": -0.10837036278920763},
                {"t": 10.053096491487338, "x": -0.09717507559642745},
                {"t": 11.309733552923255, "x": -0.07035823865195263},
            ],
        ),
    ],
)
def test_underdamped_harmonic_oscillator(omega, zeta, expected):
    ho = DampedHarmonicOscillator(system={"omega": omega, "zeta": zeta})

    df = ho(n_periods=1, n_samples_per_period=10)

    pd.testing.assert_frame_equal(df, pd.DataFrame(expected))


@pytest.mark.parametrize(
    "omega,zeta,expected",
    [
        (
            0.5,
            1.5,
            [
                {"t": 0.0, "x": 1.0},
                {"t": 1.2566370614359172, "x": 0.8082046531458691},
                {"t": 2.5132741228718345, "x": 0.5412092834154191},
                {"t": 3.7699111843077517, "x": 0.34137898430328123},
                {"t": 5.026548245743669, "x": 0.21056158643886722},
                {"t": 6.283185307179586, "x": 0.12872550838026245},
                {"t": 7.5398223686155035, "x": 0.07841286437267142},
                {"t": 8.79645943005142, "x": 0.04769482134764518},
                {"t": 10.053096491487338, "x": 0.028992995409657336},
                {"t": 11.309733552923255, "x": 0.017620056330381287},
            ],
        ),
    ],
)
def test_overdamped_harmonic_oscillator(omega, zeta, expected):
    ho = DampedHarmonicOscillator(system={"omega": omega, "zeta": zeta})

    df = ho(n_periods=1, n_samples_per_period=10)

    pd.testing.assert_frame_equal(df, pd.DataFrame(expected))


@pytest.mark.parametrize(
    "omega,zeta,expected",
    [
        (
            0.5,
            1.0,
            [
                {"t": 0.0, "x": 1.0},
                {"t": 1.2566370614359172, "x": 0.5334880910911033},
                {"t": 2.5132741228718345, "x": 0.2846095433360293},
                {"t": 3.7699111843077517, "x": 0.1518358019806489},
                {"t": 5.026548245743669, "x": 0.08100259215794314},
                {"t": 6.283185307179586, "x": 0.04321391826377226},
                {"t": 7.5398223686155035, "x": 0.023054110763106823},
                {"t": 8.79645943005142, "x": 0.012299093542812717},
                {"t": 10.053096491487338, "x": 0.006561419936306071},
                {"t": 11.309733552923255, "x": 0.003500439396667034},
            ],
        ),
    ],
)
def test_criticaldamped_harmonic_oscillator(omega, zeta, expected):
    ho = DampedHarmonicOscillator(system={"omega": omega, "zeta": zeta})

    df = ho(n_periods=1, n_samples_per_period=10)

    pd.testing.assert_frame_equal(df, pd.DataFrame(expected))
