import math

import pytest

from hamilflow.models.harmonic_oscillator.initial_conditions import parse_ic_for_sho


@pytest.fixture()
def omega() -> float:
    return 2 * math.pi


class TestParseICForSHO:
    @pytest.mark.parametrize(
        ("input", "expected"),
        [
            (
                dict(x0=1.0, v0=1.0, phi=7.0),
                dict(x0=1.0, v0=1.0, phi=7 % (2 * math.pi)),
            ),
            (dict(x0=1.0, t0=1.0), dict(x0=1.0, v0=0.0, phi=0.0)),
            (
                dict(E=1.0, t0=1.0),
                dict(x0=math.sqrt(2.0) / (2 * math.pi), v0=0.0, phi=0.0),
            ),
        ],
    )
    def test_output(
        self, omega: float, input: dict[str, float], expected: dict[str, float]
    ) -> None:
        assert parse_ic_for_sho(omega, **input) == expected

    def test_raise(self, omega: float) -> None:
        with pytest.raises(ValueError):
            parse_ic_for_sho(omega, **dict(x0=1.0, E=2.0))
