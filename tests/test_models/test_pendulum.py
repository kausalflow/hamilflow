import pytest

from hamilflow.models.pendulum import Pendulum, PendulumIC, PendulumSystem


class TestPendulumSystem:
    def test_omega_range(self) -> None:
        with pytest.raises(ValueError):
            PendulumSystem(omega0=-1)


class TestPendulumIC:
    @pytest.mark.parametrize("theta0", [-2.0, 2.0])
    def test_theta0_range(self, theta0: float) -> None:
        with pytest.raises(ValueError):
            PendulumIC(theta0=theta0)


class TestPendulum:
    def test_pendulum(self) -> None:
        assert Pendulum
