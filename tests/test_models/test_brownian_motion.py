import numpy as np
import pytest

from hamilflow.models.brownian_motion import BrownianMotionIC, BrownianMotionSystem


@pytest.mark.parametrize(
    "x0,expected",
    [(0.1, np.array(0.1)), (1, np.array(1.0)), ([1, 2], np.array([1, 2]))],
)
def test_brownian_motion_ic(x0, expected):

    brownian_motion_ic = BrownianMotionIC(x0=x0)

    np.testing.assert_equal(brownian_motion_ic.x0, expected)


@pytest.mark.parametrize(
    "sigma, delta_t, gaussian_scale", [(1, 1, 1), (1, 2, 2), (2, 1, 4)]
)
def test_brownian_motion_system(sigma, delta_t, gaussian_scale):
    bms = BrownianMotionSystem(sigma=sigma, delta_t=delta_t)

    assert bms.gaussian_scale == gaussian_scale


@pytest.mark.parametrize(
    "sigma, delta_t",
    [
        (-1, 1),
        (1, -1),
    ],
)
def test_brownian_motion_system_failed_spec(sigma, delta_t):
    with pytest.raises(ValueError):
        BrownianMotionSystem(sigma=sigma, delta_t=delta_t)
