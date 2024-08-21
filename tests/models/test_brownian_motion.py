import numpy as np
import pandas as pd
import pytest

from hamilflow.models.brownian_motion import (
    BrownianMotion,
    BrownianMotionIC,
    BrownianMotionSystem,
)


@pytest.mark.parametrize(
    "x0,expected",
    [(0.1, np.array(0.1)), (1, np.array(1.0)), ([1, 2], np.array([1, 2]))],
)
def test_brownian_motion_ic(x0, expected):
    brownian_motion_ic = BrownianMotionIC(x0=x0)

    np.testing.assert_equal(brownian_motion_ic.x0, expected)


@pytest.mark.parametrize(
    "sigma, delta_t, gaussian_scale",
    [(1, 1, 1), (1, 2, 2), (2, 1, 4)],
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


@pytest.mark.parametrize(
    "sigma, x0, expected",
    [
        (
            1,
            0,
            {
                "t": [0.0, 1.0, 2.0, 3.0, 4.0],
                "x_0": [0.0, 0.49671415, 0.35844985, 1.00613839, 2.52916825],
            },
        ),
        (
            1,
            [1, 1],
            {
                "t": [0.0, 1.0, 2.0, 3.0, 4.0],
                "x_0": [1.0, 1.49671415, 2.14440269, 1.91024932, 3.48946213],
                "x_1": [1.0, 0.8617357, 2.38476556, 2.1506286, 2.91806333],
            },
        ),
    ],
)
def test_brownian_motion_generate_from(sigma, x0, expected):
    system = {
        "sigma": sigma,
        "delta_t": 1,
    }

    initial_condition = {"x0": x0}

    bm = BrownianMotion(system=system, initial_condition=initial_condition)

    pd.testing.assert_frame_equal(
        bm.generate_from(n_steps=5, seed=42),
        pd.DataFrame(expected),
        check_exact=False,
        check_like=True,
    )


@pytest.mark.parametrize(
    "sigma, x0, t, expected",
    [
        pytest.param(
            1,
            0,
            [0.0, 1.0, 2.0, 3.0, 4.0],
            {
                "t": [0.0, 1.0, 2.0, 3.0, 4.0],
                "x_0": [0.0, 0.49671415, 0.35844985, 1.00613839, 2.52916825],
            },
            id="1d-5-steps",
        ),
        pytest.param(
            1,
            [1, 1],
            [0.0, 1.0, 2.0, 3.0, 4.0],
            {
                "t": [0.0, 1.0, 2.0, 3.0, 4.0],
                "x_0": [1.0, 1.49671415, 2.14440269, 1.91024932, 3.48946213],
                "x_1": [1.0, 0.8617357, 2.38476556, 2.1506286, 2.91806333],
            },
            id="2d-5-steps",
        ),
        pytest.param(
            1,
            0,
            0.0,
            {
                "t": [0.0],
                "x_0": [0.0],
            },
            id="1d-scaler-t",
        ),
    ],
)
def test_brownian_motion(sigma, x0, t, expected):
    system = {
        "sigma": sigma,
        "delta_t": 1,
    }

    initial_condition = {"x0": x0}

    bm = BrownianMotion(system=system, initial_condition=initial_condition)

    pd.testing.assert_frame_equal(
        bm(t=t, seed=42),
        pd.DataFrame(expected),
        check_exact=False,
        check_like=True,
    )
