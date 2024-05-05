import numpy as np
import pandas as pd
import pytest

from hamilflow.models.central_field import (
    CentralField2D,
    CentralField2DIC,
    CentralField2DSystem,
)


@pytest.fixture
def central_field_2d_system_params():
    return {"alpha": 1.0, "mass": 1.0}


@pytest.fixture
def central_field_2d_ic_params():
    return {"r_0": 1.0, "phi_0": 1.0, "v_r_0": 1.0, "v_phi_0": 1.0}


def test_central_field_2d(central_field_2d_ic_params, central_field_2d_system_params):
    cf = CentralField2D(
        system=central_field_2d_system_params,
        initial_condition=central_field_2d_ic_params,
    )

    assert cf._energy == 0.0
    assert cf._angular_momentum == 1.0
