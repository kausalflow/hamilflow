from typing import TYPE_CHECKING, ClassVar, cast

import numpy as np
import pytest
from numpy.testing import assert_approx_equal, assert_array_almost_equal

from hamilflow.maths.trigonometrics import acos_with_shift

if TYPE_CHECKING:
    from numpy import typing as npt


class TestAcosWithShift:
    _some_numbers: ClassVar[list[float]] = [-7.0, -3.0, -1.0, 1.0, 3.0, 7.0]

    @pytest.fixture(params=[*_some_numbers, _some_numbers])
    def phi(self, request: pytest.FixtureRequest) -> float | list[float]:
        return request.param

    def test_acos_with_shift(self, phi: float | list[float]) -> None:
        y = acos_with_shift(np.cos(phi), np.array(phi, copy=False) / 2 / np.pi)
        if isinstance(phi, list):
            assert_array_almost_equal(phi, cast("npt.NDArray[np.float32]", y))
        else:
            assert_approx_equal(phi, cast("np.float32", y))
