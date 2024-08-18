from typing import TYPE_CHECKING, ClassVar

import numpy as np
import pytest
from numpy.testing import assert_array_almost_equal

from hamilflow.maths.trigonometrics import acos_with_shift

if TYPE_CHECKING:
    from typing import Collection


class TestAcosWithShift:
    _some_numbers: ClassVar[list[float]] = [-7.0, -3.0, -1.0, 1.0, 3.0, 7.0]

    @pytest.fixture(params=[*_some_numbers, _some_numbers])
    def phi(self, request: pytest.FixtureRequest) -> float | list[float]:
        return request.param

    def test_acos_with_shift(self, phi: "float | Collection[float]") -> None:
        phi = np.array(phi, copy=False)
        actual = np.array(acos_with_shift(np.cos(phi), phi / 2 / np.pi), copy=False)
        assert_array_almost_equal(phi, actual)
