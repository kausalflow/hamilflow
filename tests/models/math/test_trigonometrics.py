
from typing import TYPE_CHECKING, ClassVar
import numpy as np
import pytest
from hamilflow.maths.trigonometrics import acos_with_shift
from numpy.testing import assert_approx_equal

if TYPE_CHECKING:
    from typing import Collection

class TestAcosWithShift:
    _some_numbers: ClassVar[list[float]] = [-7., -3., -1., 1., 3., 7.]
    @pytest.fixture(params=[*_some_numbers, _some_numbers])
    def phi(self, request: pytest.FixtureRequest) -> int | list[int]:
        return request.param
    
    def test_acos_with_shift(self, phi: "int | Collection[int]") -> None:
        assert_approx_equal(phi, acos_with_shift(np.cos(phi), phi / 2 / np.pi))
