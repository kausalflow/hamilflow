from typing import Mapping, Sequence

import pandas as pd
import pytest
from pandas.testing import assert_frame_equal

from hamilflow.models.free_particle import FreeParticle, FreeParticleIC


class TestFreeParticleIC:
    @pytest.mark.parametrize(("x0", "v0"), [(1, 2), ((1,), (2,)), ((1, 2), (2, 3))])
    def test_constructor(
        self, x0: int | Sequence[int], v0: int | Sequence[int]
    ) -> None:
        assert FreeParticleIC(x0=x0, v0=v0)

    @pytest.mark.parametrize(
        ("x0", "v0", "expected"), [(1, (2,), TypeError), ((1,), (2, 3), ValueError)]
    )
    def test_raise(
        self, x0: int | Sequence[int], v0: Sequence[int], expected: type[Exception]
    ) -> None:
        with pytest.raises(expected):
            FreeParticleIC(x0=x0, v0=v0)


class TestFreeParticle:
    @pytest.mark.parametrize(
        ("x0", "v0", "expected"),
        [
            (1, 2, dict(initial_condition=dict(x0=1, v0=2))),
            ((1,), (2,), dict(initial_condition=dict(x0=(1,), v0=(2,)))),
        ],
    )
    def test_definition(
        self,
        x0: int | Sequence[int],
        v0: int | Sequence[int],
        expected: Mapping[str, Mapping[str, int | Sequence[int]]],
    ) -> None:
        assert FreeParticle(initial_condition=dict(x0=x0, v0=v0)).definition == expected

    @pytest.mark.parametrize(
        ("x0", "v0", "t", "expected"),
        [
            (1, 2, (3,), pd.DataFrame(dict(t=[3], x1=[7]))),
            (
                (1, 2),
                (2, 3),
                (3, 4),
                pd.DataFrame(dict(t=(3, 4), x1=(7, 9), x2=(11, 14))),
            ),
        ],
    )
    def test_call(
        self,
        x0: int | Sequence[int],
        v0: int | Sequence[int],
        t: int | Sequence[int],
        expected: pd.DataFrame,
    ) -> None:
        assert_frame_equal(
            FreeParticle(initial_condition=dict(x0=x0, v0=v0))(t), expected
        )
