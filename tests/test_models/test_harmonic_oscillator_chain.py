from itertools import chain, product
from typing import Iterable, Mapping, Sequence

import numpy as np
import pytest

from hamilflow.models.harmonic_oscillator_chain import HarmonicOscillatorsChain

_wave_params: list[tuple[tuple[int, int], ...]] = [
    ((0, 0),),
    ((1, 0),),
    ((0, 1), (0, 1)),
]
_possible_wave_modes: list[dict[str, tuple[int, int]]] = [
    dict(zip(("amp", "phi"), param, strict=False)) for param in _wave_params
]


class TestHarmonicOscillatorChain:
    @pytest.fixture(params=(1, 2))
    def omega(self, request: pytest.FixtureRequest) -> int:
        return request.param

    @pytest.fixture(params=((0, 0), (0, 1), (1, 0), (1, 1)))
    def free_mode(self, request: pytest.FixtureRequest) -> dict[str, int]:
        return dict(zip(("x0", "v0"), request.param))

    @pytest.fixture(
        params=chain.from_iterable(
            product(_possible_wave_modes, repeat=r) for r in range(3)
        )
    )
    def wave_modes(self, request: pytest.FixtureRequest) -> list[dict[str, int]]:
        return request.param

    @pytest.fixture(params=(False, True))
    def odd_dof(self, request: pytest.FixtureRequest) -> bool:
        return request.param

    @pytest.fixture(params=(0, 1, (0, 1)))
    def times(self, request: pytest.FixtureRequest) -> int | tuple[int]:
        return request.param

    def test_init(
        self,
        omega: int,
        free_mode: Mapping[str, int],
        wave_modes: Iterable[Mapping[str, int]],
        odd_dof: bool,
    ) -> None:
        assert HarmonicOscillatorsChain(omega, [free_mode, *wave_modes], odd_dof)

    def test_real(
        self,
        omega: int,
        free_mode: Mapping[str, int],
        wave_modes: Iterable[Mapping[str, int]],
        odd_dof: bool,
        times: int | Sequence[int],
    ) -> None:
        hoc = HarmonicOscillatorsChain(omega, [free_mode, *wave_modes], odd_dof)
        original_zs, _ = hoc._z(times)
        assert np.all(original_zs.imag == 0.0)
