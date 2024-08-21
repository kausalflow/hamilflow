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
        params=chain.from_iterable(product(_possible_wave_modes, repeat=r) for r in range(3)),
    )
    def wave_modes(
        self,
        request: pytest.FixtureRequest,
    ) -> list[dict[str, tuple[int, int]]]:
        return request.param

    @pytest.fixture(params=(False, True))
    def odd_dof(self, request: pytest.FixtureRequest) -> bool:
        return request.param

    @pytest.fixture()
    def legal_wave_modes_and_odd_def(
        self,
        wave_modes: Iterable[Mapping[str, tuple[int, int]]],
        odd_dof: bool,
    ) -> tuple[Iterable[Mapping[str, tuple[int, int]]], bool]:
        return wave_modes if odd_dof else chain(wave_modes, [dict(amp=(1, 1))]), odd_dof

    @pytest.fixture(params=(0, 1, (0, 1)))
    def times(self, request: pytest.FixtureRequest) -> int | tuple[int]:
        return request.param

    def test_init(
        self,
        omega: int,
        free_mode: Mapping[str, int],
        legal_wave_modes_and_odd_def: tuple[
            Iterable[Mapping[str, tuple[int, int]]],
            bool,
        ],
    ) -> None:
        wave_modes, odd_dof = legal_wave_modes_and_odd_def
        assert HarmonicOscillatorsChain(omega, [free_mode, *wave_modes], odd_dof)

    @pytest.fixture()
    def hoc_and_zs(
        self,
        omega: int,
        free_mode: Mapping[str, int],
        legal_wave_modes_and_odd_def: tuple[
            Iterable[Mapping[str, tuple[int, int]]],
            bool,
        ],
        times: int | Sequence[int],
    ) -> tuple[HarmonicOscillatorsChain, np.ndarray, np.ndarray]:
        wave_modes, odd_dof = legal_wave_modes_and_odd_def
        hoc = HarmonicOscillatorsChain(omega, [free_mode, *wave_modes], odd_dof)
        return (hoc, *hoc._z(times))

    def test_real(
        self,
        hoc_and_zs: tuple[HarmonicOscillatorsChain, np.ndarray, np.ndarray],
    ) -> None:
        _, original_zs, _ = hoc_and_zs
        assert np.all(original_zs.imag == 0.0)

    def test_dof(
        self,
        hoc_and_zs: tuple[HarmonicOscillatorsChain, np.ndarray, np.ndarray],
    ) -> None:
        hoc, original_zs, _ = hoc_and_zs
        assert original_zs.shape[0] == hoc.n_dof

    @pytest.mark.parametrize("wave_mode", [None, *_possible_wave_modes[1:]])
    def test_raise(
        self,
        omega: int,
        free_mode: Mapping[str, int],
        wave_mode: Mapping[str, int],
    ) -> None:
        ics = [free_mode, *([wave_mode] if wave_mode else [])]
        with pytest.raises(ValueError):
            HarmonicOscillatorsChain(omega, ics, False)
