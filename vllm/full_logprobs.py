# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import msgspec
import numpy as np


class FullLogprobsParams(
    msgspec.Struct,
    array_like=True,  # type: ignore[call-arg]
    omit_defaults=True,  # type: ignore[call-arg]
    gc=False,
):  # type: ignore[call-arg]
    """Engine-level description of a teacher-mode logprobs request."""

    enabled: bool = False
    positions: list[int] | None = None
    dtype: Literal["fp16"] = "fp16"
    format: Literal["base64_dense"] = "base64_dense"


class FullLogprobsChunk(
    msgspec.Struct,
    array_like=True,  # type: ignore[call-arg]
    omit_defaults=True,  # type: ignore[call-arg]
    gc=False,
):  # type: ignore[call-arg]
    """Opaque chunk of logprobs emitted by the backend."""

    positions: list[int]
    data: bytes


@dataclass
class _RequestState:
    vocab_size: int
    params: FullLogprobsParams
    rows: dict[int, bytes]
    row_byte_length: int
    positions_filter: tuple[int, ...] | None
    positions_filter_set: set[int] | None


class FullLogprobsBuffer:
    """Host-side accumulator for per-request full logprobs."""

    def __init__(self) -> None:
        self._states: dict[str, _RequestState] = {}

    def register(
        self,
        request_id: str,
        *,
        vocab_size: int,
        params: FullLogprobsParams,
    ) -> None:
        """Register a teacher-mode request with the buffer."""
        if not params.enabled:
            raise ValueError("Attempted to register disabled full-logprobs request.")
        if request_id in self._states:
            raise ValueError(f"full logprobs already registered for {request_id}.")
        positions_filter: tuple[int, ...] | None = None
        positions_filter_set: set[int] | None = None
        if params.positions is not None:
            positions_filter = tuple(params.positions)
            positions_filter_set = set(positions_filter)
        row_byte_length = vocab_size * np.dtype(np.float16).itemsize
        self._states[request_id] = _RequestState(
            vocab_size=vocab_size,
            params=params,
            rows={},
            row_byte_length=row_byte_length,
            positions_filter=positions_filter,
            positions_filter_set=positions_filter_set,
        )

    def cleanup(self, request_id: str) -> None:
        """Release any host memory tracked for the request."""
        self._states.pop(request_id, None)

    def add_chunk(self, request_id: str, chunk: FullLogprobsChunk) -> None:
        """Add a chunk of logprobs for the given request."""
        state = self._states.get(request_id)
        if state is None:
            raise ValueError(f"full logprobs not registered for {request_id}.")
        if not chunk.positions:
            return
        expected_len = state.row_byte_length * len(chunk.positions)
        if len(chunk.data) != expected_len:
            raise ValueError(
                "full logprobs chunk has incorrect byte length: "
                f"expected {expected_len}, got {len(chunk.data)}"
            )
        for idx, position in enumerate(chunk.positions):
            if state.positions_filter_set is not None and position not in state.positions_filter_set:
                continue
            if position in state.rows:
                raise ValueError(
                    f"full logprobs already recorded for position {position}."
                )
            start = idx * state.row_byte_length
            end = start + state.row_byte_length
            state.rows[position] = chunk.data[start:end]

    def build_dense_array(self, request_id: str) -> np.ndarray:
        """Build a contiguous [L_eff, V] float16 matrix for the request."""
        state = self._states.get(request_id)
        if state is None:
            raise ValueError(f"full logprobs not registered for {request_id}.")

        if state.positions_filter is not None:
            ordered_positions = state.positions_filter
        else:
            if not state.rows:
                raise ValueError("No logprobs recorded for request.")
            ordered_positions = tuple(sorted(state.rows))

        vocab_size = state.vocab_size
        matrix = np.empty((len(ordered_positions), vocab_size), dtype=np.float16)
        for row_idx, position in enumerate(ordered_positions):
            row_bytes = state.rows.get(position)
            if row_bytes is None:
                raise ValueError(
                    f"Missing logprobs for required position {position}."
                )
            row = np.frombuffer(row_bytes, dtype=np.float16, count=vocab_size)
            matrix[row_idx] = row

        if state.positions_filter is None:
            # Ensure positions form a contiguous range starting at zero.
            for expected, actual in enumerate(ordered_positions):
                if expected != actual:
                    raise ValueError(
                        "Logprobs are missing for some prompt positions; "
                        f"expected position {expected}, found {actual}."
                    )
        return np.ascontiguousarray(matrix)

    def params_for(self, request_id: str) -> FullLogprobsParams:
        state = self._states.get(request_id)
        if state is None:
            raise ValueError(f"full logprobs not registered for {request_id}.")
        return state.params
