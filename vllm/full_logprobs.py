# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import msgspec


class FullLogprobsParams(
    msgspec.Struct,
    array_like=True,  # type: ignore[call-arg]
    omit_defaults=True,  # type: ignore[call-arg]
    gc=False,
):  # type: ignore[call-arg]
    """Engine-level description of a teacher-mode logprobs request."""

    enabled: bool = False
    positions: list[int] | None = None
    top_p: float = 0.9999
    max_top_k: int = 512
    dtype: Literal["fp16"] = "fp16"
    format: Literal["top_p"] = "top_p"


class FullLogprobsRow(
    msgspec.Struct,
    array_like=True,  # type: ignore[call-arg]
    omit_defaults=True,  # type: ignore[call-arg]
    gc=False,
):  # type: ignore[call-arg]
    """Sparse logprobs for a single prompt position."""

    position: int
    token_ids: list[int]
    logprobs: list[float]
    tail_mass: float


class FullLogprobsChunk(
    msgspec.Struct,
    array_like=True,  # type: ignore[call-arg]
    omit_defaults=True,  # type: ignore[call-arg]
    gc=False,
):  # type: ignore[call-arg]
    """Opaque chunk of logprobs emitted by the backend."""

    rows: list[FullLogprobsRow]


@dataclass
class _RequestState:
    vocab_size: int
    params: FullLogprobsParams
    rows: dict[int, FullLogprobsRow]
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
        self._states[request_id] = _RequestState(
            vocab_size=vocab_size,
            params=params,
            rows={},
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
        for row in chunk.rows:
            position = row.position
            if state.positions_filter_set is not None and position not in state.positions_filter_set:
                continue
            if position in state.rows:
                raise ValueError(
                    f"full logprobs already recorded for position {position}."
                )
            if not row.token_ids:
                raise ValueError("full logprobs row must include at least one token.")
            if len(row.token_ids) != len(row.logprobs):
                raise ValueError(
                    "full logprobs row has mismatched token/logprob lengths."
                )
            state.rows[position] = row

    def build_response_payload(
        self,
        request_id: str,
    ) -> tuple[list[int] | None, list[list[int]], list[list[float]], list[float]]:
        """Build sparse arrays for the response payload."""
        state = self._states.get(request_id)
        if state is None:
            raise ValueError(f"full logprobs not registered for {request_id}.")

        if state.positions_filter is not None:
            ordered_positions = state.positions_filter
        else:
            if not state.rows:
                raise ValueError("No logprobs recorded for request.")
            ordered_positions = tuple(sorted(state.rows))

        token_ids: list[list[int]] = []
        logprobs: list[list[float]] = []
        tail_mass: list[float] = []
        for position in ordered_positions:
            row = state.rows.get(position)
            if row is None:
                raise ValueError(
                    f"Missing logprobs for required position {position}."
                )
            token_ids.append(row.token_ids)
            logprobs.append(row.logprobs)
            tail_mass.append(row.tail_mass)

        if state.positions_filter is None:
            # Ensure positions form a contiguous range starting at zero.
            for expected, actual in enumerate(ordered_positions):
                if expected != actual:
                    raise ValueError(
                        "Logprobs are missing for some prompt positions; "
                        f"expected position {expected}, found {actual}."
                    )
        positions = list(ordered_positions) if state.positions_filter is not None else None
        return positions, token_ids, logprobs, tail_mass

    def params_for(self, request_id: str) -> FullLogprobsParams:
        state = self._states.get(request_id)
        if state is None:
            raise ValueError(f"full logprobs not registered for {request_id}.")
        return state.params
