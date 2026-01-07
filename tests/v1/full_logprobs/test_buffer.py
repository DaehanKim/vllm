# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest

from vllm.full_logprobs import (
    FullLogprobsBuffer,
    FullLogprobsChunk,
    FullLogprobsParams,
    FullLogprobsRow,
)


def _row(position: int, offset: float) -> FullLogprobsRow:
    return FullLogprobsRow(
        position=position,
        token_ids=[0, 1, 2],
        logprobs=[offset, offset + 1.0, offset + 2.0],
        tail_mass=0.1 + position * 0.05,
    )


def test_build_response_payload_full_sequence():
    buffer = FullLogprobsBuffer()
    params = FullLogprobsParams(enabled=True, positions=None)
    request_id = "req-1"
    buffer.register(request_id, vocab_size=4, params=params)

    chunk = FullLogprobsChunk(rows=[_row(0, 0.0), _row(1, 10.0)])
    buffer.add_chunk(request_id, chunk)

    positions, token_ids, logprobs, tail_mass = buffer.build_response_payload(
        request_id
    )
    assert positions is None
    assert token_ids == [[0, 1, 2], [0, 1, 2]]
    assert logprobs == [[0.0, 1.0, 2.0], [10.0, 11.0, 12.0]]
    assert tail_mass == [0.1, 0.15]


def test_positions_filter_only_keeps_requested_rows():
    buffer = FullLogprobsBuffer()
    params = FullLogprobsParams(enabled=True, positions=[1, 3])
    request_id = "req-2"
    buffer.register(request_id, vocab_size=3, params=params)

    chunk = FullLogprobsChunk(rows=[_row(1, 1.0), _row(2, 2.0), _row(3, 3.0)])
    buffer.add_chunk(request_id, chunk)

    positions, token_ids, logprobs, tail_mass = buffer.build_response_payload(
        request_id
    )
    assert positions == [1, 3]
    assert token_ids == [[0, 1, 2], [0, 1, 2]]
    assert logprobs == [[1.0, 2.0, 3.0], [3.0, 4.0, 5.0]]
    assert tail_mass == [0.15, 0.25]


def test_positions_filter_missing_row_raises():
    buffer = FullLogprobsBuffer()
    params = FullLogprobsParams(enabled=True, positions=[0, 1])
    request_id = "req-3"
    buffer.register(request_id, vocab_size=2, params=params)

    chunk = FullLogprobsChunk(rows=[_row(0, 0.0)])
    buffer.add_chunk(request_id, chunk)

    with pytest.raises(ValueError, match="Missing logprobs"):
        buffer.build_response_payload(request_id)


def test_duplicate_registration_rejected():
    buffer = FullLogprobsBuffer()
    params = FullLogprobsParams(enabled=True)
    buffer.register("req-4", vocab_size=1, params=params)
    with pytest.raises(ValueError):
        buffer.register("req-4", vocab_size=1, params=params)


def test_build_response_payload_collects_out_of_order_chunks():
    buffer = FullLogprobsBuffer()
    params = FullLogprobsParams(enabled=True, positions=None)
    request_id = "req-5"
    buffer.register(request_id, vocab_size=2, params=params)

    # Deliver chunks out of order; buffer should sort and enforce contiguity.
    buffer.add_chunk(request_id, FullLogprobsChunk(rows=[_row(2, 20.0)]))
    buffer.add_chunk(request_id, FullLogprobsChunk(rows=[_row(0, 0.0), _row(1, 10.0)]))

    positions, token_ids, logprobs, tail_mass = buffer.build_response_payload(
        request_id
    )
    assert positions is None
    assert token_ids == [[0, 1, 2], [0, 1, 2], [0, 1, 2]]
    assert logprobs == [
        [0.0, 1.0, 2.0],
        [10.0, 11.0, 12.0],
        [20.0, 21.0, 22.0],
    ]
    assert tail_mass == [0.1, 0.15, 0.2]


def test_positions_filter_across_multiple_chunks():
    buffer = FullLogprobsBuffer()
    params = FullLogprobsParams(enabled=True, positions=[0, 2])
    request_id = "req-6"
    buffer.register(request_id, vocab_size=3, params=params)

    buffer.add_chunk(request_id, FullLogprobsChunk(rows=[_row(1, 1.0)]))
    buffer.add_chunk(request_id, FullLogprobsChunk(rows=[_row(0, 0.0), _row(2, 2.0)]))

    positions, token_ids, logprobs, tail_mass = buffer.build_response_payload(
        request_id
    )
    assert positions == [0, 2]
    assert token_ids == [[0, 1, 2], [0, 1, 2]]
    assert logprobs == [[0.0, 1.0, 2.0], [2.0, 3.0, 4.0]]
    assert tail_mass == [0.1, 0.2]
