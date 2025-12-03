# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import numpy as np
import pytest

from vllm.full_logprobs import (
    FullLogprobsBuffer,
    FullLogprobsChunk,
    FullLogprobsParams,
)


def _row_bytes(vocab_size: int, offset: float) -> bytes:
    row = np.arange(vocab_size, dtype=np.float16) + offset
    return row.tobytes()


def test_build_dense_array_full_sequence():
    buffer = FullLogprobsBuffer()
    params = FullLogprobsParams(enabled=True, positions=None)
    request_id = "req-1"
    vocab_size = 4
    buffer.register(request_id, vocab_size=vocab_size, params=params)

    chunk = FullLogprobsChunk(
        positions=[0, 1],
        data=_row_bytes(vocab_size, 0.0) + _row_bytes(vocab_size, 10.0),
    )
    buffer.add_chunk(request_id, chunk)

    matrix = buffer.build_dense_array(request_id)
    assert matrix.shape == (2, vocab_size)
    np.testing.assert_allclose(matrix[0], np.arange(vocab_size, dtype=np.float16))
    np.testing.assert_allclose(matrix[1], np.arange(vocab_size, dtype=np.float16) + 10)


def test_positions_filter_only_keeps_requested_rows():
    buffer = FullLogprobsBuffer()
    params = FullLogprobsParams(enabled=True, positions=[1, 3])
    request_id = "req-2"
    vocab_size = 3
    buffer.register(request_id, vocab_size=vocab_size, params=params)

    chunk = FullLogprobsChunk(
        positions=[1, 2, 3],
        data=(
            _row_bytes(vocab_size, 1.0)
            + _row_bytes(vocab_size, 2.0)
            + _row_bytes(vocab_size, 3.0)
        ),
    )
    buffer.add_chunk(request_id, chunk)

    matrix = buffer.build_dense_array(request_id)
    assert matrix.shape == (2, vocab_size)
    np.testing.assert_allclose(matrix[0], np.arange(vocab_size, dtype=np.float16) + 1)
    np.testing.assert_allclose(matrix[1], np.arange(vocab_size, dtype=np.float16) + 3)


def test_positions_filter_missing_row_raises():
    buffer = FullLogprobsBuffer()
    params = FullLogprobsParams(enabled=True, positions=[0, 1])
    request_id = "req-3"
    buffer.register(request_id, vocab_size=2, params=params)

    chunk = FullLogprobsChunk(positions=[0], data=_row_bytes(2, 0.0))
    buffer.add_chunk(request_id, chunk)

    with pytest.raises(ValueError, match="Missing logprobs"):
        buffer.build_dense_array(request_id)


def test_duplicate_registration_rejected():
    buffer = FullLogprobsBuffer()
    params = FullLogprobsParams(enabled=True)
    buffer.register("req-4", vocab_size=1, params=params)
    with pytest.raises(ValueError):
        buffer.register("req-4", vocab_size=1, params=params)
