# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest

from vllm.full_logprobs import (
    FullLogprobsBuffer,
    FullLogprobsChunk,
    FullLogprobsParams,
    FullLogprobsRow,
)
from vllm.sampling_params import RequestOutputKind, SamplingParams
from vllm.v1.engine import EngineCoreOutput, EngineCoreRequest, FinishReason
from vllm.v1.engine.output_processor import OutputProcessor


def _make_engine_request(
    request_id: str, params: FullLogprobsParams
) -> EngineCoreRequest:
    sampling_params = SamplingParams(
        skip_special_tokens=False,
        spaces_between_special_tokens=False,
        output_kind=RequestOutputKind.DELTA,
        stop=[],
        include_stop_str_in_output=False,
        detokenize=False,
    )
    return EngineCoreRequest(
        request_id=request_id,
        prompt_token_ids=[1, 2],
        mm_features=None,
        sampling_params=sampling_params,
        pooling_params=None,
        eos_token_id=None,
        arrival_time=0.0,
        lora_request=None,
        cache_salt=None,
        data_parallel_rank=None,
        full_logprobs_params=params,
    )


def test_full_logprobs_chunks_are_buffered():
    buffer = FullLogprobsBuffer()
    request_id = "request-full-logprobs"
    params = FullLogprobsParams(enabled=True)
    vocab_size = 2
    buffer.register(request_id, vocab_size=vocab_size, params=params)

    output_processor = OutputProcessor(
        tokenizer=None,
        log_stats=False,
        full_logprobs_buffer=buffer,
    )

    engine_request = _make_engine_request(request_id, params)
    output_processor.add_request(engine_request, prompt=None)

    row = FullLogprobsRow(
        position=0,
        token_ids=[0, 1],
        logprobs=[0.25, -0.5],
        tail_mass=0.1,
    )
    chunk = FullLogprobsChunk(rows=[row])
    engine_output = EngineCoreOutput(
        request_id=request_id,
        new_token_ids=[],
        full_logprobs_chunks=[chunk],
    )

    output_processor.process_outputs([engine_output])
    positions, token_ids, logprobs, tail_mass = buffer.build_response_payload(
        request_id
    )
    assert positions is None
    assert token_ids == [[0, 1]]
    assert logprobs == [[0.25, -0.5]]
    assert tail_mass == [0.1]


def test_full_logprobs_cleanup_on_finish():
    buffer = FullLogprobsBuffer()
    params = FullLogprobsParams(enabled=True)
    request_id = "request-full-logprobs-finish-cleanup"
    vocab_size = 2
    buffer.register(request_id, vocab_size=vocab_size, params=params)

    output_processor = OutputProcessor(
        tokenizer=None,
        log_stats=False,
        full_logprobs_buffer=buffer,
    )
    output_processor.add_request(_make_engine_request(request_id, params), prompt=None)

    row = FullLogprobsRow(
        position=0,
        token_ids=[0, 1],
        logprobs=[0.0, 1.0],
        tail_mass=0.0,
    )
    chunk = FullLogprobsChunk(rows=[row])
    engine_output = EngineCoreOutput(
        request_id=request_id,
        new_token_ids=[],
        full_logprobs_chunks=[chunk],
        finish_reason=FinishReason.LENGTH,
    )

    output_processor.process_outputs([engine_output])

    positions, token_ids, logprobs, tail_mass = buffer.build_response_payload(
        request_id
    )
    assert positions is None
    assert token_ids == [[0, 1]]
    assert logprobs == [[0.0, 1.0]]
    assert tail_mass == [0.0]
    output_processor._cleanup_full_logprobs(request_id)
    with pytest.raises(ValueError, match="full logprobs not registered"):
        buffer.params_for(request_id)


def test_full_logprobs_cleanup_on_abort():
    buffer = FullLogprobsBuffer()
    params = FullLogprobsParams(enabled=True)
    request_id = "request-full-logprobs-abort-cleanup"
    buffer.register(request_id, vocab_size=1, params=params)

    output_processor = OutputProcessor(
        tokenizer=None,
        log_stats=False,
        full_logprobs_buffer=buffer,
    )
    output_processor.add_request(_make_engine_request(request_id, params), prompt=None)

    output_processor.abort_requests([request_id])

    with pytest.raises(ValueError, match="full logprobs not registered"):
        buffer.params_for(request_id)
