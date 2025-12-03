# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import numpy as np

from vllm.full_logprobs import (
    FullLogprobsBuffer,
    FullLogprobsChunk,
    FullLogprobsParams,
)
from vllm.sampling_params import RequestOutputKind, SamplingParams
from vllm.v1.engine import EngineCoreOutput, EngineCoreRequest
from vllm.v1.engine.output_processor import OutputProcessor


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

    sampling_params = SamplingParams(
        skip_special_tokens=False,
        spaces_between_special_tokens=False,
        output_kind=RequestOutputKind.DELTA,
        stop=[],
        include_stop_str_in_output=False,
        detokenize=False,
    )
    engine_request = EngineCoreRequest(
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
    output_processor.add_request(engine_request, prompt=None)

    row = np.array([0.25, -0.5], dtype=np.float16)
    chunk = FullLogprobsChunk(positions=[0], data=row.tobytes())
    engine_output = EngineCoreOutput(
        request_id=request_id,
        new_token_ids=[],
        full_logprobs_chunks=[chunk],
    )

    output_processor.process_outputs([engine_output])
    dense = buffer.build_dense_array(request_id)
    np.testing.assert_array_equal(dense, row.reshape(1, vocab_size))
