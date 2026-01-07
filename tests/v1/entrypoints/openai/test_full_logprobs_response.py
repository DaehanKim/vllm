# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace

import pytest

from vllm.entrypoints.openai.protocol import (
    CompletionRequest,
    FullLogprobsRequest,
    RequestResponseMetadata,
)
from vllm.entrypoints.openai.serving_completion import OpenAIServingCompletion
from vllm.full_logprobs import (
    FullLogprobsBuffer,
    FullLogprobsChunk,
    FullLogprobsParams,
    FullLogprobsRow,
)
from vllm.outputs import CompletionOutput, RequestOutput


def _make_buffer_with_rows(request_id: str, vocab_size: int) -> FullLogprobsBuffer:
    buffer = FullLogprobsBuffer()
    params = FullLogprobsParams(enabled=True, positions=[0, 2])
    buffer.register(request_id, vocab_size=vocab_size, params=params)

    rows = [
        FullLogprobsRow(
            position=0,
            token_ids=[0, 1],
            logprobs=[0.0, 1.0],
            tail_mass=0.0,
        ),
        FullLogprobsRow(
            position=2,
            token_ids=[0, 1],
            logprobs=[2.0, 3.0],
            tail_mass=0.1,
        ),
    ]
    buffer.add_chunk(request_id, FullLogprobsChunk(rows=rows))
    return buffer


def test_completion_response_includes_full_logprobs_and_cleans_buffer():
    request_id = "req-full-logprobs"
    vocab_size = 2
    buffer = _make_buffer_with_rows(request_id, vocab_size)

    serving = object.__new__(OpenAIServingCompletion)
    serving.input_processor = SimpleNamespace(full_logprobs_buffer=buffer)
    serving._full_logprobs_request_ids = {request_id}
    serving.enable_prompt_tokens_details = False
    serving.return_tokens_as_token_ids = False

    cleanup_calls: list[str] = []

    def _cleanup(req_id: str) -> None:
        cleanup_calls.append(req_id)
        serving._full_logprobs_request_ids.discard(req_id)
        buffer.cleanup(req_id)

    serving._cleanup_full_logprobs_request = _cleanup

    request = CompletionRequest(
        model="test-model",
        prompt=[1, 2],
        max_tokens=0,
        full_logprobs=FullLogprobsRequest(enabled=True),
        stream=False,
    )
    request_metadata = RequestResponseMetadata(request_id=request_id)

    completion_output = CompletionOutput(
        index=0,
        text="",
        token_ids=[5],
        logprobs=None,
        finish_reason="stop",
        stop_reason=None,
        cumulative_logprob=0.0,
    )
    final_res = RequestOutput(
        request_id=request_id,
        prompt="hi",
        prompt_token_ids=[1, 2],
        prompt_logprobs=None,
        outputs=[completion_output],
        finished=True,
        metrics=None,
        lora_request=None,
        encoder_prompt=None,
        encoder_prompt_token_ids=None,
        num_cached_tokens=None,
    )

    response = serving.request_output_to_completion_response(
        [final_res],
        request,
        request_id="cmpl-1",
        created_time=123,
        model_name="model",
        tokenizer=None,
        request_metadata=request_metadata,
    )

    choice = response.choices[0]
    assert choice.full_logprobs is not None
    assert choice.full_logprobs.positions == [0, 2]
    assert choice.full_logprobs.token_ids == [[0, 1], [0, 1]]
    assert choice.full_logprobs.logprobs == [[0.0, 1.0], [2.0, 3.0]]
    assert choice.full_logprobs.tail_mass == [0.0, 0.1]
    assert choice.full_logprobs.top_p == 0.9999
    assert choice.full_logprobs.max_top_k == 512

    assert cleanup_calls == [request_id]
    with pytest.raises(ValueError, match="full logprobs not registered"):
        buffer.params_for(request_id)
