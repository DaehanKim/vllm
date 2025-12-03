# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from vllm.entrypoints.openai.protocol import (
    ChatCompletionRequest,
    ChatCompletionResponseChoice,
    ChatMessage,
    CompletionRequest,
    CompletionResponseChoice,
    FullLogprobsResponse,
)


def test_completion_request_accepts_full_logprobs() -> None:
    req = CompletionRequest(
        model="test-model",
        prompt="hello",
        max_tokens=0,
        full_logprobs={"enabled": True, "positions": [0, 2]},
    )

    assert req.full_logprobs is not None
    assert req.full_logprobs.enabled is True
    assert req.full_logprobs.positions == [0, 2]


def test_chat_request_accepts_full_logprobs() -> None:
    req = ChatCompletionRequest(
        model="test-model",
        messages=[{"role": "user", "content": "hello"}],
        max_tokens=0,
        full_logprobs={"enabled": True},
    )

    assert req.full_logprobs is not None
    assert req.full_logprobs.enabled is True
    assert req.full_logprobs.positions is None


def test_response_choices_include_full_logprobs_payload() -> None:
    payload = FullLogprobsResponse(shape=(1, 2), positions=[0], data="AAAA")

    completion_choice = CompletionResponseChoice(
        index=0,
        text="",
        full_logprobs=payload,
    )
    assert completion_choice.full_logprobs is payload
    assert completion_choice.full_logprobs.encoding == "base64"

    chat_choice = ChatCompletionResponseChoice(
        index=0,
        message=ChatMessage(role="assistant", content="hi"),
        full_logprobs=payload,
    )
    assert chat_choice.full_logprobs is payload
