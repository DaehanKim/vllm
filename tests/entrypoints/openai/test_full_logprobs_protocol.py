# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest
from pydantic import ValidationError

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
        prompt=[0, 1, 2],
        max_tokens=0,
        full_logprobs={"enabled": True, "positions": [0, 2]},
    )

    assert req.full_logprobs is not None
    assert req.full_logprobs.enabled is True
    assert req.full_logprobs.positions == [0, 2]
    assert req.full_logprobs.top_p == 0.9999
    assert req.full_logprobs.max_top_k == 512


def test_completion_full_logprobs_requires_token_ids() -> None:
    with pytest.raises(ValidationError, match="token ID"):
        CompletionRequest(
            model="test-model",
            prompt="hello world",
            max_tokens=0,
            full_logprobs={"enabled": True},
        )


def test_completion_full_logprobs_requires_max_tokens_zero() -> None:
    with pytest.raises(ValidationError, match="requires `max_tokens`"):
        CompletionRequest(
            model="test-model",
            prompt=[1, 2, 3],
            max_tokens=1,
            full_logprobs={"enabled": True},
        )


def test_completion_full_logprobs_positions_validation() -> None:
    with pytest.raises(ValidationError, match="strictly increasing"):
        CompletionRequest(
            model="test-model",
            prompt=[1, 2, 3],
            max_tokens=0,
            full_logprobs={"enabled": True, "positions": [0, 0]},
        )


def test_chat_request_accepts_full_logprobs() -> None:
    req = ChatCompletionRequest(
        model="test-model",
        messages=[{"role": "user", "content": "hello"}],
        max_completion_tokens=0,
        full_logprobs={"enabled": True},
    )

    assert req.full_logprobs is not None
    assert req.full_logprobs.enabled is True
    assert req.full_logprobs.positions is None


def test_chat_full_logprobs_requires_zero_tokens_and_single_choice() -> None:
    with pytest.raises(ValidationError, match="requires `max_completion_tokens`"):
        ChatCompletionRequest(
            model="test-model",
            messages=[{"role": "user", "content": "hi"}],
            max_completion_tokens=1,
            full_logprobs={"enabled": True},
        )

    with pytest.raises(ValidationError, match="only supports n=1"):
        ChatCompletionRequest(
            model="test-model",
            messages=[{"role": "user", "content": "hi"}],
            max_completion_tokens=0,
            n=2,
            full_logprobs={"enabled": True},
        )


def test_response_choices_include_full_logprobs_payload() -> None:
    payload = FullLogprobsResponse(
        positions=[0],
        token_ids=[[1]],
        logprobs=[[-0.25]],
        tail_mass=[0.0],
        top_p=0.9,
        max_top_k=4,
    )

    completion_choice = CompletionResponseChoice(
        index=0,
        text="",
        full_logprobs=payload,
    )
    assert completion_choice.full_logprobs is payload
    assert completion_choice.full_logprobs.format == "top_p"

    chat_choice = ChatCompletionResponseChoice(
        index=0,
        message=ChatMessage(role="assistant", content="hi"),
        full_logprobs=payload,
    )
    assert chat_choice.full_logprobs is payload
