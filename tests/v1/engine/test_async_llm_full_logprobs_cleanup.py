# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest

from vllm.v1.engine.async_llm import AsyncLLM


class _DummyOutputProcessor:
    def __init__(self, return_ids: list[str]):
        self.return_ids = return_ids
        self.abort_calls: list[list[str]] = []

    def abort_requests(self, request_ids):
        self.abort_calls.append(list(request_ids))
        return self.return_ids


class _DummyInputProcessor:
    def __init__(self):
        self.cleaned: list[str] = []

    def cleanup_full_logprobs_request(self, request_id: str) -> None:
        self.cleaned.append(request_id)


class _DummyEngineCore:
    def __init__(self):
        self.aborted: list[list[str]] = []

    async def abort_requests_async(self, request_ids):
        self.aborted.append(list(request_ids))

    def shutdown(self):
        pass


@pytest.mark.asyncio
async def test_async_llm_abort_cleans_full_logprobs():
    llm = object.__new__(AsyncLLM)
    llm.output_processor = _DummyOutputProcessor(["b", "c"])
    llm.input_processor = _DummyInputProcessor()
    llm.engine_core = _DummyEngineCore()
    llm.log_requests = False

    await AsyncLLM.abort(llm, ["a", "b"])

    assert llm.output_processor.abort_calls == [["a", "b"]]
    assert llm.engine_core.aborted == [["b", "c"]]
    assert set(llm.input_processor.cleaned) == {"a", "b", "c"}
