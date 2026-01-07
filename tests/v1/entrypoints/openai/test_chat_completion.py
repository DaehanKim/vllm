# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import numpy as np
import openai  # use the official client for correctness check
import pytest
import pytest_asyncio
import torch

try:
    _ = torch.ops.vllm.dequant_mxfp4  # type: ignore[attr-defined]
except (AttributeError, RuntimeError):
    pytest.skip("custom ops not available; skipping OpenAI server tests", allow_module_level=True)

from tests.utils import RemoteOpenAIServer
from vllm.entrypoints.openai.protocol import ChatCompletionRequest

# any model with a chat template defined in tokenizer_config should work here
MODEL_NAME = "Qwen/Qwen2.5-1.5B-Instruct"


@pytest.fixture(scope="module")
def default_server_args():
    return [
        # use half precision for speed and memory savings in CI environment
        "--max-model-len",
        "2048",
        "--max-num-seqs",
        "128",
        "--enforce-eager",
        "--enable-full-logprobs-api",
    ]


@pytest.fixture(scope="module")
def server(default_server_args):
    with RemoteOpenAIServer(MODEL_NAME, default_server_args) as remote_server:
        yield remote_server


@pytest_asyncio.fixture
async def client(server):
    async with server.get_async_client() as async_client:
        yield async_client


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "model_name",
    [MODEL_NAME],
)
async def test_invalid_json_schema(client: openai.AsyncOpenAI, model_name: str) -> None:
    invalid_json_schema = {
        "$defs": {
            "CarType": {
                "enum": ["sedan", "SUV", "Truck", "Coupe"],
                "title": "CarType",
                "type": "string",
            }
        },
        "properties": {
            "brand": {"title": "Brand", "type": "string"},
            "model": {"title": "Model", "type": "string"},
            "car_type": {"$ref": "#/$defs/CarType"},
            "foo": "bar",
        },
        "required": ["brand", "model", "car_type"],
        "title": "CarDescription",
        "type": "object",
    }
    prompt = (
        "Generate a JSON with the brand, model and car_type of"
        "the most iconic car from the 90's"
    )
    with pytest.raises((openai.BadRequestError, openai.APIError)):
        await client.chat.completions.create(
            model=model_name,
            messages=[
                {
                    "role": "user",
                    "content": prompt,
                }
            ],
            extra_body={"structured_outputs": {"json": invalid_json_schema}},
        )


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "model_name",
    [MODEL_NAME],
)
async def test_invalid_regex(client: openai.AsyncOpenAI, model_name: str):
    prompt = (
        "Generate an email address for Alan Turing, who works in Enigma."
        "End in .com and new line. Example result:"
        "alan.turing@enigma.com\n"
    )

    with pytest.raises((openai.BadRequestError, openai.APIError)):
        await client.chat.completions.create(
            model=model_name,
            messages=[
                {
                    "role": "user",
                    "content": prompt,
                }
            ],
            extra_body={"structured_outputs": {"regex": r"[.*"}, "stop": ["\n"]},
        )


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "model_name",
    [MODEL_NAME],
)
async def test_invalid_grammar(client: openai.AsyncOpenAI, model_name: str):
    invalid_simplified_sql_grammar = """
        root ::= select_statementinvalidsyntax

        select_statement ::= "SELECT " column " from " table " where " condition

        column ::= "col_1 " | "col_2 "

        table ::= "table_1 " | "table_2 "

        condition ::= column "= " number

        number ::= "1 " | "2 "
    """

    prompt = (
        "Generate an SQL query to show the 'username' and 'email'"
        "from the 'users' table."
    )
    with pytest.raises((openai.BadRequestError, openai.APIError)):
        await client.chat.completions.create(
            model=model_name,
            messages=[
                {
                    "role": "user",
                    "content": prompt,
                }
            ],
            extra_body={
                "structured_outputs": {"grammar": invalid_simplified_sql_grammar}
            },
        )


def _assert_full_logprobs_mass(choice) -> None:
    assert choice.full_logprobs is not None
    for row_logprobs, tail_mass in zip(
        choice.full_logprobs.logprobs, choice.full_logprobs.tail_mass
    ):
        total_mass = float(np.exp(np.array(row_logprobs)).sum()) + tail_mass
        np.testing.assert_allclose(total_mass, 1.0, rtol=1e-2, atol=1e-2)


@pytest.mark.asyncio
async def test_chat_full_logprobs(client: openai.AsyncOpenAI):
    messages = [{"role": "user", "content": "hello"}]
    chat = await client.chat.completions.create(
        model=MODEL_NAME,
        messages=messages,
        max_tokens=0,
        stream=False,
        extra_body={"full_logprobs": {"enabled": True}},
    )

    choice = chat.choices[0]
    full_lp = choice.full_logprobs
    assert full_lp is not None

    _assert_full_logprobs_mass(choice)
    assert chat.usage.completion_tokens == 0
    assert chat.usage.total_tokens == chat.usage.prompt_tokens


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "model_name",
    [MODEL_NAME],
)
async def test_empty_grammar(client: openai.AsyncOpenAI, model_name: str) -> None:
    prompt = "Say hello"
    with pytest.raises((openai.BadRequestError, openai.APIError)):
        await client.chat.completions.create(
            model=model_name,
            messages=[
                {
                    "role": "user",
                    "content": prompt,
                }
            ],
            extra_body={"structured_outputs": {"grammar": ""}},
        )


def test_chat_sampling_params_include_full_logprobs_extra_args() -> None:
    req = ChatCompletionRequest(
        model="test-model",
        messages=[{"role": "user", "content": "hi"}],
        max_completion_tokens=0,
        full_logprobs={"enabled": True},
    )

    params = req.to_sampling_params(
        max_tokens=1,
        logits_processor_pattern=None,
        default_sampling_params={},
    )

    assert params.extra_args is not None
    payload = params.extra_args.get("full_logprobs")
    assert payload == {
        "enabled": True,
        "top_p": 0.9999,
        "max_top_k": 512,
        "positions": None,
        "dtype": "fp16",
        "format": "top_p",
    }
