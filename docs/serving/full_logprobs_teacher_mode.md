# Full Logprobs Teacher Mode (Experimental)

This mode turns vLLM into a forward-only “teacher” that returns full-vocabulary log-probabilities for every prompt token (an `[L, V]` matrix). It is opt-in because it increases model extraction risk and host memory usage.

## Enable the API

Start the server with the extra flag:

```bash
vllm serve <model> --enable-full-logprobs-api ...
```

Key constraints:

- `max_tokens=0`, `stream=false`, `n=1` (teacher mode bypasses the usual `max_tokens>=1` guard)
- `prompt` must be token IDs (not text) to avoid tokenizer drift
- Host RAM scales with `L * V` fp16 per request; use `positions` to limit rows
- `full_logprobs` is a top-level field (not under `extra_body`)

## Completion request example

```json
POST /v1/completions
{
  "model": "teacher-model",
  "prompt": [101, 2045, 102],
  "max_tokens": 0,
  "stream": false,
  "n": 1,
  "full_logprobs": {
    "enabled": true,
    "positions": [0, 2],       // optional subset; omit/null for all
    "dtype": "fp16",
    "format": "base64_dense"
  }
}
```

## Response shape

Each choice gets a `full_logprobs` block:

```json
"full_logprobs": {
  "shape": [2, V],
  "dtype": "fp16",
  "format": "base64_dense",
  "encoding": "base64",
  "positions": [0, 2],
  "data": "<base64 of row-major little-endian fp16 matrix>"
}
```

Decode on the client:

```python
import base64, numpy as np
raw = base64.b64decode(full_lp["data"])
arr = np.frombuffer(raw, dtype="<f2").reshape(full_lp["shape"])
```

Notes:

- Values are **normalized log-probabilities** (`log_softmax`), not logits.
- Chat completions use the same top-level `full_logprobs` payload.
- Usage accounting remains prompt-only (`completion_tokens=0`).
