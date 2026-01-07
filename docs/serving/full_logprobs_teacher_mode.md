# Full Logprobs Teacher Mode (Experimental)

This mode turns vLLM into a forward-only “teacher” that returns sparse top-p log-probabilities plus a tail-mass bucket for every prompt token. It is opt-in because it increases model extraction risk and host memory usage.

## Enable the API

Start the server with the extra flag:

```bash
vllm serve <model> --enable-full-logprobs-api ...
```

Key constraints:

- `max_tokens=0`, `stream=false`, `n=1` (teacher mode bypasses the usual `max_tokens>=1` guard)
- `prompt` must be token IDs (not text) to avoid tokenizer drift
- Host RAM scales with `L * K` where `K = min(max_top_k, top_p token count)`; use `positions` to limit rows
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
    "top_p": 0.9999,           // optional; defaults to 0.9999
    "max_top_k": 512,          // optional; defaults to 512
    "dtype": "fp16",
    "format": "top_p"
  }
}
```

## Response shape

Each choice gets a `full_logprobs` block:

```json
"full_logprobs": {
  "dtype": "fp16",
  "format": "top_p",
  "top_p": 0.9999,
  "max_top_k": 512,
  "positions": [0, 2],
  "token_ids": [[123, 456], [789, 101]],
  "logprobs": [[-0.1, -1.2], [-0.3, -2.0]],
  "tail_mass": [0.0008, 0.0005]
}
```

Notes on the payload:

- `token_ids[i]`, `logprobs[i]`, and `tail_mass[i]` align to `positions[i]`.
- If `positions` is `null`, rows are ordered by absolute position `[0..L-1]`.
- `tail_mass[i]` is the remaining probability mass not included in `token_ids[i]`.

Notes:

- Values are **normalized log-probabilities** (`log_softmax`) for the top-p subset, not logits.
- Chat completions use the same top-level `full_logprobs` payload.
- Usage accounting remains prompt-only (`completion_tokens=0`).
