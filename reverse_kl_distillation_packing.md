# Reverse-KL On-Policy Distillation: Bandwidth & Latency-Oriented Redesign

## Context
We are performing **on-policy distillation with reverse KL**:

\[ KL(p_teacher || p_student) \]

Original setup:
- Teacher sends **all-token logprobs**
- Payload encoded as **base64(JSON)**
- Result: excessive bandwidth + unnecessary latency

Goal:
- Preserve **meaningful reverse-KL training signal**
- **Minimize latency first**, compression ratio second
- Keep implementation practical in a **FastAPI-based HTTP pipeline**

---

## 1. Distribution Truncation Strategy

### Why truncation is dangerous for reverse KL
Reverse KL strongly penalizes cases where:
- Teacher assigns non-trivial probability
- Student assigns near-zero probability

Naive top-k truncation (e.g. top-5, top-10) destroys this signal.

### Adopt **top-p with tail preservation**
**Recommended configuration**:
- `top_p = 0.9999`
- `k_max = 512` (hard cap for worst-case entropy spikes)

This ensures:
- Teacher tail probability mass \( < 1e^{-4} \)
- Reverse KL gradient distortion is minimal

**Full logit service defaults**:
- `top_p` and `max_top_k` are optional in the request payload
- Defaults are `top_p=0.9999` and `max_top_k=512`

### Tail handling ("other bucket")
Instead of discarding the tail:
- Compute:
  - `tail_mass = 1 - sum(p_topk)`
- Treat tail as a single `"other"` bucket in KL computation

Approximate KL term:
```
Σ p_t(i) * (log p_t(i) - log p_s(i))  for i in top-k
+ p_t(other) * (log p_t(other) - log p_s(other))
```

This preserves reverse-KL behavior with minimal payload increase.

---

## 2. Full Logit Service Payload

The OpenAI-compatible full-logprobs response is JSON, so:
- **Remove base64 entirely**
- Send **sparse per-position arrays** instead of a dense matrix

Response structure (per choice):
```
full_logprobs = {
  format: "top_p",
  dtype: "fp16",
  top_p: 0.9999,
  max_top_k: 512,
  positions: [0, 2],         # null if all positions
  token_ids: [[...], [...]],
  logprobs: [[...], [...]],
  tail_mass: [.., ..]
}
```

Semantics:
- `token_ids[i]`, `logprobs[i]`, and `tail_mass[i]` align to `positions[i]`
- If `positions` is `null`, rows map to absolute prompt positions `[0..L-1]`
- `tail_mass[i] = 1 - sum(exp(logprobs[i]))`

---

## 3. Optional Binary Transport (Future)

If you control both client and server, you can further shrink payloads:
- msgpack for binary serialization
- zstd (level 1–3) for fast compression

This is optional and not required for the current full-logprobs JSON response.

---

## 4. Receiver-Side Changes (FastAPI, binary-only)

Instead of:
```python
@app.post(...)
async def endpoint(payload: PydanticModel):
    ...
```

Use:
```python
@app.post(...)
async def endpoint(req: Request):
    body = await req.body()
    body = zstd.decompress(body)
    payload = msgpack.unpackb(body, raw=False)
```

Key change:
- JSON parsing → raw byte handling
- No Pydantic model at ingress layer

---

## 5. Optional Optimizations (Future)

Only if bandwidth becomes critical again:

### a) Chunk multiple positions per request
- Compress 16–64 token positions together
- Dramatically improves compression ratio

### b) Logprob quantization
- fp16 → int8
- Per-chunk `(scale, offset)`
- ~2× reduction on logprob payload

### c) Union trick
- Transmit:
  - teacher top-k ∪ student top-m
- Reduces reverse-KL blind spots

---

## 6. Migration Summary

### Before
- base64(JSON)
- all-token logprobs
- high bandwidth
- unnecessary latency

### After (recommended)
- JSON sparse payload (token IDs + logprobs + tail_mass)
- No base64
- top_p = 0.9999 + k_max = 512
- explicit tail mass handling

Result:
- Bandwidth drastically reduced
- Latency improved
- Reverse-KL signal preserved

---

## TL;DR
> **Top-p + tail bucket + no base64**  
> Optional binary transport can come later if needed.
