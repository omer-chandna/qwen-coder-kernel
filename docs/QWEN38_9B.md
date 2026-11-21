# Qwen3.8 9B on the AdaKernel runtime

The model is Qwen3.8 9B. "qwen35" in this document and in the code (`qwen35_model.cu`, `Qwen35Model`, `ak_q35_*`)
is the name of its GGUF architecture: Qwen3.8 9B is a fine-tune of Qwen3.5-9B and keeps its hybrid layout. It does
not refer to the stock `qwen3.5:9b` model, which was not used here.

Machine: RTX 4060 8 GB, Ryzen 5 7600X, Windows 11. Model: Ollama `eriktonon/qwen3.8-uncensored:9b`, blob
`sha256-c7f6915f…3408` (5.78 GB, Q4_K_M, GGUF architecture `qwen35`, a fine-tune of Qwen3.5-9B). 32 layers: 24 Gated
DeltaNet, 8 gated full attention (every fourth), plus one MTP (multi-token prediction) block, used as the speculative draft. llama.cpp commit
`5d4c8ee`, `-ngl 99 -fa 1`. Raw numbers (gitignored): `experiments/results/qwen38_9b_decode.json`, `llamacpp_qwen38_9b.json`.

## Runtime

`native/cuda/qwen35_model.cu`, Python `adakernel.qwen35.Qwen35Model`. Matrices go through the existing dp4a GEMVs.
New kernels: causal conv1d with rolling state, gated delta rule (fp32 128x128 state per value head, one warp per
state row), gated output RMSNorm, q/k RMSNorm with NeoX RoPE on 64 of 256 dims, split-key GQA attention for head
size 256 with the sigmoid output gate. The token embedding (546 MB) stays in pinned host memory; 4.93 GB on the
device at max_seq 2048. One decode step is one CUDA graph replay. Prompts are fed in chunks of 6 tokens through the multi-column path.

## Correctness

Greedy, 64 tokens, against `llama-completion --temp 0`: three of four prompts match character for character. The
fourth ("Explain why the sky is blue…") matches for 59 tokens, then picks " (" where llama.cpp picks " of"; the two
logits are 27.314 and 27.303 (probabilities 0.500 / 0.494), a tie that rounding decides. Graph and eager decode
produce identical tokens.

## Decode, back to back in one session

| Context | AdaKernel tok/s (median) | llama.cpp tok/s (median) | ratio |
| --- | ---: | ---: | ---: |
| 5 tokens, 128 generated | 46.20 (45.8–47.0, 5 runs) | 44.32 (43.1–44.7, 5 runs) | 1.04 |
| 1024 tokens, 128 generated | 46.33 (45.9–46.7, 2 runs) | 43.64 (43.3–43.7, 3 runs) | 1.06 |

llama.cpp prefill for this model is 2230 tok/s at 512 tokens. This runtime prefills in 6-token chunks (the verify
path): 205 tok/s on a 1024-token prompt, up from about 45 tok/s token by token, still 11x behind llama.cpp.

## Where one step goes (eager, ~1 k context, 22.6 ms of GPU time)

| Kernel | ms | share |
| --- | ---: | ---: |
| FFN gate+up GEMV | 7.57 | 33.5% |
| FFN down GEMV | 4.62 | 20.5% |
| LM head GEMV (Q6_K, 248k rows) | 3.30 | 14.6% |
| DeltaNet qkv+z GEMV | 3.26 | 14.5% |
| other GEMVs | 2.34 | 10.4% |
| norms, conv, delta rule, attention, rest | 1.49 | 6.6% |

GEMVs are 93% of the step and read 5.04 GB in about 21 ms (about 235 GB/s, against a 246 GB/s copy peak on this
card). The delta-rule recurrence, which reads and writes 2 MB of state per layer, is 0.31 ms in total.

## MTP speculative decoding

`ak_q35_generate_mtp` / `Qwen35Model.generate_mtp(prompt, n, n_draft=k)`. Each round the MTP block drafts k tokens
(the first from the target's last hidden state, the rest chained on its own output), then the target verifies all
k+1 positions in one multi-column pass, so its 4.9 GB of weights are read once per round. Conv and delta-rule state
is snapshotted after every column, so a rejection resumes from the last accepted column with no recompute. Output is
identical to plain greedy decoding for every prompt and k tested (asserted in the benchmark and in
`tests/test_qwen35.py`). Extra VRAM: about 0.4 GB (5.3 GB total). Each draft step reads 0.99 GB, most of it the shared
Q6_K LM head.

256 generated tokens, median of 3 runs, tok/s (speedup over plain decode in this runtime). Raw: `experiments/results/qwen38_9b_mtp.json`.

| Prompt | plain | k=1 | k=2 | k=3 | k=4 | k=5 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| The capital of France is | 46.1 | 75.7 (1.64x) | 96.8 (2.10x) | 111.0 (2.41x) | 115.6 (2.51x) | **118.7 (2.58x)** |
| def quicksort(arr): ... | 46.0 | 74.8 (1.62x) | 92.5 (2.01x) | **103.6 (2.25x)** | 101.8 (2.21x) | 98.9 (2.15x) |
| Explain why the sky is blue ... | 46.2 | 71.7 (1.55x) | 85.6 (1.85x) | **92.4 (2.00x)** | 87.9 (1.90x) | 81.8 (1.77x) |
| Write a short story about a robot ... | 46.1 | 64.8 (1.41x) | 69.2 (1.50x) | **72.2 (1.57x)** | 62.2 (1.35x) | 57.0 (1.24x) |

Draft acceptance at k=3: 0.99, 0.90, 0.76, 0.53 (same row order); tokens per round 3.98, 3.70, 3.27, 2.58. k=3 is the
best single default: it is the best or within 7% of the best on every prompt. Against llama.cpp's plain decode
(44.3 tok/s), k=3 is 1.6x to 2.5x.

llama.cpp has its own MTP mode (`llama-speculative --spec-type draft-mtp`), but with `-md` pointing at the same file it
loads the full model a second time; two copies do not fit in 8 GB and it ran at 1.4 tok/s, so it is not a usable
comparison on this card.

## Coding workload (`benchmarks/coding_bench_q35.py`)

Three chat requests carrying real source from this repo, 256 generated tokens each, median of 2, RTX 4060 8 GB.
`spec` = `generate_spec`: per-round choice between MTP drafts and prompt lookup, draft length picked from measured
costs and per-depth acceptance (up to 11), MTP head scoring only the first 65536 token ids. Every mode produced
exactly the greedy tokens. Prefill chunk was 512 for this table.

| task (prompt tok) | greedy | mtp k=3 | **spec** | spec, full-vocab drafts | llama.cpp |
|---|---|---|---|---|---|
| write tests (1799) | 7.17 s, 46 tok/s | 4.36 s, 97 tok/s | **3.90 s, 119 tok/s** | 4.33 s, 98 tok/s | 6.62 s, 44 tok/s |
| refactor (740) | 6.24 s, 46 tok/s | 3.29 s, 100 tok/s | **2.74 s, 127 tok/s** | 3.11 s, 107 tok/s | 6.10 s, 44 tok/s |
| explain (948) | 6.33 s, 46 tok/s | 3.65 s, 92 tok/s | **3.27 s, 106 tok/s** | 3.71 s, 90 tok/s | 6.20 s, 44 tok/s |

- End to end, `spec` is 1.7-2.2x faster than llama.cpp; decode alone is 2.3-2.9x.
- The 64K draft vocabulary is a clear win (8-19% faster decode than full-vocab drafts) and accepts as many tokens
  per round or more, so low token ids do cover the common code tokens.
- Batched prefill: 205 -> ~1080 tok/s at 512-token chunks. Raising the chunk to 1024 (now the default) cut
  first-token time a further 14-22% (1799-token prompt 1.67 s -> 1.44 s; 740-token 0.70 s -> 0.57 s).
  llama.cpp is still about 2x faster at prompt reading (0.80 s / 0.35 s), and for these tasks first-token time is now
  ~40% of the total.

## Using it from OpenCode (or any OpenAI-compatible client)

```powershell
powershell -ExecutionPolicy Bypass -File scripts\serve.ps1   # -Ctx 32768 -Port 8090
```

`adakernel/serve.py` serves `POST /v1/chat/completions` (streaming, tools) and `GET /v1/models` on
http://127.0.0.1:8090/v1 with model id `qwen3.8-9b`. Port 8090 because OpenCode auto-discovers :8000 as vLLM with tools
off. OpenCode provider entry (`~/.config/opencode/opencode.json`):

```json
{
  "$schema": "https://opencode.ai/config.json",
  "providers": {
    "adakernel": {
      "name": "AdaKernel (local)",
      "package": "@opencode/ai/providers/openai-compatible",
      "settings": { "baseURL": "http://127.0.0.1:8090/v1" },
      "models": {
        "qwen3.8-9b": {
          "modelID": "qwen3.8-9b",
          "name": "Qwen3.8 9B (AdaKernel)",
          "capabilities": { "tools": true, "input": ["text"], "output": ["text"] },
          "limit": { "context": 32768, "output": 8192 }
        }
      }
    }
  }
}
```

How it works:
- Prompts are rendered with the GGUF's own Jinja chat template; tool calls come back as Qwen's
  `<tool_call><function=..><parameter=..>` XML and are converted to OpenAI `tool_calls` using each tool's JSON schema
  for argument types. Generation stops at `<|im_end|>`; a client disconnect (Esc in OpenCode) stops it too.
- Tokenizing is in-process (`adakernel/qwen_tokenizer.py`, byte-level BPE from the GGUF; identical ids to
  llama-tokenize on the test corpus, 33k tokens in ~75 ms).
- **Prefix cache.** Agents resend the whole conversation every step. KV entries are position-indexed and stay valid,
  but the Gated DeltaNet state cannot be rewound, so the runtime snapshots the recurrent state (~50 MB, 4 slots) at up
  to three `<|im_start|>` boundaries per request: end of the system prompt, the first reply after the last real user
  message (the template drops the empty think block of those replies once a new user message arrives) and the
  generation prompt. The next request restores the longest matching snapshot and only processes the rest
  (`ak_q35_generate_spec_ex`, `Qwen35Model.generate_cached`). Output equals uncached greedy (tested).
- One request at a time; greedy only (temperature / top_p ignored).

Measured (RTX 4060, ctx 32768, VRAM 6.2 GB for the model + ~0.4 GB prefill scratch):

| | first prompt | follow-up turn (cached) | decode |
|---|---|---|---|
| 4k-token prompt | 3.6 s | 0.32 s | 75-97 tok/s |
| 12k | 12.7 s | 0.34 s | 54-64 tok/s |
| 24k | 33.5 s | 0.50 s | 47 tok/s |

A real OpenCode session ("fix two bugs in stats.py, run it"): 4 requests, 6.6k-token system prompt read once in 6.3 s,
then 0.3-0.55 s per step with 6.6-7.1k tokens cached, 82-100 tok/s, edits applied and verified; 13.6 s end to end.

Context: the file declares 262144 tokens, but KV costs ~36 KB/token (8 attention layers + MTP, fp16), so 8 GB fits
about 40k next to the weights. 8-bit KV would roughly double that.

## Next

- Long-context attention: both the prefill kernel (one block per column, re-reading all keys) and the verify kernel
  (one block per column) scale with context x columns; decode drops from ~95 to 47 tok/s and prefill from ~1100 to
  ~700 tok/s by 24k. A flash-style tiled prefill kernel and a multi-column verify kernel that loads each key once
  would fix both.
- 8-bit KV cache for 64k+ contexts.
- Prefill is now the bottleneck. Even one 740-token chunk only reaches ~1300 tok/s, so the cost is in the per-token
  kernels (sequential Gated DeltaNet recurrence over T, fp16 dequant per chunk, attention), not chunking: profile
  prefill and move to a chunked (WY / UT-transform) delta rule so the recurrence also runs on tensor cores.
- Keep the prompt's KV/recurrent state across requests (prefix cache) so follow-up edits on the same file skip
  prefill entirely.
