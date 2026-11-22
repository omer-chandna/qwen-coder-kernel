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

## Tensor-core attention (`k_attn_mma`)

The first attention kernels gave every column its own block and did scalar FMAs, so prefill re-read the whole KV
cache once per prompt token and a 12-column verify pass read it 12 times. `k_attn_mma` is FlashAttention-style:

- A block holds 64 query rows: 16 columns x the 4 query heads that share one KV head (GQA packing).
- Keys stream through shared memory 32 at a time, so each key is read once per block.
- S = QK^T and O += PV are `mma.m16n8k16` (fp16 in, fp32 accumulate) with an online softmax. P never leaves
  registers, since the S accumulator layout of two 8-key tiles is the A-operand layout of one 16-key step.
- The same kernel with a key split (256-key chunks, merged by `k_attn_combine`) handles decode and verify, so one
  pass reads the cache once whatever the draft length.
- 168 registers per thread, no spills, 66 KB of shared memory.
- `set_attention(False)` / `AK_Q35_ATTN_OLD=1` switch back to the scalar kernels.

`benchmarks/long_context.py` (ctx 40960, 512-token prompt chunks, fresh prompts, 128 tokens of speculative decoding).
The first 32 generated tokens were identical between the two kernels at every size:

| prompt | first read: scalar -> mma | decode: scalar -> mma |
|---|---|---|
| 4k | 3.90 s -> 3.64 s (1.07x) | 68.4 -> 67.6 tok/s |
| 12k | 14.3 s -> 12.0 s (1.20x) | 61.3 -> 74.2 tok/s |
| 24k | 36.4 s -> 27.2 s (1.34x) | 51.9 -> 55.3 tok/s |
| 36k | 66.4 s -> 45.8 s (1.45x) | 42.9 -> 52.6 tok/s |

## Where prefill time goes (`Qwen35Model.profile(T, pos)`)

One 512-token chunk, RTX 4060:

| kernel kind | at pos 0 | at pos 32000 |
|---|---|---|
| matrix products (FFN, projections) | 353 ms (84%) | 345 ms (45%) |
| attention (`k_attn_mma`) | 4 ms | 360 ms (47%) |
| Gated DeltaNet recurrence | 43 ms (10%) | 42 ms (5%) |
| everything else | 22 ms | 21 ms |

So the sequential delta-rule recurrence is not the bottleneck: a chunked (WY / UT-transform) rewrite could save at
most ~10%. The matrix products are, and their time was a fixed ~130-180 ms per chunk (expanding every Q4_K / Q6_K
matrix to fp16, ~13 GB written and read back) plus ~0.43 ms per token in cuBLAS. That fixed part made short chunks,
which is what cached agent turns produce, cost nearly as much as 512-token ones.

## Fused quantized-weight GEMM (`qgemm.cuh`)

`k_qgemm` decodes each 128-row x 64-K weight tile straight from the Q4_K / Q6_K blocks into shared memory, with the
same fp16 rounding as the expansion kernel, and multiplies it with `mma.m16n8k16` (fp32 accumulate). So no fp16 copy
of the matrix is ever written. The next tile's raw bytes are fetched into registers while the current one is
multiplied. Token tiles are 64 wide for chunks of up to 128 tokens and 128 wide above that. Chunks of up to 768
tokens use it; cuBLAS remains for longer ones (`set_qgemm` / `AK_Q35_QGEMM_MAX`). The 100 MB fp16 scratch is now
only allocated if a cuBLAS chunk ever runs.

GEMM ms per chunk, cuBLAS incl. expansion -> fused:

| T = 64 | 128 | 256 | 512 | 1024 | 2048 |
|---|---|---|---|---|---|
| 189 -> 63 | 198 -> 101 | 244 -> 175 | 354 -> 320 | 572 -> 611 | 1018 -> 1213 |

Logits match cuBLAS to a correlation of 0.999994. End to end (`benchmarks/followup.py`, 40k context, 512-token
chunks):
- First agent turn, 6.0k tokens: 5.66 -> 5.12 s.
- Follow-up turns adding 387 / 771 / 1432 tokens: 0.45 -> 0.38, 0.88 -> 0.77 and 1.57 -> 1.49 s.
- First read of a 36k prompt: 45.8 -> 43.5 s.

The kernel is still ~35% less efficient per token than cuBLAS at large T, because decoding and multiplying are not
overlapped across tiles.

I also tried fp16 accumulation in cuBLAS (2x tensor rate on GeForce). It cut GEMM time per 512-token chunk only from
350 to 282 ms, and greedy output diverged after 55 tokens on one of three prompts. Not adopted.

## int8 tensor cores (`k_mmq4` / `k_mmq6`, the default)

Prefill matrix products run on int8 tensor cores with the decode path's arithmetic:
- Activations are quantized to int8 per 32 values (Q8_1: scale and scaled sum), once per input.
- Q4_K: an mma operand is a 4-byte nibble word, masked or shifted, so there is no decode pass at all. One
  `mma.m16n8k32.s8` runs per 32-wide sub-block, then `acc += d*sc*xd*dot - dmin*m*xsd`.
- Q6_K: operands are rebuilt from one `ql`, one `ql+32` and one `qh` word (`__vsub4` for q - 32), with one
  `mma.m16n8k16.s8` per 16-group.

Accuracy is the GEMV prefill path's (logit correlation 0.9994 against cuBLAS; it was already the decode path's
arithmetic). The fp16 kernels' output also drifts from cuBLAS after a few dozen greedy tokens on some prompts, so
both are near-tie flips. All tests pass with it, including speculative == greedy and the prefix cache.

How it got there, GEMM ms per 512-token chunk:

| step | GEMM ms | why |
|---|---|---|
| cuBLAS + fp16 expansion | 355 | |
| fp16 fused (`k_qgemm`) | 309 | no expansion |
| int8, first version | 457 | I2F conversions (quarter rate) + 186 registers or spills |
| int8, `__int_as_float(v + 0x4B400000) - 1.5*2^23` | 265 | exact int -> float in two full-rate ops (sums < 2^22) |
| int8, cp.async pipeline (Q4_K) | 235 | the plain kernel spent 143 of 276 ms just loading / decoding / syncing, at ~75 GB/s |
| int8, cp.async pipeline (Q6_K too) | **229-234** | |

Q6_K super-blocks are 210 bytes, so every other one starts 2 bytes off 4-byte alignment. The Q6_K kernel copies
words from the aligned-down address with 4-byte `cp.async` and realigns them with a 16-bit funnel shift. The shift
is uniform per stage, because it depends only on the super-block index.

Tile shape beat occupancy. GEMM ms at T = 64 / 128 / 256 / 512 / 1024:
- BN 64, 3 stages: 49 / 78 / 150 / 289 / 561.
- BN 64, 2 blocks per SM: 55 / 84 / 155 / 296 / 572.
- BN 128, 2 stages: 70 / **66** / **119** / **234** / **441**.

So chunks of up to 64 tokens use BN 64 and longer ones BN 128.

Per-256 activation scales with integer accumulation across sub-blocks were dropped. On Ada an integer multiply-add
issues at half the FP32 rate, so 2 IMADs instead of ~5 float ops per output per 32 values saves little, and it would
cost activation precision.

End to end (40k context, 512-token chunks), `benchmarks/followup.py`, cuBLAS -> fp16 fused -> int8:

| turn | cuBLAS | fp16 fused | int8 |
|---|---|---|---|
| first, 6.0k tokens | 5.60 s | 4.94 s | **4.03 s** |
| follow-up, 347 tokens | 0.44 s | 0.35 s | **0.29 s** |
| follow-up, 748 tokens | 0.87 s | 0.74 s | **0.61 s** |
| follow-up, 1446 tokens | 1.55 s | 1.43 s | **1.19 s** |

`benchmarks/long_context.py` first reads: 4k 2.61 s, 12k 8.86 s, 24k 21.0 s, 36k 36.6 s (45.8 s before this round,
66.4 s with the original kernels). Prompt reading at the start of a context is ~1700 tok/s, against llama.cpp's
~2230.

Switches: `set_int8(False)` / `AK_Q35_INT8=0` (fp16 fused / cuBLAS), `AK_Q35_MMQ_SHORT` / `AK_Q35_MMQ_LONG` (tile
configurations) and `AK_Q35_QDIAG` (diagnostics).

## Attention v2 (`k_attn_mma2`, the default)

v2 keeps v1's tiling (64 query rows x 32-key tiles, 4 warps, one block per SM) and its arithmetic. It changes only
how data reaches the tensor cores. Its output is **bit-identical** to v1 in every case tested. `set_attention(1)` /
`AK_Q35_ATTN=1` switch back to v1, and `AK_Q35_ATTN=0` to the scalar kernels.

### Results

Kernel alone (`scripts\attn_bench.ps1`: random Q/K/V, model shapes, median of 20, all outputs bit-identical). The
times cover one attention layer:

| case | v1 ms | v2 ms | speedup |
|---|---:|---:|---:|
| prefill chunk 512 @ pos 0 | 0.51 | 0.34 | 1.49x |
| prefill chunk 512 @ pos 4000 | 4.73 | 2.16 | 2.20x |
| prefill chunk 512 @ pos 16000 | 22.1 | 6.60 | 3.35x |
| prefill chunk 512 @ pos 32000 | 43.6 | 13.0 | 3.35x |
| prefill chunk 1024 @ pos 32000 | 80.5 | 24.4 | 3.30x |
| follow-up chunk 64 @ pos 32000 | 7.28 | 2.14 | 3.41x |
| decode, 1 column @ pos 4000 / 32000 | 0.129 / 1.31 | 0.064 / 0.559 | 2.03x / 2.35x |
| verify, 12 columns @ pos 4000 / 32000 | 0.161 / 1.55 | 0.094 / 0.682 | 1.71x / 2.27x |

One 512-token prefill chunk in the model (`profile(512, pos)`, ms):

| | pos 0 | pos 16000 | pos 32000 |
|---|---:|---:|---:|
| whole chunk, v1 | 296 | 468 | 633 |
| whole chunk, v2 | 297 | 348 | **401** |
| attention, v1 -> v2 | 4.1 -> 2.9 | 182 -> 56 | 355 -> **108** |
| matrix products | 226 | 222-227 | 216-227 |

At 32k, attention goes from 56% of a chunk to 27%. Matrix products (57%) are the largest cost again at every length.
One eager decode step at pos 32000 goes from 32.9 to 26.5 ms (attention 11.0 -> 4.6 ms). At pos 4000 it goes from
23.5 to 22.7 ms.

End to end, `benchmarks/long_context.py --kernels 1,2` (ctx 40960, 512-token chunks, fresh prompt, 128 tokens of
speculative decoding; first 32 tokens identical between kernels at every size). Raw:
`experiments/results/attn_v2_long_context.json`.

| prompt | first read: v1 -> v2 | decode: v1 -> v2 |
|---|---|---|
| 4k | 2.60 s -> 2.53 s | 80.4 -> 84.1 tok/s |
| 12k | 8.84 s -> 7.88 s (1.12x) | 75.6 -> 81.9 tok/s |
| 24k | 21.0 s -> 16.7 s (1.26x) | 56.1 -> 64.2 tok/s |
| 36k | 36.6 s -> 26.5 s (1.38x) | 62.5 -> 78.3 tok/s (1.25x) |

The first read of a 36k prompt has now gone 66.4 s (scalar) -> 45.8 s (tensor-core v1) -> 36.6 s (int8 GEMMs) ->
26.5 s, which is ~1360 tok/s averaged over the whole prompt. Decode rates differ between sizes mostly because draft
acceptance depends on the prompt. Compare the two kernels within a row, not rows with each other.

All 18 model tests pass with v2 as the default, including `test_attention_v2_matches_v1` (identical logits and
speculative tokens, batched and GEMV prefill), speculative == greedy, and the prefix cache.

### Why v1 was slow

Per 32-key tile, each warp runs 128 `mma.m16n8k16`: 64 for S = QK^T and 64 for PV. A 512-token chunk at pos 32000
does 2.7e11 FLOPs of attention per layer. v1 ran that at **6.1 TFLOPS** (44 ms), and v2 runs it at **20 TFLOPS**
(13.5 ms). Before measuring, I assumed the 4060's fp16-input / fp32-accumulate tensor rate was ~15 TFLOPS. That would
have put v1 at 40% of peak and capped v2 at ~2x. v2 sustaining 20 TFLOPS shows the assumption was too low, so v1 was
running at under a third of what the hardware does. The costs v2 removes:

1. **Exposed load latency.** v1 loads the K and V tile through registers (`ld.global` -> `st.shared`), then calls
   `__syncthreads`, then computes. There are only 4 warps per SM, because the 66 KB of shared memory allows one block,
   so nothing covers the L2 round trips. The tensor cores idle for every tile load.
2. **Shared-load instruction count.** Per tile and warp, v1 issues 64 `LDS.32` for Q fragments and 128 for K
   fragments. For PV, V is stored key-major while the B operand wants pairs along the key axis, so v1 issues 256
   `LDS.U16` plus 128 packs. That is 448 load instructions per 128 mmas, all from one warp per scheduler.

### What v2 changes, and why

- **`ldmatrix` for every operand.** `ldmatrix.x4` loads four 8x8 fp16 matrices, and each lane receives exactly the
  register layout of an mma fragment:
  - Q: one instruction per 16-dim step, giving a[0..3].
  - K: one per pair of 8-key tiles, since K rows are keys, which is already the layout B needs.
  - V: `ldmatrix.trans`, one per pair of 8-dim tiles. The transpose happens in the load, which removes the 16-bit
    gathers and packs.

  That is 80 instead of 448 load instructions per 128 mmas. The padded 528-byte row stride (`kHd + 8` halves) is
  4 banks per row, so the 8 row addresses of an 8x8 matrix hit 8 different 4-bank groups, with no conflicts. That is
  why the padding was kept rather than switching to an XOR swizzle.
- **`cp.async`, staggered by operand instead of double-buffered.** A full double buffer (Q + 2 K + 2 V) needs
  101,376 bytes, exactly the 99 KB per-block limit. v2 keeps one K and one V buffer (66 KB, same as v1) and staggers
  them:
  - copy V(t) while S = Q K(t)^T and the softmax run;
  - copy K(t+1) while O += P V(t) runs.

  Each copy is hidden behind ~3k cycles of math. It costs two barriers per tile, the same as v1. Rows past the range
  are zero-filled by `cp.async` (src-size 0), so masked keys never multiply garbage (0 x NaN = NaN) in PV.
- **The arithmetic is unchanged on purpose.** The mma order per accumulator, the fp16 rounding of Q and P, and the
  softmax are the same as v1, so v2 must be bit-identical. That made the check strict and cheap (any difference is a
  bug, not a rounding question). It also keeps "speculative output == greedy" without new tolerance arguments.
- **Not done:**
  - fp16 accumulation for S (2x tensor rate on GeForce). Logits reach ~30, where the fp16 spacing is 0.016-0.03,
    which would shift attention weights by 1-3%. This is the same trade rejected for cuBLAS.
  - 64-key tiles. They halve the barriers, but Q + K + V at 64 keys is again exactly 99 KB, plus 16 more registers.
  - More warps per block. Q (33 KB) dominates shared memory, so 8 warps (128 Q rows) would not fit with K/V.

Registers rose from 168 to 236-242 per thread, because the unrolled `ldmatrix` loads are hoisted. There are no
spills, and occupancy is unchanged: shared memory already limits the SM to one block.

How it was checked: `benchmarks/attn_bench.cu` includes `qwen35_model.cu`, so it launches the runtime's own
file-local kernels with the model's shapes, with no model load (~150 MB of VRAM). It runs v1 and v2 on identical
inputs and compares the outputs byte for byte. A kernel change can be validated in seconds, and while another
process holds most of the GPU.

## Agent turns: where the int8 GEMM time goes, split-K, snapshots inside a chunk

This round started from the int8 GEMM (the largest prefill cost after attention v2) and ended up mostly speeding up
**agent follow-up turns** (100-800 new tokens on a cached prefix), which is what an OpenCode session spends most of
its prompt time on.

### Measuring the GEMM in isolation (`benchmarks/gemm_bench.cu`)

`scripts\cuda_bench.ps1 -Bench gemm` runs `k_mmq4` / `k_mmq6` on random, well-formed Q4_K / Q6_K weights at the
model's shapes (no model load). It checks sampled outputs against a CPU fp64 reference that uses the same Q8_1
activations: the max relative error is ~1e-7, so the arithmetic is exact up to float rounding. It also times
diagnostic variants (template parameter `DIAG`, compiled out of the runtime) that are wrong by design but show
where time goes. Matrix shapes and types come from the GGUF tensor table:
- Q4_K: FFN gate / up, half of FFN down, half of the DeltaNet qkv projections.
- Q6_K: the other halves, plus some attn_v.

FFN gate (12288 x 4096, Q4_K), 512 tokens: ideal int8 tensor time 0.43 ms (60 TMAC/s).

| variant | ms |
|---|---:|
| full kernel | 1.40 |
| float epilogue reduced to one add per output (DIAG 1) | 0.97 |
| no min term (DIAG 2) | 1.21 |
| no mma and minimal epilogue: copies, syncs, fragment loads only (DIAG 4) | 0.79 |
| no global -> shared copies, compute only (DIAG 5) | 1.13 |

So global memory is hidden (DIAG 5 is close to the full kernel). The per-sub-block float epilogue is ~30%, and the
"plumbing" (syncs, scale decode, shared-memory fragment loads) alone is over half. Q6_K is ~2x slower than Q4_K
per MAC (4.6-11 TMAC/s against 10-18): its scales cover 16 weights, so every `m16n8k16` mma (half the MACs of the
Q4_K `k32` one) carries the same 4-output float fixup.

What I tried, and what came of it:
- **Issuing all independent mmas of a step before their epilogues:** 1-4%, within noise. ptxas already schedules
  them well. Dropped.
- **128-bit loads for the scale factors.** My count was 160 of Q4_K's 240 shared loads per warp and stage being
  scalar broadcasts of scales. For Q4_K this was a regression: 1.40 -> 1.46 ms, because splitting the activation
  scales into 8-byte copies doubled the `cp.async` count, and the compiler had probably already merged the other
  loads. Reverted for Q4_K. Kept for Q6_K: weight scales stored `[row][lg][q]`, activation scales read as `float4`,
  and Q6_K no longer copies the unused `xsd`. Together with replacing `__vsub4`, this gave 5-17% in the kernel bench
  (noisy) and was within noise in the model.
- **`q - 32` per byte without `__vsub4`.** sm_89 has no SIMD byte-subtract instruction, so `__vsub4` is emulated.
  `(q + 0x60606060) ^ 0x80808080` is exact for q in [0, 63]: adding 0x60 cannot carry into the next byte, and
  flipping bit 7 subtracts 128 mod 256.
- Not tried, with reasons:
  - Combining the two 16-groups of a Q6_K 32-block in integer arithmetic before converting to float. The sum can
    reach 2^24, beyond the exact 2-op int -> float trick (|v| < 2^22), and I2F runs at a quarter of the FMA rate.
  - fp16 mma for Q6_K. On GeForce Ada, fp16 with fp32 accumulate runs at a quarter of the int8 rate, so its ceiling
    (~15 TMAC/s) is barely above today's 11.

### Split-K for short chunks

The same kernel ran ffn_down (4096 rows) at 10 TMAC/s and ffn_gate (12288 rows) at 14.4 for 64 tokens. One block
(128 rows x up to 128 tokens) runs per SM, and a 4096-row matrix gives 32 blocks for chunks of up to 128 tokens.
On 24 SMs that is 1.33 waves, so the second wave runs on 8 SMs and leaves 16 idle.

Split-K: `blockIdx.z` takes 1/z of K and writes partial sums to a scratch plane, and `k_splitk_sum` adds the z
planes in a fixed order, so results stay deterministic. `splitk_choose` models time as ceil(z * blocks / slots) / z
and picks the best z (2-4, dividing K's super-blocks, at least 1024 K per split). It splits only when the model
predicts at least 20% less time. Without that margin, the model's small predicted gains turned into measured
losses (the reduction pass and the shorter K loops cost something), for example on the 8192-row qkv at 64 tokens.

| matrix, tokens | no split | split-K | z |
|---|---:|---:|---:|
| ffn_down Q4_K, 64 / 128 | 0.298 / 0.498 ms | 0.234 / 0.340 ms | 3 |
| ffn_down Q6_K, 64 / 128 | 0.580 / 0.811 ms | 0.406 / 0.566 ms | 3 |
| ssm_out Q4_K (4096 x 4096), 64 / 128 | 0.100 / 0.165 ms | 0.083 / 0.136 ms | 2 |

In the model, a 64-token chunk takes 57.5 -> 49.0 ms and a 128-token chunk 83.5 -> 74.4 ms. Chunks of 256 or more
are unchanged, as designed.

Side effect: a split changes the float summation order, so a token's activations now depend slightly on the length
of the chunk it was processed in. Before, a column's GEMM result was independent of T. The output is still
deterministic for a given prompt and chunking. `set_splitk(1)` / `AK_Q35_SPLITK=1` turn split-K off.

### Snapshots inside a chunk

`benchmarks/chunk_profile.py` replays an agent session on a frozen corpus (`benchmarks/data/agent_source.txt`,
so turn sizes stay fixed as the source changes). A 148-token follow-up took 177 ms, while a 128-token chunk takes
75 ms and a 256-token one 150 ms. The extra time came from how the prefix cache took its snapshots:
- `snap_points` asks for the recurrent state at up to three `<|im_start|>` positions per request, the last being the
  generation prompt.
- `generate_spec_ex` ended a prefill segment at each of them and called `snap_copy`.
- So every turn ended in a separate pass for the 3-5 tokens of `<|im_start|>assistant\n`.

Any pass streams all 4.9 GB of weights (~22 ms), however few tokens it has. Segment tails of <= 24 tokens also fell
back to GEMV passes of 12 tokens at ~22 ms each.

Now the snapshot is taken inside the pass that covers it:
- `SnapReq` lists up to 4 columns plus slot pointers.
- `k_conv` and `k_gdn` hold the state rows in registers while they walk the chunk's columns. When they pass a
  requested column, they also store that state into the slot, at its per-layer offset in `snap_copy`'s layout.
- `prefill` processes the whole new range in one call, and chunks are balanced: rem -> ceil(rem / pf_cols) equal
  chunks, so no short remainder is left for the GEMV path.
- GEMV passes that take a snapshot run eagerly, since a CUDA graph would bake in the request.
- The snapshot path is a template flag (`k_gdn<true>`). The first version checked `sr.n` inside `k_gdn`'s inner
  loop and made that kernel 9% slower on every pass (43.9 -> 48.0 ms per 512-token chunk). Interleaved A/B runs
  caught it.

Correctness:
- `test_snapshot_inside_chunk` takes a snapshot at column 699 of a 1000-token chunk and checks that it is
  bit-identical to the state after a prefill of exactly 700 tokens. It runs with split-K off, because otherwise the
  two chunk lengths use different splits and the recurrence amplifies 1e-7 rounding differences to ~0.2 over 700
  tokens. That is legitimate rounding, not a bug: GEMV vs batched prefill differ by 0.35 the same way.
- A token-level check was not enough. With the snapshot column off by one, the 24 greedy tokens after restoring
  still matched, so the first version of the test passed a wrong implementation. The state comparison fails it.
- The test also checks that greedy tokens after restoring the snapshot equal a fresh run.
- New API for this: `ak_q35_cache_save`, `ak_q35_cache_read` (`save_state`, `cache_state`) and `ak_q35_set_splitk`.

### Results (agent session replay, ctx 40960, 512-token chunks, interleaved A/B against `main`)

| turn (new tokens) | before | after | |
|---|---:|---:|---:|
| first, 5968 | 3.84 s | 3.87 s | within noise |
| follow-up, 148 | 0.177 s | **0.148 s** | -17% |
| follow-up, 375 | 0.268 s | **0.245 s** | -9% |
| follow-up, 748 | 0.530 s | **0.494 s** | -7% |
| follow-up, 1446 | 1.028 s | 1.017 s | -1% |

Raw: `experiments/results/turns_main_{0,1}.json` (before, two interleaved rounds) and `turns_splitk_snap.json` (after).

Also in this round, the attention-vs-scalar test read its prompt from the live `qwen35_model.cu`, so editing that
file's header moved a correlation from just above to just below the 0.9995 threshold. v1 and v2 gave identical logits
either way, so the prompt changed, not the kernels. Its prompt now comes from a frozen copy
(`tests/data/long_prompt_source.txt`).

## Next

- **int8 GEMM plumbing** (over half of `k_mmq4`'s time per the DIAG runs): each stage has two `__syncthreads` around
  a cooperative scale-decode phase that the tensor cores sit out. Decoding the scales for stage t+1 during stage t,
  or per thread in registers, would remove one barrier per stage.
- **Gated DeltaNet recurrence**: 43 ms of a 512-token chunk (15%) and ~20% of short chunks, the next largest item.
  A chunked (WY / UT-transform) formulation would move most of it onto tensor cores.
- **The GEMV / GEMM threshold** (`kPfMin` = 24): a whole new range of 13-24 tokens still takes two GEMV passes
  (~44 ms). If a 24-token GEMM chunk is cheaper (likely, since its fixed cost is the ~22 ms weight read), lower it.
- 8-bit KV cache: halves the bytes decode reads per pass at long context, and fits 64k+ contexts in 8 GB.
