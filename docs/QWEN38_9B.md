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

## Attention v2 (`k_attn_mma2`): status and reasoning

**Status: written, compiled (0 spills), not yet measured.** The GPU was busy with a running server during this round,
so the kernel is opt-in (`AK_Q35_ATTN=2` / `set_attention(2)`). The default stays `k_attn_mma` (v1) until the
checks below pass on the card. Then the default becomes a one-line change (`int attn = 1` in `qwen35_model.cu`).

How to check it, with no model load (random Q/K/V, ~150 MB of VRAM):

```powershell
powershell -ExecutionPolicy Bypass -File scriptsttn_bench.ps1
```

`benchmarks/attn_bench.cu` includes `qwen35_model.cu`, so it launches the runtime's own kernels with the model's
shapes (4 KV heads x 4 query heads, head dim 256, max_seq 40960). It runs v1 and v2 on identical inputs for prefill
chunks (T = 64 / 512 / 1024 at positions 0-32000) and decode / verify passes (T = 1 / 12). It prints the time of
each and whether the outputs are bit-identical. With the model: `pytest tests/test_qwen35.py -k attention_v2`
(identical logits and speculative tokens, batched and GEMV prefill).

### Why v1 runs at ~40% of the tensor-core rate

A 512-token chunk at position 32000 spends 360 ms in attention over 8 layers. Each layer runs 4 KV heads x 32 row
tiles = 128 blocks, and each block walks ~1000 key tiles of 32 keys. That is ~1.02 M block-tiles. With one block per
SM (66 KB of shared memory each, 24 SMs), that comes to ~42.7k tiles per SM, or **~8.4 us (~20.7k cycles) per tile**.

The math per tile is 128 `mma.m16n8k16` per warp: 64 for S = QK^T and 64 for PV. On GeForce Ada, fp16 with fp32
accumulate runs at about 127 FMA / clock / SM (~15 TFLOPS on the 4060), so each sub-partition (one warp here) needs
about 64 cycles per mma. That makes **~8.2k cycles of tensor work per tile**, about 40% of the measured time. The
other ~60% goes to three costs:

1. **Load latency, fully exposed.** v1 loads the K/V tile through registers (`for i ... ld.global; st.shared`),
   then `__syncthreads`, then computes. Only 4 warps are on the SM, so nothing else can run while ~16 KB come back
   from L2. That costs a few thousand cycles per tile, serialized with the math.
2. **Shared-load instruction count.** Per 32-key tile and warp, v1 issues 64 `LDS.32` for Q fragments and 128 for K
   fragments. For PV it issues 256 `LDS.U16` plus 128 pack instructions, because V is stored key-major and the B
   operand wants pairs along the key axis. That makes **448 shared-load instructions per 128 mmas**, all from 1 warp
   per scheduler, which crowds out mma issue.
3. Softmax and rescale (~0.6k cycles): masking, 4 shuffles, 16 exps and 128 `FMUL`s on `o` per tile. This part stays.

### What v2 changes, and why these choices

- **`ldmatrix` for every operand.** `ldmatrix.x4` loads four 8x8 fp16 matrices, and each lane ends up with exactly
  the register layout an mma fragment needs:
  - Q: 1 instruction per 16-dim step, giving a[0..3].
  - K: 1 instruction per pair of 8-key tiles, since K rows are keys, which is already the "B transposed" layout.
  - V: `ldmatrix.trans`, 1 instruction per pair of 8-dim tiles. The transpose happens in the load, so the
    16-bit gathers and packs go away.

  That gives **80 instead of 448 load instructions per 128 mmas** (5.6x fewer). The 528-byte padded row stride
  (`kHd + 8` halves) stays: it is 4 banks per row, so the 8 row addresses of each 8x8 matrix land in 8 different
  4-bank groups, with no bank conflicts. That is why the padding was kept instead of switching to an XOR swizzle.
- **`cp.async`, split by operand instead of double-buffered.** A full double buffer (2 x K + 2 x V + Q) needs
  101,376 bytes, which is exactly the 99 KB per-block limit. It would leave no slack and no room for anything else.
  v2 keeps one K and one V buffer (66 KB, same as v1) and staggers them:
  - copy V(t) while S = Q K(t)^T and the softmax run;
  - copy K(t+1) while O += P V(t) runs.

  Each copy overlaps one compute phase of about 4k cycles, which is much longer than an L2 round trip, so the load
  latency is hidden. It costs two barriers per tile, the same number as v1. Rows past the end of the range are
  zero-filled by `cp.async` (src-size 0), so masked keys never multiply garbage (0 x NaN = NaN) in PV.
- **The arithmetic is unchanged on purpose.** The same mma order per accumulator, the same fp16 rounding of Q and P,
  and the same softmax code. So v2 must be **bit-identical** to v1. That makes the check strict and cheap: any
  difference is a bug, not a rounding question. It also keeps the "speculative output == greedy" guarantee intact
  without new tolerance arguments.
- **Not done:**
  - fp16 accumulation for S (2x tensor rate on GeForce). Logits reach ~30, where the fp16 spacing is 0.016-0.03,
    so attention weights would be off by 1-3%. That is the same accuracy trade that was rejected for cuBLAS.
  - A larger key tile (BK = 64). It halves the barriers, but Q + K + V at 64 keys is again exactly 99 KB, and the S
    accumulators would add 16 more registers to the 242 used now.
  - More warps per block. Q (33 KB) dominates shared memory, so 8 warps would need 128 rows of Q and would not fit
    with K/V tiles.

Expected effect, an estimate until measured: tensor work plus softmax is ~9-10k cycles per tile, against 20.7k
today. So attention could get up to ~2x faster: at 32k, ~360 -> ~190 ms per 512-token chunk, or ~770 -> ~600 ms
for the whole chunk. Decode and verify at long context use the same kernel (the SPLIT variant), so they gain too,
but less, since they are bound by weight bandwidth.

Registers: v2 uses 236-242 per thread against v1's 167-168, because the unrolled `ldmatrix` loads are hoisted. There
are no spills, and occupancy is unchanged: shared memory already limits the SM to one 4-warp block.

## Next

- **Measure attention v2** (`scriptsttn_bench.ps1`, then `benchmarks/long_context.py` with `AK_Q35_ATTN=2`).
  If it is bit-identical and faster, make it the default and record the numbers above.
- **The int8 GEMM is now latency-bound:** ~0.45 ms per token against an instruction-issue estimate of ~0.16. Options:
  - a 256-row block of 16 warps, for more reuse per loaded activation tile (Q4_K fits in shared memory; Q6_K needs a
    single stage);
  - `ldmatrix` for the activation fragments;
  - moving the Q4_K min term out of the inner loop. Note: doing that exactly needs the same number of FMAs elsewhere
    (it is a rows x T x K/32 product). It only pays off if that product goes on tensor cores, in fp16, which costs
    accuracy against the decode path. So do this one last.
- 8-bit KV cache: halves the bytes decode reads per pass at long context, and fits 64k+ contexts in 8 GB.
