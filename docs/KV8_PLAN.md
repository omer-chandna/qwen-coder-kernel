# Plan: 8-bit KV cache (bigger context window)

Status: not started. The working repo is `C:\Users\OmerC\qwen-coder-kernel` (GitHub `omer-chandna/qwen-coder-kernel`),
branch `main`. Do not push to kernel-sanders. Ask before any GPU use (model loads, benchmarks, tests).

## Goal

Raise the usable context from 40,960 tokens to ~75-80k on the 8 GB RTX 4060 by storing the attention KV cache as
int8 with one fp16 scale per 32 values (Q8_0-style), instead of fp16. The model weights (Q4_K_M) are not touched.

Why: KV costs ~36 KB per token in fp16 (8 attention layers + the MTP layer, 4 KV heads x 256 dims, K and V), so
40k tokens is ~1.4 GB, and the server already uses 6.5 GB. int8 + per-32 scales is 272 bytes per head-vector
instead of 512 (-47%). Side benefit: decode at 32k reads the cache once per step (1.07 GB, 4.8 ms of 26.7 ms), so
~9% faster decode at long context.

## Where the cache lives in the code

- `native/cuda/qwen35_model.cu`
  - `k_attn_prep`: writes K (after q/k norm and RoPE) and V into `L.kc` / `L.vc` (`__half [max_seq][Hkv][256]`).
  - `k_attn_mma2` (default attention, v2): prefill (`!SPLIT`) and decode / verify (`SPLIT`). K/V tiles are loaded
    with `cp.async` into shared memory (fp16), then `ldmatrix` + `mma.m16n8k16`.
  - `k_attn_mma` (v1), `k_attn_part` / `k_attn_pf` (scalar): fallbacks (`AK_Q35_ATTN=1/0`). Simplest is to require
    v2 when the int8 cache is on.
  - `k_attn_combine`: merges split partials (unchanged).
  - MTP block attention: check `emit_mtp` (it has its own KV cache) - it must use the same format.
  - Allocation of `kc` / `vc` and the VRAM estimate: search `kc` in model load.
- `adakernel/serve.py`, `scripts/serve.ps1`: `--ctx`; OpenCode's `"limit": {"context": ...}` in `opencode.json`.

## Design

1. Storage: `int8 kq[max_seq][Hkv][256]` + `__half ks[max_seq][Hkv][8]` (scale per 32 dims), same for V. Quantize
   once in `k_attn_prep` (warp absmax per 32 values, like `quant32`). Decode and verify then read identical values,
   so speculative output still equals greedy.
2. Attention v2 with int8: `cp.async` cannot convert, so copy the int8 tile + scales into a staging buffer, then
   convert to fp16 into the existing `Ks` / `Vs` tiles (exact int8 -> fp16 via `__byte_perm` / `HSUB2` magic or
   `__int2half_rn`, then `HMUL2` by the scale), then a barrier, then `ldmatrix` as today. Shared memory today is
   66 KB; staging adds ~17 KB (fits in 99 KB). Keep the staggered pipeline (V copy during S, K copy during PV) and
   overlap the conversion with mma work where possible.
3. Opt-in first: env `AK_Q35_KV8=1` / server `--kv8`, default off until validated.

## Validation (needs GPU - ask first)

- `python -m pytest tests -q` with `LLAMA_CPP_BIN=C:\Users\OmerC\kernel-sanders\third_party\llama.cpp\build\bin`
  (add `C:\Users\OmerC\kernel-sanders\.conda-cuda\Library\bin` to PATH). All 26 must pass with KV8 on.
- `python -m benchmarks.llama_match`: currently 1351/1351 characters agree; KV8 should stay close.
- Logit correlation KV8 vs fp16 on the long test prompt and at 16k / 32k contexts (new test).
- Speed: `python -m benchmarks.chunk_profile --pos 32000 --sizes 1,12,512`, `benchmarks.long_context`,
  `benchmarks.decode_bench`. Watch long-context prefill attention (108 ms per 512-token chunk at 32k today).
- VRAM: load at `--ctx 65536` and `81920`, check `info["vram_mb"]`.

## Afterwards

Make it the default if it passes; raise `serve.ps1` / README default ctx and the OpenCode limit; document in
`docs/QWEN38_9B.md` (new section + update "Current state" and "Next"); commit and push to qwen-coder-kernel `main`.

## Practical notes from earlier sessions

- Build: `powershell -ExecutionPolicy Bypass -File scripts\build.ps1 -Cuda C:\Users\OmerC\kernel-sanders\.conda-cuda\Library -CMake C:\Users\OmerC\kernel-sanders\.venv\Lib\site-packages\cmake\data\bin\cmake.exe -Ninja C:\Users\OmerC\kernel-sanders\.venv\Scripts\ninja.exe`
  (`-BuildDir build-dev` while the server holds `build\adakernel.dll`; then set `ADAKERNEL_DLL` to the build-dev DLL).
- The server (`python -m adakernel.serve --ctx 40960 --port 8090 --pf-cols 512`) must be stopped to rebuild `build\`.
- Kernel-only harnesses: `scripts\cuda_bench.ps1 -Bench attn|gemm` (no model load) - extend `benchmarks/attn_bench.cu`
  with an int8-KV case to validate the kernel before the model.
- Profiler: `C:\Users\OmerC\kernel-sanders\.conda-ncu\Library\nsight-compute\2025.1.1\target\windows-desktop-win7-x64\ncu.exe`
  (call the exe, not `ncu.bat`); summarize with `python -m benchmarks.sass_hotspots`.
- Source files are CRLF; edit with Python that preserves line endings, avoid writing through PowerShell `Set-Content`
  (adds a BOM) and watch backslash escapes in heredocs.
