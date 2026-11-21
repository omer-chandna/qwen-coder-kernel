# qwen-coder-kernel

A CUDA runtime for Qwen3.5-architecture GGUF files (`qwen35`: Gated DeltaNet layers with gated attention every
fourth layer, plus the MTP head), tuned for coding on an 8 GB RTX 4060, with an OpenAI-compatible server so
OpenCode (or Continue, Aider, ...) can use it.

Built and measured against `eriktonon/qwen3.8-uncensored:9b` (Q4_K_M, from Ollama). Split out of the
`kernel-sanders` experiments repo (branch `qwen35-gdn`), where the benchmark scripts and llama.cpp comparisons live.

What is in it:
- **Decode:** dp4a GEMVs on Q4_K / Q6_K / Q8_0 weights, CUDA graphs, fused ops: 46 tok/s plain greedy (llama.cpp:
  44).
- **Speculative decoding:** MTP-head drafts and prompt-lookup drafts, chosen per round from measured costs,
  verified up to 12 tokens per pass. Output is bit-identical to greedy. 106-127 tok/s on coding tasks.
- **Rollback:** replay-commit for the recurrent state, so rejected drafts cost no state copies.
- **Batched prefill:** fp16 tensor-core GEMMs in 1024-token chunks (~1100 tok/s at 4k context).
- **Prefix cache:** recurrent-state snapshots at chat-message boundaries, so each agent step only processes what
  is new (0.3-0.5 s per step instead of re-reading 7k-24k tokens).
- **Server:** `adakernel/serve.py`, OpenAI chat completions with streaming and tool calls, rendered with the GGUF's
  own chat template.

Details, measurements and next steps: [docs/QWEN35_9B.md](docs/QWEN35_9B.md).

## Build

Needs CUDA 12.8+ (for sm_89), VS 2022 Build Tools, CMake 3.24+ and Ninja.

```powershell
powershell -ExecutionPolicy Bypass -File scripts\build.ps1 -Cuda <cuda toolkit root>
```

The script records the toolkit's `bin` folder in `build/cuda_bin.txt`, and the Python loader uses it to find cudart
and cuBLAS. Python needs `numpy` and `jinja2`; the tests also need `pytest`.

## Run with OpenCode

```powershell
powershell -ExecutionPolicy Bypass -File scripts\serve.ps1            # -Ctx 32768 -Port 8090 -Model <gguf>
```

Then pick `adakernel/qwen3.8-9b` in OpenCode. The provider entry for `~/.config/opencode/opencode.json` is in the
doc. The default model path is the Ollama blob of qwen3.8-uncensored:9b; pass `-Model` for another qwen35 GGUF.

Limits: one request at a time, greedy decoding only (temperature is ignored), 32k context by default (about 40k fits
in 8 GB).

## Tests

```powershell
python -m pytest tests -q
```

`tests/test_serve.py` runs on the CPU. `tests/test_qwen35.py` loads the model on the GPU (~6 GB). Set
`LLAMA_CPP_BIN` to a llama.cpp `build/bin` folder to also check token and greedy-output parity with llama.cpp.
