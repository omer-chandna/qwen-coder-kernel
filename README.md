<a id="readme-top"></a>

<p align="center">
  <img src=".github/readme/hero.svg" alt="qwen-coder-kernel: a custom CUDA kernel that runs Qwen faster in OpenCode. 119 tok/s vs llama.cpp's 44 on an RTX 4060." width="100%">
</p>

<p align="center">
  <b>A custom CUDA kernel that runs Qwen faster in OpenCode.</b><br>
  106-127 tok/s on coding tasks on an 8 GB RTX 4060 (llama.cpp: 44), with output bit-identical to greedy decoding.
</p>

<p align="center">
  <a href="#build"><img alt="CUDA 12.8+ (sm_89)" src="https://img.shields.io/badge/CUDA-12.8%2B%20%C2%B7%20sm__89-3ee6a0?style=flat-square&labelColor=11161e&logo=nvidia&logoColor=white"></a>
  <a href="pyproject.toml"><img alt="Python 3.10+" src="https://img.shields.io/badge/python-3.10%2B-3ee6a0?style=flat-square&labelColor=11161e&logo=python&logoColor=white"></a>
  <a href="#build"><img alt="Windows" src="https://img.shields.io/badge/platform-Windows-3ee6a0?style=flat-square&labelColor=11161e&logo=windows&logoColor=white"></a>
</p>

<p align="center">
  <a href="#demo">Demo</a> · <a href="#quickstart">Quickstart</a> · <a href="#numbers">Numbers</a> · <a href="#how-it-works">How it works</a> · <a href="docs/QWEN38_9B.md">Full write-up</a>
</p>

## Demo

<p align="center">
  <a href=".github/readme/brag.mp4"><img src=".github/readme/demo.gif" alt="Two terminals stream the same pytest file: llama.cpp at 44 tok/s, qwen-coder-kernel at 119 tok/s." width="100%"></a>
  <!-- Playable video: drag .github/readme/brag.mp4 into the GitHub README editor and paste the user-attachments URL here. -->
</p>

<p align="center"><sub>Same model, same card, same coding prompt. <a href=".github/readme/brag.mp4">Watch the 20-second video</a>.</sub></p>

<img src=".github/readme/divider.svg" width="100%" alt="">

## Why

A CUDA runtime for **Qwen3.8 9B** (`eriktonon/qwen3.8-uncensored:9b` from Ollama), tuned for coding on an 8 GB
RTX 4060, with an OpenAI-compatible server so OpenCode (or Continue, Aider, ...) can use it as a local provider.

> [!NOTE]
> **Why the code says "qwen35".** Qwen3.8 9B is a fine-tune of Qwen3.5-9B, so it keeps that model's layout: Gated
> DeltaNet layers with gated attention every fourth layer, plus the MTP head. Its GGUF file labels that layout
> `qwen35`, and the runtime is named after it (`qwen35_model.cu`, `Qwen35Model`). The name means the layout, not the
> model version. The stock `qwen3.5:9b` from Ollama is not what this repo was built or measured on.

```powershell
powershell -ExecutionPolicy Bypass -File scripts\serve.ps1   # http://127.0.0.1:8090/v1
# OpenCode → adakernel/qwen3.8-9b
#   106-127 tok/s on coding tasks · agent steps start in 0.3-0.55 s
```

## Features

<p align="center">
  <img src=".github/readme/feature-spec.png" alt="Speculative decoding: 106-127 tok/s on coding tasks, up to 12 drafted tokens verified per pass, output bit-identical to greedy." width="49%">
  <img src=".github/readme/feature-cache.png" alt="Prefix cache: first request with a 6.6k-token prompt takes 6.3 s, each agent step after takes 0.3-0.55 s." width="49%">
</p>
<p align="center">
  <img src=".github/readme/feature-prefill.png" alt="Batched prefill: fp16 tensor-core GEMMs read prompts at about 1100 tok/s at 4k context." width="49%">
  <img src=".github/readme/feature-server.png" alt="Drop-in for OpenCode: OpenAI-compatible server with streaming and tool calls." width="49%">
</p>

- **Decode:** dp4a GEMVs on Q4_K / Q6_K / Q8_0 weights, CUDA graphs, fused ops: 46 tok/s plain greedy (llama.cpp: 44).
- **Speculative decoding:** MTP-head drafts and prompt-lookup drafts, chosen per round from measured costs, verified up
  to 12 tokens per pass. Output is bit-identical to greedy. 106-127 tok/s on coding tasks.
- **Rollback:** replay-commit for the recurrent state, so rejected drafts cost no state copies.
- **Batched prefill:** fp16 tensor-core GEMMs in 1024-token chunks (~1100 tok/s at 4k context).
- **Prefix cache:** recurrent-state snapshots at chat-message boundaries, so each agent step only processes what is new
  (0.3-0.5 s per step instead of re-reading 7k-24k tokens).
- **Server:** `adakernel/serve.py`, OpenAI chat completions with streaming and tool calls, rendered with the GGUF's own
  chat template.

## Numbers

Three chat requests carrying real source code, 256 generated tokens each, RTX 4060 8 GB, model
`eriktonon/qwen3.8-uncensored:9b` (Q4_K_M). Every mode produced exactly the greedy tokens.

| Task (prompt tokens) | llama.cpp | this kernel, greedy | **this kernel, spec** |
|---|---:|---:|---:|
| write tests (1799) | 6.62 s, 44 tok/s | 7.17 s, 46 tok/s | **3.90 s, 119 tok/s** |
| refactor (740) | 6.10 s, 44 tok/s | 6.24 s, 46 tok/s | **2.74 s, 127 tok/s** |
| explain (948) | 6.20 s, 44 tok/s | 6.33 s, 46 tok/s | **3.27 s, 106 tok/s** |

In a real OpenCode session ("fix two bugs in stats.py, run it"), the 6.6k-token system prompt was read once in 6.3 s;
each of the following steps started in 0.3-0.55 s and decoded at 82-100 tok/s, for 13.6 s end to end.

Methodology, per-kernel profiles, long-context numbers and next steps are in [docs/QWEN38_9B.md](docs/QWEN38_9B.md).

<img src=".github/readme/divider.svg" width="100%" alt="">

## Quickstart

### Build

Needs CUDA 12.8+ (for sm_89), VS 2022 Build Tools, CMake 3.24+ and Ninja.

```powershell
powershell -ExecutionPolicy Bypass -File scripts\build.ps1 -Cuda <cuda toolkit root>
```

The script records the toolkit's `bin` folder in `build/cuda_bin.txt`, and the Python loader uses it to find cudart and
cuBLAS. Python needs `numpy` and `jinja2`; the tests also need `pytest`.

### Get the model

```powershell
ollama pull eriktonon/qwen3.8-uncensored:9b
```

The server reads the weights straight from Ollama's store
(`~/.ollama/models/blobs/sha256-c7f6915f…3408`, 5.78 GB, Q4_K_M), so Ollama itself does not need to be running.

### Run with OpenCode

```powershell
powershell -ExecutionPolicy Bypass -File scripts\serve.ps1            # -Ctx 40960 -PfCols 512 -Port 8090 -Model <gguf>
```

Then pick `adakernel/qwen3.8-9b` in OpenCode. To load a different GGUF with the same `qwen35` layout, pass its path
with `-Model` (only Qwen3.8 9B has been tested).

<details>
<summary>OpenCode provider entry (<code>~/.config/opencode/opencode.json</code>)</summary>

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
          "limit": { "context": 40960, "output": 4096 }
        }
      }
    }
  }
}
```

The server uses port 8090 because OpenCode auto-discovers :8000 as vLLM with tools off.

</details>

> [!NOTE]
> One request at a time, greedy decoding only (temperature is ignored), 40k context by default (peaks at ~7.7 GB on an 8 GB card;
> `-Ctx 32768 -PfCols 1024` leaves more room for other apps).

### Tests

```powershell
python -m pytest tests -q
```

`tests/test_serve.py` runs on the CPU. `tests/test_qwen35.py` loads the model on the GPU (~6 GB). Set `LLAMA_CPP_BIN`
to a llama.cpp `build/bin` folder to also check token and greedy-output parity with llama.cpp.

## How it works

```mermaid
flowchart LR
  A[OpenCode request] --> B[Prefix cache:<br>restore longest<br>matching snapshot]
  B --> C[Batched prefill<br>of the new tokens]
  C --> D{Draft:<br>MTP head or<br>prompt lookup}
  D --> E[Verify all drafts<br>in one pass]
  E -->|accepted| F[Stream tokens<br>and tool calls]
  E -->|rejected| G[Replay-commit<br>rollback]
  G --> D
  F --> D
```

Each round drafts tokens, then the target model verifies all of them in one multi-column pass, so its 4.9 GB of
weights are read once per round instead of once per token. Rejected drafts resume from the last accepted column.
Prompts are rendered with the GGUF's own Jinja chat template, and Qwen's `<tool_call>` XML is converted to OpenAI
`tool_calls`.

## Origin

Built and measured against `eriktonon/qwen3.8-uncensored:9b` (Q4_K_M, from Ollama). Split out of the `kernel-sanders`
experiments repo (branch `qwen35-gdn`), where the benchmark scripts and llama.cpp comparisons live.

<p align="right"><a href="#readme-top">↑ back to top</a></p>

<p align="center">
  <img src=".github/readme/footer.svg" alt="Same model. Same card. 2.7 times the tokens." width="100%">
</p>
