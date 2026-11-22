"""How long greedy output agrees with llama.cpp (the reference implementation) on a few prompts.

    set LLAMA_CPP_BIN=<llama.cpp build\\bin>   (its CUDA runtime DLLs must be on PATH)
    python -m benchmarks.llama_match [--n 64]

Prints, per prompt, how many leading characters of our greedy continuation match llama-completion --temp 0.
Exact agreement over long runs is not expected: near-tied logits (for example 27.314 vs 27.303) flip on rounding
differences between two correct implementations. Use it to check that an arithmetic change does not make
agreement worse.
"""

from __future__ import annotations

import argparse
import os
import subprocess
from pathlib import Path

PROMPTS = [
    "The capital of France is",
    "def quicksort(arr):\n    \"\"\"Sort a list.\"\"\"\n",
    "Explain why the sky is blue in two sentences.",
    "Write a short story about a robot learning to paint.",
    "#include <vector>\n// Return the indices of the two numbers that add up to target.\nstd::vector<int> two_sum(",
    "import json\n\ndef load_config(path):\n",
]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=64)
    args = ap.parse_args()
    from adakernel.qwen35 import Qwen35Model
    from adakernel.qwen_tokenizer import QwenTokenizer
    from adakernel.serve import DEFAULT_MODEL

    tok = QwenTokenizer(DEFAULT_MODEL)
    llama = Path(os.environ["LLAMA_CPP_BIN"]) / "llama-completion"
    m = Qwen35Model(DEFAULT_MODEL, max_seq=1024)
    total = 0
    for p in PROMPTS:
        ours = tok.decode(m.generate(tok.encode(p), args.n)["tokens"].tolist())
        ref = subprocess.run([str(llama), "-m", str(DEFAULT_MODEL), "-p", p, "-n", str(args.n), "--temp", "0",
                              "-ngl", "99", "-fa", "on", "-no-cnv", "--no-warmup", "--no-display-prompt"],
                             capture_output=True, text=True, encoding="utf-8", errors="replace",
                             stdin=subprocess.DEVNULL, timeout=300, check=True).stdout
        n = next((i for i, (a, b) in enumerate(zip(ours, ref)) if a != b), min(len(ours), len(ref)))
        total += n
        print(f"{n:4d} / {len(ours):4d} chars agree | {p[:40]!r}", flush=True)
    print("total agreeing chars:", total)
    m.close()


if __name__ == "__main__":
    main()
