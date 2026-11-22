"""Speculative decode speed on coding chat prompts (frozen corpus), plus a bit-exactness fingerprint.

    python -m benchmarks.decode_bench [--n-gen 256] [--reps 2] [--output out.json]

For each prompt: generate_spec (MTP + prompt-lookup drafts, adaptive length) and report decode tok/s, drafted
tokens per round and accepted tokens per round. It also stores the generated tokens and the GEMV-path logits of a
fixed prompt, so two builds can be compared for identical output (compare "tokens" and "logits_sha" of two
--output files).
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]


def prompts(fmt, src: str) -> dict[str, list[int]]:
    tasks = {
        "write tests": ("Write pytest unit tests for the prefill function below.", src[52000:58000]),
        "refactor": ("Refactor this function to be shorter without changing behavior.", src[30000:33000]),
        "explain": ("Explain what this code does, step by step.", src[42000:46000]),
        "add comments": ("Add a short comment above each block of this code and return the whole code.",
                         src[46000:48500]),
        "implement": ("Write a C++ function `int lookup(const std::vector<int>& h, int n_min, int n_max)` that returns "
                      "the index after the most recent earlier occurrence of the longest suffix of h with length in "
                      "[n_min, n_max], or -1. Then write three unit tests for it.", ""),
        "to python": ("Translate this code to idiomatic Python with type hints.", src[36000:38500]),
    }
    out = {}
    for name, (ask, code) in tasks.items():
        body = ask + ("\n\n```cpp\n" + code + "\n```" if code else "")
        msgs = [{"role": "system", "content": "You are a helpful coding assistant."},
                {"role": "user", "content": body}]
        out[name] = fmt.tok.encode(fmt.render(msgs, None))
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ctx", type=int, default=16384)
    ap.add_argument("--n-gen", type=int, default=256)
    ap.add_argument("--reps", type=int, default=2)
    ap.add_argument("--output", default=None)
    args = ap.parse_args()

    from adakernel.qwen35 import Qwen35Model
    from adakernel.serve import DEFAULT_MODEL, ChatFormat

    fmt = ChatFormat(DEFAULT_MODEL)
    src = (ROOT / "benchmarks" / "data" / "agent_source.txt").read_text(encoding="utf-8")
    m = Qwen35Model(DEFAULT_MODEL, max_seq=args.ctx)
    res = {"tasks": {}}
    for name, ids in prompts(fmt, src).items():
        m.generate_spec(ids[:64], 8)  # warm-up
        runs = [m.generate_spec(ids, args.n_gen) for _ in range(args.reps)]
        best = max(runs, key=lambda r: r["decode_tps"])
        res["tasks"][name] = {"prompt_tokens": len(ids), "decode_tps": [r["decode_tps"] for r in runs],
                              "mean_draft": best["mean_draft"], "tokens_per_round": best["tokens_per_round"],
                              "tokens": best["tokens"].tolist()}
        extra = {k: round(float(best[k]), 2) for k in ("mean_draft", "tokens_per_round", "lookup_rounds")}
        print(f"{name:12s} {len(ids):5d} tok | decode {max(r['decode_tps'] for r in runs):6.1f} tok/s | {extra}",
              flush=True)
    # GEMV-path logits (verify / decode arithmetic) of a fixed prompt, as a hash for bit-exactness checks
    m.set_prefill(False)
    lg = m.logits(prompts(fmt, src)["refactor"][:300])
    m.set_prefill(True)
    res["logits_sha"] = hashlib.sha256(np.ascontiguousarray(lg).tobytes()).hexdigest()
    print("GEMV-path logits sha256:", res["logits_sha"][:16])
    m.close()
    if args.output:
        Path(args.output).write_text(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
