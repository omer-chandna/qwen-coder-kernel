"""Long-context prompt reading and decode speed: tensor-core attention vs the older scalar kernels.

    python -m benchmarks.long_context [--ctx 40960] [--pf-cols 512] [--sizes 4000,12000,24000,36000]

Each size is a fresh (uncached) chat prompt built from this repo's sources, then 128 tokens of speculative decoding.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ctx", type=int, default=40960)
    ap.add_argument("--pf-cols", type=int, default=512)
    ap.add_argument("--sizes", default="4000,12000,24000,36000")
    ap.add_argument("--n-gen", type=int, default=128)
    ap.add_argument("--output", default=None)
    args = ap.parse_args()
    os.environ["AK_Q35_PF_COLS"] = str(args.pf_cols)

    from adakernel.qwen35 import Qwen35Model
    from adakernel.serve import DEFAULT_MODEL, ChatFormat

    fmt = ChatFormat(DEFAULT_MODEL)
    corpus = "".join((ROOT / p).read_text(encoding="utf-8") for p in
                     ["native/cuda/qwen35_model.cu", "adakernel/serve.py", "adakernel/qwen35.py",
                      "adakernel/qwen_tokenizer.py", "native/cuda/ops.cuh"])
    corpus_ids = fmt.tok.encode(corpus)
    m = Qwen35Model(DEFAULT_MODEL, max_seq=args.ctx)
    rows = []
    for n in (int(x) for x in args.sizes.split(",")):
        body = fmt.tok.decode(corpus_ids[:n])
        msgs = [{"role": "system", "content": "You are a coding agent."},
                {"role": "user", "content": "Source:\n```\n" + body + "\n```\nWhat does spec_run do? Be brief."}]
        ids = fmt.tok.encode(fmt.render(msgs, None))
        if len(ids) + args.n_gen + 16 > args.ctx:
            break
        row = {"prompt_tokens": len(ids)}
        for name, tc in (("old", False), ("new", True)):
            m.set_attention(tc)
            m.generate_spec(ids[:64], 8)  # warm graphs for this kernel choice
            r = m.generate_spec(ids, args.n_gen)
            row[name] = {"prefill_s": r["prompt_ms"] / 1000, "prefill_tps": len(ids) / r["prompt_ms"] * 1000,
                         "decode_tps": r["decode_tps"], "tokens": r["tokens"][:32].tolist()}
        row["same_first_32"] = row["old"]["tokens"] == row["new"]["tokens"]
        rows.append(row)
        o, w = row["old"], row["new"]
        print(f"prompt {len(ids):6d} | prefill {o['prefill_s']:6.2f}s -> {w['prefill_s']:6.2f}s "
              f"({o['prefill_s'] / w['prefill_s']:.2f}x) | decode {o['decode_tps']:5.1f} -> {w['decode_tps']:5.1f} tok/s "
              f"| same first 32: {row['same_first_32']}", flush=True)
    m.close()
    if args.output:
        Path(args.output).write_text(json.dumps({"ctx": args.ctx, "pf_cols": args.pf_cols, "rows": rows}, indent=1))


if __name__ == "__main__":
    main()
