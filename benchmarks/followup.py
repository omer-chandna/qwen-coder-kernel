"""Agent-style follow-up turns: a long first prompt, then turns that add a few hundred tokens each (the prefix cache
restores the earlier state, so only the new tokens are processed). Compares prefill GEMM paths.

    python -m benchmarks.followup [--ctx 40960] [--pf-cols 512]
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ctx", type=int, default=40960)
    ap.add_argument("--pf-cols", type=int, default=512)
    args = ap.parse_args()
    os.environ["AK_Q35_PF_COLS"] = str(args.pf_cols)

    from adakernel.qwen35 import Qwen35Model
    from adakernel.serve import DEFAULT_MODEL, ChatFormat

    fmt = ChatFormat(DEFAULT_MODEL)
    src = (ROOT / "native" / "cuda" / "qwen35_model.cu").read_text(encoding="utf-8")
    m = Qwen35Model(DEFAULT_MODEL, max_seq=args.ctx)
    tools_out = [src[i:i + n] for i, n in ((30000, 900), (42000, 1800), (52000, 3600))]  # ~300 / 600 / 1200 tokens
    for name, q, i8 in (("cuBLAS", 0, False), ("fp16", 768, False), ("int8", 768, True)):
        m.set_qgemm(q)
        m.set_int8(i8)
        m.reset_cache()
        msgs = [{"role": "system", "content": "You are a coding agent. " + src[:18000]},
                {"role": "user", "content": "Find the bug in spec_run."}]
        line = []
        for turn, extra in enumerate([None] + tools_out):
            if extra is not None:
                msgs += [{"role": "assistant", "content": "Let me look at more of the file."},
                         {"role": "user", "content": "<tool_response>\n" + extra + "\n</tool_response>"}]
            ids = fmt.tok.encode(fmt.render(msgs, None))
            r = m.generate_cached(ids, 16, snap_at=fmt.snap_points(ids))
            new = r["prompt_tokens"] - r["cached_tokens"]
            line.append(f"{new:5d} new tok {r['prompt_ms'] / 1000:5.2f}s")
        print(f"{name:7s}: " + " | ".join(line), flush=True)


if __name__ == "__main__":
    main()
