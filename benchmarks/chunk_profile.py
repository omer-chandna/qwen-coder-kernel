"""GPU time of one prefill chunk by kernel group, for several chunk lengths, plus agent-style follow-up turns.

    python -m benchmarks.chunk_profile [--ctx 40960] [--pf-cols 512] [--sizes 64,128,256,512] [--pos 0]

Environment switches (AK_Q35_SPLITK=1, AK_Q35_ATTN=1, ...) are read when the model loads, so compare two settings
with two runs. The follow-up part mirrors benchmarks/followup.py with the default (int8) GEMM path only.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
GROUPS = {
    "matrix": lambda k: k.startswith("gemv"),
    "attention": lambda k: k in ("attn_part", "attn_combine", "attn_prep"),
    "delta rule": lambda k: k == "gated_delta",
}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ctx", type=int, default=40960)
    ap.add_argument("--pf-cols", type=int, default=512)
    ap.add_argument("--sizes", default="64,128,256,512")
    ap.add_argument("--pos", type=int, default=0)
    ap.add_argument("--no-followup", action="store_true")
    ap.add_argument("--output", default=None)
    args = ap.parse_args()
    os.environ["AK_Q35_PF_COLS"] = str(args.pf_cols)

    from adakernel.qwen35 import Qwen35Model
    from adakernel.serve import DEFAULT_MODEL, ChatFormat

    m = Qwen35Model(DEFAULT_MODEL, max_seq=args.ctx)
    out = {"env": {k: v for k, v in os.environ.items() if k.startswith("AK_Q35")}, "chunks": [], "turns": []}
    for T in (int(x) for x in args.sizes.split(",")):
        m.profile(T, args.pos)  # warm-up
        p = m.profile(T, args.pos)
        total = sum(v[0] for v in p.values())
        row = {"T": T, "total_ms": total}
        for g, f in GROUPS.items():
            row[g] = sum(v[0] for k, v in p.items() if f(k))
        row["rest"] = total - sum(row[g] for g in GROUPS)
        out["chunks"].append(row)
        print(f"T {T:5d}: {total:7.2f} ms | " + " | ".join(f"{g} {row[g]:6.2f}" for g in [*GROUPS, "rest"])
              + f" | {total / T:.3f} ms/token", flush=True)

    if not args.no_followup:
        fmt = ChatFormat(DEFAULT_MODEL)
        # frozen copy of qwen35_model.cu at 629018b, so turn sizes do not change as the source is edited
        src = (ROOT / "benchmarks" / "data" / "agent_source.txt").read_text(encoding="utf-8")
        extras = [src[i:i + n] for i, n in ((30000, 300), (36000, 900), (42000, 1800), (52000, 3600))]
        for rep in range(2):
            m.reset_cache()
            msgs = [{"role": "system", "content": "You are a coding agent. " + src[:18000]},
                    {"role": "user", "content": "Find the bug in spec_run."}]
            line = []
            for extra in [None] + extras:
                if extra is not None:
                    msgs += [{"role": "assistant", "content": "Let me look at more of the file."},
                             {"role": "user", "content": "<tool_response>\n" + extra + "\n</tool_response>"}]
                ids = fmt.tok.encode(fmt.render(msgs, None))
                r = m.generate_cached(ids, 16, snap_at=fmt.snap_points(ids))
                new = r["prompt_tokens"] - r["cached_tokens"]
                line.append((new, r["prompt_ms"] / 1000))
            if rep == 1:  # second pass: warm graphs / allocations
                out["turns"] = [{"new_tokens": n, "prompt_s": t} for n, t in line]
                print("turns: " + " | ".join(f"{n:5d} new tok {t:5.3f}s" for n, t in line), flush=True)
    m.close()
    if args.output:
        Path(args.output).write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
