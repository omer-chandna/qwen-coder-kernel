"""8-bit KV cache (AK_Q35_KV8=1) against the fp16 cache on long prompts: accuracy, speed and VRAM.

    python -m benchmarks.kv8_compare [--ctx 40960] [--pf-cols 512] [--sizes 4000,16000,32000] [--n-gen 128]

Each mode runs in its own process (the cache format is chosen at model load). Per prompt size (a fresh chat prompt
built from this repo's sources, as in long_context.py) it records the next-token logits after the prompt, the
first 32 tokens of speculative decoding, prompt time and decode speed, then compares: logit correlation, top-1 and
top-5 agreement, max |logit difference|, matching leading tokens.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def worker(args) -> None:
    import numpy as np

    os.environ["AK_Q35_PF_COLS"] = str(args.pf_cols)
    from adakernel.qwen35 import Qwen35Model
    from adakernel.serve import DEFAULT_MODEL, ChatFormat

    fmt = ChatFormat(DEFAULT_MODEL)
    corpus = "".join((ROOT / p).read_text(encoding="utf-8") for p in
                     ["native/cuda/qwen35_model.cu", "adakernel/serve.py", "adakernel/qwen35.py",
                      "adakernel/qwen_tokenizer.py", "native/cuda/ops.cuh", "native/cuda/qgemm.cuh"])
    corpus_ids = fmt.tok.encode(corpus)
    m = Qwen35Model(DEFAULT_MODEL, max_seq=args.ctx)
    out = {"vram_mb": m.info["vram_mb"], "rows": []}
    logits = {}
    m.generate_spec(fmt.tok.encode("warm up"), 8)
    for n in (int(x) for x in args.sizes.split(",")):
        body = fmt.tok.decode(corpus_ids[:n])
        msgs = [{"role": "system", "content": "You are a coding agent."},
                {"role": "user", "content": "Source:\n```\n" + body + "\n```\nWhat does spec_run do? Be brief."}]
        ids = fmt.tok.encode(fmt.render(msgs, None))
        if len(ids) + args.n_gen + 16 > args.ctx:
            break
        logits[str(len(ids))] = m.logits(ids)
        r = m.generate_spec(ids, args.n_gen)
        out["rows"].append({"prompt_tokens": len(ids), "prefill_s": r["prompt_ms"] / 1000,
                            "decode_tps": r["decode_tps"], "tokens": r["tokens"][:32].tolist()})
        print(f"  [{args.mode}] prompt {len(ids)}: prefill {r['prompt_ms'] / 1000:.2f}s, "
              f"decode {r['decode_tps']:.1f} tok/s", flush=True)
    out["vram_peak_mb"] = m.info["vram_mb"]
    m.close()
    np.savez(args.save + ".npz", **logits)
    Path(args.save + ".json").write_text(json.dumps(out))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ctx", type=int, default=40960)
    ap.add_argument("--pf-cols", type=int, default=512)
    ap.add_argument("--sizes", default="4000,16000,32000")
    ap.add_argument("--n-gen", type=int, default=128)
    ap.add_argument("--output", default=None)
    ap.add_argument("--mode", default=None, help=argparse.SUPPRESS)
    ap.add_argument("--save", default=None, help=argparse.SUPPRESS)
    args = ap.parse_args()
    if args.mode:
        worker(args)
        return

    import numpy as np

    tmp = tempfile.mkdtemp(prefix="kv8cmp_")
    res = {}
    for mode in ("fp16", "kv8"):
        env = dict(os.environ)
        env.pop("AK_Q35_KV8", None)
        if mode == "kv8":
            env["AK_Q35_KV8"] = "1"
        save = os.path.join(tmp, mode)
        subprocess.run([sys.executable, "-m", "benchmarks.kv8_compare", "--mode", mode, "--save", save,
                        "--ctx", str(args.ctx), "--pf-cols", str(args.pf_cols), "--sizes", args.sizes,
                        "--n-gen", str(args.n_gen)], env=env, cwd=ROOT, check=True)
        res[mode] = (json.loads(Path(save + ".json").read_text()), dict(np.load(save + ".npz")))

    (a, la), (b, lb) = res["fp16"], res["kv8"]
    print(f"VRAM at ctx {args.ctx}: fp16 {a['vram_peak_mb']:.0f} MB, kv8 {b['vram_peak_mb']:.0f} MB")
    rows = []
    for ra, rb in zip(a["rows"], b["rows"]):
        n = ra["prompt_tokens"]
        x, y = la[str(n)], lb[str(n)]
        same = 0
        while same < len(ra["tokens"]) and ra["tokens"][same] == rb["tokens"][same]:
            same += 1
        row = {"prompt_tokens": n, "corr": float(np.corrcoef(x, y)[0, 1]),
               "top1_same": int(np.argmax(x)) == int(np.argmax(y)),
               "top5_overlap": len(set(np.argsort(x)[-5:].tolist()) & set(np.argsort(y)[-5:].tolist())),
               "max_abs_diff": float(np.max(np.abs(x - y))), "same_leading_tokens": same,
               "prefill_s": [ra["prefill_s"], rb["prefill_s"]], "decode_tps": [ra["decode_tps"], rb["decode_tps"]]}
        rows.append(row)
        print(f"prompt {n:6d} | logits corr {row['corr']:.6f}, top1 {'same' if row['top1_same'] else 'DIFF'}, "
              f"top5 {row['top5_overlap']}/5, max|d| {row['max_abs_diff']:.3f} | same first {same}/32 tokens | "
              f"prefill {ra['prefill_s']:.2f}s -> {rb['prefill_s']:.2f}s | decode {ra['decode_tps']:.1f} -> "
              f"{rb['decode_tps']:.1f} tok/s")
    if args.output:
        Path(args.output).write_text(json.dumps({"ctx": args.ctx, "vram_mb": [a["vram_peak_mb"], b["vram_peak_mb"]],
                                                 "rows": rows}, indent=1))


if __name__ == "__main__":
    main()
