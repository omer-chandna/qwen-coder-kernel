"""Summarize an Nsight Compute SASS source export: where a kernel's warps spend their samples, and which shared-memory
instructions replay because of bank conflicts.

    ncu --import report.ncu-rep --page source --csv --print-source sass > sass.csv
    python -m benchmarks.sass_hotspots sass.csv [--top 25]
"""

from __future__ import annotations

import argparse
import collections
import csv
import re


def num(x: str) -> float:
    try:
        return float(x.replace(",", ""))
    except ValueError:
        return 0.0


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("csv")
    ap.add_argument("--top", type=int, default=25)
    args = ap.parse_args()
    rows = list(csv.reader(open(args.csv, encoding="utf-8-sig")))
    hdr = next(r for r in rows if r and r[0] == "Address")
    data = [dict(zip(hdr, r)) for r in rows[rows.index(hdr) + 1:] if len(r) == len(hdr)]
    stall_cols = [h for h in hdr if h.startswith("stall_") and "Not Issued" not in h]
    total = sum(num(d["Warp Stall Sampling (All Samples)"]) for d in data)
    print(f"{len(data)} instructions, {total:.0f} samples")

    by_op = collections.Counter()
    by_op_stall = collections.defaultdict(collections.Counter)
    for d in data:
        op = re.sub(r"^@!?U?P\w+\s+", "", d["Source"].strip()).split(" ")[0].split(".")[0]
        s = num(d["Warp Stall Sampling (All Samples)"])
        by_op[op] += s
        for c in stall_cols:
            by_op_stall[op][c[6:]] += num(d[c])
    print("\nsamples by opcode (share, then its top stall reasons):")
    for op, s in by_op.most_common(14):
        top = ", ".join(f"{k} {v / s:.0%}" for k, v in by_op_stall[op].most_common(3) if v)
        print(f"  {op:10s} {s / total:6.1%}  {top}")

    print(f"\ntop {args.top} instructions by samples:")
    for d in sorted(data, key=lambda d: -num(d["Warp Stall Sampling (All Samples)"]))[: args.top]:
        s = num(d["Warp Stall Sampling (All Samples)"])
        top = max(stall_cols, key=lambda c: num(d[c]))
        print(f"  {s / total:5.1%}  {d['Source'].strip()[:70]:70s}  {top[6:]}")

    ex = [d for d in data if num(d.get("L1 Wavefronts Shared Excessive", "0")) > 0]
    tot_ex = sum(num(d["L1 Wavefronts Shared Excessive"]) for d in ex)
    print(f"\nshared-memory instructions with bank-conflict replays ({tot_ex:.0f} excess wavefronts):")
    for d in sorted(ex, key=lambda d: -num(d["L1 Wavefronts Shared Excessive"]))[:12]:
        print(f"  {num(d['L1 Wavefronts Shared Excessive']):12.0f} excess / {num(d['L1 Wavefronts Shared']):12.0f}  "
              f"{d['Source'].strip()[:60]}")


if __name__ == "__main__":
    main()
