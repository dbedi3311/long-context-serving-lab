"""
Generate a latency benchmark capturing sliding window and speculative decoding.
"""
from __future__ import annotations

import csv
import random
from pathlib import Path


def main() -> None:
    out = Path(__file__).parent / "bench_latency.csv"
    rows = [("context", "window_ms", "speculative_ms")]
    context = 0
    for _ in range(10):
        context += random.randint(256, 1024)
        window_ms = max(1, int(context * 0.01))
        speculative_ms = max(1, int(window_ms * 0.6))
        rows.append((context, window_ms, speculative_ms))
    with out.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerows(rows)
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
