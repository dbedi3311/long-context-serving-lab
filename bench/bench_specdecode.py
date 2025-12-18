"""
Generate a speculative decoding throughput benchmark.
"""
from __future__ import annotations

import csv
import random
from pathlib import Path


def main() -> None:
    out = Path(__file__).parent / "bench_specdecode.csv"
    rows = [("prompt_len", "draft_batch", "accept_rate")]
    prompt = 0
    for _ in range(10):
        prompt += random.randint(128, 512)
        draft_batch = random.choice([2, 4, 8])
        accept_rate = round(random.uniform(0.5, 0.9), 3)
        rows.append((prompt, draft_batch, accept_rate))
    with out.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerows(rows)
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
