"""
Generate a toy benchmark for paging throughput.
"""
from __future__ import annotations

import csv
import random
from pathlib import Path


def main() -> None:
    out = Path(__file__).parent / "bench_kv_paging.csv"
    rows = [("tokens", "page_misses", "bytes_spilled")]
    total = 0
    misses = 0
    spilled = 0
    for _ in range(10):
        step = random.randint(256, 1024)
        total += step
        misses += random.randint(0, 2)
        spilled += random.randint(0, 1) * step * 16
        rows.append((total, misses, spilled))
    with out.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerows(rows)
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
