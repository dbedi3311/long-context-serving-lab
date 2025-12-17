"""
Plot benchmark CSVs emitted by the lab scripts.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


BENCH_FILES = {
    "kv_paging": "bench_kv_paging.csv",
    "latency": "bench_latency.csv",
    "specdecode": "bench_specdecode.csv",
}


def plot_benchmarks(root: Path) -> None:
    for name, filename in BENCH_FILES.items():
        path = root / filename
        if not path.exists():
            print(f"Skipping {name}: missing {path}")
            continue
        df = pd.read_csv(path)
        plt.figure()
        df.plot(x=df.columns[0], y=df.columns[1:])
        plt.title(name)
        plt.tight_layout()
        out_path = root / f"{name}.png"
        plt.savefig(out_path)
        print(f"Wrote {out_path}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path("bench"))
    args = parser.parse_args()
    plot_benchmarks(args.root)


if __name__ == "__main__":
    main()
