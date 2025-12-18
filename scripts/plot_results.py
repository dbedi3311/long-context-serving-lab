"""
Plot benchmark CSVs emitted by the lab scripts.

The script prefers matplotlib/pandas if available, but also includes a pure-Python
SVG fallback so plots can be generated in restricted environments (e.g., no PyPI).
"""
from __future__ import annotations

import argparse
import csv
from itertools import cycle
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

try:  # Optional plotting dependencies
    import matplotlib.pyplot as plt
    import pandas as pd

    _HAS_DEPS = True
except Exception:  # pragma: no cover - offline fallback
    plt = None  # type: ignore
    pd = None  # type: ignore
    _HAS_DEPS = False


BENCH_FILES = {
    "kv_paging": "bench_kv_paging.csv",
    "latency": "bench_latency.csv",
    "specdecode": "bench_specdecode.csv",
}


def plot_with_matplotlib(path: Path, name: str) -> Path:
    df = pd.read_csv(path)  # type: ignore[arg-type]
    plt.figure()  # type: ignore[call-arg]
    df.plot(x=df.columns[0], y=df.columns[1:])  # type: ignore[call-arg]
    plt.title(name)  # type: ignore[attr-defined]
    plt.tight_layout()  # type: ignore[attr-defined]
    out_path = path.with_suffix(".png")
    plt.savefig(out_path)  # type: ignore[attr-defined]
    return out_path


def _read_csv(path: Path) -> Tuple[List[str], List[List[float]]]:
    with path.open() as f:
        reader = csv.reader(f)
        rows = list(reader)
    if not rows:
        raise ValueError(f"No data in {path}")
    header = rows[0]
    data = [[float(x) for x in row] for row in rows[1:]]
    return header, data


def _scale(values: Sequence[float], size: float, min_v: float, max_v: float) -> List[float]:
    if max_v == min_v:
        return [size / 2 for _ in values]
    return [size * (v - min_v) / (max_v - min_v) for v in values]


def plot_svg_fallback(path: Path, name: str) -> Path:
    header, data = _read_csv(path)
    x_vals = [row[0] for row in data]
    y_series: Dict[str, List[float]] = {}
    for idx, label in enumerate(header[1:], start=1):
        y_series[label] = [row[idx] for row in data]

    width, height, margin = 640, 320, 50
    inner_w, inner_h = width - 2 * margin, height - 2 * margin

    min_x, max_x = min(x_vals), max(x_vals)
    min_y = min(min(series) for series in y_series.values())
    max_y = max(max(series) for series in y_series.values())

    xs = _scale(x_vals, inner_w, min_x, max_x)
    color_cycle = cycle(["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728"])

    lines: List[str] = []
    for label, ys in y_series.items():
        ys_scaled = _scale(ys, inner_h, min_y, max_y)
        points = " ".join(
            f"{margin + x:.2f},{height - margin - y:.2f}" for x, y in zip(xs, ys_scaled)
        )
        color = next(color_cycle)
        lines.append(f'<polyline fill="none" stroke="{color}" stroke-width="2" points="{points}" />')
        lines.append(
            f'<text x="{margin + inner_w - 80}" y="{margin + 15 * (len(lines)//2)}" '
            f'fill="{color}" font-size="12">{label}</text>'
        )

    svg = f"""<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}">
<rect x="0" y="0" width="{width}" height="{height}" fill="#ffffff" stroke="#dddddd"/>
<text x="{width/2}" y="20" text-anchor="middle" font-family="Arial" font-size="16">{name}</text>
<line x1="{margin}" y1="{height-margin}" x2="{width-margin}" y2="{height-margin}" stroke="#000" stroke-width="1"/>
<line x1="{margin}" y1="{margin}" x2="{margin}" y2="{height-margin}" stroke="#000" stroke-width="1"/>
{chr(10).join(lines)}
</svg>
"""
    out_path = path.with_suffix(".svg")
    out_path.write_text(svg)
    return out_path


def plot_benchmarks(root: Path) -> None:
    for name, filename in BENCH_FILES.items():
        path = root / filename
        if not path.exists():
            print(f"Skipping {name}: missing {path}")
            continue
        if _HAS_DEPS:
            out_path = plot_with_matplotlib(path, name)
        else:
            out_path = plot_svg_fallback(path, name)
        print(f"Wrote {out_path}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path("bench"))
    args = parser.parse_args()
    plot_benchmarks(args.root)


if __name__ == "__main__":
    main()
