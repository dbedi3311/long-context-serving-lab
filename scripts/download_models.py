"""
Placeholder model downloader for the lab.
"""
from __future__ import annotations

import argparse
from pathlib import Path


def download(target_dir: Path) -> None:
    target_dir.mkdir(parents=True, exist_ok=True)
    # In a real setup we'd stream weights; here we just drop a marker file.
    marker = target_dir / "README.txt"
    marker.write_text("Models would be downloaded here.")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--path", type=Path, default=Path("models"))
    args = parser.parse_args()
    download(args.path)
    print(f"Prepared model directory at {args.path}")


if __name__ == "__main__":
    main()
