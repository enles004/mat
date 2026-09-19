"""Freeze a deterministic group-aware split manifest."""

import argparse
import json
from collections.abc import Sequence
from pathlib import Path

from src.nlp.training.dataset import DatasetStore
from src.nlp.training.splitting import DatasetSplitter


def build_parser() -> argparse.ArgumentParser:
    """Create the stable split command-line contract."""
    parser = argparse.ArgumentParser(description="Freeze group-aware evaluation splits")
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=42)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Read approved rows and write their deterministic split manifest."""
    args = build_parser().parse_args(argv)
    manifest = DatasetSplitter().build(DatasetStore().read(args.input), args.input, seed=args.seed)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(manifest.model_dump(mode="json"), ensure_ascii=False, indent=2, sort_keys=True)
        + "\n",
        encoding="utf-8",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
