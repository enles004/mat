"""Assemble the curated pool into the pending-review dataset CSV."""

import argparse
import json
from collections.abc import Sequence
from pathlib import Path

from src.nlp.training.dataset import DatasetStore
from src.nlp.training.generation import DatasetGenerator


def build_parser() -> argparse.ArgumentParser:
    """Create the stable generation command-line contract."""
    parser = argparse.ArgumentParser(description="Assemble the curated pool into dataset rows")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Write the assembled rows and print the generation evidence."""
    args = build_parser().parse_args(argv)
    rows = DatasetGenerator().generate(args.config)
    DatasetStore().write(rows, args.output)
    summary = {
        "generation_batch": rows[0].generation_batch if rows else "",
        "noise_rows": sum(bool(row.noise_types) for row in rows),
        "output": str(args.output),
        "rows": len(rows),
    }
    print(json.dumps(summary, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
