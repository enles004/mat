"""Fetch the prebuilt transformer artifact from Hub (serve-time helper).

Used by the transformer serving entrypoint: downloads the pinned Hub
revision into the artifact dir when it is missing or stale for the bundled
dataset, then leaves verification to the normal registry path at startup.
Exits nonzero when no usable artifact is available.
"""

import argparse
from collections.abc import Sequence
from pathlib import Path

from scripts.release_train_and_serve import fetch_transformer_artifact


def build_parser() -> argparse.ArgumentParser:
    """Build the fetch command-line contract."""
    parser = argparse.ArgumentParser(description="Fetch the transformer serving artifact from Hub")
    parser.add_argument("--data", type=Path, default=Path("data/dataset.csv"))
    parser.add_argument(
        "--artifact-dir",
        type=Path,
        default=Path("artifacts/transformer/champion"),
    )
    parser.add_argument(
        "--repo",
        default="faris-le/mat-bamibert-vi-automotive-sentiment",
    )
    parser.add_argument(
        "--revision",
        default="894a3de0d6eeafce3862a3f055e0c5c66e4305e1",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Fetch and report the transformer artifact source."""
    args = build_parser().parse_args(argv)
    source = fetch_transformer_artifact(args.artifact_dir, args.data, args.repo, args.revision)
    if source is None:
        print("No usable transformer artifact; cannot serve the transformer backend.")
        return 1
    print(f"Transformer artifact ready from {source}: {args.artifact_dir}.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
