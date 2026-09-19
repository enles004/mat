"""Verify local artifact payload checksums against their manifest."""

import argparse
import json
import sys
from collections.abc import Sequence
from pathlib import Path

from src.nlp.modeling.registry import ArtifactRegistry


def build_parser() -> argparse.ArgumentParser:
    """Create the artifact-registry command-line contract."""
    parser = argparse.ArgumentParser(description="Verify local sentiment artifacts")
    subparsers = parser.add_subparsers(dest="command", required=True)
    verify_parser = subparsers.add_parser(
        "verify", help="Verify payload checksums recorded in manifest.json"
    )
    verify_parser.add_argument("directory", type=Path)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Verify one local artifact and print stable JSON metadata."""
    args = build_parser().parse_args(argv)
    try:
        manifest = ArtifactRegistry().verify(args.directory)
    except ValueError as error:
        print(f"verification failed: {error}", file=sys.stderr)
        return 1
    print(
        json.dumps(
            {
                "artifact": str(args.directory),
                "model_name": manifest.model_name,
                "model_version": manifest.model_version,
                "backend": manifest.backend,
                "payload_checksum": manifest.payload_checksum,
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
