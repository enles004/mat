"""Evaluation and reporting CLI.

Thin console wrapper composing the argument contract around
``EvaluationReporter``; no evaluation logic lives here.
"""

import argparse
import json
from collections.abc import Sequence
from pathlib import Path

from src.nlp.evaluation.reporting import EvaluationReporter


def build_parser() -> argparse.ArgumentParser:
    """Build the evaluation report command-line contract."""
    parser = argparse.ArgumentParser(description="Evaluate sentiment prediction evidence")
    parser.add_argument("--predictions", required=True, type=Path)
    parser.add_argument("--data", required=True, type=Path)
    parser.add_argument("--challenge", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Evaluate one prediction artifact through the evaluation reporter service."""
    args = build_parser().parse_args(argv)
    report = EvaluationReporter().generate(
        predictions_path=args.predictions,
        data_path=args.data,
        challenge_path=args.challenge,
        output_path=args.output,
    )
    print(
        json.dumps(
            {
                "selected_view": report["selected_view"],
                "macro_f1": report["metrics"]["macro_f1"],
                "challenge_pass_rate": report["challenge"]["pass_rate"],
                "output": str(args.output),
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
