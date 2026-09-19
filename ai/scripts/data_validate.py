"""Validate a controlled dataset and optional frozen split manifest."""

import argparse
import csv
import json
from collections.abc import Sequence
from dataclasses import asdict
from pathlib import Path

from pydantic import ValidationError

from src.domain.entities import SentimentLabel
from src.domain.training import SplitManifest, ValidationReport
from src.nlp.training.dataset import DatasetStore
from src.nlp.training.validation import DatasetValidator


def build_parser() -> argparse.ArgumentParser:
    """Create the stable validation command-line contract."""
    parser = argparse.ArgumentParser(description="Validate a controlled sentiment dataset")
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--split-manifest", type=Path)
    parser.add_argument("--allow-pending", action="store_true")
    return parser


def _input_error_report(error: Exception) -> ValidationReport:
    """Render malformed CSV/Pydantic input in the legacy report shape."""
    class_counts = {label.value: 0 for label in SentimentLabel}
    if isinstance(error, ValidationError):
        errors: list[str] = []
        for detail in error.errors():
            location = tuple(str(item) for item in detail["loc"])
            if location == ("label",):
                errors.append(f"unsupported_label:{detail.get('input', '')}")
            else:
                errors.append(f"malformed_input:{'.'.join(location)}")
    elif isinstance(error, json.JSONDecodeError):
        errors = ["malformed_input:noise_types"]
    elif isinstance(error, KeyError):
        errors = [f"malformed_input:{str(error).strip(chr(39))}"]
    else:
        errors = ["malformed_input"]
    return ValidationReport(
        errors=tuple(errors), warnings=(), row_count=0, class_counts=class_counts, noise_ratio=0.0
    )


def main(argv: Sequence[str] | None = None) -> int:
    """Print validation evidence and return the established success/failure code."""
    args = build_parser().parse_args(argv)
    try:
        rows = DatasetStore().read(args.input)
    except (OSError, ValueError, KeyError, TypeError, csv.Error) as error:
        report = _input_error_report(error)
    else:
        report = DatasetValidator().validate_dataset(rows, allow_pending=args.allow_pending)
        if args.split_manifest is not None:
            split_errors: tuple[str, ...]
            try:
                manifest = SplitManifest.model_validate_json(
                    args.split_manifest.read_text(encoding="utf-8")
                )
                split_report = DatasetValidator(args.input).validate_split_manifest(rows, manifest)
            except (OSError, ValueError, TypeError):
                split_errors = ("split_manifest_malformed",)
            else:
                split_errors = split_report.errors
            report = ValidationReport(
                errors=(*report.errors, *split_errors),
                warnings=report.warnings,
                row_count=report.row_count,
                class_counts=report.class_counts,
                noise_ratio=report.noise_ratio,
            )
    print(json.dumps(asdict(report), ensure_ascii=False, sort_keys=True))
    return 0 if report.ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
