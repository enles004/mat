"""Cross-validate and persist the deterministic linear baseline evidence."""

import argparse
import hashlib
import json
from collections.abc import Sequence
from pathlib import Path

import pandas as pd  # type: ignore[import-untyped]
import yaml

from src.domain.training import BaselineRun, SplitManifest
from src.nlp.constants import _BASELINE_CONFIG, LABEL_ORDER, OOF_COLUMNS
from src.nlp.modeling.baseline import BaselineTrainer
from src.nlp.preprocessing.vietnamese_tokenizer import VietnameseSegmenter
from src.nlp.training.dataset import DatasetStore
from src.nlp.training.validation import DatasetValidator


def build_parser() -> argparse.ArgumentParser:
    """Create the stable baseline training command-line contract."""
    parser = argparse.ArgumentParser(description="Cross-validate the TF-IDF sentiment baseline")
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--splits", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--tokenizer",
        choices=("none", "pyvi"),
        default="none",
        help="none keeps the v1 champion pipeline; pyvi inserts explicit segmentation",
    )
    return parser


def _load_seed(config_path: Path, manifest: SplitManifest) -> int:
    document = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if not isinstance(document, dict):
        raise ValueError("Model configuration must be a mapping")
    seed = document.get("seed")
    labels = document.get("labels")
    baseline = document.get("baseline")
    if type(seed) is not int:
        raise ValueError("Model configuration seed must be an integer")
    if labels != LABEL_ORDER:
        raise ValueError("Model configuration labels must match the SentimentLabel enum order")
    if baseline != _BASELINE_CONFIG:
        raise ValueError("Model configuration baseline must match the frozen TF-IDF settings")
    if seed != manifest.seed:
        raise ValueError("Model configuration seed must match the persisted split manifest")
    return seed


def _checksum(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _mean_macro_f1(run: BaselineRun) -> float:
    if not run.fold_macro_f1:
        raise ValueError("Baseline run has no fold metrics")
    return sum(run.fold_macro_f1) / len(run.fold_macro_f1)


def _metrics_for(run: BaselineRun) -> dict[str, object]:
    return {
        "fold_macro_f1": list(run.fold_macro_f1),
        "mean_macro_f1": _mean_macro_f1(run),
        "oof_row_count": len(run.oof_predictions),
    }


def main(argv: Sequence[str] | None = None) -> int:
    """Train raw and normalized TF-IDF baselines on persisted development folds."""
    args = build_parser().parse_args(argv)
    store = DatasetStore()
    validator = DatasetValidator(args.data)
    segmenter = VietnameseSegmenter() if args.tokenizer == "pyvi" else None
    trainer = BaselineTrainer(dataset_store=store, validator=validator, segmenter=segmenter)

    rows = store.read(args.data)
    manifest = SplitManifest.model_validate_json(args.splits.read_text(encoding="utf-8"))
    errors = validator.validate_split_manifest(rows, manifest).errors
    if errors:
        raise ValueError(f"Frozen split manifest is invalid: {', '.join(errors)}")
    _load_seed(args.config, manifest)

    raw = trainer.cross_validate(rows, manifest, view="raw")
    normalized = trainer.cross_validate(rows, manifest, view="normalized")
    chosen = trainer.select_view(raw, normalized)
    predictions = pd.concat(
        [
            raw.oof_predictions.assign(view=raw.view),
            normalized.oof_predictions.assign(view=normalized.view),
        ],
        ignore_index=True,
    ).loc[:, ["view", *OOF_COLUMNS]]
    metrics = {
        "schema_version": "1",
        "dataset_checksum": _checksum(args.data),
        "split_manifest_checksum": _checksum(args.splits),
        "config_checksum": _checksum(args.config),
        "labels": LABEL_ORDER,
        "tokenizer": args.tokenizer,
        "chosen_view": chosen.view,
        "views": {"raw": _metrics_for(raw), "normalized": _metrics_for(normalized)},
    }

    args.output.mkdir(parents=True, exist_ok=True)
    predictions.to_csv(args.output / "cv_predictions.csv", index=False)
    (args.output / "cv_metrics.json").write_text(
        json.dumps(metrics, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({"chosen_view": chosen.view, "views": metrics["views"]}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
