"""Cross-validate the revision-pinned BamiBERT transformer candidate.

Thin console wrapper composing the dataset store, validator, challenge loader,
and model-source resolver around ``TransformerTrainer``; no duplicated logic.
"""

import argparse
import json
from collections.abc import Sequence
from pathlib import Path

import transformers
import yaml

from src.domain.exceptions import TokenLimitExceededError
from src.domain.training import SplitManifest
from src.nlp.constants import _TRANSFORMER_CONFIG
from src.nlp.evaluation.behavioral import BehavioralEvaluator
from src.nlp.modeling.model_source import ModelSourceResolver
from src.nlp.modeling.transformer import TransformerTrainer
from src.nlp.training.dataset import DatasetStore
from src.nlp.training.validation import DatasetValidator


def build_parser() -> argparse.ArgumentParser:
    """Build the transformer cross-validation command-line contract."""
    parser = argparse.ArgumentParser(description="Cross-validate revision-pinned BamiBERT")
    parser.add_argument("--model-id", required=True)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--splits", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--challenge", type=Path)
    parser.add_argument("--config", type=Path, default=Path("configs/models.yaml"))
    return parser


def _load_training_config(config_path: Path) -> None:
    """Refuse to train when YAML training args drift from the pinned code values."""
    document = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if not isinstance(document, dict):
        raise ValueError("Model configuration must be a mapping")
    transformer = document.get("transformer")
    if not isinstance(transformer, dict):
        raise ValueError("Model configuration is missing the transformer block")
    training_args = {key: transformer.get(key) for key in _TRANSFORMER_CONFIG}
    if training_args != _TRANSFORMER_CONFIG:
        raise ValueError(
            "Model configuration transformer training args must match "
            "the pinned TransformerTrainer values"
        )


def main(argv: Sequence[str] | None = None) -> int:
    """Execute the BamiBERT experiment using immutable development folds only."""
    args = build_parser().parse_args(argv)
    report_path = Path("reports") / "transformer-evaluation.md"
    license_audit_path = Path("reports") / "transformer-license-audit.md"
    trainer = TransformerTrainer()
    resolver = ModelSourceResolver()

    trainer.clear_run_artifacts(args.output, report_path)
    if license_audit_path.exists():
        license_audit_path.unlink()

    rows = DatasetStore().read(args.data)
    manifest = SplitManifest.model_validate_json(args.splits.read_text(encoding="utf-8"))
    errors = DatasetValidator(args.data).validate_split_manifest(rows, manifest).errors
    if errors:
        raise ValueError(f"Frozen split manifest is invalid: {', '.join(errors)}")
    if args.seed != manifest.seed:
        raise ValueError("Transformer seed must match the persisted split manifest")
    _load_training_config(args.config)
    args.output.mkdir(parents=True, exist_ok=True)
    source = resolver.resolve(args.model_id)
    resolver.write_source_metadata(source, args.output / "source.json")
    resolver.write_license_audit(source, license_audit_path)
    challenge_path = args.challenge or args.data.with_name("challenge_set.csv")
    challenge_cases = BehavioralEvaluator().load_cases(challenge_path)
    try:
        predictions, measurements, token_audit, challenge_scores = trainer.cross_validate(
            rows=rows,
            manifest=manifest,
            source=source,
            output_dir=args.output,
            challenge_cases=challenge_cases,
        )
    except Exception as error:
        trainer.write_failure_evidence(
            output_dir=args.output,
            report_path=report_path,
            data_path=args.data,
            splits_path=args.splits,
            source=source,
            error=error,
            installed_transformers_version=transformers.__version__,
        )
        if isinstance(error, TokenLimitExceededError):
            trainer.write_token_limit_failure(args.output, error)
        raise
    predictions.to_csv(args.output / "cv_predictions.csv", index=False)
    (args.output / "challenge_predictions.json").write_text(
        json.dumps(challenge_scores, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    trainer.write_metrics(
        output_dir=args.output,
        data_path=args.data,
        splits_path=args.splits,
        source=source,
        measurements=measurements,
        token_audit=token_audit,
        predictions=predictions,
    )
    print(
        json.dumps(
            {
                "mean_macro_f1": sum(item.macro_f1 for item in measurements) / len(measurements),
                "oof_row_count": len(predictions),
                "revision": source.revision,
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
