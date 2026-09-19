"""Champion selection, frozen-test, and transformer export CLI.

Thin console wrapper composing the argument contract around
``ChampionSelector`` and ``TransformerChampionExporter``; no selection or
training logic lives here.
"""

import argparse
import hashlib
import json
from collections.abc import Sequence
from pathlib import Path

from src.domain.artifacts import ResolvedModelSource
from src.domain.training import SplitManifest
from src.nlp.evaluation.selection import ChampionSelector
from src.nlp.modeling.registry import ArtifactRegistry
from src.nlp.modeling.saved_artifact import SavedArtifact
from src.nlp.modeling.transformer_champion import TransformerChampionExporter
from src.nlp.training.dataset import DatasetStore
from src.nlp.training.validation import DatasetValidator


def build_parser() -> argparse.ArgumentParser:
    """Build the champion selection command-line contract."""
    parser = argparse.ArgumentParser(
        description="Select the sentiment champion and export verified artifacts"
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    gates_parser = subparsers.add_parser(
        "gates", help="Record champion gates before any frozen-test access"
    )
    gates_parser.add_argument("--data", type=Path, default=Path("data/dataset.csv"))
    gates_parser.add_argument("--splits", type=Path, default=Path("data/split_manifest.json"))
    gates_parser.add_argument("--config", type=Path, default=Path("configs/models.yaml"))
    gates_parser.add_argument("--challenge", type=Path, default=Path("data/challenge_set.csv"))
    gates_parser.add_argument("--baseline-run", type=Path, default=Path("runs/baseline"))
    gates_parser.add_argument(
        "--baseline-report", type=Path, default=Path("reports/baseline-evaluation.json")
    )
    gates_parser.add_argument("--transformer-run", type=Path, default=Path("runs/transformer"))
    gates_parser.add_argument(
        "--transformer-report",
        type=Path,
        default=Path("reports/transformer-evaluation.json"),
    )
    gates_parser.add_argument("--output", type=Path, default=Path("reports/model-selection.md"))

    frozen_parser = subparsers.add_parser(
        "frozen-test", help="Fit on development, evaluate the frozen test once, export"
    )
    frozen_parser.add_argument("--data", type=Path, default=Path("data/dataset.csv"))
    frozen_parser.add_argument("--splits", type=Path, default=Path("data/split_manifest.json"))
    frozen_parser.add_argument("--baseline-run", type=Path, default=Path("runs/baseline"))
    frozen_parser.add_argument("--challenge", type=Path, default=Path("data/challenge_set.csv"))
    frozen_parser.add_argument("--artifact-dir", type=Path, default=Path("artifacts/baseline"))
    frozen_parser.add_argument("--output", type=Path, default=Path("reports/model-selection.md"))

    export_parser = subparsers.add_parser(
        "export-transformer",
        help="Fine-tune BamiBERT on all development rows and export a research artifact",
    )
    export_parser.add_argument("--model-id", default="Qualcomm-AI-Research/BamiBERT")
    export_parser.add_argument("--data", type=Path, default=Path("data/dataset.csv"))
    export_parser.add_argument("--splits", type=Path, default=Path("data/split_manifest.json"))
    export_parser.add_argument("--source", type=Path, default=Path("runs/transformer/source.json"))
    export_parser.add_argument(
        "--artifact-dir", type=Path, default=Path("artifacts/transformer/champion")
    )
    export_parser.add_argument(
        "--work-dir", type=Path, default=Path("artifacts/transformer/.champion-work")
    )
    export_parser.add_argument(
        "--report", type=Path, default=Path("reports/transformer-champion.md")
    )
    export_parser.add_argument("--seed", type=int, default=42)
    return parser


def _load_export_source(path: Path) -> ResolvedModelSource:
    """Reuse the persisted frozen source provenance; no Hub re-resolution."""
    document = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(document, dict):
        raise ValueError(f"Expected a JSON object: {path}")
    return ResolvedModelSource(**document)


def _export_reproduce_command(args: argparse.Namespace) -> str:
    """The exact command that reproduces this artifact from the frozen inputs."""
    return (
        "HF_HUB_OFFLINE=1 uv run --locked python -m scripts.release_select_champion "
        "export-transformer"
        f" --model-id {args.model_id}"
        f" --data {args.data}"
        f" --splits {args.splits}"
        f" --source {args.source}"
        f" --artifact-dir {args.artifact_dir}"
        f" --work-dir {args.work_dir}"
        f" --report {args.report}"
        f" --seed {args.seed}"
    )


def _render_export_report(
    args: argparse.Namespace,
    saved: SavedArtifact,
    source: ResolvedModelSource,
    split_manifest: SplitManifest,
) -> str:
    import transformers

    manifest = saved.manifest
    splits_checksum = hashlib.sha256(args.splits.read_bytes()).hexdigest()
    lines = [
        "# Transformer champion export",
        "",
        "## Protocol",
        "",
        f"- Fit on **all {len(split_manifest.development_ids)} development IDs** "
        "(the same champion protocol as the baseline: one fit, no epoch selection).",
        "- TrainingArguments mirror the persisted cross-validation folds: "
        "learning_rate=2e-5, per_device_train_batch_size=16, num_train_epochs=4, "
        "weight_decay=0.01, warmup_ratio=0.10, fp16 when CUDA is available.",
        '- `eval_strategy="no"`, `save_strategy="no"`: with selection complete there is '
        "no legitimate validation split and no epoch checkpoint to pick.",
        f"- The frozen test set ({len(split_manifest.test_ids)} IDs) was never touched; "
        f"seed {args.seed} matches the persisted split manifest.",
        "- Serving is the raw softmax of the fine-tuned head; no calibration was fitted.",
        "",
        "## Frozen inputs",
        "",
        f"- Dataset: `{args.data}` ({manifest.data_checksum})",
        f"- Split manifest: `{args.splits}` (SHA-256 `sha256:{splits_checksum}`)",
        f"- Source provenance: `{args.source}` (repository `{source.repo_id}`, "
        f"revision `{source.revision}`, model-card SHA-256 `{source.model_card_checksum}`)",
        f"- Installed transformers: `{transformers.__version__}`",
        "",
        "## Exported artifact",
        "",
        f"- Directory: `{args.artifact_dir}` ({len(saved.payload_files)} payload files, "
        "gitignored — reproduce with the command below)",
        f"- Payload checksum: `{manifest.payload_checksum}`",
        f"- Max input tokens: {manifest.max_input_tokens}",
        f"- Git revision at export: `{manifest.git_revision}`",
        f"- License: `{manifest.license}`",
        "",
        "### Intended use from the model card (verbatim)",
        "",
        *(
            f"> {line}" if line else ">"
            for line in (source.license_text or source.intended_use).splitlines()
        ),
        "",
        "## Serving",
        "",
        "```bash",
        "MAT_MODEL_BACKEND=transformer uv run --locked python main.py",
        "```",
        "",
        "A fixed `transformer` backend never falls back to the baseline: a missing or "
        "corrupt artifact fails startup loudly. The default `baseline` backend keeps "
        "serving the linear champion exactly as before.",
        "",
        "## Reproduction",
        "",
        "```bash",
        _export_reproduce_command(args),
        "```",
        "",
        "The champion reuses `runs/transformer/source.json` (revision-pinned, already "
        "resolved once) so the re-run needs no Hub access.",
        "",
    ]
    return "\n".join(lines) + "\n"


def _run_export_transformer(args: argparse.Namespace) -> int:
    """Export, verify, and record the transformer champion artifact."""
    rows = DatasetStore().read(args.data)
    manifest = SplitManifest.model_validate_json(args.splits.read_text(encoding="utf-8"))
    errors = DatasetValidator(args.data).validate_split_manifest(rows, manifest).errors
    if errors:
        raise ValueError(f"Frozen split manifest is invalid: {', '.join(errors)}")
    if args.seed != manifest.seed:
        raise ValueError("Export seed must match the persisted split manifest")
    source = _load_export_source(args.source)
    saved = TransformerChampionExporter().export(
        rows=rows,
        manifest=manifest,
        source=source,
        data_path=args.data,
        artifact_dir=args.artifact_dir,
        work_dir=args.work_dir,
    )
    verified = ArtifactRegistry().verify(args.artifact_dir)
    if verified != saved.manifest:
        raise ValueError("Re-verified manifest on disk does not match the exported manifest")
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(_render_export_report(args, saved, source, manifest), encoding="utf-8")
    print(
        json.dumps(
            {
                "artifact_dir": str(saved.directory),
                "payload_files": len(saved.payload_files),
                "payload_checksum": saved.manifest.payload_checksum,
                "development_rows": len(manifest.development_ids),
                "revision": source.revision,
            },
            sort_keys=True,
        )
    )
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    """Run champion selection gates, the one-shot frozen test, or transformer export."""
    args = build_parser().parse_args(argv)
    selector = ChampionSelector()

    if args.command == "gates":
        result = selector.run_gates(
            data_path=args.data,
            splits_path=args.splits,
            config_path=args.config,
            challenge_path=args.challenge,
            baseline_run=args.baseline_run,
            baseline_report=args.baseline_report,
            transformer_run=args.transformer_run,
            transformer_report=args.transformer_report,
            output_path=args.output,
        )
        print(
            json.dumps(
                {
                    "champion": result["champion"].name,
                    "calibration_method": result["calibration"].method,
                    "baseline_eligible": result["baseline"].eligible,
                    "transformer_eligible": result["transformer"].eligible,
                    "output": str(args.output),
                },
                sort_keys=True,
            )
        )
        return 0

    if args.command == "export-transformer":
        return _run_export_transformer(args)

    result = selector.run_frozen_test(
        data_path=args.data,
        splits_path=args.splits,
        baseline_run=args.baseline_run,
        challenge_path=args.challenge,
        artifact_dir=args.artifact_dir,
        output_path=args.output,
    )
    print(
        json.dumps(
            {
                "frozen_macro_f1": result["metrics"]["macro_f1"],
                "frozen_row_count": result["row_count"],
                "error_count": len(result["errors"]),
                "behavior_pass_rate": result["behavior"]["pass_rate"],
                "payload_checksum": result["artifact"].manifest.payload_checksum,
                "output": str(args.output),
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
