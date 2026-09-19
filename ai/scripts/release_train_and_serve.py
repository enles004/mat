"""Train-everything entrypoint for the GPU container image.

Composes the existing CLI mains in pipeline order so a fresh checkout (or a
user-provided ``dataset.csv``) becomes a serving artifact without manual
steps: validate -> split (only when missing) -> train baseline -> evaluate ->
train transformer (GPU only) -> gates -> one-shot frozen-test -> verify.

All trainer outputs stay under ``--work-dir``; evaluation reports and the
champion decision go under ``--report-dir`` (``runs/`` and ``artifacts/`` are
frozen roots that the reporter refuses to write into).

Skip logic: when the artifact verifies AND its recorded ``data_checksum``
matches the current dataset, training is skipped and the container serves
immediately. A checksum mismatch wipes the work and artifact dirs and retrains
from the split step, loudly. A dataset-matching transformer champion (local,
or fetched from the pinned Hub repo) also skips BamiBERT cross-validation —
retraining would only reproduce the same deterministic evidence.
"""

import argparse
import hashlib
import json
import os
import shutil
import subprocess
from collections.abc import Sequence
from csv import DictReader
from pathlib import Path

from scripts import data_split, data_validate, eval_predictions, train_baseline, train_transformer
from scripts import release_select_champion as select_champion
from scripts import release_verify_artifact as verify_artifact
from src.nlp.modeling.registry import ArtifactRegistry


def build_parser() -> argparse.ArgumentParser:
    """Build the train-and-serve command-line contract."""
    parser = argparse.ArgumentParser(
        description="Train all candidates from a dataset CSV, export, and verify"
    )
    parser.add_argument("--data", type=Path, default=Path("data/dataset.csv"))
    parser.add_argument("--splits", type=Path, default=Path("runs/container/split.json"))
    parser.add_argument("--config", type=Path, default=Path("configs/models.yaml"))
    parser.add_argument("--challenge", type=Path, default=Path("data/challenge_set.csv"))
    parser.add_argument("--work-dir", type=Path, default=Path("runs/container"))
    parser.add_argument("--report-dir", type=Path, default=Path("reports/container"))
    parser.add_argument("--artifact-dir", type=Path, default=Path("artifacts/container"))
    parser.add_argument("--model-id", default="Qualcomm-AI-Research/BamiBERT")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--transformer-mode",
        choices=("auto", "always", "never"),
        default="auto",
        help="auto trains BamiBERT only when CUDA is available",
    )
    parser.add_argument(
        "--transformer-artifact-dir",
        type=Path,
        default=Path("artifacts/transformer/champion"),
        help="serving dir for the transformer artifact (Hub fetch lands here)",
    )
    parser.add_argument(
        "--transformer-hf-repo",
        default="faris-le/mat-bamibert-vi-automotive-sentiment",
        help="Hub repo used to fetch a prebuilt transformer artifact",
    )
    parser.add_argument(
        "--transformer-hf-revision",
        default="894a3de0d6eeafce3862a3f055e0c5c66e4305e1",
        help="pinned Hub commit for the transformer artifact fetch",
    )
    parser.add_argument(
        "--fetch-transformer",
        choices=("auto", "always", "never"),
        default="auto",
        help="auto fetches from Hub only when the transformer dir is empty",
    )
    return parser


def sha256_file(path: Path) -> str:
    """Return the hex SHA-256 of a file."""
    return hashlib.sha256(path.read_bytes()).hexdigest()


def cuda_available() -> bool:
    """Return True only when torch with CUDA is importable and usable."""
    try:
        import torch
    except ImportError:
        return False
    try:
        return bool(torch.cuda.is_available())
    except Exception:
        return False


def artifact_matches_dataset(artifact_dir: Path, data_path: Path) -> bool:
    """Verify the artifact and confirm it was trained on the current dataset."""
    try:
        manifest = ArtifactRegistry().verify(artifact_dir)
    except Exception:
        return False
    return manifest.data_checksum == f"sha256:{sha256_file(data_path)}"


def wipe_dir(path: Path) -> None:
    """Remove every entry inside a directory, keeping the directory itself."""
    if not path.is_dir():
        return
    for entry in path.iterdir():
        if entry.is_dir() and not entry.is_symlink():
            shutil.rmtree(entry)
        else:
            entry.unlink()


def transformer_fetch_needed(artifact_dir: Path, mode: str) -> bool:
    """Fetch when forced, or (auto) when the transformer dir is absent/empty."""
    if mode == "never":
        return False
    if mode == "always":
        return True
    return not artifact_dir.is_dir() or not any(artifact_dir.iterdir())


def _strip_hub_fetch_extras(artifact_dir: Path) -> None:
    """Drop Hub fetch residue that is not payload.

    ``snapshot_download(local_dir=...)`` writes resume metadata under
    ``<dir>/.cache/huggingface/`` (locks, .metadata, .incomplete) and Hub repos
    carry a generated ``.gitattributes``. The registry's payload checksum covers
    every file in the dir, so leaving them behind breaks verification on every
    boot and turns a valid fetch into a wipe-restart loop.
    """
    hub_cache = artifact_dir / ".cache"
    if hub_cache.is_dir():
        shutil.rmtree(hub_cache)
    gitattributes = artifact_dir / ".gitattributes"
    if gitattributes.is_file():
        gitattributes.unlink()


def fetch_transformer_artifact(
    artifact_dir: Path, data_path: Path, repo: str, revision: str
) -> str | None:
    """Fetch a prebuilt transformer artifact from Hub when it matches this dataset.

    Returns "local" if a verified dataset-matching artifact already exists,
    "hub" after a successful fetch, or None when fetching is impossible or the
    Hub artifact was trained on different data (never served stale: the dir is
    wiped on checksum mismatch). Network and auth problems degrade to None.
    """
    artifact_dir.mkdir(parents=True, exist_ok=True)
    _strip_hub_fetch_extras(artifact_dir)
    if artifact_matches_dataset(artifact_dir, data_path):
        return "local"
    try:
        from huggingface_hub import snapshot_download
    except ImportError:
        print("huggingface_hub is missing; skipping transformer Hub fetch.")
        return None
    try:
        snapshot_download(repo_id=repo, revision=revision, local_dir=str(artifact_dir))
    except Exception as error:
        print(f"Transformer Hub fetch failed ({error}); continuing without it.")
        return None
    _strip_hub_fetch_extras(artifact_dir)
    if artifact_matches_dataset(artifact_dir, data_path):
        print(f"Transformer artifact fetched from Hub repo {repo}.")
        return "hub"
    print("Hub transformer artifact is stale for this dataset; wiping it.")
    wipe_dir(artifact_dir)
    return None


def transformer_evidence_fresh(
    transformer_run: Path, transformer_report: Path, challenge_path: Path
) -> bool:
    """Reuse transformer CV evidence when it already matches this challenge set.

    Fresh means the cross-validation predictions exist and the stored report
    was evaluated on exactly the current challenge rows — then retraining
    would only reproduce the same evidence (deterministic seeds).
    """
    if not (transformer_run / "cv_predictions.csv").is_file():
        return False
    try:
        report = json.loads(transformer_report.read_text(encoding="utf-8"))
        with challenge_path.open(newline="", encoding="utf-8") as handle:
            challenge_rows = sum(1 for _ in DictReader(handle))
    except (OSError, ValueError, KeyError):
        return False
    challenge_count = report.get("challenge", {}).get("count")
    return isinstance(challenge_count, int) and challenge_count == challenge_rows


def _check(result: int, step: str) -> None:
    if result != 0:
        raise ValueError(f"Pipeline step failed: {step} (exit {result})")


def ensure_provenance() -> None:
    """Fail fast when no git revision is available for gate provenance."""
    if os.environ.get("MAT_GIT_REVISION", "").strip():
        return
    probe = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        capture_output=True,
        text=True,
        check=False,
    )
    if probe.returncode != 0:
        raise ValueError(
            "No git revision available: rebuild with "
            "--build-arg GIT_REVISION=$(git rev-parse HEAD) "
            "or set MAT_GIT_REVISION."
        )


def main(argv: Sequence[str] | None = None) -> int:
    """Run the full train-then-verify pipeline, skipping when already fresh."""
    args = build_parser().parse_args(argv)
    args.work_dir.mkdir(parents=True, exist_ok=True)
    args.report_dir.mkdir(parents=True, exist_ok=True)
    args.artifact_dir.mkdir(parents=True, exist_ok=True)

    transformer_source: str | None = None
    if transformer_fetch_needed(args.transformer_artifact_dir, args.fetch_transformer):
        transformer_source = fetch_transformer_artifact(
            args.transformer_artifact_dir,
            args.data,
            args.transformer_hf_repo,
            args.transformer_hf_revision,
        )
    # A verified dataset-matching champion (local or fetched) reproduces the
    # same deterministic evidence retraining would, so skip the CV entirely.
    transformer_ready = transformer_source in ("local", "hub")

    if artifact_matches_dataset(args.artifact_dir, args.data):
        print(f"Artifact {args.artifact_dir} is fresh for {args.data}; skipping training.")
        return 0
    ensure_provenance()
    if any(args.artifact_dir.iterdir()) or any(args.work_dir.iterdir()):
        print("Artifact is stale or missing; wiping work dirs and retraining.")
        wipe_dir(args.artifact_dir)
        wipe_dir(args.work_dir)

    data = str(args.data)
    splits = str(args.splits)
    config = str(args.config)
    challenge = str(args.challenge)
    baseline_run = str(args.work_dir / "baseline")
    baseline_report = str(args.report_dir / "baseline-evaluation.json")
    transformer_run = str(args.work_dir / "transformer")
    transformer_report = str(args.report_dir / "transformer-evaluation.json")
    decision_report = str(args.report_dir / "model-selection.md")

    _check(data_validate.main(["--input", data]), "validate-data")
    if Path(splits).is_file():
        _check(
            data_validate.main(["--input", data, "--split-manifest", splits]),
            "validate-split",
        )
    else:
        _check(
            data_split.main(["--input", data, "--output", splits, "--seed", str(args.seed)]),
            "split-data",
        )
    _check(
        train_baseline.main(
            ["--data", data, "--splits", splits, "--config", config, "--output", baseline_run]
        ),
        "train-baseline",
    )
    _check(
        eval_predictions.main(
            [
                "--predictions",
                f"{baseline_run}/cv_predictions.csv",
                "--data",
                data,
                "--challenge",
                challenge,
                "--output",
                str(args.report_dir / "baseline-evaluation.md"),
            ]
        ),
        "evaluate-baseline",
    )

    want_transformer = args.transformer_mode == "always" or (
        args.transformer_mode == "auto" and cuda_available()
    )
    if transformer_ready:
        print("Transformer artifact verified for this dataset; skipping BamiBERT training.")
        want_transformer = False
    elif args.transformer_mode == "always" and not cuda_available():
        raise ValueError("transformer-mode=always but no CUDA device is available")
    if want_transformer and transformer_evidence_fresh(
        Path(transformer_run), Path(transformer_report), Path(challenge)
    ):
        print("Transformer evidence already matches this challenge set; skipping retrain.")
        want_transformer = False
        retrained_transformer = False
    else:
        retrained_transformer = want_transformer
    if retrained_transformer:
        _check(
            train_transformer.main(
                [
                    "--model-id",
                    args.model_id,
                    "--data",
                    data,
                    "--splits",
                    splits,
                    "--output",
                    transformer_run,
                    "--seed",
                    str(args.seed),
                    "--challenge",
                    challenge,
                    "--config",
                    config,
                ]
            ),
            "train-transformer",
        )
        _check(
            eval_predictions.main(
                [
                    "--predictions",
                    f"{transformer_run}/cv_predictions.csv",
                    "--data",
                    data,
                    "--challenge",
                    challenge,
                    "--output",
                    str(args.report_dir / "transformer-evaluation.md"),
                ]
            ),
            "evaluate-transformer",
        )
    if want_transformer:
        _check(
            select_champion.main(
                [
                    "gates",
                    "--data",
                    data,
                    "--splits",
                    splits,
                    "--config",
                    config,
                    "--challenge",
                    challenge,
                    "--baseline-run",
                    baseline_run,
                    "--baseline-report",
                    baseline_report,
                    "--transformer-run",
                    transformer_run,
                    "--transformer-report",
                    transformer_report,
                    "--output",
                    decision_report,
                ]
            ),
            "select-gates",
        )
    else:
        print("No CUDA device; skipping BamiBERT (baseline-only container run).")

    _check(
        select_champion.main(
            [
                "frozen-test",
                "--data",
                data,
                "--splits",
                splits,
                "--baseline-run",
                baseline_run,
                "--challenge",
                challenge,
                "--artifact-dir",
                str(args.artifact_dir),
                "--output",
                decision_report,
            ]
        ),
        "frozen-test",
    )
    _check(
        verify_artifact.main(["verify", str(args.artifact_dir)]),
        "verify-artifact",
    )
    print(f"Pipeline complete; artifact ready at {args.artifact_dir}.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
