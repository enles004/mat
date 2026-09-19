import re
from pathlib import Path
from typing import Literal

from src.domain.entities import SentimentLabel

# Label vocabulary — single source for every pipeline stage.
LABELS: tuple[str, str, str] = ("negative", "neutral", "positive")
LABEL_ORDER: list[str] = [label.value for label in SentimentLabel]
LABEL2ID: dict[str, int] = {"negative": 0, "neutral": 1, "positive": 2}
ID2LABEL: dict[int, str] = {value: key for key, value in LABEL2ID.items()}

# Column layout shared by the baseline and transformer trainers.
OOF_COLUMNS: list[str] = ["id", "fold", "y_true", *LABEL_ORDER]

# Baseline configuration mirrored from configs/models.yaml.
# Rule: YAML owns the values, code owns the schema; run_train.py refuses to
# run when the two disagree, so drift fails loudly instead of silently.
_VIEWS: tuple[Literal["raw", "normalized"], ...] = ("raw", "normalized")
_BASELINE_CONFIG = {
    "word_ngram_range": [1, 2],
    "char_ngram_range": [3, 5],
    "min_df": 2,
    "sublinear_tf": True,
    "classifier_c": 2.0,
    "max_iter": 2000,
}

# Transformer training arguments mirrored from configs/models.yaml.
# Same rule: run_transformer.py gates on equality (resolver identity keys
# family/repo_id/revision_policy stay YAML-only; the six numeric training
# args below must match build_training_arguments).
_TRANSFORMER_CONFIG = {
    "learning_rate": 2e-5,
    "train_batch_size": 16,
    "eval_batch_size": 32,
    "epochs": 4,
    "weight_decay": 0.01,
    "warmup_ratio": 0.10,
}

# Dataset CSV columns (training persistence).
_CSV_FIELDS: tuple[str, ...] = (
    "id",
    "raw_text",
    "normalized_text",
    "label",
    "aspect",
    "style",
    "noise_types",
    "difficulty",
    "canonical_group_id",
    "generation_batch",
    "review_status",
)

# Behavioral challenge CSV columns.
_CHALLENGE_COLUMNS: list[str] = [
    "id",
    "kind",
    "text",
    "paired_text",
    "expected_label",
    "paired_expected_label",
    "minimum_delta",
    "phenomenon",
]

# Slice dimensions reported by the evaluation.
_SLICE_FIELDS: tuple[str, ...] = ("noise_types", "aspect", "style", "difficulty")

# Demo response-contract validation.
_REQUEST_ID_PATTERN = re.compile(r"^req_[A-Za-z0-9]{16,}$")
_SCORE_SUM_TOLERANCE = 1e-6
# Semantic fields that must be byte-equal across the two repeat calls;
# ``request_id`` legitimately differs per request and is excluded.
_SEMANTIC_FIELDS: tuple[str, ...] = ("label", "confidence", "scores", "uncertain", "model")

# Champion identity, gate titles, and selection inputs.
MODEL_NAME: str = "tfidf-word-char-logreg"
MODEL_VERSION: str = "1.0.0"
BASELINE_LICENSE: str = "internal-use-only"
EVALUATION_REPORT_PATH: str = "reports/model-selection.md"
CHALLENGE_CASE_COUNT: int = 74
_GATE_TITLES: dict[str, str] = {
    "reproducible_five_fold_run": "Reproducible five-fold run",
    "no_group_leakage": "No group leakage",
    "all_required_labels": "All required labels",
    "finite_probabilities": "Finite probabilities",
    "behavior_report_present": "Behavior report present",
    "artifact_export_supported": "Artifact export supported",
    "source_revision_recorded": "Exact source revision recorded",
    "license_approved": "License/intended use approved",
}

# Report output constraints.
FROZEN_OUTPUT_ROOTS: tuple[str, ...] = ("data", "artifacts", "runs")
_SERVICE_ROOT: Path = Path(__file__).resolve().parents[2]

# Artifact persistence names.
MODEL_PAYLOAD_NAME: str = "model.joblib"
MANIFEST_NAME: str = "manifest.json"

# Numerical guards.
_PROBABILITY_EPSILON: float = 1e-12

# Transformer source provenance.
_COMMIT_SHA = re.compile(r"[0-9a-f]{40}\Z")
