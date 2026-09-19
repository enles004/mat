from dataclasses import dataclass
from typing import Literal

import pandas as pd  # type: ignore[import-untyped]
from pydantic import BaseModel


class FoldManifest(BaseModel):
    """One group-disjoint training and validation partition of development data."""

    fold: int
    train_ids: list[str]
    validation_ids: list[str]


class SplitManifest(BaseModel):
    """Immutable dataset membership for model development and final evaluation."""

    schema_version: Literal["1"] = "1"
    seed: int
    dataset_checksum: str
    development_ids: list[str]
    test_ids: list[str]
    folds: list[FoldManifest]


@dataclass(frozen=True)
class TokenLengthAudit:
    """The untruncated token-length audit for development text only."""

    row_count: int
    maximum_observed_tokens: int
    model_limit: int
    tokenizer_limit: int | None
    config_limit: int | None


@dataclass(frozen=True)
class FoldMeasurement:
    """One validation fold's resource and quality evidence."""

    fold: int
    duration_seconds: float
    peak_allocated_bytes: int
    macro_f1: float
    validation_row_count: int
    source_loading: dict[str, list[str]]
    best_checkpoint_loading: dict[str, list[str]]


@dataclass(frozen=True)
class BaselineRun:
    """OOF predictions and per-fold macro-F1 for one text representation."""

    view: Literal["raw", "normalized"]
    oof_predictions: pd.DataFrame
    fold_macro_f1: tuple[float, ...]


@dataclass(frozen=True)
class ValidationReport:
    errors: tuple[str, ...]
    warnings: tuple[str, ...]
    row_count: int
    class_counts: dict[str, int]
    noise_ratio: float

    @property
    def ok(self) -> bool:
        return not self.errors
