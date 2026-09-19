from dataclasses import dataclass
from typing import Literal

MACRO_F1_DROP_TOLERANCE = 0.01


@dataclass(frozen=True)
class CalibrationDecision:
    """Gate evidence comparing native and cross-fitted calibrated OOF scores."""

    method: Literal["sigmoid", "none"]
    native_log_loss: float
    calibrated_log_loss: float
    native_brier: float
    calibrated_brier: float
    native_macro_f1: float
    calibrated_macro_f1: float

    @property
    def keep(self) -> bool:
        proper_score_improves = (
            self.calibrated_log_loss < self.native_log_loss
            or self.calibrated_brier < self.native_brier
        )
        macro_f1_kept = (
            self.calibrated_macro_f1 >= self.native_macro_f1 - MACRO_F1_DROP_TOLERANCE
        )
        return proper_score_improves and macro_f1_kept


@dataclass(frozen=True)
class ChallengeResult:
    """Auditable outcome for one behavioral challenge."""

    id: str
    kind: str
    passed: bool
    detail: str


@dataclass(frozen=True)
class CandidateEvidence:
    name: str
    eligible: bool
    macro_f1: float
    behavior_pass_rate: float
    license_status: str


_GATE_FIELDS = (
    "reproducible_five_fold_run",
    "no_group_leakage",
    "all_required_labels",
    "finite_probabilities",
    "behavior_report_present",
    "artifact_export_supported",
    "source_revision_recorded",
    "license_approved",
)


@dataclass(frozen=True)
class CandidateGates:
    """Per-gate evidence for one candidate, recorded before frozen-test access."""

    name: str
    reproducible_five_fold_run: bool
    no_group_leakage: bool
    all_required_labels: bool
    finite_probabilities: bool
    behavior_report_present: bool
    artifact_export_supported: bool
    source_revision_recorded: bool
    license_approved: bool
    macro_f1: float
    behavior_pass_rate: float
    license_status: str
    evidence: dict[str, str]

    @property
    def eligible(self) -> bool:
        return all(getattr(self, field) for field in _GATE_FIELDS)

    def as_evidence(self) -> CandidateEvidence:
        return CandidateEvidence(
            name=self.name,
            eligible=self.eligible,
            macro_f1=self.macro_f1,
            behavior_pass_rate=self.behavior_pass_rate,
            license_status=self.license_status,
        )
