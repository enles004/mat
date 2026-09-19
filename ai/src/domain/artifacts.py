from dataclasses import asdict, dataclass, field
from typing import Any

from src.domain.contracts import Normalizer

_HEX_COMMIT_DIGITS = frozenset("0123456789abcdef")


@dataclass(frozen=True)
class ResolvedModelSource:
    """A Hub model source pinned to one immutable commit."""

    repo_id: str
    revision: str
    license_id: str
    intended_use: str
    license_text: str = ""
    license_metadata: dict[str, object] = field(default_factory=dict)
    file_checksums: dict[str, str] = field(default_factory=dict)
    model_card_checksum: str = ""
    transformers_requirement: str = ""

    def __post_init__(self) -> None:
        if len(self.revision) != 40 or any(
            char not in _HEX_COMMIT_DIGITS for char in self.revision
        ):
            raise ValueError("Model revision must be a 40-character commit SHA")
        if not self.repo_id:
            raise ValueError("Model repository ID must not be empty")
        if not self.license_id:
            raise ValueError("Model license metadata must not be empty")
        if not self.intended_use:
            raise ValueError("Model card intended-use text must not be empty")

    def as_dict(self) -> dict[str, Any]:
        """Return a stable JSON-safe representation."""
        return asdict(self)


@dataclass(frozen=True)
class CatalogActivation:
    """Immutable record of one successful catalog activation."""

    normalizer: Normalizer
    dictionary_version: str
    catalog_checksum: str
    rule_count: int
