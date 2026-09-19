from pathlib import Path
from typing import cast

import yaml

from src.domain.entities import DatasetRow, Difficulty, ReviewStatus, SentimentLabel
from src.nlp.preprocessing.catalog_loader import CatalogLoader
from src.nlp.preprocessing.matcher import RegexMatcher
from src.nlp.preprocessing.normalizer import TextNormalizer

_SUPPORTED_VERSION = "2.0.0"
_SUPPORTED_NOISE = frozenset(
    {"teencode", "brand_alias", "elongation", "emoji", "code_switching", "missing_diacritics"}
)


class DatasetGenerator:
    """Assemble the curated pool into deterministic pending-review rows.

    Every pool sentence — clean and noisy alike — is written complete by
    the reviewer; this class applies no text transformation. It assigns
    stable ids and groups, normalizes each raw text through the catalog
    that ships next to the pool config, and marks every row pending
    review. The pool is static, so the output is fully determined by the
    config alone.
    """

    @staticmethod
    def _load_config(config_path: Path) -> tuple[str, list[dict[str, object]]]:
        document = cast(
            dict[str, object], yaml.safe_load(config_path.read_text(encoding="utf-8"))
        )
        version = cast(str, document["version"])
        if version != _SUPPORTED_VERSION:
            raise ValueError(f"Unsupported pool config version: {version}")
        batch = cast(str, document["generation_batch"])
        aspects = set(cast(list[str], document["aspects"]))
        styles = set(cast(list[str], document["styles"]))
        pool = cast(list[dict[str, object]], document["pool"])
        if not pool:
            raise ValueError("Pool configuration must contain at least one sentence")
        seen: set[str] = set()
        for entry in pool:
            text = cast(str, entry["text"])
            folded = text.casefold()
            if folded in seen:
                raise ValueError(f"Duplicate pool text: {text}")
            seen.add(folded)
            cast(str, entry["label"])
            cast(str, entry["difficulty"])
            if entry["aspect"] not in aspects:
                raise ValueError(f"Pool aspect is not declared: {entry['aspect']}")
            if entry["style"] not in styles:
                raise ValueError(f"Pool style is not declared: {entry['style']}")
            if "noise_types" not in entry:
                raise ValueError(f"Pool entry must declare noise_types explicitly: {text!r}")
            unknown = set(cast(list[str], entry["noise_types"])) - _SUPPORTED_NOISE
            if unknown:
                raise ValueError(f"Unknown noise_type label: {sorted(unknown)[0]}")
        return batch, pool

    def generate(self, config_path: Path) -> list[DatasetRow]:
        """Return the pool as pending rows with catalog-normalized text."""
        batch, pool = self._load_config(config_path)
        normalizer = TextNormalizer(
            RegexMatcher.from_catalog(
                CatalogLoader().load(config_path.parent / "normalization.yaml").catalog
            )
        )
        return [
            DatasetRow(
                id=f"{batch}-{index:04d}",
                raw_text=cast(str, entry["text"]),
                normalized_text=normalizer.normalize(cast(str, entry["text"])).normalized_text,
                label=SentimentLabel(cast(str, entry["label"])),
                aspect=cast(str, entry["aspect"]),
                style=cast(str, entry["style"]),
                noise_types=list(cast(list[str], entry["noise_types"])),
                difficulty=Difficulty(cast(str, entry["difficulty"])),
                canonical_group_id=f"{batch}-g{index:04d}",
                generation_batch=batch,
                review_status=ReviewStatus.PENDING,
            )
            for index, entry in enumerate(pool)
        ]
