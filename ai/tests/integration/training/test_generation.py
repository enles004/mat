"""Curated-pool assembly contract (dataset v2).

Every pool sentence — clean and noisy alike — is written complete by the
reviewer; the generator applies no text transformation. Its whole job is
deterministic assembly: stable ids and groups, catalog normalization, and
pending-review rows.
"""

import shutil
from pathlib import Path

import pytest
import yaml

from src.domain.entities import DatasetRow, ReviewStatus
from src.nlp.training.dataset import DatasetStore
from src.nlp.training.generation import DatasetGenerator

PROD_NORMALIZATION = Path("configs/normalization.yaml")

# Noisy entries carry their surface forms already written in ("toy", "ko",
# the emoji, the code-switching tail) — exactly what the reviewer approved.
POOL: list[dict[str, object]] = [
    {
        "text": "Động cơ Vinfast chạy êm, lên dốc không ì",
        "label": "positive",
        "aspect": "động cơ",
        "style": "review",
        "difficulty": "easy",
        "noise_types": [],
    },
    {
        "text": "toy này máy vẫn bốc lắm nha",
        "label": "positive",
        "aspect": "động cơ",
        "style": "comment",
        "difficulty": "medium",
        "noise_types": ["teencode", "brand_alias"],
    },
    {
        "text": "Nội thất Hyundai đi hai năm bắt đầu kêu ồn",
        "label": "negative",
        "aspect": "nội thất",
        "style": "review",
        "difficulty": "easy",
        "noise_types": [],
    },
    {
        "text": "Giá Mazda ở mức tạm ổn so với đối thủ",
        "label": "neutral",
        "aspect": "giá cả",
        "style": "comparison",
        "difficulty": "medium",
        "noise_types": [],
    },
    {
        "text": "Dịch vụ hậu mãi Vinfast trả lời chậm, ko hài lòng 😡",
        "label": "negative",
        "aspect": "dịch vụ hậu mãi",
        "style": "comment",
        "difficulty": "easy",
        "noise_types": ["teencode", "emoji"],
    },
    {
        "text": "Xe tiêu hao khoảng 7 lít trên 100km, overall average thôi",
        "label": "neutral",
        "aspect": "mức tiêu hao nhiên liệu",
        "style": "review",
        "difficulty": "hard",
        "noise_types": ["code_switching"],
    },
]


def _write_config(
    tmp_path: Path,
    *,
    pool: list[dict[str, object]] | None = None,
    version: str = "2.0.0",
) -> Path:
    shutil.copy(PROD_NORMALIZATION, tmp_path / "normalization.yaml")
    document = {
        "version": version,
        "generation_batch": "curated-test",
        "aspects": [
            "động cơ",
            "nội thất",
            "giá cả",
            "dịch vụ hậu mãi",
            "mức tiêu hao nhiên liệu",
        ],
        "styles": ["review", "comment", "comparison", "question", "short", "conversational"],
        "pool": POOL if pool is None else pool,
    }
    path = tmp_path / "data.yaml"
    path.write_text(yaml.safe_dump(document, allow_unicode=True), encoding="utf-8")
    return path


def test_assembly_is_deterministic(tmp_path: Path) -> None:
    config = _write_config(tmp_path)
    first = DatasetGenerator().generate(config)
    second = DatasetGenerator().generate(config)
    assert [row.model_dump() for row in first] == [row.model_dump() for row in second]


def test_ids_and_groups_use_batch_and_index(tmp_path: Path) -> None:
    config = _write_config(tmp_path)
    rows = DatasetGenerator().generate(config)
    assert [row.id for row in rows] == [
        f"curated-test-{index:04d}" for index in range(len(POOL))
    ]
    assert [row.canonical_group_id for row in rows] == [
        f"curated-test-g{index:04d}" for index in range(len(POOL))
    ]
    assert len({row.canonical_group_id for row in rows}) == len(POOL)


def test_raw_text_is_taken_verbatim_from_the_pool(tmp_path: Path) -> None:
    config = _write_config(tmp_path)
    rows = DatasetGenerator().generate(config)
    for index, row in enumerate(rows):
        assert row.raw_text == POOL[index]["text"]


def test_noise_labels_pass_through_from_the_pool(tmp_path: Path) -> None:
    config = _write_config(tmp_path)
    rows = DatasetGenerator().generate(config)
    for index, row in enumerate(rows):
        assert row.noise_types == POOL[index]["noise_types"]
    assert any(row.noise_types for row in rows)


def test_normalized_text_uses_the_neighbor_catalog(tmp_path: Path) -> None:
    config = _write_config(tmp_path)
    rows = DatasetGenerator().generate(config)
    teencode_row = next(row for row in rows if "ko " in row.raw_text)
    assert "không" in teencode_row.normalized_text
    assert "ko " not in teencode_row.normalized_text
    alias_row = next(row for row in rows if row.raw_text.startswith("toy"))
    assert alias_row.normalized_text.startswith("Toyota")


def test_rows_are_pending_with_batch_metadata(tmp_path: Path) -> None:
    config = _write_config(tmp_path)
    rows = DatasetGenerator().generate(config)
    assert all(row.review_status is ReviewStatus.PENDING for row in rows)
    assert all(row.generation_batch == "curated-test" for row in rows)


def test_store_round_trip_preserves_generated_rows(tmp_path: Path) -> None:
    config = _write_config(tmp_path)
    rows = DatasetGenerator().generate(config)
    out = tmp_path / "dataset.csv"
    DatasetStore().write(rows, out)
    read_back: list[DatasetRow] = DatasetStore().read(out)
    assert [row.model_dump() for row in read_back] == [row.model_dump() for row in rows]


def test_duplicate_pool_texts_are_rejected(tmp_path: Path) -> None:
    duplicated = [POOL[0], POOL[1], dict(POOL[0])]
    config = _write_config(tmp_path, pool=duplicated)
    with pytest.raises(ValueError, match="Duplicate pool text"):
        DatasetGenerator().generate(config)


def test_unknown_noise_label_is_rejected(tmp_path: Path) -> None:
    stray = dict(POOL[0])
    stray["noise_types"] = ["magic_dust"]
    config = _write_config(tmp_path, pool=[POOL[2], stray])
    with pytest.raises(ValueError, match="noise_type"):
        DatasetGenerator().generate(config)


def test_missing_noise_label_declaration_is_rejected(tmp_path: Path) -> None:
    undeclared = {key: value for key, value in POOL[0].items() if key != "noise_types"}
    config = _write_config(tmp_path, pool=[POOL[2], undeclared])
    with pytest.raises(ValueError, match="noise_types"):
        DatasetGenerator().generate(config)


def test_pool_entry_outside_declared_aspects_is_rejected(tmp_path: Path) -> None:
    stray = dict(POOL[0])
    stray["aspect"] = "âm thanh"
    config = _write_config(tmp_path, pool=[POOL[2], stray])
    with pytest.raises(ValueError, match="aspect"):
        DatasetGenerator().generate(config)


def test_pool_entry_outside_declared_styles_is_rejected(tmp_path: Path) -> None:
    stray = dict(POOL[0])
    stray["style"] = "poem"
    config = _write_config(tmp_path, pool=[POOL[2], stray])
    with pytest.raises(ValueError, match="style"):
        DatasetGenerator().generate(config)


def test_unsupported_config_version_is_rejected(tmp_path: Path) -> None:
    config = _write_config(tmp_path, version="1.0.0")
    with pytest.raises(ValueError, match="version"):
        DatasetGenerator().generate(config)
