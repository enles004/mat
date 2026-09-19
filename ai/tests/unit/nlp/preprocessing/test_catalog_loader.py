"""Typed catalog loader and checksum lock tests (Plan 1 Task 2)."""

import hashlib
import json
from pathlib import Path

import pytest

from src.domain.exceptions import CatalogError
from src.nlp.preprocessing.catalog_loader import CatalogLoader

VALID_ONE_RULE = """schema_version: "1"
dictionary_version: "1.0.0"
rules:
  - id: ko_to_khong
    category: negation
    pattern: '(?<!\\w)(ko|khum|k)(?!\\w)'
    replacement: không
    context_any: []
    flags: IGNORECASE
    source:
      kind: reviewed_internal
      reference: MAT-NORM-NEG-001
"""


def write_catalog(tmp_path: Path, body: str) -> Path:
    path = tmp_path / "normalization.yaml"
    path.write_text(body, encoding="utf-8")
    return path


def load_valid(tmp_path: Path):
    return CatalogLoader().load(write_catalog(tmp_path, VALID_ONE_RULE))


def write_lock(tmp_path: Path, schema: str, dictionary: str, checksum: str) -> Path:
    path = tmp_path / "normalization.lock.json"
    path.write_text(
        json.dumps(
            {
                "schema_version": schema,
                "dictionary_version": dictionary,
                "catalog_checksum": checksum,
            }
        )
        + "\n",
        encoding="utf-8",
    )
    return path


def test_load_catalog_checksums_exact_bytes(tmp_path: Path) -> None:
    path = write_catalog(tmp_path, VALID_ONE_RULE)
    loaded = CatalogLoader().load(path)
    expected = "sha256:" + hashlib.sha256(path.read_bytes()).hexdigest()
    assert loaded.checksum == expected
    assert loaded.catalog.dictionary_version == "1.0.0"
    assert [rule.id for rule in loaded.catalog.rules] == ["ko_to_khong"]


def test_load_catalog_rejects_malformed_yaml(tmp_path: Path) -> None:
    path = write_catalog(tmp_path, "rules: [broken")
    with pytest.raises(CatalogError, match="malformed"):
        CatalogLoader().load(path)


def test_load_catalog_rejects_non_mapping_yaml(tmp_path: Path) -> None:
    path = write_catalog(tmp_path, "- just\n- a\n- list\n")
    with pytest.raises(CatalogError, match="YAML mapping"):
        CatalogLoader().load(path)


def test_load_catalog_rejects_duplicate_yaml_keys(tmp_path: Path) -> None:
    body = (
        'schema_version: "1"\n'
        'schema_version: "2"\n'
        'dictionary_version: "1.0.0"\n'
        "rules:\n"
        "  - id: ko_to_khong\n"
        "    category: negation\n"
        "    pattern: 'x'\n"
        "    replacement: 'y'\n"
        "    context_any: []\n"
        "    flags: NONE\n"
        "    source:\n"
        "      kind: reviewed_internal\n"
        "      reference: MAT-NORM-NEG-001\n"
    )
    path = write_catalog(tmp_path, body)
    with pytest.raises(CatalogError, match="duplicate"):
        CatalogLoader().load(path)


def test_load_catalog_rejects_uncompilable_pattern(tmp_path: Path) -> None:
    body = VALID_ONE_RULE.replace(r"(?<!\w)(ko|khum|k)(?!\w)", "(unbalanced")
    path = write_catalog(tmp_path, body)
    with pytest.raises(CatalogError, match="ko_to_khong"):
        CatalogLoader().load(path)


def test_load_catalog_rejects_missing_file(tmp_path: Path) -> None:
    with pytest.raises(CatalogError, match="unreadable"):
        CatalogLoader().load(tmp_path / "does-not-exist.yaml")


def test_load_catalog_rejects_invalid_utf8(tmp_path: Path) -> None:
    path = tmp_path / "normalization.yaml"
    path.write_bytes(b"schema_version: \xff\xfe\n")
    with pytest.raises(CatalogError, match="UTF-8"):
        CatalogLoader().load(path)


def test_verify_catalog_lock_accepts_matching_lock(tmp_path: Path) -> None:
    loaded = load_valid(tmp_path)
    lock_path = write_lock(tmp_path, "1", "1.0.0", loaded.checksum)
    assert CatalogLoader().verify_lock(loaded, lock_path) is None


def test_verify_catalog_lock_rejects_version_mismatch(tmp_path: Path) -> None:
    loaded = load_valid(tmp_path)
    lock_path = write_lock(tmp_path, "1", "0.9.0", loaded.checksum)
    with pytest.raises(CatalogError, match="dictionary_version"):
        CatalogLoader().verify_lock(loaded, lock_path)


def test_verify_catalog_lock_rejects_schema_version_mismatch(tmp_path: Path) -> None:
    loaded = load_valid(tmp_path)
    lock_path = write_lock(tmp_path, "2", "1.0.0", loaded.checksum)
    with pytest.raises(CatalogError, match="schema_version"):
        CatalogLoader().verify_lock(loaded, lock_path)


def test_verify_catalog_lock_rejects_checksum_mismatch(tmp_path: Path) -> None:
    loaded = load_valid(tmp_path)
    drifted = "sha256:" + "0" * 64
    lock_path = write_lock(tmp_path, "1", "1.0.0", drifted)
    with pytest.raises(CatalogError, match="checksum"):
        CatalogLoader().verify_lock(loaded, lock_path)


def test_checked_in_lock_matches_checked_in_catalog() -> None:
    loaded = CatalogLoader().load(Path("configs/normalization.yaml"))
    CatalogLoader().verify_lock(loaded, Path("configs/normalization.lock.json"))
    assert [rule.id for rule in loaded.catalog.rules] == [
        "hyundai_h_dot",
        "toy_to_toyota",
        "vf_to_vinfast",
        "mitsu_to_mitsubishi",
        "mec_to_mercedes",
        "ko_to_khong",
        "cx_to_cung",
        "dc_to_duoc",
        "elong_qua_to_qua",
        "elong_rat_to_rat",
        "overall_good_to_tot",
        "overall_average_to_tam",
        "overall_bad_to_te",
    ]


def test_catalog_identity_is_content_addressed_across_symlinks(tmp_path: Path) -> None:
    """Canonical path policy: a catalog's identity is its exact-byte checksum,
    not its path. Reading through a symlink is transparent, and repointing a
    symlink at drifted content can never pass the reviewed lock."""
    reviewed = write_catalog(tmp_path, VALID_ONE_RULE)
    loaded = CatalogLoader().load(reviewed)

    link = tmp_path / "active-normalization.yaml"
    link.symlink_to(reviewed)
    via_link = CatalogLoader().load(link)
    assert via_link.checksum == loaded.checksum
    assert via_link.catalog == loaded.catalog

    drifted = write_catalog(tmp_path, VALID_ONE_RULE.replace("không", "khôngDRIFT"))
    drifted_link = tmp_path / "active-normalization.yaml"
    drifted_link.unlink()
    drifted_link.symlink_to(drifted)
    assert CatalogLoader().load(drifted_link).checksum != loaded.checksum
    lock = write_lock(tmp_path, "1", "1.0.0", loaded.checksum)
    with pytest.raises(CatalogError, match="checksum"):
        CatalogLoader().verify_lock(CatalogLoader().load(drifted_link), lock)
