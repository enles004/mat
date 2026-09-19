"""Typed normalization catalog schema tests (Plan 1 Task 1)."""

import importlib
import inspect

import pytest
import yaml
from pydantic import ValidationError

from src.domain.exceptions import CatalogError
from src.nlp.preprocessing.catalog import NormalizationCatalog
from src.nlp.preprocessing.catalog_loader import CatalogLoader

SOURCE = {"kind": "reviewed_internal", "reference": "MAT-NORM-NEG-001"}


def valid_document() -> dict:
    return {
        "schema_version": "1",
        "dictionary_version": "1.0.0",
        "rules": [
            {
                "id": "hyundai_h_dot",
                "category": "brand",
                "pattern": r"(?<!\w)H\.(?!\w)",
                "replacement": "Hyundai",
                "context_any": ["xe", "hãng", "nhà", "ô tô", "oto", "car"],
                "flags": "IGNORECASE",
                "source": {"kind": "reviewed_internal", "reference": "MAT-NORM-BRAND-001"},
            },
            {
                "id": "ko_to_khong",
                "category": "negation",
                "pattern": r"(?<!\w)(ko|khum|k)(?!\w)",
                "replacement": "không",
                "context_any": [],
                "flags": "IGNORECASE",
                "source": dict(SOURCE),
            },
            {
                "id": "cx_to_cung",
                "category": "teencode",
                "pattern": r"(?<!\w)cx(?!\w)",
                "replacement": "cũng",
                "context_any": [],
                "flags": "IGNORECASE",
                "source": {"kind": "reviewed_internal", "reference": "MAT-NORM-TEEN-001"},
            },
            {
                "id": "dc_to_duoc",
                "category": "teencode",
                "pattern": r"(?<!\w)(dc|đc)(?!\w)",
                "replacement": "được",
                "context_any": [],
                "flags": "IGNORECASE",
                "source": {"kind": "reviewed_internal", "reference": "MAT-NORM-TEEN-002"},
            },
        ],
    }


def test_catalog_parses_four_rules_in_yaml_order() -> None:
    catalog = NormalizationCatalog.model_validate(valid_document())
    assert catalog.schema_version == "1"
    assert catalog.dictionary_version == "1.0.0"
    assert [rule.id for rule in catalog.rules] == [
        "hyundai_h_dot",
        "ko_to_khong",
        "cx_to_cung",
        "dc_to_duoc",
    ]
    first = catalog.rules[0]
    assert first.category == "brand"
    assert first.pattern == r"(?<!\w)H\.(?!\w)"
    assert first.replacement == "Hyundai"
    assert first.context_any == ("xe", "hãng", "nhà", "ô tô", "oto", "car")
    assert first.flags == "IGNORECASE"
    assert first.source.kind == "reviewed_internal"
    assert first.source.reference == "MAT-NORM-BRAND-001"


def test_catalog_preserves_rule_payloads() -> None:
    catalog = NormalizationCatalog.model_validate(valid_document())
    assert catalog.rules[1].pattern == r"(?<!\w)(ko|khum|k)(?!\w)"
    assert catalog.rules[1].replacement == "không"
    assert catalog.rules[1].context_any == ()
    assert catalog.rules[1].category == "negation"
    assert catalog.rules[3].pattern == r"(?<!\w)(dc|đc)(?!\w)"
    assert catalog.rules[3].replacement == "được"


def test_catalog_models_are_frozen() -> None:
    catalog = NormalizationCatalog.model_validate(valid_document())
    with pytest.raises(ValidationError):
        catalog.dictionary_version = "9.9.9"  # type: ignore[misc]
    with pytest.raises(ValidationError):
        catalog.rules[0].pattern = "other"  # type: ignore[misc]


def test_catalog_rejects_duplicate_rule_ids() -> None:
    document = valid_document()
    document["rules"].append(document["rules"][1].copy())
    with pytest.raises(ValidationError, match="duplicate rule id: ko_to_khong"):
        NormalizationCatalog.model_validate(document)


def test_catalog_rejects_empty_pattern() -> None:
    document = valid_document()
    document["rules"][0]["pattern"] = "   "
    with pytest.raises(ValidationError, match="hyundai_h_dot"):
        NormalizationCatalog.model_validate(document)


def test_catalog_rejects_empty_replacement() -> None:
    document = valid_document()
    document["rules"][0]["replacement"] = ""
    with pytest.raises(ValidationError, match="hyundai_h_dot"):
        NormalizationCatalog.model_validate(document)


def test_catalog_rejects_unsupported_flag() -> None:
    document = valid_document()
    document["rules"][0]["flags"] = "DOTALL"
    with pytest.raises(ValidationError, match="flags"):
        NormalizationCatalog.model_validate(document)


def test_catalog_rejects_invalid_category() -> None:
    document = valid_document()
    document["rules"][0]["category"] = "slang"
    with pytest.raises(ValidationError, match="category"):
        NormalizationCatalog.model_validate(document)


def test_catalog_rejects_empty_source_reference() -> None:
    document = valid_document()
    document["rules"][0]["source"]["reference"] = "  "
    with pytest.raises(ValidationError, match="reference"):
        NormalizationCatalog.model_validate(document)


def test_catalog_rejects_unsupported_source_kind() -> None:
    document = valid_document()
    document["rules"][0]["source"]["kind"] = "guessed"
    with pytest.raises(ValidationError, match="kind"):
        NormalizationCatalog.model_validate(document)


def test_catalog_rejects_zero_rules() -> None:
    document = valid_document()
    document["rules"] = []
    with pytest.raises(ValidationError, match="rules"):
        NormalizationCatalog.model_validate(document)


def test_catalog_rejects_uncompilable_pattern() -> None:
    document = valid_document()
    document["rules"][0]["pattern"] = "(?<!\\w)H.(?!\\w"
    with pytest.raises(ValidationError, match="uncompilable pattern"):
        NormalizationCatalog.model_validate(document)


def test_loader_wraps_uncompilable_pattern_in_catalog_error(tmp_path) -> None:
    document = valid_document()
    document["rules"][0]["pattern"] = "(?<!\\w)H.(?!\\w"
    catalog_path = tmp_path / "catalog.yaml"
    catalog_path.write_text(yaml.safe_dump(document), encoding="utf-8")
    with pytest.raises(CatalogError, match="uncompilable pattern"):
        CatalogLoader().load(catalog_path)


def test_preprocessing_modules_expose_only_designated_public_class() -> None:
    expected = {
        "catalog": "NormalizationCatalog",
        "catalog_loader": "CatalogLoader",
        "loaded_catalog": "LoadedCatalog",
        "matcher": "RegexMatcher",
        "normalizer": "TextNormalizer",
        "provider": "NormalizerProvider",
    }
    for module_name, class_name in expected.items():
        module = importlib.import_module(f"src.nlp.preprocessing.{module_name}")
        public_bindings = sorted(
            name
            for name, value in vars(module).items()
            if not name.startswith("_") and (inspect.isclass(value) or inspect.isroutine(value))
        )
        assert public_bindings == [class_name]


def test_preprocessing_modules_do_not_reexport_legacy_bindings() -> None:
    legacy_bindings = {
        "catalog_loader": ("CatalogError", "LoadedCatalog", "load_catalog", "verify_catalog_lock"),
        "matcher": ("RuleMatch", "Rule", "CompiledMatcher"),
        "normalizer": ("NormalizationResult", "RegexMatcher", "load_rules", "normalize"),
        "provider": ("CatalogError", "CatalogLoader", "RegexMatcher", "TextNormalizer"),
    }
    for module_name, bindings in legacy_bindings.items():
        module = importlib.import_module(f"src.nlp.preprocessing.{module_name}")
        for binding in bindings:
            assert not hasattr(module, binding)
