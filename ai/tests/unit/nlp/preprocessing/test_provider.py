"""Atomic activation provider tests (Plan 1 Task 4)."""

import hashlib
import json
import threading
from pathlib import Path

import pytest

from src.domain.artifacts import CatalogActivation
from src.domain.contracts import Normalizer
from src.domain.exceptions import CatalogError
from src.nlp.preprocessing.catalog_loader import CatalogLoader
from src.nlp.preprocessing.matcher import RegexMatcher
from src.nlp.preprocessing.normalizer import TextNormalizer
from src.nlp.preprocessing.provider import NormalizerProvider

PROD_CATALOG = Path("configs/normalization.yaml")
VARIANT_SUFFIX = "V2MARKER"


def _checksum(path: Path) -> str:
    return "sha256:" + hashlib.sha256(path.read_bytes()).hexdigest()


def write_variant_catalog(tmp_path: Path) -> Path:
    """A complete second catalog: same shape, different ko replacement."""
    body = PROD_CATALOG.read_text(encoding="utf-8").replace(
        "replacement: không",
        f"replacement: không{VARIANT_SUFFIX}",
    )
    path = tmp_path / "normalization-v2.yaml"
    path.write_text(body, encoding="utf-8")
    return path


def write_lock(tmp_path: Path, catalog_path: Path) -> Path:
    lock_path = tmp_path / "normalization.lock.json"
    lock_path.write_text(
        json.dumps(
            {
                "schema_version": "1",
                "dictionary_version": "2.0.0",
                "catalog_checksum": _checksum(catalog_path),
            }
        )
        + "\n",
        encoding="utf-8",
    )
    return lock_path


def make_provider(tmp_path: Path, catalog_path: Path = PROD_CATALOG) -> NormalizerProvider:
    return NormalizerProvider(lock_path=write_lock(tmp_path, catalog_path))


def accept_normalizer(normalizer: Normalizer) -> Normalizer:
    """Structural protocol conformance: arguments must satisfy Normalizer."""
    return normalizer


def test_provider_requires_activation_before_get(tmp_path: Path) -> None:
    provider = make_provider(tmp_path)
    with pytest.raises(CatalogError, match="not activated"):
        provider.get()


def test_activation_publishes_version_checksum_and_rule_count(tmp_path: Path) -> None:
    provider = make_provider(tmp_path)
    activation = provider.activate(PROD_CATALOG, _checksum(PROD_CATALOG))
    assert isinstance(activation, CatalogActivation)
    assert activation.dictionary_version == "2.0.0"
    assert activation.catalog_checksum == _checksum(PROD_CATALOG)
    assert activation.rule_count == 13
    assert provider.get() is activation.normalizer
    result = accept_normalizer(provider.get()).normalize("xe ko ổn")
    assert result.normalized_text == "xe không ổn"


def test_activation_accepts_path_keyword(tmp_path: Path) -> None:
    provider = make_provider(tmp_path)
    activation = provider.activate(path=PROD_CATALOG, expected_checksum=_checksum(PROD_CATALOG))
    assert provider.get() is activation.normalizer


def test_text_normalizer_accepts_injected_matcher() -> None:
    loaded = CatalogLoader().load(PROD_CATALOG)
    normalizer = accept_normalizer(TextNormalizer(RegexMatcher.from_catalog(loaded.catalog)))
    assert normalizer.normalize("xe ko ổn").normalized_text == "xe không ổn"


@pytest.mark.parametrize(
    "breakage",
    ["wrong_checksum", "malformed_yaml", "bad_regex", "lock_drift"],
)
def test_failed_activation_preserves_previous_identity_and_behavior(
    tmp_path: Path,
    breakage: str,
) -> None:
    provider = make_provider(tmp_path)
    provider.activate(PROD_CATALOG, _checksum(PROD_CATALOG))
    previous = provider.get()
    before = previous.normalize("xe ko ổn")

    variant = write_variant_catalog(tmp_path)
    if breakage == "wrong_checksum":
        with pytest.raises(CatalogError, match="expected catalog checksum mismatch"):
            provider.activate(PROD_CATALOG, "sha256:" + "0" * 64)
    elif breakage == "malformed_yaml":
        broken = tmp_path / "broken.yaml"
        broken.write_text("rules: [oops", encoding="utf-8")
        with pytest.raises(CatalogError, match="malformed"):
            provider.activate(broken, _checksum(broken))
    elif breakage == "bad_regex":
        bad = tmp_path / "bad-regex.yaml"
        bad.write_text(
            PROD_CATALOG.read_text(encoding="utf-8").replace(r"(?<!\w)cx(?!\w)", "(unbalanced"),
            encoding="utf-8",
        )
        with pytest.raises(CatalogError, match="uncompilable pattern"):
            provider.activate(bad, _checksum(bad))
    else:
        with pytest.raises(CatalogError, match="checksum mismatch"):
            provider.activate(variant, _checksum(variant))

    assert provider.get() is previous
    after = provider.get().normalize("xe ko ổn")
    assert (after.normalized_text, after.applied_rules, after.warnings) == (
        before.normalized_text,
        before.applied_rules,
        before.warnings,
    )


def test_successful_activation_swaps_to_new_complete_version(tmp_path: Path) -> None:
    provider = make_provider(tmp_path)
    old_activation = provider.activate(PROD_CATALOG, _checksum(PROD_CATALOG))

    variant = write_variant_catalog(tmp_path)
    write_lock(tmp_path, variant)
    new_activation = provider.activate(variant, _checksum(variant))

    assert new_activation.normalizer is not old_activation.normalizer
    assert provider.get() is new_activation.normalizer
    result = provider.get().normalize("xe ko ổn")
    assert result.normalized_text == f"xe không{VARIANT_SUFFIX} ổn"


def test_concurrent_readers_observe_only_complete_versions(tmp_path: Path) -> None:
    provider = make_provider(tmp_path)
    old_activation = provider.activate(PROD_CATALOG, _checksum(PROD_CATALOG))
    old_text = old_activation.normalizer.normalize("xe ko ổn").normalized_text
    assert old_text == "xe không ổn"

    variant = write_variant_catalog(tmp_path)
    write_lock(tmp_path, variant)
    new_text = f"xe không{VARIANT_SUFFIX} ổn"

    observations: list[str] = []
    identities: list[int] = []
    errors: list[Exception] = []
    stop = threading.Event()

    def reader() -> None:
        while not stop.is_set():
            try:
                normalizer = provider.get()
                observations.append(normalizer.normalize("xe ko ổn").normalized_text)
                identities.append(id(normalizer))
            except Exception as exc:  # the test collects any reader failure
                errors.append(exc)
                return

    threads = [threading.Thread(target=reader) for _ in range(4)]
    for thread in threads:
        thread.start()
    try:
        for _ in range(25):
            # The operator contract: catalog + lock move together, then the
            # swap is requested. Readers never touch the filesystem.
            write_lock(tmp_path, variant)
            provider.activate(variant, _checksum(variant))
            write_lock(tmp_path, PROD_CATALOG)
            provider.activate(PROD_CATALOG, _checksum(PROD_CATALOG))
    finally:
        stop.set()
        for thread in threads:
            thread.join()

    assert errors == []
    assert observations, "readers produced no observations"
    observed = set(observations)
    assert observed <= {old_text, new_text}, f"partial versions: {sorted(observed)}"
    assert identities  # every read returned a live normalizer reference


def test_lock_replacement_race_never_half_applies(tmp_path: Path) -> None:
    """Two activators racing the lock replacement can each lose to the other's
    lock file, but a lost race is a typed ``CatalogError`` and the provider's
    active reference is always one complete version — never a torn swap."""
    provider = make_provider(tmp_path)
    old_activation = provider.activate(PROD_CATALOG, _checksum(PROD_CATALOG))
    old_text = old_activation.normalizer.normalize("xe ko ổn").normalized_text

    variant = write_variant_catalog(tmp_path)
    new_text = f"xe không{VARIANT_SUFFIX} ổn"

    errors: list[Exception] = []
    observations: list[str] = []
    identities: list[int] = []
    stop = threading.Event()

    def activator(catalog: Path) -> None:
        for _ in range(25):
            write_lock(tmp_path, catalog)  # lock moves with its catalog, per operator contract
            try:
                provider.activate(catalog, _checksum(catalog))
            except Exception as exc:  # collect; classified below
                errors.append(exc)

    def reader() -> None:
        while not stop.is_set():
            try:
                normalizer = provider.get()
                observations.append(normalizer.normalize("xe ko ổn").normalized_text)
                identities.append(id(normalizer))
            except Exception as exc:  # the test collects any reader failure
                errors.append(exc)
                return

    threads = [
        threading.Thread(target=activator, args=(variant,)),
        threading.Thread(target=activator, args=(PROD_CATALOG,)),
        threading.Thread(target=reader),
    ]
    for thread in threads:
        thread.start()
    try:
        for thread in threads[:2]:  # activators are cycle-bounded
            thread.join()
    finally:
        stop.set()
        threads[2].join()

    assert all(isinstance(exc, CatalogError) for exc in errors), [
        repr(exc) for exc in errors
    ]
    assert observations, "readers produced no observations"
    observed = set(observations)
    assert observed <= {old_text, new_text}, f"partial versions: {sorted(observed)}"
    assert identities

    # After the race settles, one clean activation still lands completely.
    write_lock(tmp_path, variant)
    final = provider.activate(variant, _checksum(variant))
    assert provider.get() is final.normalizer
    assert provider.get().normalize("xe ko ổn").normalized_text == new_text
