import hashlib
import json
import re
from pathlib import Path as _Path
from typing import Any as _Any

import yaml

from src.domain.exceptions import CatalogError as _CatalogError
from src.nlp.preprocessing.catalog import NormalizationCatalog as _NormalizationCatalog
from src.nlp.preprocessing.loaded_catalog import LoadedCatalog as _LoadedCatalog


class _StrictSafeLoader(yaml.SafeLoader):
    """SafeLoader that rejects duplicate mapping keys before validation."""


class CatalogLoader:
    """Load reviewed catalogs and verify their caller-supplied lock files."""

    @staticmethod
    def _construct_mapping(
        loader: _StrictSafeLoader,
        node: yaml.MappingNode,
        deep: bool = False,
    ) -> dict[_Any, _Any]:
        loader.flatten_mapping(node)
        mapping: dict[_Any, _Any] = {}
        for key_node, value_node in node.value:
            key = loader.construct_object(key_node, deep=deep)
            try:
                duplicate = key in mapping
            except TypeError as exc:
                raise yaml.constructor.ConstructorError(
                    "while constructing a mapping",
                    node.start_mark,
                    f"found unhashable key: {key!r}",
                    key_node.start_mark,
                ) from exc
            if duplicate:
                raise yaml.constructor.ConstructorError(
                    "while constructing a mapping",
                    node.start_mark,
                    f"found duplicate key: {key!r}",
                    key_node.start_mark,
                )
            mapping[key] = loader.construct_object(value_node, deep=deep)
        return mapping

    @staticmethod
    def _read_yaml(raw: bytes, path: _Path) -> dict[_Any, _Any]:
        try:
            text = raw.decode("utf-8")
        except UnicodeDecodeError as exc:
            raise _CatalogError(f"catalog is not valid UTF-8: {path}") from exc
        try:
            document = yaml.load(text, Loader=_StrictSafeLoader)
        except yaml.YAMLError as exc:
            raise _CatalogError(f"catalog YAML is malformed: {path}: {exc}") from exc
        if not isinstance(document, dict):
            raise _CatalogError(f"catalog must be a YAML mapping: {path}")
        return document

    def load(self, path: _Path) -> _LoadedCatalog:
        """Load and validate the catalog from ``path`` with no fallback."""
        try:
            raw = path.read_bytes()
        except OSError as exc:
            raise _CatalogError(f"catalog file unreadable: {path}") from exc
        checksum = "sha256:" + hashlib.sha256(raw).hexdigest()
        document = self._read_yaml(raw, path)
        try:
            catalog = _NormalizationCatalog.model_validate(document)
        except ValueError as exc:  # pydantic.ValidationError subclasses ValueError
            raise _CatalogError(f"catalog validation failed: {path}: {exc}") from exc

        for rule in catalog.rules:
            flags = re.IGNORECASE if rule.flags == "IGNORECASE" else 0
            try:
                re.compile(rule.pattern, flags)
            except re.error as exc:
                raise _CatalogError(
                    f"catalog validation failed: {path}: rule '{rule.id}': "
                    f"uncompilable pattern: {exc}"
                ) from exc
        return _LoadedCatalog(catalog=catalog, checksum=checksum)

    def verify_lock(self, loaded: _LoadedCatalog, lock_path: _Path) -> None:
        """Fail when a loaded catalog differs from its reviewed lock."""
        try:
            raw = lock_path.read_text(encoding="utf-8")
        except OSError as exc:
            raise _CatalogError(f"normalization lock unreadable: {lock_path}") from exc
        try:
            lock = json.loads(raw)
        except json.JSONDecodeError as exc:
            raise _CatalogError(f"normalization lock is malformed: {lock_path}: {exc}") from exc
        if not isinstance(lock, dict):
            raise _CatalogError(f"normalization lock must be a JSON object: {lock_path}")
        expected_schema = loaded.catalog.schema_version
        if lock.get("schema_version") != expected_schema:
            raise _CatalogError(
                f"normalization lock schema_version mismatch: catalog {expected_schema!r}, "
                f"lock {lock.get('schema_version')!r}"
            )
        expected_dictionary = loaded.catalog.dictionary_version
        if lock.get("dictionary_version") != expected_dictionary:
            raise _CatalogError(
                f"normalization lock dictionary_version mismatch: catalog {expected_dictionary!r}, "
                f"lock {lock.get('dictionary_version')!r}"
            )
        if lock.get("catalog_checksum") != loaded.checksum:
            raise _CatalogError(
                f"normalization lock checksum mismatch: catalog {loaded.checksum}, "
                f"lock {lock.get('catalog_checksum')} — catalog drifted from its reviewed lock"
            )


_StrictSafeLoader.add_constructor(
    yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG,
    CatalogLoader._construct_mapping,
)
