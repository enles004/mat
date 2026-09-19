from dataclasses import dataclass as _dataclass

from src.nlp.preprocessing.catalog import NormalizationCatalog as _NormalizationCatalog


@_dataclass(frozen=True)
class LoadedCatalog:
    """An immutable loaded catalog bound to its exact-byte checksum."""

    catalog: _NormalizationCatalog
    checksum: str
