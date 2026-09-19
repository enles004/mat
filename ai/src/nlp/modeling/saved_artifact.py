from dataclasses import dataclass as _dataclass
from pathlib import Path as _Path

from src.domain.entities import ArtifactManifest as _ArtifactManifest


@_dataclass(frozen=True)
class SavedArtifact:
    """One exported artifact directory and its verified manifest."""

    directory: _Path
    manifest: _ArtifactManifest
    payload_files: tuple[_Path, ...]
