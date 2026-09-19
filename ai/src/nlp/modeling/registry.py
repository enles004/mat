import hashlib as _hashlib
import json as _json
from collections.abc import Callable as _Callable
from collections.abc import Sequence as _Sequence
from pathlib import Path as _Path
from typing import Any as _Any

import joblib as _joblib  # type: ignore[import-untyped]

from src.domain.entities import ArtifactManifest as _ArtifactManifest
from src.nlp.constants import MANIFEST_NAME, MODEL_PAYLOAD_NAME
from src.nlp.modeling.saved_artifact import SavedArtifact as _SavedArtifact


class ArtifactRegistry:
    """Save, enumerate, verify, and load local model artifacts."""

    @staticmethod
    def _display_key(path: _Path, base: _Path | None) -> str:
        """Return the path exactly as it enters the checksum, relative to the base."""
        if base is None:
            return path.as_posix()
        return path.resolve().relative_to(base.resolve()).as_posix()

    @classmethod
    def payload_checksum(cls, payload_files: _Sequence[_Path], base: _Path | None = None) -> str:
        """SHA-256 over payload files in lexicographic relative-path order."""
        digest = _hashlib.sha256()
        for path in sorted(payload_files, key=lambda item: cls._display_key(item, base)):
            digest.update(cls._display_key(path, base).encode("utf-8"))
            digest.update(b"\x00")
            digest.update(path.read_bytes())
            digest.update(b"\n")
        return f"sha256:{digest.hexdigest()}"

    @classmethod
    def verify_payload_checksum(
        cls,
        payload_files: _Sequence[_Path],
        base: _Path | None = None,
        expected: str | None = None,
    ) -> str:
        """Return the computed checksum, rejecting a mismatch with the expected value."""
        actual = cls.payload_checksum(payload_files, base=base)
        if expected is not None and actual != expected:
            raise ValueError(f"payload checksum mismatch: expected {expected}, computed {actual}")
        return actual

    def save(
        self,
        model: object,
        directory: _Path,
        manifest_for: _Callable[[str], _ArtifactManifest],
    ) -> _SavedArtifact:
        """Persist the payload, verify it on disk, then write its manifest."""
        directory.mkdir(parents=True, exist_ok=True)
        payload = directory / MODEL_PAYLOAD_NAME
        _joblib.dump(model, payload)
        checksum = self.payload_checksum([payload], base=directory)
        manifest = manifest_for(checksum)
        (directory / MANIFEST_NAME).write_text(
            _json.dumps(
                manifest.model_dump(mode="json"),
                ensure_ascii=False,
                indent=2,
                sort_keys=True,
            )
            + "\n",
            encoding="utf-8",
        )
        return _SavedArtifact(
            directory=directory,
            manifest=manifest,
            payload_files=self.payload_files(directory),
        )

    @classmethod
    def payload_files(cls, directory: _Path) -> tuple[_Path, ...]:
        """List payload files except manifest.json in lexicographic order."""
        if not directory.is_dir():
            raise ValueError(f"Artifact directory does not exist: {directory}")
        files = [
            path for path in directory.rglob("*") if path.is_file() and path.name != MANIFEST_NAME
        ]
        if not files:
            raise ValueError(f"Artifact directory has no payload files: {directory}")
        return tuple(sorted(files, key=lambda item: cls._display_key(item, directory)))

    def verify(self, directory: _Path) -> _ArtifactManifest:
        """Validate manifest.json and recompute every payload checksum against it."""
        manifest_path = directory / MANIFEST_NAME
        if not manifest_path.is_file():
            raise ValueError(f"Artifact manifest is missing: {manifest_path}")
        manifest = _ArtifactManifest.model_validate_json(manifest_path.read_text(encoding="utf-8"))
        self.verify_payload_checksum(
            self.payload_files(directory),
            base=directory,
            expected=manifest.payload_checksum,
        )
        return manifest

    def load(self, directory: _Path) -> tuple[_Any, _ArtifactManifest]:
        """Load a model from a trusted local artifact after verification."""
        manifest = self.verify(directory)
        model = _joblib.load(directory / MODEL_PAYLOAD_NAME)
        return model, manifest
