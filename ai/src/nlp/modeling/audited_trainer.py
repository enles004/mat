from collections.abc import Callable, Mapping
from typing import Any

from transformers import Trainer as _Trainer

from src.domain.exceptions import (
    ModelLoadingProvenanceError as _ModelLoadingProvenanceError,
)


class AuditedTrainer(_Trainer):
    """Reload best checkpoints through from_pretrained so legacy weight names are converted."""

    best_checkpoint_diagnostics: dict[str, list[str]] | None = None

    def __init__(
        self,
        *args: Any,
        source_revision: str,
        loading_validator: Callable[[Mapping[str, object]], dict[str, list[str]]],
        **kwargs: Any,
    ) -> None:
        super().__init__(*args, **kwargs)
        self.source_revision = source_revision
        self._loading_validator = loading_validator

    def _load_best_model(self) -> None:
        checkpoint = self.state.best_model_checkpoint
        if checkpoint is None:
            raise _ModelLoadingProvenanceError("Trainer did not record a best-model checkpoint")
        if self.model is None:
            raise _ModelLoadingProvenanceError("Trainer does not have a model to reload")
        model = self.model
        loader = getattr(model, "from_pretrained", None)
        if not callable(loader):
            raise _ModelLoadingProvenanceError("Trainer model does not support from_pretrained")
        loaded = loader(
            checkpoint,
            revision=self.source_revision,
            output_loading_info=True,
            local_files_only=True,
        )
        if not isinstance(loaded, tuple) or len(loaded) != 2:
            raise _ModelLoadingProvenanceError(
                "Checkpoint loader did not return loading diagnostics"
            )
        checkpoint_model, loading_info = loaded
        if not isinstance(loading_info, Mapping):
            raise _ModelLoadingProvenanceError("Checkpoint loading diagnostics are not a mapping")
        self.best_checkpoint_diagnostics = self._loading_validator(loading_info)
        model.load_state_dict(checkpoint_model.state_dict(), strict=True)
