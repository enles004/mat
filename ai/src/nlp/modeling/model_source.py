import json as _json
import re as _re
from collections.abc import Callable as _Callable
from collections.abc import Mapping as _Mapping
from hashlib import sha256 as _sha256
from pathlib import Path as _Path
from typing import Any as _Any

import yaml as _yaml
from huggingface_hub import HfApi as _HfApi
from huggingface_hub import hf_hub_download as _hf_hub_download

from src.domain.artifacts import ResolvedModelSource as _ResolvedModelSource
from src.nlp.constants import _COMMIT_SHA


class ModelSourceResolver:
    """Resolve, validate, and persist immutable transformer source provenance."""

    def __init__(
        self,
        api_factory: _Callable[[], _Any] = _HfApi,
        download: _Callable[..., str] = _hf_hub_download,
    ) -> None:
        self._api_factory = api_factory
        self._download = download

    @staticmethod
    def _card_metadata(info: object) -> _Mapping[str, _Any]:
        card_data = getattr(info, "card_data", None)
        if card_data is None:
            card_data = getattr(info, "cardData", None)
        return card_data if isinstance(card_data, _Mapping) else {}

    @staticmethod
    def _front_matter(card_text: str) -> dict[str, object]:
        match = _re.match(
            r"\A---[ \t]*\n(.*?)\n---[ \t]*(?:\n|\Z)",
            card_text.lstrip("\n"),
            _re.DOTALL,
        )
        if not match:
            return {}
        parsed = _yaml.safe_load(match.group(1))
        if not isinstance(parsed, _Mapping):
            return {}
        return {str(key): value for key, value in parsed.items()}

    @staticmethod
    def _card_section(card_text: str, *heading_fragments: str) -> str:
        headings = list(
            _re.finditer(r"(?m)^#{1,6}[ \t]+(?P<title>[^\n]+?)[ \t]*$", card_text)
        )
        for index, heading in enumerate(headings):
            title = heading.group("title").casefold()
            if not any(fragment.casefold() in title for fragment in heading_fragments):
                continue
            next_start = (
                headings[index + 1].start()
                if index + 1 < len(headings)
                else len(card_text)
            )
            return card_text[heading.end() : next_start].strip()
        return ""

    @staticmethod
    def _fallback_restriction(card_text: str) -> str:
        for paragraph in _re.split(r"\n\s*\n", card_text):
            normalized = " ".join(paragraph.split())
            lowered = normalized.lower()
            if "research" in lowered and "education" in lowered:
                return normalized
        return ""

    @staticmethod
    def _license_id(value: object) -> str:
        if isinstance(value, str):
            return value.strip() or "unknown"
        if isinstance(value, list) and all(isinstance(item, str) for item in value):
            return ", ".join(item.strip() for item in value if item.strip()) or "unknown"
        return "unknown"

    @staticmethod
    def _file_checksums(info: object) -> dict[str, str]:
        siblings = getattr(info, "siblings", ())
        checksums: dict[str, str] = {}
        if not isinstance(siblings, list):
            return checksums
        for sibling in siblings:
            filename = getattr(sibling, "rfilename", None)
            lfs = getattr(sibling, "lfs", None)
            checksum = getattr(lfs, "sha256", None)
            if isinstance(filename, str) and isinstance(checksum, str):
                checksums[filename] = checksum
        return dict(sorted(checksums.items()))

    @staticmethod
    def _transformers_requirement(card_text: str) -> str:
        match = _re.search(
            r"\btransformers\s*((?:[<>=!~]=?\s*\d+(?:\.\d+){0,2})+)", card_text
        )
        if not match:
            return ""
        return _re.sub(r"\s+", "", match.group(1))

    def resolve(self, repo_id: str) -> _ResolvedModelSource:
        """Resolve the Hub commit before fetching tokenizer or model weights."""
        info = self._api_factory().model_info(repo_id, files_metadata=True)
        revision = getattr(info, "sha", None)
        if not isinstance(revision, str) or not _COMMIT_SHA.fullmatch(revision):
            raise ValueError("Hugging Face did not return a 40-character commit SHA")

        card_path = _Path(
            self._download(repo_id=repo_id, filename="README.md", revision=revision)
        )
        card_text = card_path.read_text(encoding="utf-8")
        metadata = {**self._card_metadata(info), **self._front_matter(card_text)}
        license_metadata = {
            key: value for key, value in metadata.items() if key.casefold().startswith("license")
        }
        license_id = self._license_id(license_metadata.get("license", "unknown"))
        intended_use = self._card_section(
            card_text, "intended use", "uses"
        ) or self._fallback_restriction(card_text)
        if not intended_use:
            raise ValueError("Model card does not contain an intended-use restriction")
        license_text = self._card_section(card_text, "license")
        return _ResolvedModelSource(
            repo_id=repo_id,
            revision=revision,
            license_id=license_id,
            intended_use=intended_use,
            license_text=license_text,
            license_metadata=license_metadata,
            file_checksums=self._file_checksums(info),
            model_card_checksum=_sha256(card_text.encode("utf-8")).hexdigest(),
            transformers_requirement=self._transformers_requirement(card_text),
        )

    @staticmethod
    def write_source_metadata(source: _ResolvedModelSource, output_path: _Path) -> None:
        """Write immutable source metadata before model/tokenizer weight downloads."""
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(
            _json.dumps(source.as_dict(), ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )

    @staticmethod
    def write_license_audit(source: _ResolvedModelSource, output_path: _Path) -> None:
        """Record the exact source restrictions and deployment decision."""
        restriction = source.intended_use.lower()
        restricted = "research" in restriction or "education" in restriction
        lines = [
            "# BamiBERT license and intended-use audit",
            "",
            "## Resolved immutable source",
            "",
            f"- Repository: `{source.repo_id}`",
            f"- Revision: `{source.revision}`",
            f"- Model-card license metadata: `{source.license_id}`",
            f"- Downloaded `README.md` SHA-256: `{source.model_card_checksum}`",
        ]
        if source.transformers_requirement:
            lines.append(
                "- Model-card transformer compatibility: "
                f"`transformers{source.transformers_requirement}`"
            )
        lines.extend(["", "## License terms from the model card (verbatim)", ""])
        if source.license_text:
            lines.extend(
                f"> {line}" if line else ">" for line in source.license_text.splitlines()
            )
        else:
            lines.extend(
                [
                    "> No dedicated `License` section was found; the model-card license metadata",
                    "> above is retained.",
                ]
            )
        if source.license_metadata:
            lines.extend(
                [
                    "",
                    "## License metadata from the pinned model card",
                    "",
                    "```json",
                    _json.dumps(
                        source.license_metadata,
                        ensure_ascii=False,
                        indent=2,
                        sort_keys=True,
                    ),
                    "```",
                ]
            )
        if source.file_checksums:
            lines.extend(
                [
                    "",
                    "## Hub LFS file checksums",
                    "",
                    "| File | SHA-256 |",
                    "| --- | --- |",
                ]
            )
            lines.extend(
                f"| `{filename}` | `{checksum}` |"
                for filename, checksum in source.file_checksums.items()
            )
        lines.extend(["", "## Intended use from the model card (verbatim)", ""])
        lines.extend(
            f"> {line}" if line else ">" for line in source.intended_use.splitlines()
        )
        lines.extend(["", "## Deployment decision", ""])
        if restricted:
            lines.append(
                "**Non-deployable candidate:** the quoted intended-use restriction requires "
                "explicit, documented approval of the exact terms before any deployment decision."
            )
        else:
            lines.append(
                "No research/educational-only restriction was detected automatically; "
                "legal review is still required."
            )
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
