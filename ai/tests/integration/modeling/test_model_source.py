from pathlib import Path
from types import SimpleNamespace

import pytest

from src.domain.artifacts import ResolvedModelSource
from src.nlp.modeling.model_source import ModelSourceResolver


def test_resolved_source_requires_commit_sha() -> None:
    source = ResolvedModelSource(
        repo_id="Qualcomm-AI-Research/BamiBERT",
        revision="a" * 40,
        license_id="bsd-3-clause-clear",
        intended_use="research and educational purposes",
    )

    assert len(source.revision) == 40


def test_resolved_source_rejects_non_commit_revision() -> None:
    with pytest.raises(ValueError, match="40-character commit SHA"):
        ResolvedModelSource(
            repo_id="Qualcomm-AI-Research/BamiBERT",
            revision="main",
            license_id="bsd-3-clause-clear",
            intended_use="research and educational purposes",
        )


def test_source_resolver_pins_hub_sha_and_records_model_card_terms(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    sha = "b" * 40
    calls: list[tuple[str, bool]] = []

    class FakeApi:
        def model_info(self, repo_id: str, *, files_metadata: bool) -> SimpleNamespace:
            calls.append((repo_id, files_metadata))
            return SimpleNamespace(
                sha=sha,
                card_data={"license": "bsd-3-clause-clear"},
                siblings=[
                    SimpleNamespace(
                        rfilename="model.safetensors",
                        lfs=SimpleNamespace(sha256="d" * 64),
                    )
                ],
            )

    card_path = tmp_path / "README.md"
    card_path.write_text(
        "# BamiBERT\n\n"
        "## Intended Use\n\n"
        "For research and educational purposes only.\n\n"
        "## License\n\n"
        "BSD 3-Clause Clear plus Qualcomm Responsible AI License.\n",
        encoding="utf-8",
    )
    download_calls: list[tuple[str, str, str]] = []

    def fake_download(*, repo_id: str, filename: str, revision: str) -> str:
        download_calls.append((repo_id, filename, revision))
        return str(card_path)

    resolver = ModelSourceResolver(api_factory=FakeApi, download=fake_download)
    source = resolver.resolve("Qualcomm-AI-Research/BamiBERT")
    output_path = tmp_path / "source.json"
    resolver.write_source_metadata(source, output_path)

    assert calls == [("Qualcomm-AI-Research/BamiBERT", True)]
    assert download_calls == [("Qualcomm-AI-Research/BamiBERT", "README.md", sha)]
    assert source.revision == sha
    assert source.license_id == "bsd-3-clause-clear"
    assert source.intended_use == "For research and educational purposes only."
    assert source.license_text == "BSD 3-Clause Clear plus Qualcomm Responsible AI License."
    assert source.file_checksums == {"model.safetensors": "d" * 64}
    assert len(source.model_card_checksum) == 64
    assert sha in output_path.read_text(encoding="utf-8")


def test_source_resolver_reads_card_front_matter_and_real_section_names(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    sha = "c" * 40

    class FakeApi:
        def model_info(self, repo_id: str, *, files_metadata: bool) -> SimpleNamespace:
            return SimpleNamespace(sha=sha, card_data={})

    card_path = tmp_path / "README.md"
    card_path.write_text(
        "\n---\n"
        "license:\n"
        "- bsd-3-clause-clear\n"
        "- other\n"
        "license_name: qualcomm-responsible-ai-license\n"
        "---\n\n"
        "# BamiBERT\n\n"
        "## Model Loading\n"
        "Using transformers<=5.5.0\n\n"
        "## License/Terms of Use\n"
        "This model is released under BSD 3-Clause Clear and Qualcomm Responsible AI License.\n\n"
        "## Uses\n"
        "The model is intended for research and educational purposes.\n",
        encoding="utf-8",
    )

    resolver = ModelSourceResolver(
        api_factory=FakeApi,
        download=lambda *, repo_id, filename, revision: str(card_path),
    )
    source = resolver.resolve("Qualcomm-AI-Research/BamiBERT")

    assert source.license_id == "bsd-3-clause-clear, other"
    assert source.license_metadata == {
        "license": ["bsd-3-clause-clear", "other"],
        "license_name": "qualcomm-responsible-ai-license",
    }
    assert source.license_text == (
        "This model is released under BSD 3-Clause Clear and Qualcomm Responsible AI License."
    )
    assert source.intended_use == "The model is intended for research and educational purposes."
    assert source.transformers_requirement == "<=5.5.0"
