"""Unit tests for the train-and-serve orchestrator (all training mocked)."""

from pathlib import Path
from types import SimpleNamespace

from scripts import release_train_and_serve as train_and_serve


def test_cuda_available_returns_bool() -> None:
    assert isinstance(train_and_serve.cuda_available(), bool)


def test_cuda_available_is_false_without_torch(monkeypatch) -> None:
    import builtins

    real_import = builtins.__import__

    def _no_torch(name: str, *args: object, **kwargs: object):
        if name == "torch":
            raise ImportError("No module named 'torch'")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", _no_torch)
    assert train_and_serve.cuda_available() is False


def test_skips_training_when_artifact_is_fresh(tmp_path: Path, monkeypatch) -> None:
    data = tmp_path / "dataset.csv"
    data.write_text("id,raw_text\n", encoding="utf-8")
    artifact_dir = tmp_path / "artifact"
    artifact_dir.mkdir()
    checksum = f"sha256:{train_and_serve.sha256_file(data)}"
    monkeypatch.setattr(
        train_and_serve.ArtifactRegistry,
        "verify",
        lambda self, directory: SimpleNamespace(data_checksum=checksum),
    )

    def _forbidden(*args: object, **kwargs: object) -> int:
        raise AssertionError("training must be skipped for a fresh artifact")

    monkeypatch.setattr(train_and_serve.data_validate, "main", _forbidden)
    assert train_and_serve.main(["--data", str(data), "--artifact-dir", str(artifact_dir)]) == 0


def test_stale_artifact_runs_baseline_only_pipeline(tmp_path: Path, monkeypatch) -> None:
    data = tmp_path / "dataset.csv"
    data.write_text("id,raw_text\n", encoding="utf-8")
    work_dir = tmp_path / "work"
    artifact_dir = tmp_path / "artifact"
    calls: list[str] = []

    def _recorder(name: str):
        def _fake(argv: object) -> int:
            calls.append(name)
            return 0

        return _fake

    monkeypatch.setattr(
        train_and_serve.ArtifactRegistry,
        "verify",
        lambda self, d: (_ for _ in ()).throw(ValueError("no artifact")),
    )
    monkeypatch.setattr(train_and_serve.data_validate, "main", _recorder("validate"))
    monkeypatch.setattr(train_and_serve.data_split, "main", _recorder("split"))
    monkeypatch.setattr(train_and_serve.train_baseline, "main", _recorder("train"))
    monkeypatch.setattr(train_and_serve.eval_predictions, "main", _recorder("eval"))
    monkeypatch.setattr(train_and_serve.select_champion, "main", _recorder("champion"))
    monkeypatch.setattr(train_and_serve.verify_artifact, "main", _recorder("verify"))

    result = train_and_serve.main(
        [
            "--data",
            str(data),
            "--splits",
            str(work_dir / "split.json"),
            "--work-dir",
            str(work_dir),
            "--report-dir",
            str(tmp_path / "reports"),
            "--artifact-dir",
            str(artifact_dir),
            "--transformer-mode",
            "never",
        ]
    )
    assert result == 0
    # validate runs twice (data, then split), gates skipped without transformer,
    # frozen-test + verify always run last.
    assert calls == [
        "validate",
        "split",
        "train",
        "eval",
        "champion",
        "verify",
    ]


def _record_pipeline(tmp_path: Path, monkeypatch, fetch_result: str | None) -> list[str]:
    """Run the pipeline with every trainer mocked; return the recorded steps."""
    data = tmp_path / "dataset.csv"
    data.write_text("id,raw_text\n", encoding="utf-8")
    work_dir = tmp_path / "work"
    calls: list[str] = []

    def _recorder(name: str):
        def _fake(argv: object) -> int:
            step = argv[0] if isinstance(argv, list) and argv else ""
            calls.append(f"{name}:{step}")
            return 0

        return _fake

    monkeypatch.setattr(train_and_serve, "cuda_available", lambda: True)
    monkeypatch.setattr(
        train_and_serve, "fetch_transformer_artifact", lambda *_a, **_k: fetch_result
    )
    monkeypatch.setattr(
        train_and_serve.ArtifactRegistry,
        "verify",
        lambda self, d: (_ for _ in ()).throw(ValueError("no artifact")),
    )
    monkeypatch.setattr(train_and_serve.data_validate, "main", _recorder("validate"))
    monkeypatch.setattr(train_and_serve.data_split, "main", _recorder("split"))
    monkeypatch.setattr(train_and_serve.train_baseline, "main", _recorder("train"))
    monkeypatch.setattr(train_and_serve.eval_predictions, "main", _recorder("eval"))
    monkeypatch.setattr(train_and_serve.train_transformer, "main", _recorder("train-transformer"))
    monkeypatch.setattr(train_and_serve.select_champion, "main", _recorder("champion"))
    monkeypatch.setattr(train_and_serve.verify_artifact, "main", _recorder("verify"))
    assert (
        train_and_serve.main(
            [
                "--data",
                str(data),
                "--splits",
                str(work_dir / "split.json"),
                "--work-dir",
                str(work_dir),
                "--report-dir",
                str(tmp_path / "reports"),
                "--artifact-dir",
                str(tmp_path / "artifact"),
                "--transformer-artifact-dir",
                str(tmp_path / "tr-champion"),
            ]
        )
        == 0
    )
    return calls


def test_hub_champion_skips_bamibert_training(tmp_path: Path, monkeypatch) -> None:
    """A dataset-matching Hub champion replaces BamiBERT CV training, even with CUDA."""
    calls = _record_pipeline(tmp_path, monkeypatch, fetch_result="hub")
    assert not any(call.startswith("train-transformer:") for call in calls)
    assert "champion:gates" not in calls, "gates need CV evidence; skip when fetched"
    assert "champion:frozen-test" in calls
    assert "verify:verify" in calls


def test_cuda_trains_bamibert_when_hub_fetch_fails(tmp_path: Path, monkeypatch) -> None:
    """A failed Hub fetch falls back to full BamiBERT CV training on CUDA."""
    calls = _record_pipeline(tmp_path, monkeypatch, fetch_result=None)
    assert any(call.startswith("train-transformer:") for call in calls)
    assert "champion:gates" in calls
    assert "champion:frozen-test" in calls


def test_transformer_fetch_needed_matrix(tmp_path: Path) -> None:
    missing = tmp_path / "missing"
    empty = tmp_path / "empty"
    empty.mkdir()
    full = tmp_path / "full"
    full.mkdir()
    (full / "x").write_text("x", encoding="utf-8")
    assert train_and_serve.transformer_fetch_needed(missing, "never") is False
    assert train_and_serve.transformer_fetch_needed(missing, "always") is True
    assert train_and_serve.transformer_fetch_needed(missing, "auto") is True
    assert train_and_serve.transformer_fetch_needed(empty, "auto") is True
    assert train_and_serve.transformer_fetch_needed(full, "auto") is False


def test_fetch_returns_local_without_network(tmp_path: Path, monkeypatch) -> None:
    artifact_dir = tmp_path / "tr"
    artifact_dir.mkdir()
    monkeypatch.setattr(train_and_serve, "artifact_matches_dataset", lambda _d, _p: True)

    def _forbidden(*args: object, **kwargs: object):
        raise AssertionError("no download needed for a fresh artifact")

    monkeypatch.setitem(__import__("sys").modules, "huggingface_hub", _forbidden)
    assert (
        train_and_serve.fetch_transformer_artifact(
            artifact_dir, tmp_path / "data.csv", "org/repo", "abc123"
        )
        == "local"
    )


def test_fetch_degrades_without_huggingface_hub(tmp_path: Path, monkeypatch) -> None:
    import builtins

    artifact_dir = tmp_path / "tr"
    real_import = builtins.__import__

    def _no_hub(name: str, *args: object, **kwargs: object):
        if name == "huggingface_hub":
            raise ImportError("No module named 'huggingface_hub'")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", _no_hub)
    assert (
        train_and_serve.fetch_transformer_artifact(
            artifact_dir, tmp_path / "data.csv", "org/repo", "abc123"
        )
        is None
    )


def test_fetch_sanitizes_hub_download_extras(tmp_path: Path, monkeypatch) -> None:
    """Hub fetch drops resume metadata (.cache/) + .gitattributes into the dir.

    Neither is payload: leaving them behind breaks the payload checksum the
    registry verifies on every boot, which would wipe the artifact and loop.
    """
    import sys

    artifact_dir = tmp_path / "tr"
    data = tmp_path / "data.csv"
    data.write_text("id,raw_text\n", encoding="utf-8")

    def _fake_snapshot_download(*args: object, **kwargs: object) -> str:
        local_dir = Path(str(kwargs["local_dir"]))
        (local_dir / "manifest.json").write_text("{}", encoding="utf-8")
        (local_dir / "model.safetensors").write_bytes(b"payload")
        (local_dir / ".gitattributes").write_text("git-lfs metadata", encoding="utf-8")
        resume = local_dir / ".cache" / "huggingface" / "download"
        resume.mkdir(parents=True)
        (resume / "model.safetensors.incomplete").write_bytes(b"partial")
        return str(local_dir)

    monkeypatch.setitem(
        sys.modules, "huggingface_hub", SimpleNamespace(snapshot_download=_fake_snapshot_download)
    )
    manifest = SimpleNamespace(data_checksum=f"sha256:{train_and_serve.sha256_file(data)}")

    def _verify(self: object, directory: Path) -> SimpleNamespace:
        payload = Path(directory)
        has_payload = (payload / "manifest.json").exists() and (
            payload / "model.safetensors"
        ).exists()
        leftover = (payload / ".cache").exists() or (payload / ".gitattributes").exists()
        if leftover or not has_payload:
            raise ValueError("payload checksum mismatch (empty dir or fetch metadata present)")
        return manifest

    monkeypatch.setattr(train_and_serve.ArtifactRegistry, "verify", _verify)

    assert (
        train_and_serve.fetch_transformer_artifact(artifact_dir, data, "org/repo", "abc123")
        == "hub"
    )
    assert not (artifact_dir / ".gitattributes").exists()
    assert not (artifact_dir / ".cache").exists()


def test_strip_hub_fetch_extras_cleans_stale_metadata(tmp_path: Path) -> None:
    """A pre-existing artifact dir carrying old fetch metadata is sanitized too."""
    artifact_dir = tmp_path / "tr"
    artifact_dir.mkdir()
    stale_cache = artifact_dir / ".cache" / "huggingface" / "download"
    stale_cache.mkdir(parents=True)
    (stale_cache / "old.incomplete").write_bytes(b"partial")
    (artifact_dir / ".gitattributes").write_text("git-lfs metadata", encoding="utf-8")

    train_and_serve._strip_hub_fetch_extras(artifact_dir)

    assert not (artifact_dir / ".cache").exists()
    assert not (artifact_dir / ".gitattributes").exists()
    assert artifact_dir.is_dir()


def test_transformer_evidence_fresh_matrix(tmp_path: Path) -> None:
    import csv as csv_module
    import json as json_module

    run = tmp_path / "tr"
    challenge = tmp_path / "challenge.csv"
    with challenge.open("w", newline="", encoding="utf-8") as handle:
        writer = csv_module.writer(handle)
        writer.writerow(["id"])
        writer.writerows([["c1"], ["c2"]])
    report = tmp_path / "report.json"

    assert train_and_serve.transformer_evidence_fresh(run, report, challenge) is False
    run.mkdir()
    (run / "cv_predictions.csv").write_text("x", encoding="utf-8")
    assert train_and_serve.transformer_evidence_fresh(run, report, challenge) is False
    report.write_text(json_module.dumps({"challenge": {"count": 2}}), encoding="utf-8")
    assert train_and_serve.transformer_evidence_fresh(run, report, challenge) is True
    report.write_text(json_module.dumps({"challenge": {"count": 99}}), encoding="utf-8")
    assert train_and_serve.transformer_evidence_fresh(run, report, challenge) is False


def test_fetch_script_reports_source_and_status(tmp_path: Path, monkeypatch) -> None:
    from scripts import release_fetch_transformer as fetch_script

    monkeypatch.setattr(fetch_script, "fetch_transformer_artifact", lambda _d, _p, _r, _v: "hub")
    assert (
        fetch_script.main(["--data", str(tmp_path / "d.csv"), "--artifact-dir", str(tmp_path)]) == 0
    )
    monkeypatch.setattr(fetch_script, "fetch_transformer_artifact", lambda _d, _p, _r, _v: None)
    assert (
        fetch_script.main(["--data", str(tmp_path / "d.csv"), "--artifact-dir", str(tmp_path)]) == 1
    )
