from pathlib import Path

import pytest

from src.core.settings import Settings


def test_settings_default_to_baseline_and_64_kib() -> None:
    settings = Settings()
    assert settings.model_backend == "baseline"
    assert settings.max_request_bytes == 65_536


def test_settings_read_mat_prefixed_environment(monkeypatch) -> None:
    monkeypatch.setenv("MAT_MODEL_BACKEND", "transformer")
    monkeypatch.setenv("MAT_MAX_REQUEST_BYTES", "1024")

    settings = Settings()

    assert settings.model_backend == "transformer"
    assert settings.max_request_bytes == 1024


SERVICE_ROOT = Path(__file__).resolve().parents[2]


def test_catalog_and_lock_settings_resolve_from_service_root_regardless_of_cwd(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Catalog/lock settings stay anchored to the service root from any cwd.

    Defaults are service-root-relative literals, never cwd-captured absolute
    paths, so relocating the working directory cannot bake a stale absolute
    location into the settings. The operator anchors a deployment elsewhere
    with absolute service-root paths through the ``MAT_`` environment (the
    Docker Compose contract), and those must resolve to the real service
    root files no matter which directory the process was started in.
    """
    monkeypatch.chdir(tmp_path)  # any directory that is not the service root

    defaults = Settings()
    assert defaults.normalization_catalog_path == Path("configs/normalization.yaml")
    assert defaults.normalization_lock_path == Path("configs/normalization.lock.json")
    assert not defaults.normalization_catalog_path.is_absolute()
    assert not defaults.normalization_lock_path.is_absolute()

    monkeypatch.setenv(
        "MAT_NORMALIZATION_CATALOG_PATH",
        str(SERVICE_ROOT / "configs" / "normalization.yaml"),
    )
    monkeypatch.setenv(
        "MAT_NORMALIZATION_LOCK_PATH",
        str(SERVICE_ROOT / "configs" / "normalization.lock.json"),
    )
    anchored = Settings()
    assert anchored.normalization_catalog_path == (
        SERVICE_ROOT / "configs" / "normalization.yaml"
    )
    assert anchored.normalization_lock_path == (
        SERVICE_ROOT / "configs" / "normalization.lock.json"
    )
    assert anchored.normalization_catalog_path.is_file()
    assert anchored.normalization_lock_path.is_file()
