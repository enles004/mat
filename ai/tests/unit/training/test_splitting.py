from hashlib import sha256
from pathlib import Path

from scripts import data_split
from src.domain.training import SplitManifest
from src.nlp.training.dataset import DatasetStore
from src.nlp.training.splitting import DatasetSplitter


def test_checked_in_split_manifest_matches_deterministic_rebuild() -> None:
    """The committed freeze point must equal a fresh deterministic rebuild."""
    dataset_path = Path("data/dataset.csv")
    rows = DatasetStore().read(dataset_path)
    frozen = SplitManifest.model_validate_json(
        Path("data/split_manifest.json").read_text(encoding="utf-8")
    )

    assert DatasetSplitter().build(rows, dataset_path, seed=42) == frozen


def test_split_manifest_is_group_disjoint_and_deterministic() -> None:
    """A changed group split must be caught before model evaluation uses it."""
    dataset_path = Path("data/dataset.csv")
    rows = DatasetStore().read(dataset_path)
    first = DatasetSplitter().build(rows, dataset_path, seed=42)
    second = DatasetSplitter().build(rows, dataset_path, seed=42)
    assert first == second

    groups = {row.id: row.canonical_group_id for row in rows}
    test_groups = {groups[row_id] for row_id in first.test_ids}
    development_groups = {groups[row_id] for row_id in first.development_ids}
    assert test_groups.isdisjoint(development_groups)

    for fold in first.folds:
        train_groups = {groups[row_id] for row_id in fold.train_ids}
        validation_groups = {groups[row_id] for row_id in fold.validation_ids}
        assert train_groups.isdisjoint(validation_groups)


def test_split_manifest_covers_dataset_with_sorted_ids_and_byte_checksum() -> None:
    """Changing the source CSV or omitting a row must invalidate the freeze point."""
    dataset_path = Path("data/dataset.csv")
    rows = DatasetStore().read(dataset_path)

    manifest = DatasetSplitter().build(rows, dataset_path, seed=42)

    assert manifest.dataset_checksum == sha256(dataset_path.read_bytes()).hexdigest()
    assert manifest.test_ids == sorted(manifest.test_ids)
    assert manifest.development_ids == sorted(manifest.development_ids)
    assert len(manifest.test_ids) == 180
    assert len(manifest.development_ids) == 720
    assert set(manifest.test_ids).isdisjoint(manifest.development_ids)
    assert set(manifest.test_ids) | set(manifest.development_ids) == {row.id for row in rows}
    assert [fold.fold for fold in manifest.folds] == [0, 1, 2, 3, 4]
    for fold in manifest.folds:
        assert fold.train_ids == sorted(fold.train_ids)
        assert fold.validation_ids == sorted(fold.validation_ids)
        assert set(fold.train_ids).isdisjoint(fold.validation_ids)
        assert set(fold.train_ids) | set(fold.validation_ids) == set(manifest.development_ids)


def test_split_script_parses_arguments_and_writes_requested_manifest(tmp_path: Path) -> None:
    output = tmp_path / "split.json"

    result = data_split.main(
        ["--input", "data/dataset.csv", "--output", str(output), "--seed", "42"]
    )

    assert result == 0
    assert SplitManifest.model_validate_json(output.read_text(encoding="utf-8")).seed == 42
