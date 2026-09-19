import csv as _csv
import json as _json
from pathlib import Path as _Path

from src.domain.entities import DatasetRow as _DatasetRow
from src.nlp.constants import _CSV_FIELDS


class DatasetStore:
    """Read and write typed dataset rows at paths selected by the caller."""

    def write(self, rows: list[_DatasetRow], path: _Path) -> None:
        """Write UTF-8 CSV; serialize noise types as a JSON array."""
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8", newline="") as file:
            writer = _csv.DictWriter(file, fieldnames=_CSV_FIELDS, lineterminator="\n")
            writer.writeheader()
            for row in rows:
                payload = row.model_dump(mode="json")
                payload["noise_types"] = _json.dumps(payload["noise_types"], ensure_ascii=False)
                writer.writerow(payload)

    def read(self, path: _Path) -> list[_DatasetRow]:
        """Read UTF-8 CSV and restore noise types before Pydantic validation."""
        with path.open(encoding="utf-8", newline="") as file:
            reader = _csv.DictReader(file)
            rows: list[_DatasetRow] = []
            for payload in reader:
                payload["noise_types"] = _json.loads(payload["noise_types"])
                rows.append(_DatasetRow.model_validate(payload))
        return rows
