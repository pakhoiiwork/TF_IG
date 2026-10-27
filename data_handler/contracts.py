from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class DatasetSpec:
    name: str
    data_path: str


@dataclass(frozen=True)
class EvaluationSpec:
    protocol: str = "intra_dataset"
    test_dataset_name: str | None = None
    test_data_path: str | None = None


def record_subject_id(record: dict[str, Any]) -> str:
    return str(record["id"])
