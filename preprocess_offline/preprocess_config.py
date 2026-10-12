from __future__ import annotations
from pathlib import Path
from typing import Any, Collection
import yaml


def load_config(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as file:
        config = yaml.safe_load(file) or {}
    if not isinstance(config, dict):
        raise ValueError(f"configuration root must be a mapping：{path}")
    return config


def _positive_int(value: Any, field: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError(f"{field} must be a positive integer，actual {value!r}")
    return value


def load_classification_fabric_ids(config_path: Path) -> list[int]:
    values = load_config(config_path).get("fabric_ids")
    if not isinstance(values, list) or not values:
        raise ValueError(f"fabric_ids must be a non-empty list：{config_path}")
    fabric_ids = [
        _positive_int(value, f"fabric_ids[{index}]")
        for index, value in enumerate(values)
    ]
    if len(set(fabric_ids)) != len(fabric_ids):
        raise ValueError(f"fabric_ids contains duplicate values：{config_path}")
    return fabric_ids


def load_interaction_fabric_tasks(
    config_path: Path,
    valid_tasks: Collection[str],
) -> dict[int, str]:
    entries = load_config(config_path).get("tasks")
    if not isinstance(entries, list) or not entries:
        raise ValueError(f"tasks must be a non-empty list：{config_path}")

    valid = set(valid_tasks)
    mapping: dict[int, str] = {}
    names: list[str] = []
    for index, entry in enumerate(entries):
        if not isinstance(entry, dict):
            raise ValueError(f"tasks[{index}] must be a mapping")
        name = str(entry.get("name", ""))
        if name not in valid:
            raise ValueError(f"tasks[{index}].name is unsupported：{name!r}")
        target_label = str(entry.get("target_label", ""))
        if target_label != name:
            raise ValueError(
                f"tasks[{index}].target_label must equal its task name：{name!r}"
            )
        fabric_id = _positive_int(
            entry.get("fabric_id"), f"tasks[{index}].fabric_id"
        )
        if fabric_id in mapping:
            raise ValueError(f"duplicate interaction fabric_id：{fabric_id}")
        if name in names:
            raise ValueError(f"duplicate interaction task：{name}")
        mapping[fabric_id] = name
        names.append(name)

    configured = set(names)
    if configured != valid:
        raise ValueError(
            f"interaction tasks must be {sorted(valid)}，actual {sorted(configured)}"
        )
    return mapping
