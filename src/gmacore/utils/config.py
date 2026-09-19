"""YAML configuration loading with command line overrides."""

from __future__ import annotations

import copy
from typing import Any, Dict, List

import yaml


def load_yaml(path: str) -> Dict[str, Any]:
    """Read a YAML configuration file into a dictionary."""
    with open(path, encoding="utf-8") as handle:
        return yaml.safe_load(handle) or {}


def _coerce(value: str) -> Any:
    """Convert a command line string into an int, float, bool, None or string."""
    lowered = value.lower()
    if lowered in {"true", "false"}:
        return lowered == "true"
    if lowered in {"none", "null"}:
        return None
    for caster in (int, float):
        try:
            return caster(value)
        except ValueError:
            continue
    return value


def apply_overrides(config: Dict[str, Any], overrides: List[str]) -> Dict[str, Any]:
    """Apply `section.key=value` overrides to a nested configuration.

    Overrides let a single YAML file cover a sweep, for example
    `--set pretrain.epochs=5 framework.decay=0.999`.
    """
    result = copy.deepcopy(config)

    for override in overrides:
        if "=" not in override:
            raise ValueError(f"Override '{override}' is not of the form key=value")

        key_path, raw_value = override.split("=", 1)
        keys = key_path.split(".")

        cursor = result
        for key in keys[:-1]:
            cursor = cursor.setdefault(key, {})
        cursor[keys[-1]] = _coerce(raw_value)

    return result


def merge(base: Dict[str, Any], updates: Dict[str, Any]) -> Dict[str, Any]:
    """Recursively merge `updates` into `base`, returning a new dictionary."""
    result = copy.deepcopy(base)
    for key, value in updates.items():
        if isinstance(value, dict) and isinstance(result.get(key), dict):
            result[key] = merge(result[key], value)
        else:
            result[key] = value
    return result
