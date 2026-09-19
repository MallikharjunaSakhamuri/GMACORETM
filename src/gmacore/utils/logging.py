"""Logging configuration and training metric history."""

from __future__ import annotations

import json
import logging
import os
from typing import Dict, List, Optional


def configure_logging(level: int = logging.INFO, log_file: Optional[str] = None) -> None:
    """Configure root logging for the command line entry points."""
    handlers: List[logging.Handler] = [logging.StreamHandler()]
    if log_file is not None:
        os.makedirs(os.path.dirname(log_file) or ".", exist_ok=True)
        handlers.append(logging.FileHandler(log_file))

    logging.basicConfig(
        level=level,
        format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=handlers,
        force=True,
    )


class MetricHistory:
    """Per-epoch record of training metrics, serializable to JSON."""

    def __init__(self) -> None:
        self.records: List[Dict[str, float]] = []

    def append(self, epoch: int, metrics: Dict[str, float]) -> None:
        record = {"epoch": epoch}
        record.update({key: float(value) for key, value in metrics.items()})
        self.records.append(record)

    def last(self) -> Dict[str, float]:
        return self.records[-1] if self.records else {}

    def series(self, key: str) -> List[float]:
        return [record[key] for record in self.records if key in record]

    def to_json(self, path: str) -> None:
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        with open(path, "w", encoding="utf-8") as handle:
            json.dump(self.records, handle, indent=2)

    @classmethod
    def from_json(cls, path: str) -> MetricHistory:
        history = cls()
        with open(path, encoding="utf-8") as handle:
            history.records = json.load(handle)
        return history
