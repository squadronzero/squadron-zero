"""Load config/settings.yaml and resolve paths relative to the repo root."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_SETTINGS = REPO_ROOT / "config" / "settings.yaml"


@dataclass
class Settings:
    raw: dict[str, Any]
    root: Path

    def __getitem__(self, key: str) -> Any:
        return self.raw[key]

    @property
    def dataset(self) -> str:
        return str(self.raw.get("dataset", "global"))

    def path(self, value: str) -> Path:
        p = Path(value.format(dataset=self.dataset))
        return p if p.is_absolute() else self.root / p

    @property
    def raw_dir(self) -> Path:
        return self.path(self.raw["inputs"]["raw_dir"])

    @property
    def processed_dir(self) -> Path:
        return self.path(self.raw["outputs"]["processed_dir"])

    @property
    def detections_dir(self) -> Path:
        return self.processed_dir / self.raw["outputs"]["detections_subdir"]

    @property
    def celldays_dir(self) -> Path:
        return self.processed_dir / self.raw["outputs"]["celldays_subdir"]

    @property
    def resolution(self) -> float:
        return float(self.raw["grid"]["resolution_deg"])


def load_settings(path: str | Path | None = None, root: str | Path | None = None,
                  dataset: str | None = None) -> Settings:
    path = Path(path) if path else DEFAULT_SETTINGS
    with open(path, encoding="utf-8") as fh:
        raw = yaml.safe_load(fh)
    if dataset:
        raw["dataset"] = dataset
    return Settings(raw=raw, root=Path(root) if root else REPO_ROOT)
