"""Provenance stamp written into every stage report and into the exported data (must-fix 8):
which code (git commit, and whether code or settings had uncommitted changes), which settings
(SHA-256 of the settings as loaded), which software, when."""
from __future__ import annotations

import hashlib
import platform
import subprocess
from datetime import datetime, timezone

import yaml

from .config import REPO_ROOT


def _git(*args: str) -> str | None:
    try:
        return subprocess.run(["git", "-C", str(REPO_ROOT), *args], capture_output=True, text=True,
                              timeout=20, check=True).stdout.strip()
    except Exception:  # noqa: BLE001 - no git or not a repo: recorded as unknown
        return None


def settings_hash(raw: dict) -> str:
    text = yaml.safe_dump(raw, sort_keys=True, allow_unicode=True)
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def stamp(settings) -> dict:
    import numpy
    import pandas
    commit = _git("rev-parse", "HEAD")
    dirty = _git("status", "--porcelain", "--", "pipeline", "config")
    return {"git_commit": commit, "code_or_settings_changed_since_commit": None if dirty is None else bool(dirty),
            "settings_sha256": settings_hash(settings.raw), "dataset": settings.dataset,
            "python": platform.python_version(), "numpy": numpy.__version__, "pandas": pandas.__version__,
            "run_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")}
