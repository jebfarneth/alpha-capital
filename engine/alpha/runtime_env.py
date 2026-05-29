"""Runtime environment loading helpers.

Engine entrypoints can be launched from cron, shells, or repo-root scripts.
Do not make provider credentials depend on the current working directory.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Iterable


def default_env_paths() -> list[Path]:
    """Return deterministic dotenv search paths for engine runtimes."""

    engine_dir = Path(__file__).resolve().parents[1]
    repo_root = engine_dir.parent
    return [engine_dir / ".env", repo_root / ".env"]


def load_runtime_env(paths: Iterable[Path | str] | None = None) -> list[Path]:
    """Load runtime env files without overriding existing process values.

    Existing ``os.environ`` values remain authoritative. By default this loads
    ``engine/.env`` and then a repo-root ``.env`` when present, regardless of
    the caller's current working directory. Returns the files that were read.
    """

    loaded: list[Path] = []
    for raw_path in paths if paths is not None else default_env_paths():
        path = Path(raw_path)
        if not path.exists():
            continue
        with path.open(encoding="utf-8") as handle:
            for raw_line in handle:
                line = raw_line.strip()
                if not line or line.startswith("#") or "=" not in line:
                    continue
                key, value = line.split("=", 1)
                os.environ.setdefault(key, _clean_env_value(value))
        loaded.append(path)
    return loaded


def _clean_env_value(value: str) -> str:
    value = value.strip()
    if len(value) >= 2 and value[0] == value[-1] and value[0] in {"'", '"'}:
        return value[1:-1]
    return value
