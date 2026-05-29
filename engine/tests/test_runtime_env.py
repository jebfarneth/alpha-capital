"""Runtime environment loading tests."""

from __future__ import annotations

import os
from pathlib import Path

from alpha.runtime_env import default_env_paths, load_runtime_env


def test_default_env_paths_are_independent_of_cwd(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)

    paths = default_env_paths()

    assert paths[0].name == ".env"
    assert paths[0].parent.name == "engine"
    assert paths[1] == paths[0].parent.parent / ".env"


def test_load_runtime_env_reads_env_file_without_overriding_process_env(
    monkeypatch, tmp_path
):
    env_file = tmp_path / ".env"
    env_file.write_text(
        "\n".join(
            [
                "# comment",
                "FMP_API_KEY=file-fmp-key",
                "POLYGON_API_KEY='file-polygon-key'",
                'BENZINGA_API_KEY="file-benzinga-key"',
                "IGNORED_LINE_WITHOUT_EQUALS",
            ]
        ),
        encoding="utf-8",
    )
    monkeypatch.setenv("FMP_API_KEY", "process-fmp-key")
    monkeypatch.delenv("POLYGON_API_KEY", raising=False)
    monkeypatch.delenv("BENZINGA_API_KEY", raising=False)

    loaded = load_runtime_env([env_file])

    assert loaded == [env_file]
    assert os.environ["FMP_API_KEY"] == "process-fmp-key"
    assert os.environ["POLYGON_API_KEY"] == "file-polygon-key"
    assert os.environ["BENZINGA_API_KEY"] == "file-benzinga-key"


def test_load_runtime_env_skips_missing_files(monkeypatch, tmp_path):
    missing = tmp_path / "missing.env"
    monkeypatch.delenv("FMP_API_KEY", raising=False)

    loaded = load_runtime_env([missing])

    assert loaded == []
    assert "FMP_API_KEY" not in os.environ
