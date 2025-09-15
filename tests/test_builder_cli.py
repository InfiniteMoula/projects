
import argparse
from pathlib import Path

import pytest

import builder_cli


def _namespace(tmp_path, **overrides):
    defaults = dict(
        job=str(tmp_path / "job.yml"),
        out=str(tmp_path / "out"),
        input=None,
        run_id=None,
        dry_run=False,
        sample=0,
        time_budget_min=0,
        workers=2,
        json=False,
        resume=False,
        verbose=False,
        max_ram_mb=0,
    )
    defaults.update(overrides)
    return argparse.Namespace(**defaults)


def test_build_context_creates_dirs(tmp_path, monkeypatch):
    input_file = tmp_path / "input.parquet"
    input_file.write_bytes(b"")
    args = _namespace(tmp_path, input=str(input_file))
    monkeypatch.setattr(builder_cli.config, "load_env", lambda: {"ENV": "1"})
    ctx = builder_cli.build_context(args, {"output": {"lang": "fr"}})
    assert Path(ctx["outdir"]).exists()
    assert Path(ctx["logs"]).parent.exists()
    assert ctx["input"] == str(input_file.resolve())
    assert ctx["env"] == {"ENV": "1"}


def test_build_context_missing_input(tmp_path, monkeypatch):
    args = _namespace(tmp_path, input=str(tmp_path / "missing.parquet"))
    monkeypatch.setattr(builder_cli.config, "load_env", lambda: {})
    with pytest.raises(FileNotFoundError):
        builder_cli.build_context(args, {})
