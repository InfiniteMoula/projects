
import json
from pathlib import Path

from utils import io


def test_atomic_write(tmp_path):
    path = tmp_path / "data.txt"
    io.atomic_write(path, b"hello")
    assert path.read_text(encoding="utf-8") == "hello"


def test_atomic_write_iter(tmp_path):
    path = tmp_path / "data.bin"

    def chunks():
        for part in (b"a", b"b", b"c"):
            yield part

    io.atomic_write_iter(path, chunks())
    assert path.read_bytes() == b"abc"


def test_log_json_appends(tmp_path):
    path = tmp_path / "log.jsonl"
    io.log_json(path, {"a": 1})
    io.log_json(path, {"b": 2})
    lines = path.read_text(encoding="utf-8").strip().splitlines()
    assert json.loads(lines[0]) == {"a": 1}
    assert json.loads(lines[1]) == {"b": 2}
