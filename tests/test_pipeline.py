
import json
import logging

import pytest

from utils import pipeline


def test_resolve_target_json_loads():
    func = pipeline.resolve_target("json:loads")
    assert func("[]") == []


def test_resolve_step_uses_registry():
    registry = {"demo.step": "json:dumps"}
    func = pipeline.resolve_step("demo.step", registry)
    assert func({"ok": True}) == json.dumps({"ok": True})


def test_topo_sort_respects_dependencies():
    steps = ["step.c", "step.a", "step.b"]
    deps = {
        "step.c": ["step.b"],
        "step.b": ["step.a"],
    }
    ordered = pipeline.topo_sort(steps, deps)
    assert ordered == ["step.a", "step.b", "step.c"]


def test_topo_sort_detects_cycle():
    steps = ["a", "b"]
    deps = {"a": ["b"], "b": ["a"]}
    with pytest.raises(ValueError):
        pipeline.topo_sort(steps, deps)


def test_log_step_event_levels():
    logger = logging.getLogger("pipeline.test")
    logger.handlers.clear()
    logger.setLevel(logging.DEBUG)
    logger.propagate = False

    class Collector(logging.Handler):
        def __init__(self) -> None:
            super().__init__()
            self.records: list[logging.LogRecord] = []

        def emit(self, record: logging.LogRecord) -> None:
            self.records.append(record)

    handler = Collector()
    logger.addHandler(handler)

    pipeline.log_step_event(logger, "demo", "end", status="WARN")
    pipeline.log_step_event(logger, "demo", "end", status="FAIL")

    assert handler.records[0].levelno == logging.WARNING
    assert handler.records[1].levelno == logging.ERROR
