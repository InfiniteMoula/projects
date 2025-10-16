from __future__ import annotations

import json
import logging
import sys
from datetime import datetime, timezone
from typing import Any, Dict

from utils.logging_filters import SensitiveDataFilter

_DEFAULT_LEVEL = logging.INFO


class JsonFormatter(logging.Formatter):
    """Minimal JSON formatter compatible with stdlib logging."""

    def format(self, record: logging.LogRecord) -> str:  # pragma: no cover - trivial wrapper
        payload: Dict[str, Any] = {
            "timestamp": datetime.fromtimestamp(record.created, tz=timezone.utc).isoformat(),
            "level": record.levelname,
            "name": record.name,
            "message": record.getMessage(),
        }
        if record.exc_info:
            payload["exc_info"] = self.formatException(record.exc_info)
        for key, value in record.__dict__.items():
            if key.startswith("_") or key in {"args", "msg", "message", "created", "levelname", "levelno", "msecs", "relativeCreated", "stack_info", "exc_info", "exc_text", "name", "thread", "threadName", "process", "processName"}:
                continue
            try:
                json.dumps(value)
                payload[key] = value
            except TypeError:
                payload[key] = repr(value)
        return json.dumps(payload, ensure_ascii=False)


def get_logger(name: str, level: int = _DEFAULT_LEVEL) -> logging.Logger:
    """Return a logger configured with a JSON stream handler."""

    logger = logging.getLogger(name)
    logger.setLevel(level)
    logger.propagate = False

    handler_exists = False
    for handler in logger.handlers:
        if getattr(handler, "_loggingx_json", False):
            handler_exists = True
            handler.setLevel(level)
    if not handler_exists:
        handler = logging.StreamHandler(stream=sys.stdout)
        handler.setLevel(level)
        formatter = JsonFormatter()
        handler.setFormatter(formatter)
        if not any(isinstance(filt, SensitiveDataFilter) for filt in handler.filters):
            handler.addFilter(SensitiveDataFilter())
        handler._loggingx_json = True  # type: ignore[attr-defined]
        logger.addHandler(handler)
    else:
        for handler in logger.handlers:
            if not any(isinstance(filt, SensitiveDataFilter) for filt in handler.filters):
                handler.addFilter(SensitiveDataFilter())
    return logger
