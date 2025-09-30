"""User-Agent rotation helpers."""
from __future__ import annotations

import random
import threading
from pathlib import Path
from typing import Iterable, Sequence

from utils import io

_DEFAULT_USER_AGENTS: Sequence[str] = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/127.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 14_5_0) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/17.4 Safari/605.1.15",
    "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/126.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:126.0) Gecko/20100101 Firefox/126.0",
    "Mozilla/5.0 (iPhone; CPU iPhone OS 17_4 like Mac OS X) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/17.4 Mobile/15E148 Safari/604.1",
    "Mozilla/5.0 (Linux; Android 14; SM-S921B) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/125.0.6422.112 Mobile Safari/537.36",
    "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 13_6) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/123.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Edge/124.0.2478.51",
    "Mozilla/5.0 (iPad; CPU OS 17_4 like Mac OS X) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/17.4 Mobile/15E148 Safari/604.1",
)


class UserAgentPool:
    """Thread-safe pool of User-Agent strings."""

    def __init__(self, user_agents: Iterable[str] | None = None) -> None:
        agents = [ua.strip() for ua in (user_agents or []) if ua and ua.strip()]
        if not agents:
            agents = list(_DEFAULT_USER_AGENTS)
        self._user_agents = tuple(dict.fromkeys(agents))
        self._lock = threading.Lock()
        self._last: str | None = None

    def get(self) -> str:
        """Return a random User-Agent string, avoiding immediate repeats."""

        with self._lock:
            if len(self._user_agents) == 1:
                choice = self._user_agents[0]
            else:
                choice = random.choice(self._user_agents)
                if choice == self._last:
                    choice = random.choice(self._user_agents)
            self._last = choice
            return choice

    @classmethod
    def from_file(cls, path: str | Path, fallback: Iterable[str] | None = None) -> "UserAgentPool":
        target = Path(path)
        if not target.exists():
            return cls(fallback)
        try:
            lines = target.read_text(encoding="utf-8").splitlines()
        except OSError as exc:
            raise io.IoError(f"unable to read user agent file: {target}") from exc
        return cls(lines or fallback)


def load_user_agent_pool(path: str | None) -> UserAgentPool:
    """Return a `UserAgentPool`, optionally loading from *path*."""

    if path:
        return UserAgentPool.from_file(path)
    return UserAgentPool()
