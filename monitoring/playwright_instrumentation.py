"""Prometheus instrumentation helpers for Playwright sessions."""
from __future__ import annotations

import logging
import time
from typing import Any, Optional, Sequence, TYPE_CHECKING

try:
    from prometheus_client import Counter, Histogram

    PROMETHEUS_AVAILABLE = True
except ImportError:  # pragma: no cover - optional dependency
    PROMETHEUS_AVAILABLE = False

if TYPE_CHECKING:  # pragma: no cover - typing only
    from playwright.async_api import Page
else:  # pragma: no cover - avoids runtime dependency when unused
    Page = Any  # type: ignore[assignment]


LOG = logging.getLogger("monitoring.playwright")

if PROMETHEUS_AVAILABLE:  # pragma: no cover - guarded initialisation
    _NAVIGATION_DURATION = Histogram(
        "playwright_navigation_duration_seconds",
        "Total time spent waiting for Playwright page navigation",
        labelnames=("provider", "operation", "status"),
    )
    _DOM_READY = Histogram(
        "playwright_dom_ready_seconds",
        "Browser reported DOMContentLoaded delta for Playwright pages",
        labelnames=("provider", "operation", "status"),
    )
    _JS_ERRORS = Counter(
        "playwright_js_errors_total",
        "Number of JavaScript errors captured during Playwright sessions",
        labelnames=("provider", "operation", "status"),
    )
else:  # pragma: no cover - no dependency
    _NAVIGATION_DURATION = None
    _DOM_READY = None
    _JS_ERRORS = None


class PlaywrightSessionMetrics:
    """Collect navigation, DOM timing and JS error statistics."""

    def __init__(self, provider: str, operation: str = "navigation") -> None:
        self.provider = provider or "unknown"
        self.operation = operation or "navigation"
        self._start = time.perf_counter()
        self._js_errors = 0
        self._finalised = False
        self._page_error_callback: Optional[Any] = None
        self._console_callback: Optional[Any] = None

    def attach(self, page: Page) -> None:
        """Attach listeners to capture runtime errors."""

        if not PROMETHEUS_AVAILABLE:
            return
        if page is None:
            return
        if self._page_error_callback is not None:
            return
        self._page_error_callback = self._on_page_error
        self._console_callback = self._on_console_message
        try:
            page.on("pageerror", self._page_error_callback)
            page.on("console", self._console_callback)
        except Exception:  # pragma: no cover - defensive
            LOG.debug("Failed to attach Playwright listeners", exc_info=True)
            self._page_error_callback = None
            self._console_callback = None

    async def finalise(self, page: Page | None, *, status: str = "success") -> None:
        """Record collected metrics and detach listeners."""

        if self._finalised:
            return
        self._finalised = True

        if not PROMETHEUS_AVAILABLE:
            self.detach(page)
            return

        duration = max(time.perf_counter() - self._start, 0.0)
        labels = dict(provider=self.provider, operation=self.operation, status=status)
        try:
            _NAVIGATION_DURATION.labels(**labels).observe(duration)
        except Exception:  # pragma: no cover - defensive
            LOG.debug("Prometheus navigation metric failed", exc_info=True)

        dom_ready = None
        if page is not None:
            dom_ready = await self._extract_dom_ready(page)
            if dom_ready is not None:
                try:
                    _DOM_READY.labels(**labels).observe(dom_ready)
                except Exception:  # pragma: no cover - defensive
                    LOG.debug("Prometheus DOM metric failed", exc_info=True)

        if self._js_errors:
            try:
                _JS_ERRORS.labels(**labels).inc(self._js_errors)
            except Exception:  # pragma: no cover - defensive
                LOG.debug("Prometheus JS error metric failed", exc_info=True)

        self.detach(page)

    async def _extract_dom_ready(self, page: Page) -> Optional[float]:
        if not PROMETHEUS_AVAILABLE:
            return None
        try:
            result = await page.evaluate(
                """
                () => {
                    const navEntry = performance.getEntriesByType('navigation')[0];
                    if (navEntry && navEntry.domContentLoadedEventEnd) {
                        return navEntry.domContentLoadedEventEnd / 1000;
                    }
                    if (!performance.timing) {
                        return null;
                    }
                    const timing = performance.timing;
                    if (!timing.domContentLoadedEventEnd || !timing.navigationStart) {
                        return null;
                    }
                    return (timing.domContentLoadedEventEnd - timing.navigationStart) / 1000;
                }
                """
            )
        except Exception:
            LOG.debug("Playwright DOM metrics evaluation failed", exc_info=True)
            return None

        if isinstance(result, (int, float)) and result >= 0:
            return float(result)
        return None

    def _on_page_error(self, _exception: BaseException) -> None:
        self._js_errors += 1

    def _on_console_message(self, message: Any) -> None:
        try:
            type_attr = getattr(message, "type", None)
            msg_type = type_attr() if callable(type_attr) else type_attr
            if msg_type == "error":
                self._js_errors += 1
        except Exception:  # pragma: no cover - defensive
            LOG.debug("Failed to inspect console message", exc_info=True)

    def detach(self, page: Page | None) -> None:
        if not PROMETHEUS_AVAILABLE:
            return
        if page is None:
            return
        if self._page_error_callback is None and self._console_callback is None:
            return
        try:
            if self._page_error_callback is not None:
                page.off("pageerror", self._page_error_callback)
            if self._console_callback is not None:
                page.off("console", self._console_callback)
        except Exception:  # pragma: no cover - defensive
            LOG.debug("Failed to detach Playwright listeners", exc_info=True)
        finally:
            self._page_error_callback = None
            self._console_callback = None


__all__: Sequence[str] = ["PlaywrightSessionMetrics", "PROMETHEUS_AVAILABLE"]
