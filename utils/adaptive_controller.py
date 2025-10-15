"""Adaptive controller for tuning batch concurrency and chunk sizes."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Optional


LOGGER = logging.getLogger(__name__)


@dataclass(slots=True)
class AdaptiveState:
    """Internal state returned by :class:`AdaptiveController.observe`."""

    concurrency: int
    chunk_size: int
    concurrency_changed: bool = False
    chunk_size_changed: bool = False


class AdaptiveController:
    """Adaptive heuristics to steer concurrency and chunk size.

    The controller consumes coarse telemetry about the last processing window
    (error rate, throughput and RAM usage) and applies simple rules:

    * if the observed error rate exceeds ``5%`` the concurrency is reduced by
      ``20%`` to alleviate pressure.
    * if the error rate stays below ``3%`` and the request rate is stable (less
      than ``5%`` deviation compared to the previous measurement) the
      concurrency is increased by ``10%`` to improve throughput.
    * if the RAM consumption exceeds the configured limit the chunk size is
      reduced by ``20%`` in an attempt to limit memory pressure.

    The resulting concurrency and chunk size are always clamped within the
    bounds defined by the enrichment configuration.
    """

    STABLE_DELTA = 0.05
    REDUCTION_FACTOR = 0.8
    INCREASE_FACTOR = 1.1

    def __init__(
        self,
        cfg,
        *,
        initial_concurrency: int,
        initial_chunk_size: int,
    ) -> None:
        self._cfg = cfg
        self.current_concurrency = self._clamp(
            int(initial_concurrency or cfg.min_concurrency),
            cfg.min_concurrency,
            cfg.max_concurrency,
        )
        self.current_chunk_size = self._clamp(
            int(initial_chunk_size or cfg.min_chunk),
            cfg.min_chunk,
            cfg.max_chunk,
        )
        self._last_req_per_min: Optional[float] = None

    @staticmethod
    def _clamp(value: int, minimum: int, maximum: int) -> int:
        return max(minimum, min(maximum, int(value)))

    def observe(
        self,
        *,
        error_rate: Optional[float],
        req_per_min: Optional[float],
        ram_used: Optional[float],
    ) -> AdaptiveState:
        """Update the controller with fresh telemetry and return new targets."""

        if not getattr(self._cfg, "enabled", False):
            if req_per_min is not None:
                self._last_req_per_min = req_per_min
            return AdaptiveState(
                concurrency=self.current_concurrency,
                chunk_size=self.current_chunk_size,
            )

        concurrency_changed = False
        chunk_size_changed = False

        if error_rate is not None:
            if error_rate > 0.05:
                lowered = max(
                    self._cfg.min_concurrency,
                    int(round(self.current_concurrency * self.REDUCTION_FACTOR)),
                )
                if lowered < self.current_concurrency:
                    LOGGER.info(
                        "Adaptive controller lowering concurrency from %s to %s due to %.1f%% error rate",
                        self.current_concurrency,
                        lowered,
                        error_rate * 100.0,
                    )
                    self.current_concurrency = lowered
                    concurrency_changed = True
            elif (
                error_rate < 0.03
                and req_per_min is not None
                and self._last_req_per_min is not None
            ):
                baseline = max(self._last_req_per_min, 1e-6)
                delta = abs(req_per_min - self._last_req_per_min) / baseline
                if delta <= self.STABLE_DELTA:
                    increased = max(
                        self.current_concurrency + 1,
                        int(round(self.current_concurrency * self.INCREASE_FACTOR)),
                    )
                    increased = min(increased, self._cfg.max_concurrency)
                    if increased > self.current_concurrency:
                        LOGGER.info(
                            "Adaptive controller raising concurrency from %s to %s (stable throughput %.1f req/min)",
                            self.current_concurrency,
                            increased,
                            req_per_min,
                        )
                        self.current_concurrency = increased
                        concurrency_changed = True

        if ram_used is not None and ram_used > self._cfg.max_ram_gb:
            reduced = max(
                self._cfg.min_chunk,
                int(round(self.current_chunk_size * self.REDUCTION_FACTOR)),
            )
            if reduced < self.current_chunk_size:
                LOGGER.info(
                    "Adaptive controller reducing chunk size from %s to %s due to %.2f GiB RAM",
                    self.current_chunk_size,
                    reduced,
                    ram_used,
                )
                self.current_chunk_size = reduced
                chunk_size_changed = True

        if req_per_min is not None:
            self._last_req_per_min = req_per_min

        self.current_concurrency = self._clamp(
            self.current_concurrency,
            self._cfg.min_concurrency,
            self._cfg.max_concurrency,
        )
        self.current_chunk_size = self._clamp(
            self.current_chunk_size,
            self._cfg.min_chunk,
            self._cfg.max_chunk,
        )

        return AdaptiveState(
            concurrency=self.current_concurrency,
            chunk_size=self.current_chunk_size,
            concurrency_changed=concurrency_changed,
            chunk_size_changed=chunk_size_changed,
        )


__all__ = ["AdaptiveController", "AdaptiveState"]

