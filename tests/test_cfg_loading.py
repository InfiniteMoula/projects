from __future__ import annotations

import textwrap
from pathlib import Path

import pytest

from config.enrichment_config import load_enrichment_config


@pytest.fixture(autouse=True)
def clear_cache():
    load_enrichment_config.cache_clear()
    yield
    load_enrichment_config.cache_clear()


def test_enrichment_config_defaults_include_new_sections(tmp_path: Path) -> None:
    cfg_path = tmp_path / "enrichment.yaml"
    cfg_path.write_text("use_domains: false\n", encoding="utf-8")

    cfg = load_enrichment_config(cfg_path)

    assert cfg.cache.enabled is False
    assert cfg.circuit_breaker.failure_threshold == 5
    assert cfg.adaptive.max_parallelism >= cfg.adaptive.min_parallelism
    assert cfg.embeddings.enabled is True
    assert cfg.ai.enabled is True


def test_enrichment_config_section_overrides(tmp_path: Path) -> None:
    cfg_path = tmp_path / "custom.yaml"
    cfg_path.write_text(
        textwrap.dedent(
            """
            cache:
              enabled: true
              backend: sqlite
              dsn: "/tmp/cache.sqlite"
              ttl_sec: 1800
            circuit_breaker:
              enabled: true
              failure_threshold: 3
              recovery_time_sec: 25.0
              sample_window_sec: 12.0
            adaptive:
              enabled: true
              min_parallelism: 2
              max_parallelism: 6
              target_latency_sec: 1.5
              smoothing_factor: 0.75
            embeddings:
              enabled: false
              provider: test-provider
              model: embed-test
              batch_size: 16
              request_timeout: 12.0
            ai:
              enabled: true
              provider: local
              model: mini-llm
              temperature: 0.4
              max_output_tokens: 256
            """
        ).strip(),
        encoding="utf-8",
    )

    cfg = load_enrichment_config(cfg_path)

    assert cfg.cache.enabled is True
    assert cfg.cache.backend == "sqlite"
    assert cfg.cache.location == Path("/tmp/cache.sqlite")
    assert cfg.cache.ttl_seconds == 1800
    assert cfg.circuit_breaker.enabled is True
    assert cfg.circuit_breaker.failure_threshold == 3
    assert cfg.circuit_breaker.recovery_time_sec == 25.0
    assert cfg.adaptive.enabled is True
    assert cfg.adaptive.min_parallelism == 2
    assert cfg.adaptive.max_parallelism == 6
    assert pytest.approx(cfg.adaptive.target_latency_sec) == 1.5
    assert pytest.approx(cfg.adaptive.smoothing_factor) == 0.75
    assert cfg.embeddings.enabled is False
    assert cfg.embeddings.provider == "test-provider"
    assert cfg.embeddings.model == "embed-test"
    assert cfg.embeddings.batch_size == 16
    assert pytest.approx(cfg.embeddings.request_timeout) == 12.0
    assert cfg.ai.enabled is True
    assert cfg.ai.provider == "local"
    assert cfg.ai.model == "mini-llm"
    assert pytest.approx(cfg.ai.temperature) == 0.4
    assert cfg.ai.max_output_tokens == 256
