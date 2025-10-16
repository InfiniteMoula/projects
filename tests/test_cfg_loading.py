from pathlib import Path

import pytest

from config.enrichment_config import load_enrichment_config


@pytest.fixture(autouse=True)
def _clear_enrichment_cache():
    load_enrichment_config.cache_clear()
    yield
    load_enrichment_config.cache_clear()


def test_default_enrichment_config_sections():
    cfg = load_enrichment_config(Path("config/enrichment.yaml"))

    assert cfg.use_metrics_export is False
    assert cfg.cache.enabled is False
    assert cfg.cache.backend == "sqlite"
    assert cfg.cache.ttl_days == 7
    assert cfg.circuit_breaker.enabled is False
    assert cfg.adaptive.enabled is True
    assert cfg.embeddings.enabled is False
    assert cfg.ai.enabled is False
    assert cfg.ai.contacts is False


def test_enrichment_config_defaults_when_missing(tmp_path):
    cfg_path = tmp_path / "enrichment.yaml"
    cfg_path.write_text("use_domains: false\n", encoding="utf-8")

    cfg = load_enrichment_config(cfg_path)

    assert cfg.cache.enabled is False
    assert cfg.cache.backend == "memory"
    assert cfg.circuit_breaker.enabled is False
    assert cfg.embeddings.enabled is False
    assert cfg.ai.contacts is False
