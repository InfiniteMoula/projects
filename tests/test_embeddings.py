import numpy as np
import pandas as pd
import pytest

from enrich.enrich_domains import _apply_semantic_selection
from features import embeddings


def test_semantic_selection_prefers_high_similarity(monkeypatch):
    df = pd.DataFrame(
        [
            {
                "denomination": "Alpha Conseil",
                "ville": "Paris",
                "naf": "70.22Z",
                "site_web_candidates": [
                    {
                        "url": "https://alphaconseil.fr",
                        "homepage": "Bienvenue sur Alpha Conseil, experts 70.22Z à Paris",
                        "source": "serp",
                        "title": "Alpha Conseil - Paris",
                    },
                    {
                        "url": "https://alphaconseil-btp.fr",
                        "homepage": "Bienvenue sur Alpha Conseil BTP, entreprise de travaux à Lyon",
                        "source": "serp",
                        "title": "Alpha Conseil BTP - Lyon",
                    },
                ],
            }
        ]
    )

    series = df.iloc[0]
    company_text = embeddings.generate_text(series)
    winning_text = df.at[0, "site_web_candidates"][0]["homepage"]
    losing_text = df.at[0, "site_web_candidates"][1]["homepage"]

    vectors = {
        company_text: np.array([1.0, 0.0], dtype=np.float32),
        winning_text: np.array([1.0, 0.0], dtype=np.float32),
        losing_text: np.array([0.0, 1.0], dtype=np.float32),
    }

    def fake_embed(text: str, *, model_name: str | None = None) -> np.ndarray:
        return vectors.get(text, np.zeros(2, dtype=np.float32))

    monkeypatch.setattr(embeddings, "embed", fake_embed)

    cfg = {"enabled": True, "threshold": 0.6, "model": "stub"}
    df_out, selected = _apply_semantic_selection(df, cfg, logger=None)

    assert selected == [0]
    assert df_out.loc[0, "site_web"] == "https://alphaconseil.fr"
    assert df_out.loc[0, "site_web_source"] == "SEMANTIC:serp"
    assert pytest.approx(df_out.loc[0, "site_web_semantic_similarity"], rel=1e-3) == 1.0
    assert df_out.loc[0, "site_web_score"] > 0


def test_semantic_selection_respects_threshold(monkeypatch):
    df = pd.DataFrame(
        [
            {
                "denomination": "Beta Conseil",
                "ville": "Lyon",
                "naf": "70.22Z",
                "site_web_candidates": [
                    {
                        "url": "https://betaconseil.fr",
                        "homepage": "Bienvenue sur Beta Conseil Lyon",
                        "source": "serp",
                    },
                    {
                        "url": "https://betaconseil.com",
                        "homepage": "Discover Beta Conseil international",
                        "source": "serp",
                    },
                ],
            }
        ]
    )

    series = df.iloc[0]
    company_text = embeddings.generate_text(series)

    vectors = {
        company_text: np.array([1.0, 0.0], dtype=np.float32),
        df.at[0, "site_web_candidates"][0]["homepage"]: np.array([0.0, 1.0], dtype=np.float32),
        df.at[0, "site_web_candidates"][1]["homepage"]: np.array([0.0, 0.5], dtype=np.float32),
    }

    def fake_embed(text: str, *, model_name: str | None = None) -> np.ndarray:
        return vectors.get(text, np.zeros(2, dtype=np.float32))

    monkeypatch.setattr(embeddings, "embed", fake_embed)

    cfg = {"enabled": True, "threshold": 0.9, "model": "stub"}
    df_out, selected = _apply_semantic_selection(df, cfg, logger=None)

    assert selected == []
    assert pd.isna(df_out.loc[0, "site_web"])
    assert pd.isna(df_out.loc[0, "site_web_semantic_similarity"])
