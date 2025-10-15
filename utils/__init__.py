"""
Utility helpers and reusable components.

This module exposes high-level helpers that are commonly required by
enrichment tasks.
"""

from .normalization import generate_domain_candidates, normalize_company_name

__all__ = ["normalize_company_name", "generate_domain_candidates"]
