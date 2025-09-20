#!/usr/bin/env python3
"""
Company name processing utilities for LinkedIn optimization.

This module provides the CompanyNameProcessor class for optimizing company names
for LinkedIn searches by removing legal forms, creating variants, and scoring.
"""

import re
from typing import Dict, List, Optional
import pandas as pd


class CompanyNameProcessor:
    """Optimize company names for LinkedIn searches."""
    
    def __init__(self):
        """Initialize with French legal forms and common patterns."""
        self.legal_forms = [
            'SAS', 'SARL', 'SA', 'EURL', 'SNC', 'SCS', 'SASU',
            'SELARL', 'SELAFA', 'SELAS', 'SELCA', 'EARL', 'GAEC',
            'SCEA', 'GIE', 'EEIG', 'SCI', 'SCM', 'SCP', 'SEP'
        ]
        
        self.common_suffixes = [
            '& Associés', '& Associates', '& Co', '& Cie', '& Compagnie',
            'Conseil', 'Consulting', 'Services', 'Solutions', 'Group',
            'Groupe', 'International', 'France', 'Europe', 'Worldwide'
        ]
        
        self.noise_words = [
            'et', 'de', 'du', 'des', 'le', 'la', 'les', 'un', 'une',
            'au', 'aux', 'par', 'pour', 'avec', 'sans', 'sur', 'sous'
        ]
    
    def optimize_for_linkedin(self, company_name: str) -> Dict:
        """
        Create optimized variants for LinkedIn search.
        
        Args:
            company_name: Original company name
            
        Returns:
            Dictionary with variants and confidence score
        """
        if not company_name or pd.isna(company_name):
            return {'variants': [], 'confidence': 0.0, 'strategy': 'skip'}
        
        variants = []
        base_name = str(company_name).strip()
        
        # Original name
        variants.append(base_name)
        
        # Remove legal forms
        clean_name = self._remove_legal_forms(base_name)
        if clean_name != base_name and len(clean_name) > 2:
            variants.append(clean_name)
        
        # Remove common suffixes
        no_suffix = self._remove_suffixes(clean_name)
        if no_suffix != clean_name and len(no_suffix) > 2:
            variants.append(no_suffix)
        
        # Create acronym for long names
        if len(clean_name.split()) >= 3:
            acronym = self._create_acronym(clean_name)
            if acronym and len(acronym) >= 2:
                variants.append(acronym)
        
        # Create simplified version (remove noise words)
        simplified = self._remove_noise_words(no_suffix)
        if simplified != no_suffix and len(simplified) > 2:
            variants.append(simplified)
        
        # Create core name (first significant word)
        core_name = self._extract_core_name(simplified)
        if core_name and core_name not in variants and len(core_name) > 2:
            variants.append(core_name)
        
        # Remove duplicates and empty variants
        unique_variants = []
        seen = set()
        for variant in variants:
            variant = variant.strip()
            if variant and variant not in seen and len(variant) > 2:
                seen.add(variant)
                unique_variants.append(variant)
        
        # Calculate confidence based on variants quality
        confidence = self._calculate_confidence(base_name, unique_variants)
        strategy = self._get_search_strategy(confidence, unique_variants)
        
        return {
            'variants': unique_variants,
            'confidence': confidence,
            'strategy': strategy,
            'primary': unique_variants[0] if unique_variants else base_name
        }
    
    def _remove_legal_forms(self, company_name: str) -> str:
        """Remove legal form suffixes from company name."""
        name = company_name
        
        for legal_form in self.legal_forms:
            # Remove legal form at the end
            pattern = rf'\s+{re.escape(legal_form)}\s*$'
            name = re.sub(pattern, '', name, flags=re.IGNORECASE)
            
            # Remove legal form in parentheses
            pattern = rf'\s*\({re.escape(legal_form)}\)\s*'
            name = re.sub(pattern, ' ', name, flags=re.IGNORECASE)
        
        return name.strip()
    
    def _remove_suffixes(self, company_name: str) -> str:
        """Remove common business suffixes."""
        name = company_name
        
        for suffix in self.common_suffixes:
            pattern = rf'\s+{re.escape(suffix)}\s*$'
            name = re.sub(pattern, '', name, flags=re.IGNORECASE)
        
        return name.strip()
    
    def _remove_noise_words(self, company_name: str) -> str:
        """Remove common noise words."""
        words = company_name.split()
        filtered_words = []
        
        for word in words:
            if word.lower() not in self.noise_words:
                filtered_words.append(word)
        
        return ' '.join(filtered_words)
    
    def _create_acronym(self, company_name: str) -> Optional[str]:
        """Create acronym from company name."""
        words = company_name.split()
        
        # Only use significant words (length > 2)
        significant_words = [word for word in words if len(word) > 2]
        
        if len(significant_words) < 2:
            return None
        
        # Create acronym from first letters
        acronym = ''.join(word[0].upper() for word in significant_words[:4])
        
        # Only return if reasonable length
        return acronym if 2 <= len(acronym) <= 6 else None
    
    def _extract_core_name(self, company_name: str) -> Optional[str]:
        """Extract the core/main part of the company name."""
        words = company_name.split()
        
        if not words:
            return None
        
        # Find the longest word that's not a noise word
        core_candidates = [
            word for word in words 
            if len(word) > 3 and word.lower() not in self.noise_words
        ]
        
        if core_candidates:
            # Return the first significant word
            return core_candidates[0]
        
        # Fallback to first word if no good candidates
        return words[0] if words else None
    
    def _calculate_confidence(self, original_name: str, variants: List[str]) -> float:
        """Calculate confidence score for the variants."""
        if not variants:
            return 0.0
        
        score = 0.0
        
        # Base score for having variants
        score += 0.3
        
        # Bonus for multiple variants
        if len(variants) > 1:
            score += 0.2
        
        # Bonus for having reasonable length variants
        reasonable_variants = [v for v in variants if 3 <= len(v) <= 50]
        if reasonable_variants:
            score += 0.3
        
        # Bonus for having both full name and simplified versions
        has_full = any(len(v.split()) > 1 for v in variants)
        has_short = any(len(v.split()) == 1 for v in variants)
        if has_full and has_short:
            score += 0.2
        
        return min(score, 1.0)
    
    def _get_search_strategy(self, confidence: float, variants: List[str]) -> str:
        """Determine search strategy based on confidence and variants."""
        if confidence >= 0.8 and len(variants) >= 3:
            return 'comprehensive'
        elif confidence >= 0.6:
            return 'standard'
        elif confidence >= 0.3:
            return 'basic'
        else:
            return 'minimal'
    
    def process_dataframe(self, df: pd.DataFrame, company_column: str = 'denomination') -> pd.DataFrame:
        """
        Process company names in a DataFrame and add optimization columns.
        
        Args:
            df: Input DataFrame with company names
            company_column: Name of the column containing company names
            
        Returns:
            Enhanced DataFrame with additional company name processing columns
        """
        # Handle multiple possible company name columns
        possible_columns = [company_column, 'raison_sociale', 'company_name', 'nom_entreprise']
        
        actual_column = None
        for col in possible_columns:
            if col in df.columns:
                actual_column = col
                break
        
        if actual_column is None:
            raise ValueError(f"No company name column found. Tried: {possible_columns}")
        
        # Process company names
        optimized_companies = []
        for company_name in df[actual_column]:
            result = self.optimize_for_linkedin(company_name)
            optimized_companies.append(result)
        
        # Add new columns
        result_df = df.copy()
        result_df['company_variants'] = ['; '.join(comp['variants']) for comp in optimized_companies]
        result_df['company_primary'] = [comp['primary'] for comp in optimized_companies]
        result_df['company_confidence'] = [comp['confidence'] for comp in optimized_companies]
        result_df['company_search_strategy'] = [comp['strategy'] for comp in optimized_companies]
        
        return result_df
    
    def get_search_terms(self, company_name: str, max_terms: int = 3) -> List[str]:
        """
        Get prioritized search terms for a company name.
        
        Args:
            company_name: Company name to process
            max_terms: Maximum number of search terms to return
            
        Returns:
            List of prioritized search terms
        """
        result = self.optimize_for_linkedin(company_name)
        variants = result['variants']
        
        # Prioritize variants by length and completeness
        prioritized = sorted(variants, key=lambda x: (
            -len(x.split()),  # Prefer multi-word names
            -len(x),          # Then prefer longer names
            x.lower()         # Then alphabetical
        ))
        
        return prioritized[:max_terms]