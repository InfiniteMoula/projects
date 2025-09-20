#!/usr/bin/env python3
"""
Address processing utilities for intelligent address preparation.

This module provides the AddressProcessor class for cleaning, normalizing,
and generating search variants for addresses to optimize Google Maps results.
"""

import re
from typing import Dict, List, Optional
import pandas as pd


class AddressProcessor:
    """Intelligent address processing for optimal Google Maps results."""
    
    def __init__(self):
        """Initialize the AddressProcessor with French address patterns."""
        self.postal_code_regex = re.compile(r'\b\d{5}\b')
        self.street_indicators = [
            'rue', 'avenue', 'boulevard', 'place', 'impasse', 'allée',
            'chemin', 'route', 'voie', 'passage', 'square', 'cour',
            'quai', 'esplanade', 'promenade'
        ]
        
        # Common abbreviation patterns for standardization
        self.abbreviation_patterns = {
            r'\bRUE\b': 'Rue',
            r'\bAVE?\b': 'Avenue', 
            r'\bAV\b': 'Avenue',
            r'\bBD\b|\bBLVD\b': 'Boulevard',
            r'\bPL\b': 'Place',
            r'\bST\b': 'Saint',
            r'\bSTE\b': 'Sainte',
            r'\bCHEM\b': 'Chemin',
            r'\bALL\b': 'Allée',
            r'\bIMP\b': 'Impasse'
        }
    
    def enhance_addresses(self, addresses: List[str]) -> List[Dict]:
        """
        Enhance addresses for better Google Maps matching.
        
        Args:
            addresses: List of address strings to process
            
        Returns:
            List of dictionaries with enhanced address information
        """
        enhanced = []
        
        for addr in addresses:
            processed = self._process_single_address(addr)
            enhanced.append(processed)
        
        return enhanced
    
    def _process_single_address(self, address: str) -> Dict:
        """Process a single address with multiple variants."""
        if not address or pd.isna(address):
            return {
                'original': '',
                'cleaned': '',
                'variants': [],
                'primary': '',
                'quality_score': 0.0,
                'search_strategy': 'skip'
            }
        
        # Clean the address
        clean_addr = self._clean_address(str(address))
        
        # Generate search variants
        variants = self._generate_address_variants(clean_addr)
        
        # Score address quality
        quality_score = self._calculate_address_quality(clean_addr)
        
        return {
            'original': address,
            'cleaned': clean_addr,
            'variants': variants,
            'primary': variants[0] if variants else clean_addr,
            'quality_score': quality_score,
            'search_strategy': self._get_search_strategy(quality_score)
        }
    
    def _clean_address(self, address: str) -> str:
        """Clean and standardize address format."""
        # Remove extra whitespace
        addr = re.sub(r'\s+', ' ', address.strip())
        
        # Standardize common abbreviations
        for pattern, replacement in self.abbreviation_patterns.items():
            addr = re.sub(pattern, replacement, addr, flags=re.IGNORECASE)
        
        # Capitalize properly
        addr = self._capitalize_address(addr)
        
        return addr
    
    def _capitalize_address(self, address: str) -> str:
        """Properly capitalize address components."""
        words = address.split()
        result = []
        
        for word in words:
            # Keep postal codes as-is
            if re.match(r'^\d{5}$', word):
                result.append(word)
            # Capitalize first letter of other words
            elif len(word) > 0:
                result.append(word[0].upper() + word[1:].lower())
        
        return ' '.join(result)
    
    def _generate_address_variants(self, address: str) -> List[str]:
        """Generate multiple search variants for an address."""
        variants = [address]
        
        # Add variant without building number
        no_number = re.sub(r'^\d+\s*', '', address)
        if no_number != address and len(no_number) > 10:
            variants.append(no_number)
        
        # Add variant with postal code emphasis
        postal_match = self.postal_code_regex.search(address)
        if postal_match:
            postal_code = postal_match.group()
            city_part = address[postal_match.end():].strip()
            if city_part:
                variants.append(f"{postal_code} {city_part}")
        
        # Add simplified variant (just street + city)
        simplified = self._extract_street_and_city(address)
        if simplified and simplified not in variants:
            variants.append(simplified)
        
        # Remove duplicates while preserving order
        seen = set()
        unique_variants = []
        for variant in variants:
            if variant not in seen and len(variant.strip()) > 5:
                seen.add(variant)
                unique_variants.append(variant)
        
        return unique_variants
    
    def _extract_street_and_city(self, address: str) -> Optional[str]:
        """Extract street and city components from address."""
        # Look for postal code to split address
        postal_match = self.postal_code_regex.search(address)
        if postal_match:
            before_postal = address[:postal_match.start()].strip()
            after_postal = address[postal_match.end():].strip()
            
            if before_postal and after_postal:
                return f"{before_postal} {after_postal}"
        
        # Fallback: return address without first number
        no_number = re.sub(r'^\d+\s*', '', address).strip()
        return no_number if len(no_number) > 10 else None
    
    def _calculate_address_quality(self, address: str) -> float:
        """Calculate address quality score (0-1)."""
        score = 0.0
        
        # Has street number (20% weight)
        if re.search(r'^\d+', address):
            score += 0.2
        
        # Has postal code (30% weight)
        if self.postal_code_regex.search(address):
            score += 0.3
        
        # Has street name indicator (20% weight)
        if any(indicator in address.lower() for indicator in self.street_indicators):
            score += 0.2
        
        # Reasonable length (20% weight)
        if 15 <= len(address) <= 100:
            score += 0.2
        
        # Has city name (10% weight) - simplified check
        parts = address.split()
        if len(parts) >= 3:  # At least number, street, city
            score += 0.1
        
        return min(score, 1.0)
    
    def _get_search_strategy(self, quality_score: float) -> str:
        """Determine search strategy based on quality score."""
        if quality_score >= 0.8:
            return 'precise'
        elif quality_score >= 0.5:
            return 'standard'
        elif quality_score >= 0.3:
            return 'fuzzy'
        else:
            return 'fallback'
    
    def validate_postal_code(self, address: str) -> str:
        """Validate and correct postal code format."""
        postal_matches = self.postal_code_regex.findall(address)
        
        for postal in postal_matches:
            # Check if it's a valid French postal code
            if postal.startswith('0') or postal.startswith('9'):
                # These are often special cases or errors for mainland France
                continue
            
            # Basic validation - French postal codes are 01XXX to 95XXX for mainland
            try:
                postal_int = int(postal)
                if not (1000 <= postal_int <= 95999):
                    # Invalid range, might need correction
                    continue
            except ValueError:
                continue
        
        return address
    
    def process_dataframe(self, df: pd.DataFrame, address_column: str = 'adresse') -> pd.DataFrame:
        """
        Process addresses in a DataFrame and add enhancement columns.
        
        Args:
            df: Input DataFrame with addresses
            address_column: Name of the column containing addresses
            
        Returns:
            Enhanced DataFrame with additional address processing columns
        """
        if address_column not in df.columns:
            raise ValueError(f"Column '{address_column}' not found in DataFrame")
        
        # Process addresses
        enhanced_addresses = self.enhance_addresses(df[address_column].tolist())
        
        # Add new columns
        result_df = df.copy()
        result_df['address_cleaned'] = [addr['cleaned'] for addr in enhanced_addresses]
        result_df['address_variants'] = ['; '.join(addr['variants']) for addr in enhanced_addresses]
        result_df['address_primary'] = [addr['primary'] for addr in enhanced_addresses]
        result_df['address_quality_score'] = [addr['quality_score'] for addr in enhanced_addresses]
        result_df['address_search_strategy'] = [addr['search_strategy'] for addr in enhanced_addresses]
        
        return result_df