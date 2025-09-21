#!/usr/bin/env python3
"""
Contact extraction and validation module.

This module provides comprehensive contact information extraction and validation
with confidence scoring for business data enrichment workflows.
"""

import re
import logging
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
import httpx
import dns.resolver

from utils.http import HttpError, request_with_backoff
from urllib.parse import urlparse


@dataclass
class ContactInfo:
    """Data class for structured contact information."""
    phone: Optional[str] = None
    email: Optional[str] = None
    website: Optional[str] = None
    address: Optional[str] = None
    confidence_score: float = 0.0
    validation_issues: List[str] = None
    
    def __post_init__(self):
        if self.validation_issues is None:
            self.validation_issues = []


class ContactExtractor:
    """Contact extraction and validation with confidence scoring."""
    
    def __init__(self):
        """Initialize contact extractor with validation patterns."""
        # French phone number patterns (more comprehensive)
        self.phone_patterns = [
            # International format: +33 followed by 9 digits
            re.compile(r'(\+33\s?[1-9](?:\s?\d{2}){4})'),
            # National format: 0 followed by 9 digits
            re.compile(r'(0[1-9](?:\s?\d{2}){4})'),
            # Formatted with dots/dashes
            re.compile(r'(\+33[\s\.-]?[1-9](?:[\s\.-]?\d{2}){4})'),
            re.compile(r'(0[1-9](?:[\s\.-]?\d{2}){4})'),
            # Simple 10-digit format
            re.compile(r'(\d{2}\s?\d{2}\s?\d{2}\s?\d{2}\s?\d{2})'),
        ]
        
        # Email validation patterns
        self.email_patterns = [
            # Standard email format
            re.compile(r'([a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,})'),
            # Common contact emails
            re.compile(r'(contact@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,})'),
            re.compile(r'(info@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,})'),
            re.compile(r'(admin@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,})'),
        ]
        
        # Website URL patterns (conservative to avoid false positives)
        self.website_patterns = [
            re.compile(r'https?://[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}(?:/[^\s]*)?'),
            re.compile(r'www\.[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}(?:/[^\s]*)?'),
        ]
        
        # Common business email domains to boost confidence
        self.business_domains = {
            'gmail.com': 0.3,     # Lower confidence for generic domains
            'yahoo.fr': 0.3,
            'hotmail.com': 0.3,
            'outlook.com': 0.4,
            'free.fr': 0.4,
            'orange.fr': 0.5,
            'wanadoo.fr': 0.4,
        }
        
        self.logger = logging.getLogger(__name__)
    
    def extract_phone_numbers(self, text: str) -> List[Tuple[str, float]]:
        """
        Extract phone numbers from text with confidence scores.
        
        Returns:
            List of tuples (phone_number, confidence_score)
        """
        if not text:
            return []
        
        phones = []
        
        for pattern in self.phone_patterns:
            matches = pattern.findall(text)
            for match in matches:
                normalized = self._normalize_phone(match)
                if normalized and self._validate_phone_format(normalized):
                    confidence = self._calculate_phone_confidence(match, normalized)
                    phones.append((normalized, confidence))
        
        # Remove duplicates while preserving highest confidence
        phone_dict = {}
        for phone, conf in phones:
            if phone not in phone_dict or conf > phone_dict[phone]:
                phone_dict[phone] = conf
        
        return [(phone, conf) for phone, conf in phone_dict.items()]
    
    def extract_emails(self, text: str) -> List[Tuple[str, float]]:
        """
        Extract email addresses from text with confidence scores.
        
        Returns:
            List of tuples (email, confidence_score)
        """
        if not text:
            return []
        
        emails = []
        
        for pattern in self.email_patterns:
            matches = pattern.findall(text)
            for match in matches:
                if self._validate_email_format(match):
                    confidence = self._calculate_email_confidence(match)
                    emails.append((match.lower(), confidence))
        
        # Remove duplicates while preserving highest confidence
        email_dict = {}
        for email, conf in emails:
            if email not in email_dict or conf > email_dict[email]:
                email_dict[email] = conf
        
        return [(email, conf) for email, conf in email_dict.items()]
    
    def extract_websites(self, text: str) -> List[Tuple[str, float]]:
        """
        Extract website URLs from text with confidence scores.
        
        Returns:
            List of tuples (website_url, confidence_score)
        """
        if not text:
            return []
        
        websites = []
        
        # Process patterns in order of specificity (most specific first)
        for i, pattern in enumerate(self.website_patterns):
            matches = pattern.findall(text)
            for match in matches:
                normalized = self._normalize_website(match)
                if normalized and self._validate_website_format(normalized):
                    confidence = self._calculate_website_confidence(normalized)
                    # Add priority bonus for more specific patterns
                    if i == 0:  # http/https patterns get highest priority
                        confidence += 0.1
                    elif i == 1:  # www patterns get medium priority
                        confidence += 0.05
                    websites.append((normalized, min(confidence, 1.0)))
        
        # Remove duplicates while preserving highest confidence, and normalize domain duplicates
        website_dict = {}
        for website, conf in websites:
            # Extract domain for duplicate detection
            try:
                from urllib.parse import urlparse
                parsed = urlparse(website)
                domain = parsed.netloc.lower()
                # Remove www. prefix for comparison
                domain_key = domain.replace('www.', '') if domain.startswith('www.') else domain
            except:
                domain_key = website.lower()
            
            # Prefer original URL format over normalized format
            if domain_key not in website_dict:
                website_dict[domain_key] = (website, conf)
            else:
                existing_url, existing_conf = website_dict[domain_key]
                # Prefer URLs with explicit protocol over normalized ones
                if (website.startswith(('http://', 'https://')) and 
                    not existing_url.startswith(('http://', 'https://'))):
                    website_dict[domain_key] = (website, conf)
                elif conf > existing_conf:
                    website_dict[domain_key] = (website, conf)
        
        return [(website, conf) for website, conf in website_dict.values()]
    
    def extract_contact_info(self, text: str, source: str = "unknown") -> ContactInfo:
        """
        Extract all contact information from text with overall confidence.
        
        Args:
            text: Text content to extract from
            source: Source of the text (e.g., 'google_maps', 'website')
            
        Returns:
            ContactInfo object with extracted data and confidence
        """
        if not text:
            return ContactInfo()
        
        phones = self.extract_phone_numbers(text)
        emails = self.extract_emails(text)
        websites = self.extract_websites(text)
        
        # Select best contact info based on confidence
        best_phone = max(phones, key=lambda x: x[1]) if phones else (None, 0.0)
        best_email = max(emails, key=lambda x: x[1]) if emails else (None, 0.0)
        best_website = max(websites, key=lambda x: x[1]) if websites else (None, 0.0)
        
        # Calculate overall confidence score
        confidence_scores = [score for _, score in [best_phone, best_email, best_website] if score > 0]
        overall_confidence = sum(confidence_scores) / len(confidence_scores) if confidence_scores else 0.0
        
        # Apply source-specific confidence adjustments
        source_multiplier = {
            'google_maps': 1.0,      # High trust in Google Maps data
            'website': 0.9,          # High trust in website data
            'linkedin': 0.8,         # Good trust in LinkedIn data
            'search_result': 0.7,    # Medium trust in search results
            'unknown': 0.6           # Lower trust for unknown sources
        }.get(source, 0.6)
        
        overall_confidence *= source_multiplier
        
        # Collect validation issues
        issues = []
        if not phones and not emails and not websites:
            issues.append("No contact information found")
        if best_phone[1] < 0.5 and best_phone[0]:
            issues.append("Low confidence phone number")
        if best_email[1] < 0.5 and best_email[0]:
            issues.append("Low confidence email address")
        
        return ContactInfo(
            phone=best_phone[0],
            email=best_email[0],
            website=best_website[0],
            confidence_score=min(overall_confidence, 1.0),
            validation_issues=issues
        )
    
    def _normalize_phone(self, phone: str) -> str:
        """Normalize phone number to standard format."""
        # Remove all non-digit characters except +
        cleaned = re.sub(r'[^\d+]', '', phone)
        
        # Convert to international format
        if cleaned.startswith('0'):
            # French national format to international
            cleaned = '+33' + cleaned[1:]
        elif not cleaned.startswith('+'):
            # Assume French number if no country code
            if len(cleaned) == 9:
                cleaned = '+33' + cleaned
            elif len(cleaned) == 10 and cleaned.startswith('0'):
                cleaned = '+33' + cleaned[1:]
        
        return cleaned
    
    def _validate_phone_format(self, phone: str) -> bool:
        """Validate phone number format."""
        # French phone number should be +33 followed by 9 digits
        pattern = re.compile(r'^\+33[1-9]\d{8}$')
        return bool(pattern.match(phone))
    
    def _calculate_phone_confidence(self, original: str, normalized: str) -> float:
        """Calculate confidence score for phone number."""
        confidence = 0.7  # Base confidence
        
        # Boost confidence for properly formatted numbers
        if '+33' in original:
            confidence += 0.2
        if re.search(r'\d{2}\s\d{2}\s\d{2}\s\d{2}\s\d{2}', original):
            confidence += 0.1  # Well-formatted
        
        # Validate the area code (French mobile/landline prefixes)
        if normalized.startswith('+33'):
            area_code = normalized[3:4]
            if area_code in ['1', '2', '3', '4', '5']:  # Landline
                confidence += 0.05
            elif area_code in ['6', '7']:  # Mobile
                confidence += 0.1
        
        return min(confidence, 1.0)
    
    def _validate_email_format(self, email: str) -> bool:
        """Validate email address format."""
        pattern = re.compile(r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$')
        return bool(pattern.match(email))
    
    def _calculate_email_confidence(self, email: str) -> float:
        """Calculate confidence score for email address."""
        confidence = 0.6  # Base confidence
        
        domain = email.split('@')[1].lower() if '@' in email else ''
        
        # Check against known business domains
        if domain in self.business_domains:
            confidence = max(confidence, self.business_domains[domain])
        else:
            # Higher confidence for non-generic domains
            confidence = 0.8
        
        # Boost confidence for business-like emails
        local_part = email.split('@')[0].lower()
        if any(term in local_part for term in ['contact', 'info', 'admin', 'hello', 'support']):
            confidence += 0.1
        
        # Reduce confidence for personal-looking emails
        if any(term in local_part for term in ['prenom', 'nom', 'perso']):
            confidence -= 0.1
        
        return min(max(confidence, 0.1), 1.0)
    
    def _normalize_website(self, url: str) -> str:
        """Normalize website URL."""
        url = url.strip()
        
        # Don't modify URLs that already have protocols
        if url.startswith(('http://', 'https://')):
            return url
        
        # Add protocol if missing
        if not url.startswith(('http://', 'https://')):
            if url.startswith('www.'):
                url = 'https://' + url
            else:
                url = 'https://www.' + url
        
        return url
    
    def _validate_website_format(self, url: str) -> bool:
        """Validate website URL format."""
        try:
            parsed = urlparse(url)
            return bool(parsed.netloc and parsed.scheme in ['http', 'https'])
        except Exception:
            return False
    
    def _calculate_website_confidence(self, url: str) -> float:
        """Calculate confidence score for website URL."""
        confidence = 0.7  # Base confidence
        
        # Boost confidence for HTTPS
        if url.startswith('https://'):
            confidence += 0.1
        
        # Boost confidence for proper domain structure
        try:
            parsed = urlparse(url)
            domain_parts = parsed.netloc.split('.')
            if len(domain_parts) >= 2:
                confidence += 0.1
            
            # French domains get slight boost
            if parsed.netloc.endswith('.fr'):
                confidence += 0.05
        except Exception:
            confidence -= 0.2
        
        return min(confidence, 1.0)
    
    def validate_contact_info(self, contact_info: ContactInfo) -> Dict[str, Any]:
        """
        Perform deep validation of contact information.
        
        Returns:
            Dictionary with validation results and enhanced confidence scores
        """
        validation_result = {
            'original_confidence': contact_info.confidence_score,
            'enhanced_confidence': contact_info.confidence_score,
            'validation_details': {},
            'recommendations': []
        }
        
        # Validate phone number if present
        if contact_info.phone:
            phone_validation = self._deep_validate_phone(contact_info.phone)
            validation_result['validation_details']['phone'] = phone_validation
            if not phone_validation['is_valid']:
                validation_result['enhanced_confidence'] *= 0.8
        
        # Validate email if present
        if contact_info.email:
            email_validation = self._deep_validate_email(contact_info.email)
            validation_result['validation_details']['email'] = email_validation
            if not email_validation['is_valid']:
                validation_result['enhanced_confidence'] *= 0.8
        
        # Validate website if present
        if contact_info.website:
            website_validation = self._deep_validate_website(contact_info.website)
            validation_result['validation_details']['website'] = website_validation
            if not website_validation['is_valid']:
                validation_result['enhanced_confidence'] *= 0.8
        
        # Generate recommendations
        if contact_info.confidence_score < 0.6:
            validation_result['recommendations'].append("Low confidence contact data - manual review recommended")
        
        if not contact_info.phone and not contact_info.email:
            validation_result['recommendations'].append("No direct contact methods found")
        
        return validation_result
    
    def _deep_validate_phone(self, phone: str) -> Dict[str, Any]:
        """Perform deep validation of phone number."""
        result = {
            'is_valid': False,
            'format_valid': False,
            'confidence': 0.0,
            'issues': []
        }
        
        if not phone:
            result['issues'].append("Empty phone number")
            return result
        
        # Format validation
        if self._validate_phone_format(phone):
            result['format_valid'] = True
            result['confidence'] += 0.5
        else:
            result['issues'].append("Invalid phone format")
        
        # French number validation
        if phone.startswith('+33'):
            area_code = phone[3:4] if len(phone) > 3 else ''
            if area_code in ['1', '2', '3', '4', '5', '6', '7', '8', '9']:
                result['confidence'] += 0.3
                result['is_valid'] = True
            else:
                result['issues'].append("Invalid French area code")
        else:
            result['issues'].append("Not a French phone number")
        
        return result
    
    def _deep_validate_email(self, email: str) -> Dict[str, Any]:
        """Perform deep validation of email address."""
        result = {
            'is_valid': False,
            'format_valid': False,
            'domain_valid': False,
            'confidence': 0.0,
            'issues': []
        }
        
        if not email:
            result['issues'].append("Empty email address")
            return result
        
        # Format validation
        if self._validate_email_format(email):
            result['format_valid'] = True
            result['confidence'] += 0.4
        else:
            result['issues'].append("Invalid email format")
            return result
        
        # Domain validation
        domain = email.split('@')[1] if '@' in email else ''
        if domain:
            try:
                # Check if domain has MX record
                dns.resolver.resolve(domain, 'MX')
                result['domain_valid'] = True
                result['confidence'] += 0.4
                result['is_valid'] = True
            except Exception:
                result['issues'].append("Domain does not accept email")
                result['confidence'] += 0.2  # Still some confidence for format
        
        return result
    
    def _deep_validate_website(self, website: str) -> Dict[str, Any]:
        """Perform deep validation of website URL."""
        result = {
            'is_valid': False,
            'format_valid': False,
            'accessible': False,
            'confidence': 0.0,
            'issues': []
        }
        
        if not website:
            result['issues'].append("Empty website URL")
            return result
        
        # Format validation
        if self._validate_website_format(website):
            result['format_valid'] = True
            result['confidence'] += 0.3
        else:
            result['issues'].append("Invalid URL format")
            return result
        
        # Accessibility check (with timeout)
        try:
            with httpx.Client(follow_redirects=True, timeout=5.0) as client:
                response = request_with_backoff(
                    client,
                    "HEAD",
                    website,
                    max_attempts=3,
                    backoff_factor=0.5,
                    logger=self.logger,
                )
            if response.status_code < 400:
                result['accessible'] = True
                result['confidence'] += 0.5
                result['is_valid'] = True
            else:
                result['issues'].append(f"Website returns {response.status_code}")
                result['confidence'] += 0.2
        except HttpError:
            result['issues'].append("Website not accessible")
            result['confidence'] += 0.1
        
        return result