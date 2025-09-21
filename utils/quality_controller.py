#!/usr/bin/env python3
"""
Quality control framework for data extraction results.

This module provides comprehensive quality control, validation, and confidence scoring
for business data enrichment workflows, particularly focusing on Google Maps and LinkedIn
extraction results.
"""

import logging
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field
import pandas as pd
import numpy as np
from datetime import datetime
import json

from . import io
from .contact_extractor import ContactExtractor, ContactInfo


@dataclass
class ValidationResult:
    """Result of validation for a single record."""
    overall_score: float = 0.0
    field_scores: Dict[str, float] = field(default_factory=dict)
    validation_issues: List[str] = field(default_factory=list)
    confidence_level: str = "low"  # low, medium, high
    is_valid: bool = False
    source: str = ""
    recommendations: List[str] = field(default_factory=list)


@dataclass
class QualityReport:
    """Quality report for a batch of validation results."""
    total_records: int = 0
    valid_records: int = 0
    validation_rate: float = 0.0
    average_score: float = 0.0
    confidence_distribution: Dict[str, int] = field(default_factory=dict)
    field_coverage: Dict[str, float] = field(default_factory=dict)
    common_issues: List[Tuple[str, int]] = field(default_factory=list)
    recommendations: List[str] = field(default_factory=list)
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())


class QualityController:
    """Main quality control framework for extraction results."""
    
    def __init__(self, config: Optional[Dict] = None):
        """
        Initialize quality controller with configuration.
        
        Args:
            config: Configuration dictionary with quality thresholds and weights
        """
        self.config = config or {}
        self.contact_extractor = ContactExtractor()
        self.logger = logging.getLogger(__name__)
        
        # Quality thresholds
        self.thresholds = self.config.get('thresholds', {
            'validation_rate': 0.8,      # 80% of results should be valid
            'phone_coverage': 0.6,       # 60% should have phone numbers
            'email_coverage': 0.4,       # 40% should have emails
            'linkedin_success': 0.5,     # 50% LinkedIn search success
            'minimum_confidence': 0.5,   # Minimum confidence to be considered valid
        })
        
        # Field weights for scoring
        self.field_weights = self.config.get('field_weights', {
            'phone': 0.3,
            'email': 0.3,
            'website': 0.2,
            'address': 0.2
        })
        
        # Validation rules
        self.validation_rules = {
            'phone': self._validate_phone,
            'email': self._validate_email,
            'website': self._validate_website,
            'address': self._validate_address,
            'business_name': self._validate_business_name
        }
    
    def validate_extraction_results(self, results: List[Dict], source: str = "unknown") -> List[Dict]:
        """
        Validate and score all extraction results.
        
        Args:
            results: List of extraction result dictionaries
            source: Source of the data (e.g., 'google_maps', 'linkedin')
            
        Returns:
            List of results with validation metadata added
        """
        validated_results = []
        
        for result in results:
            validation = self._validate_single_result(result, source)
            
            # Add validation metadata
            result['validation'] = validation.__dict__
            result['quality_score'] = validation.overall_score
            result['is_valid'] = validation.is_valid
            result['confidence_level'] = validation.confidence_level
            
            validated_results.append(result)
        
        return validated_results
    
    def _validate_single_result(self, result: Dict, source: str) -> ValidationResult:
        """Validate a single extraction result."""
        validation = ValidationResult(source=source)
        
        # Extract and validate contact information
        combined_text = self._combine_text_fields(result)
        contact_info = self.contact_extractor.extract_contact_info(combined_text, source)
        
        # Validate each field based on source type
        if source == 'google_places':
            validation = self._validate_google_places_result(result, contact_info, validation)
        elif source == 'google_maps_contacts':
            validation = self._validate_google_maps_result(result, contact_info, validation)
        elif source == 'linkedin_premium':
            validation = self._validate_linkedin_result(result, contact_info, validation)
        else:
            validation = self._validate_generic_result(result, contact_info, validation)
        
        # Calculate overall score
        if validation.field_scores:
            # Apply field weights
            weighted_score = 0.0
            total_weight = 0.0
            
            for field, score in validation.field_scores.items():
                weight = self.field_weights.get(field, 0.1)
                weighted_score += score * weight
                total_weight += weight
            
            validation.overall_score = weighted_score / total_weight if total_weight > 0 else 0.0
        
        # Determine confidence level and validity
        if validation.overall_score >= 0.8:
            validation.confidence_level = 'high'
        elif validation.overall_score >= 0.6:
            validation.confidence_level = 'medium'
        else:
            validation.confidence_level = 'low'
        
        validation.is_valid = validation.overall_score >= self.thresholds['minimum_confidence']
        
        # Generate recommendations
        validation.recommendations = self._generate_recommendations(validation, result)
        
        return validation
    
    def _validate_google_places_result(self, result: Dict, contact_info: ContactInfo, validation: ValidationResult) -> ValidationResult:
        """Validate Google Places specific result."""
        
        # Title validation
        title = result.get('title', '').strip()
        if title:
            if len(title) >= 3 and not title.isupper():
                validation.field_scores['title'] = 0.9
            else:
                validation.field_scores['title'] = 0.5
                validation.validation_issues.append("Title format questionable")
        else:
            validation.field_scores['title'] = 0.0
            validation.validation_issues.append("Missing title")
        
        # Phone validation using contact extractor
        if contact_info.phone:
            validation.field_scores['phone'] = min(contact_info.confidence_score * 2, 1.0)
        else:
            phone = result.get('phone', '').strip()
            if phone:
                phones = self.contact_extractor.extract_phone_numbers(phone)
                if phones:
                    validation.field_scores['phone'] = phones[0][1]
                else:
                    validation.field_scores['phone'] = 0.3
                    validation.validation_issues.append("Invalid phone format")
            else:
                validation.field_scores['phone'] = 0.0
        
        # Address validation
        address = result.get('address', '').strip()
        if address and len(address) > 10:
            validation.field_scores['address'] = 0.8
        else:
            validation.field_scores['address'] = 0.2
            validation.validation_issues.append("Address too short or missing")
        
        # Rating validation (Google-specific)
        rating = result.get('totalScore', 0)
        if isinstance(rating, (int, float)) and 0 <= rating <= 5:
            validation.field_scores['rating'] = 0.9
        else:
            validation.field_scores['rating'] = 0.5
        
        return validation
    
    def _validate_google_maps_result(self, result: Dict, contact_info: ContactInfo, validation: ValidationResult) -> ValidationResult:
        """Validate Google Maps specific result."""
        
        # Business name validation
        name = result.get('title', '').strip()
        if name:
            validation.field_scores['business_name'] = 0.8
        else:
            validation.field_scores['business_name'] = 0.0
            validation.validation_issues.append("Missing business name")
        
        # Enhanced contact validation using contact extractor
        if contact_info.phone:
            validation.field_scores['phone'] = contact_info.confidence_score
        else:
            validation.field_scores['phone'] = 0.0
        
        if contact_info.email:
            validation.field_scores['email'] = contact_info.confidence_score
        else:
            validation.field_scores['email'] = 0.0
        
        if contact_info.website:
            validation.field_scores['website'] = contact_info.confidence_score
        else:
            validation.field_scores['website'] = 0.0
        
        # Google Maps specific fields
        hours = result.get('hours', '').strip()
        if hours:
            validation.field_scores['hours'] = 0.7
        else:
            validation.field_scores['hours'] = 0.0
        
        return validation
    
    def _validate_linkedin_result(self, result: Dict, contact_info: ContactInfo, validation: ValidationResult) -> ValidationResult:
        """Validate LinkedIn specific result."""
        
        # Name validation
        full_name = result.get('fullName', '').strip()
        if full_name:
            name_parts = full_name.strip().split()
            if len(name_parts) >= 2 and all(len(part) >= 2 for part in name_parts):
                validation.field_scores['name'] = 0.9
            else:
                validation.field_scores['name'] = 0.4
                validation.validation_issues.append("Name format questionable")
        else:
            validation.field_scores['name'] = 0.0
            validation.validation_issues.append("Missing name")
        
        # Position validation
        position = result.get('position', '').strip()
        if position:
            # Check for executive positions
            executive_keywords = ['ceo', 'cfo', 'director', 'directeur', 'gérant', 'président', 'manager']
            is_executive = any(keyword in position.lower() for keyword in executive_keywords)
            validation.field_scores['position'] = 0.9 if is_executive else 0.7
        else:
            validation.field_scores['position'] = 0.0
            validation.validation_issues.append("Missing position")
        
        # Company validation
        company = result.get('companyName', '').strip()
        if company:
            validation.field_scores['company'] = 0.8
        else:
            validation.field_scores['company'] = 0.3
            validation.validation_issues.append("Missing company information")
        
        # LinkedIn profile URL validation
        profile_url = result.get('profileUrl', '').strip()
        if profile_url and 'linkedin.com' in profile_url:
            validation.field_scores['profile_url'] = 0.9
        else:
            validation.field_scores['profile_url'] = 0.2
            validation.validation_issues.append("Invalid LinkedIn profile URL")
        
        return validation
    
    def _validate_generic_result(self, result: Dict, contact_info: ContactInfo, validation: ValidationResult) -> ValidationResult:
        """Validate generic result when source is unknown."""
        
        # Use contact extractor for all contact fields
        validation.field_scores['phone'] = contact_info.confidence_score if contact_info.phone else 0.0
        validation.field_scores['email'] = contact_info.confidence_score if contact_info.email else 0.0
        validation.field_scores['website'] = contact_info.confidence_score if contact_info.website else 0.0
        
        # Basic field presence validation
        for field in ['title', 'name', 'address']:
            if field in result and result[field] and str(result[field]).strip():
                validation.field_scores[field] = 0.7
            else:
                validation.field_scores[field] = 0.0
        
        return validation
    
    def _validate_phone(self, phone: str) -> Tuple[float, List[str]]:
        """Validate phone number format and return score and issues."""
        if not phone:
            return 0.0, ["Missing phone number"]
        
        phones = self.contact_extractor.extract_phone_numbers(phone)
        if phones:
            return phones[0][1], []
        else:
            return 0.3, ["Invalid phone format"]
    
    def _validate_email(self, email: str) -> Tuple[float, List[str]]:
        """Validate email address and return score and issues."""
        if not email:
            return 0.0, ["Missing email address"]
        
        emails = self.contact_extractor.extract_emails(email)
        if emails:
            return emails[0][1], []
        else:
            return 0.2, ["Invalid email format"]
    
    def _validate_website(self, website: str) -> Tuple[float, List[str]]:
        """Validate website URL and return score and issues."""
        if not website:
            return 0.0, ["Missing website"]
        
        websites = self.contact_extractor.extract_websites(website)
        if websites:
            return websites[0][1], []
        else:
            return 0.2, ["Invalid website format"]
    
    def _validate_address(self, address: str) -> Tuple[float, List[str]]:
        """Validate address and return score and issues."""
        if not address:
            return 0.0, ["Missing address"]
        
        address = address.strip()
        if len(address) < 10:
            return 0.3, ["Address too short"]
        
        # Check for French postal codes
        if any(char.isdigit() for char in address):
            return 0.8, []
        else:
            return 0.5, ["Address may be incomplete"]
    
    def _validate_business_name(self, name: str) -> Tuple[float, List[str]]:
        """Validate business name and return score and issues."""
        if not name:
            return 0.0, ["Missing business name"]
        
        name = name.strip()
        if len(name) < 2:
            return 0.2, ["Business name too short"]
        
        if name.isupper():
            return 0.6, ["Business name in all caps"]
        
        return 0.8, []
    
    def _combine_text_fields(self, result: Dict) -> str:
        """Combine all text fields from a result for contact extraction."""
        text_fields = []
        
        # Common fields that might contain contact info
        fields_to_check = [
            'title', 'description', 'address', 'phone', 'email', 'website',
            'contact_info', 'details', 'about', 'hours', 'location'
        ]
        
        for field in fields_to_check:
            if field in result and result[field]:
                text_fields.append(str(result[field]))
        
        return ' '.join(text_fields)
    
    def _generate_recommendations(self, validation: ValidationResult, result: Dict) -> List[str]:
        """Generate recommendations based on validation results."""
        recommendations = []
        
        if validation.overall_score < 0.5:
            recommendations.append("Low quality data - consider manual verification")
        
        if 'phone' in validation.field_scores and validation.field_scores['phone'] == 0.0:
            recommendations.append("No phone number found - search additional sources")
        
        if 'email' in validation.field_scores and validation.field_scores['email'] == 0.0:
            recommendations.append("No email address found - check company website")
        
        if len(validation.validation_issues) > 3:
            recommendations.append("Multiple validation issues - manual review recommended")
        
        return recommendations
    
    def generate_quality_report(self, validated_results: List[Dict]) -> QualityReport:
        """Generate comprehensive quality report for validation results."""
        if not validated_results:
            return QualityReport()
        
        report = QualityReport()
        report.total_records = len(validated_results)
        
        # Count valid records
        valid_results = [r for r in validated_results if r.get('is_valid', False)]
        report.valid_records = len(valid_results)
        report.validation_rate = report.valid_records / report.total_records if report.total_records > 0 else 0.0
        
        # Calculate average score
        scores = [r.get('quality_score', 0.0) for r in validated_results]
        report.average_score = np.mean(scores) if scores else 0.0
        
        # Confidence distribution
        confidence_levels = [r.get('confidence_level', 'low') for r in validated_results]
        report.confidence_distribution = {
            'high': confidence_levels.count('high'),
            'medium': confidence_levels.count('medium'),
            'low': confidence_levels.count('low')
        }
        
        # Field coverage analysis
        report.field_coverage = self._calculate_field_coverage(validated_results)
        
        # Common issues analysis
        report.common_issues = self._analyze_common_issues(validated_results)
        
        # Generate recommendations
        report.recommendations = self._generate_report_recommendations(report)
        
        return report
    
    def _calculate_field_coverage(self, results: List[Dict]) -> Dict[str, float]:
        """Calculate coverage percentage for each field."""
        if not results:
            return {}
        
        field_coverage = {}
        fields_to_check = ['phone', 'email', 'website', 'address', 'title', 'name']
        
        for field in fields_to_check:
            count = 0
            for result in results:
                validation = result.get('validation', {})
                field_scores = validation.get('field_scores', {})
                if field in field_scores and field_scores[field] > 0.5:
                    count += 1
            
            field_coverage[field] = count / len(results) if results else 0.0
        
        return field_coverage
    
    def _analyze_common_issues(self, results: List[Dict]) -> List[Tuple[str, int]]:
        """Analyze most common validation issues."""
        issue_counts = {}
        
        for result in results:
            validation = result.get('validation', {})
            issues = validation.get('validation_issues', [])
            
            for issue in issues:
                issue_counts[issue] = issue_counts.get(issue, 0) + 1
        
        # Sort by frequency and return top 10
        sorted_issues = sorted(issue_counts.items(), key=lambda x: x[1], reverse=True)
        return sorted_issues[:10]
    
    def _generate_report_recommendations(self, report: QualityReport) -> List[str]:
        """Generate recommendations based on quality report."""
        recommendations = []
        
        if report.validation_rate < self.thresholds['validation_rate']:
            recommendations.append(f"Low validation rate ({report.validation_rate:.1%}) - review data sources")
        
        if report.field_coverage.get('phone', 0) < self.thresholds['phone_coverage']:
            recommendations.append("Low phone number coverage - consider additional phone search")
        
        if report.field_coverage.get('email', 0) < self.thresholds['email_coverage']:
            recommendations.append("Low email coverage - consider website scraping for contacts")
        
        if report.average_score < 0.6:
            recommendations.append("Low average quality score - review extraction methods")
        
        if report.confidence_distribution['high'] < report.total_records * 0.3:
            recommendations.append("Few high-confidence results - manual verification may be needed")
        
        return recommendations
    
    def filter_by_quality(self, results: List[Dict], min_score: float = None, 
                         min_confidence: str = None) -> List[Dict]:
        """
        Filter results based on quality criteria.
        
        Args:
            results: List of validated results
            min_score: Minimum quality score (0.0-1.0)
            min_confidence: Minimum confidence level ('low', 'medium', 'high')
            
        Returns:
            Filtered list of results
        """
        filtered_results = results.copy()
        
        if min_score is not None:
            filtered_results = [r for r in filtered_results if r.get('quality_score', 0.0) >= min_score]
        
        if min_confidence is not None:
            confidence_order = {'low': 0, 'medium': 1, 'high': 2}
            min_level = confidence_order.get(min_confidence, 0)
            filtered_results = [
                r for r in filtered_results 
                if confidence_order.get(r.get('confidence_level', 'low'), 0) >= min_level
            ]
        
        return filtered_results
    
    def export_quality_dashboard_data(self, report: QualityReport, output_path: str) -> str:
        """Export quality report data for dashboard visualization."""
        dashboard_data = {
            'summary': {
                'total_records': report.total_records,
                'valid_records': report.valid_records,
                'validation_rate': round(report.validation_rate * 100, 1),
                'average_score': round(report.average_score * 100, 1)
            },
            'confidence_distribution': report.confidence_distribution,
            'field_coverage': {k: round(v * 100, 1) for k, v in report.field_coverage.items()},
            'common_issues': report.common_issues[:5],  # Top 5 issues
            'recommendations': report.recommendations,
            'timestamp': report.timestamp,
            'thresholds': {k: round(v * 100, 1) for k, v in self.thresholds.items()}
        }
        
        io.write_text(output_path, json.dumps(dashboard_data, indent=2, ensure_ascii=False))
        
        return output_path


class GoogleMapsQualityController(QualityController):
    """Specialized quality controller for Google Maps extraction results."""
    
    def __init__(self, config: Optional[Dict] = None):
        """Initialize Google Maps specific quality controller."""
        super().__init__(config)
        
        # Google Maps specific thresholds
        self.thresholds.update({
            'business_name_coverage': 0.9,   # 90% should have business names
            'address_coverage': 0.8,         # 80% should have addresses
            'rating_coverage': 0.6,          # 60% should have ratings
        })
        
        # Google Maps specific field weights
        self.field_weights.update({
            'business_name': 0.25,
            'address': 0.25,
            'phone': 0.2,
            'email': 0.15,
            'website': 0.1,
            'rating': 0.05
        })
    
    def validate_google_maps_batch(self, results: List[Dict]) -> Tuple[List[Dict], QualityReport]:
        """Validate a batch of Google Maps results and generate report."""
        validated_results = self.validate_extraction_results(results, 'google_maps_contacts')
        report = self.generate_quality_report(validated_results)
        
        return validated_results, report


class LinkedInQualityController(QualityController):
    """Specialized quality controller for LinkedIn extraction results."""
    
    def __init__(self, config: Optional[Dict] = None):
        """Initialize LinkedIn specific quality controller."""
        super().__init__(config)
        
        # LinkedIn specific thresholds
        self.thresholds.update({
            'name_coverage': 0.95,           # 95% should have names
            'position_coverage': 0.9,        # 90% should have positions
            'company_coverage': 0.8,         # 80% should have company info
        })
        
        # LinkedIn specific field weights
        self.field_weights.update({
            'name': 0.3,
            'position': 0.3,
            'company': 0.2,
            'profile_url': 0.2
        })
    
    def validate_linkedin_batch(self, results: List[Dict]) -> Tuple[List[Dict], QualityReport]:
        """Validate a batch of LinkedIn results and generate report."""
        validated_results = self.validate_extraction_results(results, 'linkedin_premium')
        report = self.generate_quality_report(validated_results)
        
        return validated_results, report