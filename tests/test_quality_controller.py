"""Tests for quality controller module."""

import pytest
import tempfile
import json
from pathlib import Path
from unittest.mock import Mock, patch

from utils.quality_controller import (
    QualityController, GoogleMapsQualityController, LinkedInQualityController,
    ValidationResult, QualityReport
)


@pytest.fixture
def quality_controller():
    """Create a QualityController instance for testing."""
    config = {
        'thresholds': {
            'validation_rate': 0.8,
            'phone_coverage': 0.6,
            'email_coverage': 0.4,
            'minimum_confidence': 0.5,
        },
        'field_weights': {
            'phone': 0.3,
            'email': 0.3,
            'website': 0.2,
            'address': 0.2
        }
    }
    return QualityController(config)


@pytest.fixture
def google_maps_controller():
    """Create a GoogleMapsQualityController instance for testing."""
    return GoogleMapsQualityController()


@pytest.fixture
def linkedin_controller():
    """Create a LinkedInQualityController instance for testing."""
    return LinkedInQualityController()


def test_quality_controller_initialization(quality_controller):
    """Test QualityController initialization."""
    assert quality_controller.thresholds['validation_rate'] == 0.8
    assert quality_controller.field_weights['phone'] == 0.3
    assert 'phone' in quality_controller.validation_rules
    assert 'email' in quality_controller.validation_rules


def test_validate_google_places_result(google_maps_controller):
    """Test validation of Google Places results."""
    
    # High quality result
    good_result = {
        'title': 'Excellent Company SA',
        'phone': '+33 1 42 86 87 88',
        'address': '123 Avenue des Champs-Élysées, 75008 Paris',
        'totalScore': 4.5,
        'email': 'contact@excellent.fr'
    }
    
    validation = google_maps_controller._validate_single_result(good_result, 'google_places')
    
    assert validation.overall_score > 0.7
    assert validation.confidence_level in ['medium', 'high']
    assert validation.is_valid
    assert 'title' in validation.field_scores
    assert 'phone' in validation.field_scores


def test_validate_google_places_result_poor_quality(google_maps_controller):
    """Test validation of poor quality Google Places results."""
    
    # Poor quality result
    poor_result = {
        'title': '',  # Missing title
        'phone': '123',  # Invalid phone
        'address': 'Rue',  # Too short
        'totalScore': 0
    }
    
    validation = google_maps_controller._validate_single_result(poor_result, 'google_places')
    
    assert validation.overall_score < 0.5
    assert validation.confidence_level == 'low'
    assert not validation.is_valid
    assert len(validation.validation_issues) > 0


def test_validate_linkedin_result(linkedin_controller):
    """Test validation of LinkedIn results."""
    
    # Good LinkedIn result
    good_result = {
        'fullName': 'Jean Dupont',
        'position': 'CEO',
        'companyName': 'Test Company SA',
        'profileUrl': 'https://linkedin.com/in/jean-dupont'
    }
    
    validation = linkedin_controller._validate_single_result(good_result, 'linkedin_premium')
    
    assert validation.overall_score > 0.6
    assert validation.is_valid
    assert 'name' in validation.field_scores
    assert 'position' in validation.field_scores
    assert 'company' in validation.field_scores


def test_validate_linkedin_result_poor_quality(linkedin_controller):
    """Test validation of poor quality LinkedIn results."""
    
    # Poor LinkedIn result
    poor_result = {
        'fullName': 'A',  # Too short
        'position': '',  # Missing
        'companyName': '',  # Missing
        'profileUrl': 'invalid-url'
    }
    
    validation = linkedin_controller._validate_single_result(poor_result, 'linkedin_premium')
    
    assert validation.overall_score < 0.5
    assert not validation.is_valid
    assert len(validation.validation_issues) > 0


def test_validate_extraction_results(quality_controller):
    """Test batch validation of extraction results."""
    
    results = [
        {
            'title': 'Company A',
            'phone': '+33142868788',
            'email': 'contact@company-a.fr'
        },
        {
            'title': 'Company B',
            'phone': 'invalid',
            'email': ''
        }
    ]
    
    validated_results = quality_controller.validate_extraction_results(results, 'google_maps')
    
    assert len(validated_results) == 2
    assert all('validation' in result for result in validated_results)
    assert all('quality_score' in result for result in validated_results)
    assert all('is_valid' in result for result in validated_results)
    
    # First result should have higher quality than second
    assert validated_results[0]['quality_score'] > validated_results[1]['quality_score']


def test_generate_quality_report(quality_controller):
    """Test quality report generation."""
    
    validated_results = [
        {
            'validation': {
                'overall_score': 0.8,
                'confidence_level': 'high',
                'is_valid': True,
                'validation_issues': [],
                'field_scores': {'phone': 0.8, 'email': 0.7}
            },
            'quality_score': 0.8,
            'is_valid': True,
            'confidence_level': 'high'
        },
        {
            'validation': {
                'overall_score': 0.3,
                'confidence_level': 'low',
                'is_valid': False,
                'validation_issues': ['Missing phone', 'Invalid email'],
                'field_scores': {'phone': 0.0, 'email': 0.2}
            },
            'quality_score': 0.3,
            'is_valid': False,
            'confidence_level': 'low'
        }
    ]
    
    report = quality_controller.generate_quality_report(validated_results)
    
    assert report.total_records == 2
    assert report.valid_records == 1
    assert report.validation_rate == 0.5
    assert report.average_score > 0.0
    assert 'high' in report.confidence_distribution
    assert 'low' in report.confidence_distribution


def test_filter_by_quality(quality_controller):
    """Test quality-based filtering."""
    
    results = [
        {'quality_score': 0.8, 'confidence_level': 'high', 'title': 'Good Company'},
        {'quality_score': 0.3, 'confidence_level': 'low', 'title': 'Poor Company'},
        {'quality_score': 0.6, 'confidence_level': 'medium', 'title': 'Medium Company'},
    ]
    
    # Filter by minimum score
    filtered_by_score = quality_controller.filter_by_quality(results, min_score=0.5)
    assert len(filtered_by_score) == 2  # Should keep 0.8 and 0.6
    
    # Filter by minimum confidence
    filtered_by_confidence = quality_controller.filter_by_quality(results, min_confidence='medium')
    assert len(filtered_by_confidence) == 2  # Should keep 'high' and 'medium'
    
    # Filter by both
    filtered_by_both = quality_controller.filter_by_quality(
        results, min_score=0.7, min_confidence='high'
    )
    assert len(filtered_by_both) == 1  # Should keep only 'Good Company'


def test_export_quality_dashboard_data(quality_controller):
    """Test quality dashboard data export."""
    
    report = QualityReport(
        total_records=10,
        valid_records=8,
        validation_rate=0.8,
        average_score=0.75,
        confidence_distribution={'high': 5, 'medium': 3, 'low': 2},
        field_coverage={'phone': 0.7, 'email': 0.5},
        common_issues=[('Missing phone', 3), ('Invalid email', 2)],
        recommendations=['Improve phone coverage', 'Validate email formats']
    )
    
    with tempfile.TemporaryDirectory() as tmpdir:
        output_path = Path(tmpdir) / "dashboard.json"
        quality_controller.export_quality_dashboard_data(report, str(output_path))
        
        assert output_path.exists()
        
        with open(output_path, 'r') as f:
            dashboard_data = json.load(f)
        
        assert 'summary' in dashboard_data
        assert 'confidence_distribution' in dashboard_data
        assert 'field_coverage' in dashboard_data
        assert dashboard_data['summary']['total_records'] == 10
        assert dashboard_data['summary']['validation_rate'] == 80.0  # Converted to percentage


def test_phone_validation(quality_controller):
    """Test phone number validation."""
    
    # Valid phone
    score, issues = quality_controller._validate_phone('+33142868788')
    assert score > 0.5
    assert len(issues) == 0
    
    # Invalid phone
    score, issues = quality_controller._validate_phone('123')
    assert score < 0.5
    assert len(issues) > 0
    
    # Missing phone
    score, issues = quality_controller._validate_phone('')
    assert score == 0.0
    assert 'Missing phone number' in issues


def test_email_validation(quality_controller):
    """Test email validation."""
    
    # Valid email
    score, issues = quality_controller._validate_email('contact@test.fr')
    assert score > 0.5
    assert len(issues) == 0
    
    # Invalid email
    score, issues = quality_controller._validate_email('invalid-email')
    assert score < 0.5
    assert len(issues) > 0
    
    # Missing email
    score, issues = quality_controller._validate_email('')
    assert score == 0.0
    assert 'Missing email address' in issues


def test_website_validation(quality_controller):
    """Test website validation."""
    
    # Valid website
    score, issues = quality_controller._validate_website('https://www.test.fr')
    assert score > 0.5
    assert len(issues) == 0
    
    # Invalid website
    score, issues = quality_controller._validate_website('not-a-url')
    assert score < 0.5
    assert len(issues) > 0
    
    # Missing website
    score, issues = quality_controller._validate_website('')
    assert score == 0.0
    assert 'Missing website' in issues


def test_address_validation(quality_controller):
    """Test address validation."""
    
    # Good address
    score, issues = quality_controller._validate_address('123 Avenue des Champs-Élysées, 75008 Paris')
    assert score > 0.5
    assert len(issues) == 0
    
    # Short address
    score, issues = quality_controller._validate_address('Rue')
    assert score < 0.5
    assert 'Address too short' in issues
    
    # Missing address
    score, issues = quality_controller._validate_address('')
    assert score == 0.0
    assert 'Missing address' in issues


def test_business_name_validation(quality_controller):
    """Test business name validation."""
    
    # Good business name
    score, issues = quality_controller._validate_business_name('Excellent Company SA')
    assert score > 0.5
    assert len(issues) == 0
    
    # All caps (lower score)
    score, issues = quality_controller._validate_business_name('COMPANY NAME')
    assert score == 0.6
    assert 'Business name in all caps' in issues
    
    # Too short
    score, issues = quality_controller._validate_business_name('A')
    assert score < 0.5
    assert 'Business name too short' in issues


def test_combine_text_fields(quality_controller):
    """Test text field combination for contact extraction."""
    
    result = {
        'title': 'Test Company',
        'phone': '+33142868788',
        'email': 'contact@test.fr',
        'description': 'Great company',
        'other_field': 'Should not be included'
    }
    
    combined_text = quality_controller._combine_text_fields(result)
    
    assert 'Test Company' in combined_text
    assert '+33142868788' in combined_text
    assert 'contact@test.fr' in combined_text
    assert 'Great company' in combined_text
    assert 'Should not be included' not in combined_text


def test_generate_recommendations(quality_controller):
    """Test recommendation generation."""
    
    # Low quality validation
    validation = ValidationResult(
        overall_score=0.3,
        field_scores={'phone': 0.0, 'email': 0.0},
        validation_issues=['Issue 1', 'Issue 2', 'Issue 3', 'Issue 4']
    )
    result = {'title': 'Test'}
    
    recommendations = quality_controller._generate_recommendations(validation, result)
    
    assert len(recommendations) > 0
    assert any('Low quality data' in rec for rec in recommendations)
    assert any('No phone number found' in rec for rec in recommendations)
    assert any('No email address found' in rec for rec in recommendations)


def test_google_maps_specialized_controller():
    """Test Google Maps specialized controller."""
    
    controller = GoogleMapsQualityController()
    
    # Should have Google Maps specific thresholds
    assert 'business_name_coverage' in controller.thresholds
    assert 'address_coverage' in controller.thresholds
    
    # Should have Google Maps specific field weights
    assert 'business_name' in controller.field_weights
    assert 'rating' in controller.field_weights


def test_linkedin_specialized_controller():
    """Test LinkedIn specialized controller."""
    
    controller = LinkedInQualityController()
    
    # Should have LinkedIn specific thresholds
    assert 'name_coverage' in controller.thresholds
    assert 'position_coverage' in controller.thresholds
    
    # Should have LinkedIn specific field weights
    assert 'name' in controller.field_weights
    assert 'position' in controller.field_weights


def test_validation_result_dataclass():
    """Test ValidationResult dataclass."""
    
    # Test default initialization
    validation = ValidationResult()
    assert validation.overall_score == 0.0
    assert validation.field_scores == {}
    assert validation.validation_issues == []
    assert validation.confidence_level == "low"
    assert not validation.is_valid
    
    # Test with values
    validation = ValidationResult(
        overall_score=0.8,
        confidence_level="high",
        is_valid=True
    )
    assert validation.overall_score == 0.8
    assert validation.confidence_level == "high"
    assert validation.is_valid


def test_quality_report_dataclass():
    """Test QualityReport dataclass."""
    
    # Test default initialization
    report = QualityReport()
    assert report.total_records == 0
    assert report.valid_records == 0
    assert report.validation_rate == 0.0
    assert report.confidence_distribution == {}
    
    # Test with values
    report = QualityReport(
        total_records=100,
        valid_records=80,
        validation_rate=0.8
    )
    assert report.total_records == 100
    assert report.valid_records == 80
    assert report.validation_rate == 0.8


def test_field_coverage_calculation(quality_controller):
    """Test field coverage calculation."""
    
    results = [
        {
            'validation': {
                'field_scores': {'phone': 0.8, 'email': 0.7, 'website': 0.0}
            }
        },
        {
            'validation': {
                'field_scores': {'phone': 0.0, 'email': 0.6, 'website': 0.8}
            }
        }
    ]
    
    coverage = quality_controller._calculate_field_coverage(results)
    
    assert coverage['phone'] == 0.5  # 1 out of 2 records have good phone
    assert coverage['email'] == 1.0  # 2 out of 2 records have good email
    assert coverage['website'] == 0.5  # 1 out of 2 records have good website


def test_common_issues_analysis(quality_controller):
    """Test common issues analysis."""
    
    results = [
        {
            'validation': {
                'validation_issues': ['Missing phone', 'Invalid email']
            }
        },
        {
            'validation': {
                'validation_issues': ['Missing phone', 'Short address']
            }
        },
        {
            'validation': {
                'validation_issues': ['Invalid email']
            }
        }
    ]
    
    common_issues = quality_controller._analyze_common_issues(results)
    
    # Should be sorted by frequency
    assert common_issues[0][0] == 'Missing phone'  # Appears 2 times
    assert common_issues[0][1] == 2
    assert common_issues[1][0] == 'Invalid email'  # Appears 2 times
    assert common_issues[1][1] == 2


@patch('utils.contact_extractor.requests.head')
def test_website_accessibility_check(mock_head, quality_controller):
    """Test website accessibility checking."""
    
    # Mock successful response
    mock_response = Mock()
    mock_response.status_code = 200
    mock_head.return_value = mock_response
    
    # This would normally be tested through the contact extractor
    # but we can test the validation logic
    score, issues = quality_controller._validate_website('https://www.test.fr')
    assert score > 0.0  # Should have some base confidence
    assert len(issues) == 0