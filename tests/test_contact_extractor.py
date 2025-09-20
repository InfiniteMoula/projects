"""Tests for contact extractor module."""

import pytest
from utils.contact_extractor import ContactExtractor, ContactInfo


@pytest.fixture
def extractor():
    """Create a ContactExtractor instance for testing."""
    return ContactExtractor()


def test_extract_french_phone_numbers(extractor):
    """Test French phone number extraction."""
    
    # Test various French phone formats
    test_cases = [
        ("+33 1 42 86 87 88", ["+33142868788"]),
        ("01 42 86 87 88", ["+33142868788"]),
        ("0142868788", ["+33142868788"]),
        ("+33142868788", ["+33142868788"]),
        ("Tel: 01.42.86.87.88", ["+33142868788"]),
        ("Téléphone: +33-1-42-86-87-88", ["+33142868788"]),
        ("Invalid phone: 123", []),  # Too short
        ("Mobile: 06 12 34 56 78", ["+33612345678"]),  # Mobile number
    ]
    
    for text, expected in test_cases:
        phones = extractor.extract_phone_numbers(text)
        phone_numbers = [phone for phone, _ in phones]
        assert phone_numbers == expected, f"Failed for text: {text}"


def test_extract_emails(extractor):
    """Test email extraction."""
    
    test_cases = [
        ("contact@example.fr", ["contact@example.fr"]),
        ("Contact: info@company.com", ["info@company.com"]),
        ("Email: admin@test.org", ["admin@test.org"]),
        ("Multiple emails: contact@test.fr, info@test.fr", ["contact@test.fr", "info@test.fr"]),
        ("Invalid email", []),
        ("Partial @test.com", []),
    ]
    
    for text, expected in test_cases:
        emails = extractor.extract_emails(text)
        email_addresses = [email for email, _ in emails]
        # Sort for comparison since order might vary
        assert sorted(email_addresses) == sorted(expected), f"Failed for text: {text}"


def test_extract_websites(extractor):
    """Test website URL extraction."""
    
    test_cases = [
        ("https://www.example.com", ["https://www.example.com"]),
        ("www.example.fr", ["https://www.example.fr"]),
        ("Visit our site: http://company.org", ["http://company.org"]),
        ("Website: www.example.com", ["https://www.example.com"]),  # More realistic than bare domain
        ("Not a website", []),
    ]
    
    for text, expected in test_cases:
        websites = extractor.extract_websites(text)
        website_urls = [website for website, _ in websites]
        assert website_urls == expected, f"Failed for text: {text}"


def test_phone_confidence_scoring(extractor):
    """Test phone number confidence scoring."""
    
    test_cases = [
        ("+33 1 42 86 87 88", 0.8),  # Well formatted international
        ("01 42 86 87 88", 0.7),     # National format
        ("0142868788", 0.7),         # No formatting
        ("+33 6 12 34 56 78", 0.8),  # Mobile number (higher confidence)
    ]
    
    for text, min_expected_confidence in test_cases:
        phones = extractor.extract_phone_numbers(text)
        if phones:
            _, confidence = phones[0]
            assert confidence >= min_expected_confidence, f"Low confidence for: {text} (got {confidence})"


def test_email_confidence_scoring(extractor):
    """Test email confidence scoring."""
    
    test_cases = [
        ("contact@company.fr", 0.6),      # Business email
        ("info@business.com", 0.6),       # Business email
        ("john@gmail.com", 0.3),          # Generic domain (lower confidence)
        ("admin@corporate.org", 0.6),     # Business-like
    ]
    
    for text, min_expected_confidence in test_cases:
        emails = extractor.extract_emails(text)
        if emails:
            _, confidence = emails[0]
            assert confidence >= min_expected_confidence, f"Low confidence for: {text} (got {confidence})"


def test_extract_contact_info_comprehensive(extractor):
    """Test comprehensive contact information extraction."""
    
    text = """
    Entreprise Test SARL
    Adresse: 123 Rue de la Paix, 75001 Paris
    Téléphone: +33 1 42 86 87 88
    Email: contact@test-entreprise.fr
    Site web: https://www.test-entreprise.fr
    """
    
    contact_info = extractor.extract_contact_info(text, source="website")
    
    assert contact_info.phone == "+33142868788"
    assert contact_info.email == "contact@test-entreprise.fr"
    assert contact_info.website == "https://www.test-entreprise.fr"
    assert contact_info.confidence_score > 0.5
    assert len(contact_info.validation_issues) == 0


def test_extract_contact_info_partial(extractor):
    """Test contact extraction with partial information."""
    
    text = "Société ABC - Email: info@abc.fr"
    
    contact_info = extractor.extract_contact_info(text, source="search_result")
    
    assert contact_info.phone is None
    assert contact_info.email == "info@abc.fr"
    assert contact_info.website is None
    assert contact_info.confidence_score > 0.0
    assert "No contact information found" not in contact_info.validation_issues


def test_extract_contact_info_empty(extractor):
    """Test contact extraction with no information."""
    
    contact_info = extractor.extract_contact_info("", source="unknown")
    
    assert contact_info.phone is None
    assert contact_info.email is None
    assert contact_info.website is None
    assert contact_info.confidence_score == 0.0


def test_source_confidence_adjustment(extractor):
    """Test that source type affects confidence scoring."""
    
    text = "Contact: +33 1 42 86 87 88, info@test.fr"
    
    google_maps_contact = extractor.extract_contact_info(text, source="google_maps")
    unknown_contact = extractor.extract_contact_info(text, source="unknown")
    
    # Google Maps source should have higher confidence than unknown
    assert google_maps_contact.confidence_score > unknown_contact.confidence_score


def test_validate_contact_info(extractor):
    """Test deep validation of contact information."""
    
    contact_info = ContactInfo(
        phone="+33142868788",
        email="contact@test.fr",
        website="https://www.test.fr",
        confidence_score=0.8
    )
    
    validation_result = extractor.validate_contact_info(contact_info)
    
    assert "original_confidence" in validation_result
    assert "enhanced_confidence" in validation_result
    assert "validation_details" in validation_result
    assert validation_result["original_confidence"] == 0.8


def test_phone_normalization(extractor):
    """Test phone number normalization."""
    
    test_cases = [
        ("01 42 86 87 88", "+33142868788"),
        ("0142868788", "+33142868788"),
        ("+33 1 42 86 87 88", "+33142868788"),
        ("01.42.86.87.88", "+33142868788"),
        ("01-42-86-87-88", "+33142868788"),
    ]
    
    for input_phone, expected in test_cases:
        normalized = extractor._normalize_phone(input_phone)
        assert normalized == expected, f"Failed normalization: {input_phone} -> {normalized}"


def test_website_normalization(extractor):
    """Test website URL normalization."""
    
    test_cases = [
        ("www.example.com", "https://www.example.com"),
        ("example.com", "https://www.example.com"),
        ("https://example.com", "https://example.com"),
        ("http://www.example.com", "http://www.example.com"),
    ]
    
    for input_url, expected in test_cases:
        normalized = extractor._normalize_website(input_url)
        assert normalized == expected, f"Failed normalization: {input_url} -> {normalized}"


def test_french_phone_validation(extractor):
    """Test French phone number format validation."""
    
    valid_phones = [
        "+33123456789",  # Landline
        "+33612345678",  # Mobile
        "+33712345678",  # Mobile
    ]
    
    invalid_phones = [
        "+33012345678",  # Invalid area code (0)
        "+3312345678",   # Too short
        "+331234567890", # Too long
        "123456789",     # No country code
    ]
    
    for phone in valid_phones:
        assert extractor._validate_phone_format(phone), f"Should be valid: {phone}"
    
    for phone in invalid_phones:
        assert not extractor._validate_phone_format(phone), f"Should be invalid: {phone}"


def test_multiple_phone_numbers(extractor):
    """Test extraction of multiple phone numbers with confidence."""
    
    text = "Tel: 01 42 86 87 88, Mobile: 06 12 34 56 78, Fax: 01 42 86 87 89"
    
    phones = extractor.extract_phone_numbers(text)
    
    assert len(phones) == 3
    phone_numbers = [phone for phone, _ in phones]
    assert "+33142868788" in phone_numbers
    assert "+33612345678" in phone_numbers
    assert "+33142868789" in phone_numbers


def test_email_domain_confidence(extractor):
    """Test that business domains get different confidence scores."""
    
    business_email = extractor.extract_emails("contact@company.fr")
    generic_email = extractor.extract_emails("user@gmail.com")
    
    if business_email and generic_email:
        business_confidence = business_email[0][1]
        generic_confidence = generic_email[0][1]
        
        # Business domain should have higher confidence than generic
        assert business_confidence > generic_confidence


def test_contact_info_dataclass():
    """Test ContactInfo dataclass functionality."""
    
    # Test default initialization
    contact = ContactInfo()
    assert contact.phone is None
    assert contact.email is None
    assert contact.website is None
    assert contact.confidence_score == 0.0
    assert contact.validation_issues == []
    
    # Test with values
    contact = ContactInfo(
        phone="+33123456789",
        email="test@example.com",
        confidence_score=0.8,
        validation_issues=["test issue"]
    )
    assert contact.phone == "+33123456789"
    assert contact.email == "test@example.com"
    assert contact.confidence_score == 0.8
    assert "test issue" in contact.validation_issues