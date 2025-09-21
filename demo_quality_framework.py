#!/usr/bin/env python3
"""
Quality Control Framework Demonstration

This script demonstrates the key features of the Quality Control Framework
implemented for Week 3-4 of the Apify automation roadmap.
"""

import json
from pathlib import Path
import pandas as pd

from utils import io
from utils.contact_extractor import ContactExtractor
from utils.quality_controller import QualityController, GoogleMapsQualityController, LinkedInQualityController


def demo_contact_extraction():
    """Demonstrate contact extraction capabilities."""
    print("=" * 60)
    print("CONTACT EXTRACTION DEMONSTRATION")
    print("=" * 60)
    
    extractor = ContactExtractor()
    
    # Test cases with various contact information
    test_cases = [
        {
            "name": "High Quality Google Maps Result",
            "text": """
            Entreprise Excellence SARL
            123 Avenue des Champs-Élysées, 75008 Paris
            Téléphone: +33 1 42 86 87 88
            Email: contact@excellence.fr
            Site web: https://www.excellence.fr
            Horaires: Lun-Ven 9h-18h
            """,
            "source": "google_maps"
        },
        {
            "name": "Medium Quality Website Result",
            "text": """
            Société ABC
            Tel: 01.42.86.87.89
            Contactez-nous: info@abc-company.fr
            www.abc-company.fr
            """,
            "source": "website"
        },
        {
            "name": "Low Quality Search Result",
            "text": """
            Company XYZ
            Email: admin@gmail.com
            Phone: 123456
            """,
            "source": "search_result"
        }
    ]
    
    for i, test_case in enumerate(test_cases, 1):
        print(f"\n{i}. {test_case['name']}")
        print("-" * 40)
        
        contact_info = extractor.extract_contact_info(test_case['text'], test_case['source'])
        
        print(f"Phone: {contact_info.phone}")
        print(f"Email: {contact_info.email}")
        print(f"Website: {contact_info.website}")
        print(f"Confidence Score: {contact_info.confidence_score:.2f}")
        print(f"Issues: {'; '.join(contact_info.validation_issues) if contact_info.validation_issues else 'None'}")
        
        # Deep validation
        validation = extractor.validate_contact_info(contact_info)
        print(f"Enhanced Confidence: {validation['enhanced_confidence']:.2f}")
        print(f"Recommendations: {'; '.join(validation['recommendations']) if validation['recommendations'] else 'None'}")


def demo_quality_validation():
    """Demonstrate quality validation for different data sources."""
    print("\n" + "=" * 60)
    print("QUALITY VALIDATION DEMONSTRATION")
    print("=" * 60)
    
    # Sample extraction results for different sources
    google_maps_results = [
        {
            'title': 'Excellent Company SA',
            'phone': '+33 1 42 86 87 88',
            'email': 'contact@excellent.fr',
            'address': '123 Avenue des Champs-Élysées, 75008 Paris',
            'totalScore': 4.5,
            'website': 'https://www.excellent.fr'
        },
        {
            'title': 'Poor Company',
            'phone': '123',
            'email': '',
            'address': 'Rue',
            'totalScore': 1.0,
            'website': ''
        }
    ]
    
    linkedin_results = [
        {
            'fullName': 'Jean Dupont',
            'position': 'CEO',
            'companyName': 'Excellent Company SA',
            'profileUrl': 'https://linkedin.com/in/jean-dupont'
        },
        {
            'fullName': 'A B',
            'position': '',
            'companyName': '',
            'profileUrl': 'invalid-url'
        }
    ]
    
    # Test Google Maps validation
    print("\n1. Google Maps Quality Validation")
    print("-" * 40)
    
    google_controller = GoogleMapsQualityController()
    validated_google = google_controller.validate_extraction_results(google_maps_results, 'google_maps_contacts')
    google_report = google_controller.generate_quality_report(validated_google)
    
    print(f"Total Records: {google_report.total_records}")
    print(f"Valid Records: {google_report.valid_records}")
    print(f"Validation Rate: {google_report.validation_rate:.1%}")
    print(f"Average Score: {google_report.average_score:.2f}")
    print(f"Confidence Distribution: {google_report.confidence_distribution}")
    
    # Test LinkedIn validation
    print("\n2. LinkedIn Quality Validation")
    print("-" * 40)
    
    linkedin_controller = LinkedInQualityController()
    validated_linkedin = linkedin_controller.validate_extraction_results(linkedin_results, 'linkedin_premium')
    linkedin_report = linkedin_controller.generate_quality_report(validated_linkedin)
    
    print(f"Total Records: {linkedin_report.total_records}")
    print(f"Valid Records: {linkedin_report.valid_records}")
    print(f"Validation Rate: {linkedin_report.validation_rate:.1%}")
    print(f"Average Score: {linkedin_report.average_score:.2f}")
    print(f"Confidence Distribution: {linkedin_report.confidence_distribution}")
    
    return validated_google, validated_linkedin, google_report, linkedin_report


def demo_quality_filtering():
    """Demonstrate automated quality filtering."""
    print("\n" + "=" * 60)
    print("AUTOMATED QUALITY FILTERING DEMONSTRATION")
    print("=" * 60)
    
    controller = QualityController()
    
    # Sample results with varying quality
    mixed_results = [
        {'title': 'High Quality Co', 'quality_score': 0.85, 'confidence_level': 'high'},
        {'title': 'Medium Quality Co', 'quality_score': 0.65, 'confidence_level': 'medium'},
        {'title': 'Low Quality Co', 'quality_score': 0.35, 'confidence_level': 'low'},
        {'title': 'Another Low Co', 'quality_score': 0.25, 'confidence_level': 'low'},
    ]
    
    print(f"Original Results: {len(mixed_results)} records")
    
    # Filter by minimum score
    high_quality = controller.filter_by_quality(mixed_results, min_score=0.7)
    print(f"High Quality (>= 0.7): {len(high_quality)} records")
    for result in high_quality:
        print(f"  - {result['title']} (score: {result['quality_score']})")
    
    # Filter by confidence level
    medium_plus = controller.filter_by_quality(mixed_results, min_confidence='medium')
    print(f"Medium+ Confidence: {len(medium_plus)} records")
    for result in medium_plus:
        print(f"  - {result['title']} ({result['confidence_level']})")
    
    # Combined filtering
    combined = controller.filter_by_quality(mixed_results, min_score=0.6, min_confidence='medium')
    print(f"Combined Filter (score >= 0.6 AND confidence >= medium): {len(combined)} records")
    for result in combined:
        print(f"  - {result['title']} (score: {result['quality_score']}, confidence: {result['confidence_level']})")


def demo_dashboard_export():
    """Demonstrate quality dashboard data export."""
    print("\n" + "=" * 60)
    print("QUALITY DASHBOARD EXPORT DEMONSTRATION")
    print("=" * 60)
    
    # Use results from previous demo
    controller = GoogleMapsQualityController()
    
    # Sample report data
    from utils.quality_controller import QualityReport
    sample_report = QualityReport(
        total_records=100,
        valid_records=75,
        validation_rate=0.75,
        average_score=0.68,
        confidence_distribution={'high': 25, 'medium': 50, 'low': 25},
        field_coverage={'phone': 0.8, 'email': 0.6, 'website': 0.4, 'address': 0.9},
        common_issues=[('Missing phone', 20), ('Invalid email', 15), ('Short address', 10)],
        recommendations=['Improve phone coverage', 'Validate email formats', 'Review extraction methods']
    )
    
    # Export dashboard data
    dashboard_path = "/tmp/demo_quality_dashboard.json"
    controller.export_quality_dashboard_data(sample_report, dashboard_path)
    
    print(f"Dashboard data exported to: {dashboard_path}")
    
    # Show dashboard data structure
    dashboard_data = json.loads(io.read_text(dashboard_path))
    
    print("\nDashboard Data Structure:")
    print(f"  Summary: {dashboard_data['summary']}")
    print(f"  Field Coverage: {dashboard_data['field_coverage']}")
    print(f"  Top Issues: {dashboard_data['common_issues']}")
    print(f"  Recommendations: {dashboard_data['recommendations']}")
    
    return dashboard_path


def demo_integration_workflow():
    """Demonstrate the complete integration workflow."""
    print("\n" + "=" * 60)
    print("COMPLETE INTEGRATION WORKFLOW DEMONSTRATION")
    print("=" * 60)
    
    # Simulate Apify extraction results
    sample_df = pd.DataFrame({
        'adresse': [
            '123 Rue de la Paix, 75001 Paris',
            '456 Avenue de Lyon, 69000 Lyon',
            'Incomplete Address'
        ],
        'denomination': [
            'Excellence Company SA',
            'Medium Quality Co',
            'Poor Co'
        ],
        'apify_business_names': [
            'Excellence Company SA',
            'Medium Quality Co', 
            'Poor Co'
        ],
        'apify_phones': [
            '+33 1 42 86 87 88',
            '04 72 00 00 00',
            'invalid'
        ],
        'apify_emails': [
            'contact@excellence.fr',
            'info@medium.com',
            ''
        ]
    })
    
    print("Sample Apify Results DataFrame:")
    print(sample_df.to_string(index=False))
    
    # Simulate quality validation integration
    # Note: In production, this would use _add_quality_scores_to_dataframe from api.apify_agents
    # For demo purposes, we'll simulate the enhancement manually
    
    google_controller = GoogleMapsQualityController()
    linkedin_controller = LinkedInQualityController()
    
    # Mock some results
    google_results = [
        {
            'title': 'Excellence Company SA',
            'searchString': '123 Rue de la Paix, 75001 Paris',
            'phone': '+33 1 42 86 87 88',
            'email': 'contact@excellence.fr',
            'totalScore': 4.8
        }
    ]
    
    linkedin_results = [
        {
            'fullName': 'Jean Dupont',
            'position': 'CEO',
            'companyName': 'Excellence Company SA',
            'profileUrl': 'https://linkedin.com/in/jean-dupont'
        }
    ]
    
    # Simulate quality scoring (manual for demo)
    enhanced_df = sample_df.copy()
    enhanced_df['google_maps_quality_score'] = [0.75, 0.65, 0.25]
    enhanced_df['linkedin_quality_score'] = [0.85, 0.0, 0.0]
    enhanced_df['overall_quality_score'] = [0.78, 0.46, 0.18]
    enhanced_df['quality_confidence_level'] = ['medium', 'low', 'low']
    
    print("\nEnhanced DataFrame with Quality Scores:")
    quality_columns = [
        'google_maps_quality_score',
        'linkedin_quality_score', 
        'overall_quality_score',
        'quality_confidence_level'
    ]
    
    print(enhanced_df[['denomination'] + quality_columns].to_string(index=False))
    
    # Show filtering example
    high_quality_df = enhanced_df[enhanced_df['overall_quality_score'] >= 0.5]
    print(f"\nHigh Quality Records (score >= 0.5): {len(high_quality_df)}/{len(enhanced_df)}")
    
    return enhanced_df


def main():
    """Run all demonstrations."""
    print("QUALITY CONTROL FRAMEWORK DEMONSTRATION")
    print("Implementation for Week 3-4: Quality Control Framework")
    print("=" * 80)
    
    try:
        # Run all demonstrations
        demo_contact_extraction()
        validated_google, validated_linkedin, google_report, linkedin_report = demo_quality_validation()
        demo_quality_filtering()
        dashboard_path = demo_dashboard_export()
        enhanced_df = demo_integration_workflow()
        
        print("\n" + "=" * 80)
        print("DEMONSTRATION SUMMARY")
        print("=" * 80)
        print("[OK] Contact extraction with confidence scoring")
        print("[OK] Quality validation for Google Maps and LinkedIn results")
        print("[OK] Automated quality filtering")
        print("[OK] Quality dashboard data export")
        print("[OK] Complete integration workflow")
        print(f"[OK] Dashboard data available at: {dashboard_path}")
        print("\nKey Features Demonstrated:")
        print("- French phone number extraction and validation")
        print("- Email and website validation with confidence scoring")
        print("- Source-specific quality validation")
        print("- Automated quality reporting")
        print("- Quality-based filtering and recommendations")
        print("- Dashboard data export for monitoring")
        print("- Integration with existing Apify workflow")
        
    except Exception as e:
        print(f"Error during demonstration: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
