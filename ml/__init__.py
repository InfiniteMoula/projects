"""
Machine Learning package for extraction enhancement and optimization.

This package provides ML-based components for:
- Extraction pattern learning and confidence modeling
- Address success prediction and classification  
- Automated parameter tuning and optimization
"""

from .extraction_models import (
    ExtractionFeatures,
    ExtractionResult,
    FeatureExtractor,
    ExtractionPatternLearner,
    create_extraction_learner
)

from .address_classifier import (
    AddressFeatures,
    AddressPrediction,
    AddressFeatureExtractor,
    AddressSuccessClassifier,
    create_address_classifier
)

from .ml_optimizer import (
    OptimizationResult,
    PerformanceMetrics,
    OptimizationHistory,
    ParameterSpace,
    MLParameterOptimizer,
    create_ml_optimizer
)

__all__ = [
    # Extraction models
    'ExtractionFeatures',
    'ExtractionResult', 
    'FeatureExtractor',
    'ExtractionPatternLearner',
    'create_extraction_learner',
    
    # Address classifier
    'AddressFeatures',
    'AddressPrediction',
    'AddressFeatureExtractor', 
    'AddressSuccessClassifier',
    'create_address_classifier',
    
    # ML optimizer
    'OptimizationResult',
    'PerformanceMetrics',
    'OptimizationHistory',
    'ParameterSpace',
    'MLParameterOptimizer',
    'create_ml_optimizer'
]