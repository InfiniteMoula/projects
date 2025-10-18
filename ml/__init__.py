"""
Machine Learning package for extraction enhancement and optimization.

This package provides ML-based components for:
- Extraction pattern learning and confidence modeling
- Address success prediction and classification  
- Automated parameter tuning and optimization
"""

from .extraction_models import (  # noqa: F401
    ExtractionFeatures,
    ExtractionResult,
    FeatureExtractor,
    ExtractionPatternLearner,
    create_extraction_learner
)

from .address_classifier import (  # noqa: F401
    AddressFeatures,
    AddressPrediction,
    AddressFeatureExtractor,
    AddressSuccessClassifier,
    create_address_classifier
)

try:  # Optional component – not available in all distributions.
    from .ml_optimizer import (  # type: ignore  # noqa: F401
        OptimizationResult,
        PerformanceMetrics,
        OptimizationHistory,
        ParameterSpace,
        MLParameterOptimizer,
        create_ml_optimizer
    )
except ModuleNotFoundError:  # pragma: no cover - optional dependency.
    OptimizationResult = PerformanceMetrics = OptimizationHistory = ParameterSpace = None  # type: ignore  # noqa: E501
    MLParameterOptimizer = create_ml_optimizer = None  # type: ignore

from .lead_score import (  # noqa: F401
    LeadScoreBreakdown,
    LeadScoreWeights,
    DEFAULT_WEIGHTS,
    compute_lead_score,
    add_business_score,
)
from .domain_predictor import (  # noqa: F401
    DomainPrediction,
    DomainPredictor,
    DEFAULT_MODEL_PATH,
    load_domain_predictor,
    predict_best_domain,
    train_domain_predictor,
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

    # Lead scoring
    'LeadScoreBreakdown',
    'LeadScoreWeights',
    'DEFAULT_WEIGHTS',
    'compute_lead_score',
    'add_business_score',

    # Domain predictor
    'DomainPrediction',
    'DomainPredictor',
    'DEFAULT_MODEL_PATH',
    'load_domain_predictor',
    'predict_best_domain',
    'train_domain_predictor',
]

if OptimizationResult is not None:  # pragma: no branch - simple runtime guard.
    __all__.extend([
        'OptimizationResult',
        'PerformanceMetrics',
        'OptimizationHistory',
        'ParameterSpace',
        'MLParameterOptimizer',
        'create_ml_optimizer',
    ])
