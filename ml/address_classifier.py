#!/usr/bin/env python3
"""
Address success prediction and classification using machine learning.

This module implements ML-based address quality assessment and success prediction
for scraping operations, helping optimize address selection and processing.
"""

import copy
import pickle
import logging
import re
from functools import lru_cache
from pathlib import Path
from typing import TYPE_CHECKING, Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
import pandas as pd
import numpy as np

if TYPE_CHECKING:
    from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
    from sklearn.model_selection import cross_val_score, train_test_split
    from sklearn.preprocessing import LabelEncoder, StandardScaler


logger = logging.getLogger(__name__)

@lru_cache(maxsize=1)
def _get_preprocessing():
    from sklearn.preprocessing import LabelEncoder, StandardScaler
    return LabelEncoder, StandardScaler

@lru_cache(maxsize=1)
def _get_model_selection():
    from sklearn.model_selection import cross_val_score, train_test_split
    return cross_val_score, train_test_split

@lru_cache(maxsize=1)
def _get_ensemble():
    from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
    return GradientBoostingClassifier, RandomForestClassifier

@lru_cache(maxsize=8)
def _load_pickled_models(path: str) -> dict:
    with open(path, "rb") as handle:
        return pickle.load(handle)



@dataclass
class AddressFeatures:
    """Features extracted from addresses for ML classification."""
    # Basic structure features
    length: int = 0
    word_count: int = 0
    number_count: int = 0
    has_street_number: bool = False
    has_street_type: bool = False
    has_postal_code: bool = False
    has_city: bool = False
    
    # Quality indicators
    completeness_score: float = 0.0
    formatting_score: float = 0.0
    specificity_score: float = 0.0
    
    # Pattern matches
    french_postal_pattern: bool = False
    street_type_pattern: bool = False
    building_number_pattern: bool = False
    
    # Geographic indicators
    contains_region: bool = False
    contains_department: bool = False
    geographic_specificity: float = 0.0
    
    # Text quality
    capitalization_consistency: float = 0.0
    spelling_quality: float = 0.0
    abbreviation_ratio: float = 0.0


@dataclass
class AddressPrediction:
    """Prediction result for address scraping success."""
    address: str
    success_probability: float
    quality_score: float
    recommendation: str
    features: AddressFeatures
    confidence: float


class AddressFeatureExtractor:
    """Extract features from addresses for ML classification."""
    
    def __init__(self):
        # French address patterns
        self.street_types = {
            'rue', 'avenue', 'boulevard', 'place', 'allée', 'chemin', 'impasse',
            'square', 'cours', 'quai', 'passage', 'sentier', 'voie', 'route',
            'av', 'bd', 'pl', 'all', 'ch', 'imp', 'sq', 'pass'
        }
        
        self.building_types = {
            'bis', 'ter', 'quater', 'a', 'b', 'c', 'd', 'e'
        }
        
        # Common French regions and departments
        self.regions = {
            'ile-de-france', 'provence-alpes-cote-d-azur', 'auvergne-rhone-alpes',
            'nouvelle-aquitaine', 'occitanie', 'hauts-de-france', 'grand-est',
            'pays-de-la-loire', 'bretagne', 'normandie', 'centre-val-de-loire',
            'bourgogne-franche-comte', 'corse'
        }
        
        self.departments = {
            'paris', 'marseille', 'lyon', 'toulouse', 'nice', 'nantes',
            'montpellier', 'strasbourg', 'bordeaux', 'lille', 'rennes',
            'reims', 'saint-etienne', 'toulon', 'grenoble'
        }
        
        # Common abbreviations
        self.abbreviations = {
            'st': 'saint', 'ste': 'sainte', 'dr': 'docteur', 'gen': 'general',
            'mal': 'marechal', 'pl': 'place', 'av': 'avenue', 'bd': 'boulevard'
        }
        
        # Quality keywords
        self.quality_keywords = {
            'high': {'batiment', 'residence', 'lotissement', 'zone', 'parc'},
            'medium': {'centre', 'quartier', 'secteur'},
            'low': {'vague', 'environ', 'pres', 'proche'}
        }
    
    def extract_features(self, address: str) -> AddressFeatures:
        """Extract comprehensive features from an address string."""
        if not address or not isinstance(address, str):
            return AddressFeatures()
        
        address_clean = address.strip().lower()
        words = address_clean.split()
        
        # Basic structure
        length = len(address)
        word_count = len(words)
        number_count = sum(1 for word in words if re.search(r'\d', word))
        
        # Check for essential components
        has_street_number = bool(re.search(r'^\d+', address_clean))
        has_street_type = any(st in address_clean for st in self.street_types)
        has_postal_code = bool(re.search(r'\b\d{5}\b', address_clean))
        has_city = self._has_city_name(address_clean)
        
        # Pattern matching
        french_postal_pattern = bool(re.search(r'\b\d{5}\s+[a-zA-Z\s-]+$', address))
        street_type_pattern = bool(re.search(r'\b(?:' + '|'.join(self.street_types) + r')\b', address_clean))
        building_number_pattern = bool(re.search(r'^\d+(?:\s+(?:' + '|'.join(self.building_types) + r'))?', address_clean))
        
        # Geographic indicators
        contains_region = any(region in address_clean for region in self.regions)
        contains_department = any(dept in address_clean for dept in self.departments)
        geographic_specificity = self._calculate_geographic_specificity(address_clean)
        
        # Quality scores
        completeness_score = self._calculate_completeness_score(address_clean, has_street_number, has_street_type, has_postal_code, has_city)
        formatting_score = self._calculate_formatting_score(address)
        specificity_score = self._calculate_specificity_score(address_clean)
        
        # Text quality
        capitalization_consistency = self._calculate_capitalization_consistency(address)
        spelling_quality = self._calculate_spelling_quality(address_clean)
        abbreviation_ratio = self._calculate_abbreviation_ratio(words)
        
        return AddressFeatures(
            length=length,
            word_count=word_count,
            number_count=number_count,
            has_street_number=has_street_number,
            has_street_type=has_street_type,
            has_postal_code=has_postal_code,
            has_city=has_city,
            completeness_score=completeness_score,
            formatting_score=formatting_score,
            specificity_score=specificity_score,
            french_postal_pattern=french_postal_pattern,
            street_type_pattern=street_type_pattern,
            building_number_pattern=building_number_pattern,
            contains_region=contains_region,
            contains_department=contains_department,
            geographic_specificity=geographic_specificity,
            capitalization_consistency=capitalization_consistency,
            spelling_quality=spelling_quality,
            abbreviation_ratio=abbreviation_ratio
        )
    
    def _has_city_name(self, address: str) -> bool:
        """Check if address contains a city name."""
        # Look for postal code followed by city name pattern
        postal_city_match = re.search(r'\d{5}\s+([a-zA-Z\s-]+)', address)
        if postal_city_match:
            city_part = postal_city_match.group(1).strip()
            # City name should be at least 2 characters and not all numbers
            return len(city_part) >= 2 and not city_part.isdigit()
        
        # Look for known city names
        return any(city in address for city in self.departments)
    
    def _calculate_completeness_score(self, address: str, has_number: bool, has_street: bool, has_postal: bool, has_city: bool) -> float:
        """Calculate how complete the address is."""
        score = 0.0
        
        # Essential components
        if has_number:
            score += 0.25
        if has_street:
            score += 0.25
        if has_postal:
            score += 0.25
        if has_city:
            score += 0.25
        
        return score
    
    def _calculate_formatting_score(self, address: str) -> float:
        """Calculate formatting quality score."""
        score = 0.0
        
        # Proper capitalization
        if address and address[0].isupper():
            score += 0.2
        
        # No excessive spaces
        if not re.search(r'\s{2,}', address):
            score += 0.2
        
        # Reasonable length
        if 10 <= len(address) <= 100:
            score += 0.2
        
        # No special characters at start/end
        if address and address[0].isalnum() and address[-1].isalnum():
            score += 0.2
        
        # Contains numbers (likely street number)
        if re.search(r'\d', address):
            score += 0.2
        
        return score
    
    def _calculate_specificity_score(self, address: str) -> float:
        """Calculate how specific the address is."""
        score = 0.0
        
        # Has specific street number
        if re.search(r'^\d+', address):
            score += 0.3
        
        # Has specific building indicators
        if any(bt in address for bt in self.building_types):
            score += 0.2
        
        # Has quality indicators
        for quality_level, keywords in self.quality_keywords.items():
            if any(kw in address for kw in keywords):
                if quality_level == 'high':
                    score += 0.3
                elif quality_level == 'medium':
                    score += 0.1
                else:  # low quality
                    score -= 0.1
        
        # Longer addresses tend to be more specific
        if len(address) > 30:
            score += 0.2
        
        return max(0.0, min(1.0, score))
    
    def _calculate_geographic_specificity(self, address: str) -> float:
        """Calculate geographic specificity of the address."""
        score = 0.0
        
        # Has postal code
        if re.search(r'\d{5}', address):
            score += 0.4
        
        # Has known region/department
        if any(region in address for region in self.regions):
            score += 0.3
        if any(dept in address for dept in self.departments):
            score += 0.3
        
        return min(1.0, score)
    
    def _calculate_capitalization_consistency(self, address: str) -> float:
        """Calculate consistency of capitalization."""
        if not address:
            return 0.0
        
        words = address.split()
        if not words:
            return 0.0
        
        # Check if words are consistently capitalized
        capitalized_words = sum(1 for word in words if word and word[0].isupper())
        consistency = capitalized_words / len(words)
        
        # Prefer either all capitalized or title case
        if consistency in [0.0, 1.0] or 0.7 <= consistency <= 1.0:
            return 1.0
        elif 0.3 <= consistency <= 0.7:
            return 0.5
        else:
            return 0.0
    
    def _calculate_spelling_quality(self, address: str) -> float:
        """Estimate spelling quality (simplified heuristic)."""
        score = 1.0
        
        # Penalize obvious typos
        if re.search(r'(.)\1{3,}', address):  # Repeated characters
            score -= 0.3
        
        # Penalize numbers in unexpected places
        words = address.split()
        for word in words:
            if re.search(r'[a-zA-Z]\d[a-zA-Z]', word):  # Letters mixed with numbers
                score -= 0.1
        
        # Penalize excessive special characters
        special_chars = sum(1 for c in address if not c.isalnum() and c not in ' -,.')
        if special_chars > len(address) * 0.1:
            score -= 0.2
        
        return max(0.0, score)
    
    def _calculate_abbreviation_ratio(self, words: List[str]) -> float:
        """Calculate ratio of abbreviated words."""
        if not words:
            return 0.0
        
        abbreviated_count = sum(1 for word in words if word.lower() in self.abbreviations)
        return abbreviated_count / len(words)


class AddressSuccessClassifier:
    """ML classifier for predicting address scraping success."""
    
    def __init__(self, model_dir: str = "models"):
        self.model_dir = Path(model_dir)
        LabelEncoderCls, StandardScalerCls = _get_preprocessing()
        self.feature_extractor = AddressFeatureExtractor()
        self.classifier = None
        self.scaler = StandardScalerCls()
        self.label_encoder = LabelEncoderCls()
        
        # Training data
        self.training_data = []
        
        # Success categories
        self.success_categories = ['low', 'medium', 'high']
    
    def add_training_example(
        self,
        address: str,
        scraping_success: bool,
        results_quality: float = 0.5,
        results_count: int = 0,
        error_type: Optional[str] = None
    ):
        """Add a training example from scraping results."""
        
        features = self.feature_extractor.extract_features(address)
        
        # Determine success category
        if scraping_success and results_quality > 0.7 and results_count > 0:
            success_category = 'high'
        elif scraping_success and results_quality > 0.3:
            success_category = 'medium'
        else:
            success_category = 'low'
        
        example = {
            'address': address,
            'features': features,
            'scraping_success': scraping_success,
            'results_quality': results_quality,
            'results_count': results_count,
            'error_type': error_type,
            'success_category': success_category,
            'timestamp': pd.Timestamp.now()
        }
        
        self.training_data.append(example)
    
    def train_classifier(self) -> Dict[str, Any]:
        """Train the address success classifier."""
        
        if len(self.training_data) < 20:
            logger.warning(f"Insufficient training data: {len(self.training_data)} examples")
            return {"error": "Insufficient training data"}
        
        logger.info(f"Training classifier with {len(self.training_data)} examples")
        
        # Prepare features and targets
        GradientBoostingCls, _ = _get_ensemble()
        cross_val_score_fn, train_test_split_fn = _get_model_selection()
        X_features = []
        y_categories = []
        
        for example in self.training_data:
            features = example['features']
            
            # Convert features to numerical array
            feature_vector = [
                features.length, features.word_count, features.number_count,
                int(features.has_street_number), int(features.has_street_type),
                int(features.has_postal_code), int(features.has_city),
                features.completeness_score, features.formatting_score, features.specificity_score,
                int(features.french_postal_pattern), int(features.street_type_pattern),
                int(features.building_number_pattern), int(features.contains_region),
                int(features.contains_department), features.geographic_specificity,
                features.capitalization_consistency, features.spelling_quality,
                features.abbreviation_ratio
            ]
            
            X_features.append(feature_vector)
            y_categories.append(example['success_category'])
        
        X = np.array(X_features)
        y = np.array(y_categories)
        
        # Scale features
        X_scaled = self.scaler.fit_transform(X)
        
        # Encode labels
        y_encoded = self.label_encoder.fit_transform(y)
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split_fn(
            X_scaled, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
        )
        
        # Train classifier
        self.classifier = GradientBoostingCls(
            n_estimators=100,
            learning_rate=0.1,
            max_depth=3,
            random_state=42
        )
        
        self.classifier.fit(X_train, y_train)
        
        # Evaluate
        train_score = self.classifier.score(X_train, y_train)
        test_score = self.classifier.score(X_test, y_test)
        
        # Cross-validation
        cv_scores = cross_val_score_fn(self.classifier, X_scaled, y_encoded, cv=5)
        
        # Feature importance
        feature_names = [
            'length', 'word_count', 'number_count', 'has_street_number',
            'has_street_type', 'has_postal_code', 'has_city', 'completeness_score',
            'formatting_score', 'specificity_score', 'french_postal_pattern',
            'street_type_pattern', 'building_number_pattern', 'contains_region',
            'contains_department', 'geographic_specificity', 'capitalization_consistency',
            'spelling_quality', 'abbreviation_ratio'
        ]
        
        feature_importance = dict(zip(feature_names, self.classifier.feature_importances_))
        
        # Save models
        self._save_models()
        
        results = {
            'train_score': train_score,
            'test_score': test_score,
            'cv_mean': cv_scores.mean(),
            'cv_std': cv_scores.std(),
            'feature_importance': feature_importance,
            'training_examples': len(self.training_data)
        }
        
        logger.info(f"Classifier trained - Test score: {test_score:.3f}, CV: {cv_scores.mean():.3f}±{cv_scores.std():.3f}")
        
        return results
    
    def predict_success(self, address: str) -> AddressPrediction:
        """Predict scraping success for an address."""
        
        if self.classifier is None:
            # No trained model, return conservative prediction
            features = self.feature_extractor.extract_features(address)
            return AddressPrediction(
                address=address,
                success_probability=0.5,
                quality_score=features.completeness_score,
                recommendation="No trained model available",
                features=features,
                confidence=0.0
            )
        
        # Extract features
        features = self.feature_extractor.extract_features(address)
        
        # Prepare feature vector
        feature_vector = np.array([[
            features.length, features.word_count, features.number_count,
            int(features.has_street_number), int(features.has_street_type),
            int(features.has_postal_code), int(features.has_city),
            features.completeness_score, features.formatting_score, features.specificity_score,
            int(features.french_postal_pattern), int(features.street_type_pattern),
            int(features.building_number_pattern), int(features.contains_region),
            int(features.contains_department), features.geographic_specificity,
            features.capitalization_consistency, features.spelling_quality,
            features.abbreviation_ratio
        ]])
        
        # Scale features
        feature_vector_scaled = self.scaler.transform(feature_vector)
        
        # Predict
        prediction_proba = self.classifier.predict_proba(feature_vector_scaled)[0]
        predicted_class = self.classifier.predict(feature_vector_scaled)[0]
        
        # Convert back to category name
        predicted_category = self.label_encoder.inverse_transform([predicted_class])[0]
        
        # Calculate success probability (probability of medium or high success)
        class_names = self.label_encoder.classes_
        category_to_idx = {cat: idx for idx, cat in enumerate(self.label_encoder.inverse_transform(range(len(class_names))))}
        
        success_probability = 0.0
        if 'medium' in category_to_idx:
            success_probability += prediction_proba[category_to_idx['medium']]
        if 'high' in category_to_idx:
            success_probability += prediction_proba[category_to_idx['high']]
        
        # Generate recommendation
        recommendation = self._generate_recommendation(features, predicted_category, prediction_proba)
        
        # Confidence is the maximum probability
        confidence = max(prediction_proba)
        
        return AddressPrediction(
            address=address,
            success_probability=success_probability,
            quality_score=features.completeness_score,
            recommendation=recommendation,
            features=features,
            confidence=confidence
        )
    
    def _generate_recommendation(self, features: AddressFeatures, predicted_category: str, probabilities: np.ndarray) -> str:
        """Generate actionable recommendation for address improvement."""
        
        recommendations = []
        
        if predicted_category == 'low':
            recommendations.append("Low success probability predicted.")
            
            if not features.has_street_number:
                recommendations.append("Add street number if available.")
            if not features.has_street_type:
                recommendations.append("Specify street type (rue, avenue, etc.).")
            if not features.has_postal_code:
                recommendations.append("Include postal code.")
            if not features.has_city:
                recommendations.append("Add city name.")
            if features.completeness_score < 0.5:
                recommendations.append("Address appears incomplete.")
                
        elif predicted_category == 'medium':
            recommendations.append("Moderate success probability.")
            
            if features.specificity_score < 0.5:
                recommendations.append("Consider adding more specific details.")
            if features.formatting_score < 0.7:
                recommendations.append("Check address formatting.")
                
        else:  # high
            recommendations.append("High success probability predicted.")
        
        return " ".join(recommendations) if recommendations else "Address looks good for scraping."
    
    def batch_predict(self, addresses: List[str]) -> List[AddressPrediction]:
        """Predict success for a batch of addresses."""
        return [self.predict_success(addr) for addr in addresses]
    
    def get_feature_importance(self) -> Dict[str, float]:
        """Get feature importance from trained classifier."""
        if self.classifier is None:
            return {}
        
        feature_names = [
            'length', 'word_count', 'number_count', 'has_street_number',
            'has_street_type', 'has_postal_code', 'has_city', 'completeness_score',
            'formatting_score', 'specificity_score', 'french_postal_pattern',
            'street_type_pattern', 'building_number_pattern', 'contains_region',
            'contains_department', 'geographic_specificity', 'capitalization_consistency',
            'spelling_quality', 'abbreviation_ratio'
        ]
        
        return dict(zip(feature_names, self.classifier.feature_importances_))
    
    def load_models(self) -> bool:
        """Load trained models from disk and return whether loading succeeded."""
        self.model_dir.mkdir(parents=True, exist_ok=True)
        model_file = self.model_dir / "address_classifier.pkl"

        if not model_file.exists():
            logger.info("No saved models found at %s", model_file)
            return False

        try:
            models = copy.deepcopy(_load_pickled_models(str(model_file)))
        except FileNotFoundError:
            logger.info("No saved models found at %s", model_file)
            return False
        except (OSError, pickle.UnpicklingError) as exc:
            logger.error("Failed to load models from %s: %s", model_file, exc)
            return False
        except Exception as exc:  # pragma: no cover - defensive
            logger.error("Unexpected error while loading models from %s: %s", model_file, exc)
            return False

        LabelEncoderCls, StandardScalerCls = _get_preprocessing()
        classifier = models.get("classifier")
        if classifier is None:
            logger.warning("Saved model file %s did not contain a classifier", model_file)
            return False

        self.classifier = classifier
        self.scaler = models.get("scaler") or StandardScalerCls()
        self.label_encoder = models.get("label_encoder") or LabelEncoderCls()
        self.training_data = models.get("training_data", [])

        logger.info("Models loaded from %s", model_file)
        return True
    
    def _save_models(self):
        """Persist trained models to disk."""
        models_to_save = {
            'classifier': self.classifier,
            'scaler': self.scaler,
            'label_encoder': self.label_encoder,
            'training_data': self.training_data
        }
        
        model_file = self.model_dir / "address_classifier.pkl"

        self.model_dir.mkdir(parents=True, exist_ok=True)
        
        try:
            with open(model_file, 'wb') as f:
                pickle.dump(models_to_save, f)
            _load_pickled_models.cache_clear()
            logger.info(f"Models saved to {model_file}")
        except Exception as exc:  # pragma: no cover - defensive
            logger.error("Failed to save models to %s: %s", model_file, exc)


def create_address_classifier(model_dir: str = "models") -> AddressSuccessClassifier:
    """Create and initialize an address success classifier."""
    classifier = AddressSuccessClassifier(model_dir)
    classifier.load_models()  # Try to load existing models
    return classifier


if __name__ == "__main__":
    # Example usage
    
    # Create classifier
    classifier = create_address_classifier("/tmp/address_models")
    
    # Add training examples
    classifier.add_training_example(
        address="123 rue de la Paix, 75001 Paris",
        scraping_success=True,
        results_quality=0.9,
        results_count=5
    )
    
    classifier.add_training_example(
        address="somewhere in Paris",
        scraping_success=False,
        results_quality=0.0,
        results_count=0,
        error_type="address_too_vague"
    )
    
    # Add more examples for a realistic training set
    good_addresses = [
        "45 avenue des Champs-Élysées, 75008 Paris",
        "15 boulevard Saint-Germain, 75005 Paris",
        "88 rue du Faubourg Saint-Honoré, 75008 Paris"
    ]
    
    poor_addresses = [
        "Paris",
        "somewhere near the tower",
        "123 unknown street"
    ]
    
    for addr in good_addresses:
        classifier.add_training_example(addr, True, 0.8, 3)
    
    for addr in poor_addresses:
        classifier.add_training_example(addr, False, 0.1, 0)
    
    # Train classifier
    results = classifier.train_classifier()
    print(f"Training results: {results}")
    
    # Test prediction
    test_address = "42 rue de Rivoli, 75001 Paris"
    prediction = classifier.predict_success(test_address)
    
    print(f"\nPrediction for '{test_address}':")
    print(f"Success probability: {prediction.success_probability:.3f}")
    print(f"Quality score: {prediction.quality_score:.3f}")
    print(f"Recommendation: {prediction.recommendation}")
    print(f"Confidence: {prediction.confidence:.3f}")
