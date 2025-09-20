#!/usr/bin/env python3
"""
Machine Learning models for extraction pattern learning and confidence modeling.

This module implements ML-based extraction enhancement including:
- Pattern learning from successful extractions
- Confidence modeling for extracted data
- Feature extraction from text and context
- Model training and prediction
"""

import json
import pickle
import logging
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any, Set
from dataclasses import dataclass
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, mean_squared_error
from sklearn.preprocessing import StandardScaler

logger = logging.getLogger(__name__)


@dataclass
class ExtractionFeatures:
    """Features extracted from text for ML models."""
    # Text features
    text_length: int = 0
    word_count: int = 0
    sentence_count: int = 0
    uppercase_ratio: float = 0.0
    digit_ratio: float = 0.0
    punctuation_ratio: float = 0.0
    
    # Pattern features
    phone_pattern_count: int = 0
    email_pattern_count: int = 0
    url_pattern_count: int = 0
    address_pattern_count: int = 0
    
    # Context features
    contact_keywords: int = 0
    business_keywords: int = 0
    formal_keywords: int = 0
    
    # Quality indicators
    formatting_score: float = 0.0
    completeness_score: float = 0.0
    consistency_score: float = 0.0


@dataclass
class ExtractionResult:
    """Result of an extraction with ML confidence."""
    extracted_value: str
    confidence_score: float
    extraction_type: str  # phone, email, address, etc.
    source_text: str
    features: ExtractionFeatures
    metadata: Dict[str, Any]


class FeatureExtractor:
    """Extract features from text for ML models."""
    
    def __init__(self):
        # Pattern definitions
        self.phone_patterns = [
            r'\+33\s*[1-9](?:[.\s-]*\d{2}){4}',
            r'0[1-9](?:[.\s-]*\d{2}){4}',
            r'\d{10}',
            r'\d{2}[.\s-]\d{2}[.\s-]\d{2}[.\s-]\d{2}[.\s-]\d{2}'
        ]
        
        self.email_patterns = [
            r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'
        ]
        
        self.url_patterns = [
            r'https?://[^\s]+',
            r'www\.[^\s]+',
            r'\b[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}\b'
        ]
        
        self.address_patterns = [
            r'\d+\s+[A-Za-z\s]+(?:rue|avenue|boulevard|place|allée|chemin|impasse)',
            r'(?:rue|avenue|boulevard|place|allée|chemin|impasse)\s+[A-Za-z\s]+',
            r'\d{5}\s+[A-Za-z\s]+'  # Postal code + city
        ]
        
        # Keyword sets
        self.contact_keywords = {
            'contact', 'telephone', 'phone', 'email', 'mail', 'adresse', 'address',
            'coordonnées', 'joindre', 'contacter', 'appeler', 'écrire'
        }
        
        self.business_keywords = {
            'entreprise', 'société', 'company', 'business', 'sarl', 'sas', 'sa',
            'eurl', 'auto-entrepreneur', 'artisan', 'commerce', 'boutique'
        }
        
        self.formal_keywords = {
            'monsieur', 'madame', 'mr', 'mme', 'docteur', 'professeur',
            'directeur', 'manager', 'responsable', 'chef'
        }
    
    def extract_features(self, text: str, context: str = "") -> ExtractionFeatures:
        """Extract comprehensive features from text."""
        if not text:
            return ExtractionFeatures()
        
        combined_text = f"{text} {context}".lower()
        
        # Basic text features
        text_length = len(text)
        words = text.split()
        word_count = len(words)
        sentences = text.split('.')
        sentence_count = len([s for s in sentences if s.strip()])
        
        # Character ratios
        uppercase_count = sum(1 for c in text if c.isupper())
        digit_count = sum(1 for c in text if c.isdigit())
        punct_count = sum(1 for c in text if c in '.,;:!?()-[]{}')
        
        uppercase_ratio = uppercase_count / max(text_length, 1)
        digit_ratio = digit_count / max(text_length, 1)
        punctuation_ratio = punct_count / max(text_length, 1)
        
        # Pattern counts
        phone_pattern_count = sum(len(re.findall(pattern, text, re.IGNORECASE)) 
                                 for pattern in self.phone_patterns)
        email_pattern_count = sum(len(re.findall(pattern, text, re.IGNORECASE)) 
                                 for pattern in self.email_patterns)
        url_pattern_count = sum(len(re.findall(pattern, text, re.IGNORECASE)) 
                               for pattern in self.url_patterns)
        address_pattern_count = sum(len(re.findall(pattern, text, re.IGNORECASE)) 
                                   for pattern in self.address_patterns)
        
        # Keyword counts
        contact_keywords = sum(1 for keyword in self.contact_keywords 
                              if keyword in combined_text)
        business_keywords = sum(1 for keyword in self.business_keywords 
                               if keyword in combined_text)
        formal_keywords = sum(1 for keyword in self.formal_keywords 
                             if keyword in combined_text)
        
        # Quality scores
        formatting_score = self._calculate_formatting_score(text)
        completeness_score = self._calculate_completeness_score(text)
        consistency_score = self._calculate_consistency_score(text, context)
        
        return ExtractionFeatures(
            text_length=text_length,
            word_count=word_count,
            sentence_count=sentence_count,
            uppercase_ratio=uppercase_ratio,
            digit_ratio=digit_ratio,
            punctuation_ratio=punctuation_ratio,
            phone_pattern_count=phone_pattern_count,
            email_pattern_count=email_pattern_count,
            url_pattern_count=url_pattern_count,
            address_pattern_count=address_pattern_count,
            contact_keywords=contact_keywords,
            business_keywords=business_keywords,
            formal_keywords=formal_keywords,
            formatting_score=formatting_score,
            completeness_score=completeness_score,
            consistency_score=consistency_score
        )
    
    def _calculate_formatting_score(self, text: str) -> float:
        """Calculate formatting quality score."""
        score = 0.0
        
        # Proper capitalization
        if text and text[0].isupper():
            score += 0.2
        
        # Consistent spacing
        if not re.search(r'\s{2,}', text):  # No multiple spaces
            score += 0.2
        
        # Proper punctuation
        if text.endswith('.') or text.endswith('!') or text.endswith('?'):
            score += 0.2
        
        # No excessive punctuation
        if not re.search(r'[.!?]{2,}', text):
            score += 0.2
        
        # Reasonable length
        if 10 <= len(text) <= 200:
            score += 0.2
        
        return score
    
    def _calculate_completeness_score(self, text: str) -> float:
        """Calculate completeness score based on expected patterns."""
        if not text:
            return 0.0
        
        score = 0.0
        
        # Has actual content (not just numbers or symbols)
        if re.search(r'[a-zA-Z]{3,}', text):
            score += 0.5
        
        # Has proper structure
        if len(text.split()) >= 2:
            score += 0.3
        
        # Not obviously truncated
        if not text.endswith('...') and not text.endswith('…'):
            score += 0.2
        
        return score
    
    def _calculate_consistency_score(self, text: str, context: str) -> float:
        """Calculate consistency with context."""
        if not context:
            return 0.5  # Neutral score when no context
        
        score = 0.0
        text_words = set(text.lower().split())
        context_words = set(context.lower().split())
        
        # Shared vocabulary
        if text_words and context_words:
            overlap = len(text_words.intersection(context_words))
            max_len = max(len(text_words), len(context_words))
            if max_len > 0:
                score += (overlap / max_len) * 0.5
        
        # Similar length distribution
        if abs(len(text) - len(context)) < 50:
            score += 0.3
        
        # Consistent formatting
        if (text.isupper() == context.isupper()) or (text.islower() == context.islower()):
            score += 0.2
        
        return min(score, 1.0)


class ExtractionPatternLearner:
    """Learn extraction patterns from successful examples."""
    
    def __init__(self, model_dir: str = "models"):
        self.model_dir = Path(model_dir)
        self.model_dir.mkdir(parents=True, exist_ok=True)
        
        self.feature_extractor = FeatureExtractor()
        self.vectorizer = TfidfVectorizer(max_features=1000, stop_words='english')
        self.scaler = StandardScaler()
        
        # Models for different extraction types
        self.pattern_classifiers = {}  # Type -> classifier
        self.confidence_regressors = {}  # Type -> regressor
        
        # Training data storage
        self.training_data = {}  # Type -> List[examples]
        
    def add_training_example(
        self,
        text: str,
        extracted_value: str,
        extraction_type: str,
        confidence: float,
        context: str = "",
        is_correct: bool = True
    ):
        """Add a training example."""
        if extraction_type not in self.training_data:
            self.training_data[extraction_type] = []
        
        features = self.feature_extractor.extract_features(text, context)
        
        example = {
            'text': text,
            'extracted_value': extracted_value,
            'context': context,
            'features': features,
            'confidence': confidence,
            'is_correct': is_correct,
            'timestamp': pd.Timestamp.now()
        }
        
        self.training_data[extraction_type].append(example)
    
    def train_models(self, extraction_type: str = None) -> Dict[str, Any]:
        """Train ML models for pattern recognition and confidence prediction."""
        
        if extraction_type:
            types_to_train = [extraction_type]
        else:
            types_to_train = list(self.training_data.keys())
        
        results = {}
        
        for ext_type in types_to_train:
            logger.info(f"Training models for extraction type: {ext_type}")
            
            examples = self.training_data[ext_type]
            if len(examples) < 10:
                logger.warning(f"Insufficient training data for {ext_type}: {len(examples)} examples")
                continue
            
            # Prepare features and targets
            X_features = []
            X_text = []
            y_correct = []
            y_confidence = []
            
            for example in examples:
                # Numerical features
                features = example['features']
                feature_vector = [
                    features.text_length, features.word_count, features.sentence_count,
                    features.uppercase_ratio, features.digit_ratio, features.punctuation_ratio,
                    features.phone_pattern_count, features.email_pattern_count,
                    features.url_pattern_count, features.address_pattern_count,
                    features.contact_keywords, features.business_keywords, features.formal_keywords,
                    features.formatting_score, features.completeness_score, features.consistency_score
                ]
                X_features.append(feature_vector)
                
                # Text features
                X_text.append(example['text'])
                
                # Targets
                y_correct.append(example['is_correct'])
                y_confidence.append(example['confidence'])
            
            # Convert to numpy arrays
            X_features = np.array(X_features)
            y_correct = np.array(y_correct)
            y_confidence = np.array(y_confidence)
            
            # Scale numerical features
            X_features_scaled = self.scaler.fit_transform(X_features)
            
            # Vectorize text
            X_text_vec = self.vectorizer.fit_transform(X_text).toarray()
            
            # Combine features
            X_combined = np.hstack([X_features_scaled, X_text_vec])
            
            # Train classification model (correct/incorrect)
            if len(set(y_correct)) > 1:  # Need both positive and negative examples
                X_train, X_test, y_train_correct, y_test_correct = train_test_split(
                    X_combined, y_correct, test_size=0.2, random_state=42
                )
                
                classifier = RandomForestClassifier(n_estimators=100, random_state=42)
                classifier.fit(X_train, y_train_correct)
                
                # Evaluate
                y_pred_correct = classifier.predict(X_test)
                classification_score = classifier.score(X_test, y_test_correct)
                
                self.pattern_classifiers[ext_type] = classifier
                
                logger.info(f"Classification accuracy for {ext_type}: {classification_score:.3f}")
            
            # Train confidence regression model
            X_train, X_test, y_train_conf, y_test_conf = train_test_split(
                X_combined, y_confidence, test_size=0.2, random_state=42
            )
            
            regressor = RandomForestRegressor(n_estimators=100, random_state=42)
            regressor.fit(X_train, y_train_conf)
            
            # Evaluate
            y_pred_conf = regressor.predict(X_test)
            mse = mean_squared_error(y_test_conf, y_pred_conf)
            
            self.confidence_regressors[ext_type] = regressor
            
            logger.info(f"Confidence MSE for {ext_type}: {mse:.3f}")
            
            results[ext_type] = {
                'classification_accuracy': classification_score if ext_type in self.pattern_classifiers else None,
                'confidence_mse': mse,
                'training_examples': len(examples)
            }
        
        # Save models
        self._save_models()
        
        return results
    
    def predict_extraction_quality(
        self,
        text: str,
        extracted_value: str,
        extraction_type: str,
        context: str = ""
    ) -> Tuple[bool, float]:
        """Predict if extraction is correct and its confidence."""
        
        if extraction_type not in self.confidence_regressors:
            # No trained model, return conservative estimates
            return True, 0.5
        
        # Extract features
        features = self.feature_extractor.extract_features(text, context)
        
        # Prepare feature vector
        feature_vector = np.array([[
            features.text_length, features.word_count, features.sentence_count,
            features.uppercase_ratio, features.digit_ratio, features.punctuation_ratio,
            features.phone_pattern_count, features.email_pattern_count,
            features.url_pattern_count, features.address_pattern_count,
            features.contact_keywords, features.business_keywords, features.formal_keywords,
            features.formatting_score, features.completeness_score, features.consistency_score
        ]])
        
        # Scale features
        feature_vector_scaled = self.scaler.transform(feature_vector)
        
        # Vectorize text
        text_vec = self.vectorizer.transform([text]).toarray()
        
        # Combine features
        X_combined = np.hstack([feature_vector_scaled, text_vec])
        
        # Predict correctness
        is_correct = True
        if extraction_type in self.pattern_classifiers:
            is_correct = self.pattern_classifiers[extraction_type].predict(X_combined)[0]
        
        # Predict confidence
        confidence = self.confidence_regressors[extraction_type].predict(X_combined)[0]
        confidence = max(0.0, min(1.0, confidence))  # Clamp to [0, 1]
        
        return bool(is_correct), float(confidence)
    
    def learn_from_feedback(
        self,
        text: str,
        extracted_value: str,
        extraction_type: str,
        actual_confidence: float,
        is_correct: bool,
        context: str = ""
    ):
        """Learn from user feedback on extraction quality."""
        
        # Add as training example
        self.add_training_example(
            text=text,
            extracted_value=extracted_value,
            extraction_type=extraction_type,
            confidence=actual_confidence,
            context=context,
            is_correct=is_correct
        )
        
        # Retrain if we have enough new examples
        if len(self.training_data[extraction_type]) % 50 == 0:
            logger.info(f"Retraining models for {extraction_type} with new feedback")
            self.train_models(extraction_type)
    
    def get_extraction_patterns(self, extraction_type: str) -> List[Dict[str, Any]]:
        """Get learned patterns for an extraction type."""
        
        if extraction_type not in self.training_data:
            return []
        
        examples = self.training_data[extraction_type]
        correct_examples = [ex for ex in examples if ex['is_correct']]
        
        # Analyze patterns in correct examples
        patterns = []
        
        # Common text patterns
        texts = [ex['text'] for ex in correct_examples]
        if texts:
            # Find common regex patterns
            # This is simplified - could use more sophisticated pattern mining
            common_words = {}
            for text in texts:
                words = text.lower().split()
                for word in words:
                    common_words[word] = common_words.get(word, 0) + 1
            
            # Top words that appear in >20% of examples
            threshold = max(1, len(texts) * 0.2)
            frequent_words = [word for word, count in common_words.items() if count >= threshold]
            
            if frequent_words:
                patterns.append({
                    'type': 'frequent_words',
                    'pattern': frequent_words,
                    'confidence': len(frequent_words) / len(set(word for text in texts for word in text.split()))
                })
        
        return patterns
    
    def _save_models(self):
        """Save trained models to disk."""
        models_to_save = {
            'pattern_classifiers': self.pattern_classifiers,
            'confidence_regressors': self.confidence_regressors,
            'vectorizer': self.vectorizer,
            'scaler': self.scaler,
            'training_data': self.training_data
        }
        
        model_file = self.model_dir / "extraction_models.pkl"
        
        try:
            with open(model_file, 'wb') as f:
                pickle.dump(models_to_save, f)
            logger.info(f"Models saved to {model_file}")
        except Exception as e:
            logger.error(f"Failed to save models: {e}")
    
    def load_models(self) -> bool:
        """Load trained models from disk."""
        model_file = self.model_dir / "extraction_models.pkl"
        
        if not model_file.exists():
            logger.info("No saved models found")
            return False
        
        try:
            with open(model_file, 'rb') as f:
                models = pickle.load(f)
            
            self.pattern_classifiers = models.get('pattern_classifiers', {})
            self.confidence_regressors = models.get('confidence_regressors', {})
            self.vectorizer = models.get('vectorizer', TfidfVectorizer())
            self.scaler = models.get('scaler', StandardScaler())
            self.training_data = models.get('training_data', {})
            
            logger.info(f"Models loaded from {model_file}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to load models: {e}")
            return False


def create_extraction_learner(model_dir: str = "models") -> ExtractionPatternLearner:
    """Create and initialize an extraction pattern learner."""
    learner = ExtractionPatternLearner(model_dir)
    learner.load_models()  # Try to load existing models
    return learner


if __name__ == "__main__":
    # Example usage
    
    # Create learner
    learner = create_extraction_learner("/tmp/extraction_models")
    
    # Add some training examples
    learner.add_training_example(
        text="Contact us at +33 1 42 34 56 78",
        extracted_value="+33 1 42 34 56 78",
        extraction_type="phone",
        confidence=0.95,
        context="Contact information section",
        is_correct=True
    )
    
    learner.add_training_example(
        text="Call 123abc for info",
        extracted_value="123abc",
        extraction_type="phone",
        confidence=0.2,
        context="",
        is_correct=False
    )
    
    # Train models
    results = learner.train_models("phone")
    print(f"Training results: {results}")
    
    # Test prediction
    is_correct, confidence = learner.predict_extraction_quality(
        text="Please call us at 06 12 34 56 78",
        extracted_value="06 12 34 56 78",
        extraction_type="phone",
        context="Contact section"
    )
    
    print(f"Predicted: correct={is_correct}, confidence={confidence:.3f}")