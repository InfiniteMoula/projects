#!/usr/bin/env python3
"""
ML-based optimizer for automated parameter tuning of scraping operations.

This module implements intelligent parameter optimization using historical data
and machine learning to automatically tune scraper configurations for optimal
performance and cost efficiency.
"""

import json
import pickle
import logging
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any, Union
from dataclasses import dataclass, asdict
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
from scipy.optimize import minimize
import warnings
warnings.filterwarnings('ignore')

logger = logging.getLogger(__name__)


@dataclass
class OptimizationResult:
    """Result of parameter optimization."""
    optimal_params: Dict[str, Any]
    expected_performance: Dict[str, float]
    confidence: float
    optimization_method: str
    iterations: int
    improvement_ratio: float


@dataclass
class PerformanceMetrics:
    """Performance metrics for scraping operations."""
    success_rate: float = 0.0
    results_quality: float = 0.0
    processing_time: float = 0.0
    cost_per_result: float = 0.0
    memory_usage: float = 0.0
    error_rate: float = 0.0
    throughput: float = 0.0  # results per minute


@dataclass
class OptimizationHistory:
    """Historical optimization record."""
    timestamp: str
    parameters: Dict[str, Any]
    performance: PerformanceMetrics
    context: Dict[str, Any]  # Data size, complexity, etc.
    scraper_type: str


class ParameterSpace:
    """Define parameter optimization space."""
    
    def __init__(self):
        # Define parameter ranges for different components
        self.parameter_ranges = {
            # Batch processing parameters
            'batch_size': (10, 100),
            'max_concurrent': (1, 10),
            'rate_limit_delay': (0.5, 5.0),
            'timeout_seconds': (60, 1200),
            
            # Apify scraper parameters
            'max_places_per_search': (5, 50),
            'max_linkedin_searches': (5, 50),
            'max_profiles_per_company': (1, 10),
            'max_contact_enrichments': (10, 100),
            
            # Quality thresholds
            'min_confidence_threshold': (0.3, 0.9),
            'quality_score_threshold': (0.5, 0.95),
            
            # Retry parameters
            'max_retries': (1, 5),
            'retry_delay': (1.0, 10.0),
            
            # Memory optimization
            'max_memory_mb': (1024, 8192),
            'chunk_size': (100, 2000)
        }
        
        # Parameter types
        self.parameter_types = {
            'batch_size': 'int',
            'max_concurrent': 'int',
            'rate_limit_delay': 'float',
            'timeout_seconds': 'int',
            'max_places_per_search': 'int',
            'max_linkedin_searches': 'int',
            'max_profiles_per_company': 'int',
            'max_contact_enrichments': 'int',
            'min_confidence_threshold': 'float',
            'quality_score_threshold': 'float',
            'max_retries': 'int',
            'retry_delay': 'float',
            'max_memory_mb': 'int',
            'chunk_size': 'int'
        }
        
        # Parameter dependencies (some parameters affect others)
        self.parameter_dependencies = {
            'max_concurrent': ['batch_size', 'max_memory_mb'],
            'batch_size': ['max_memory_mb', 'timeout_seconds'],
            'max_places_per_search': ['timeout_seconds', 'cost_per_result'],
            'max_linkedin_searches': ['timeout_seconds', 'cost_per_result']
        }
    
    def get_random_parameters(self) -> Dict[str, Any]:
        """Generate random parameters within valid ranges."""
        params = {}
        
        for param_name, (min_val, max_val) in self.parameter_ranges.items():
            param_type = self.parameter_types[param_name]
            
            if param_type == 'int':
                params[param_name] = np.random.randint(min_val, max_val + 1)
            else:  # float
                params[param_name] = np.random.uniform(min_val, max_val)
        
        # Apply constraints
        params = self._apply_parameter_constraints(params)
        
        return params
    
    def _apply_parameter_constraints(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Apply logical constraints between parameters."""
        
        # Ensure batch_size is reasonable for max_concurrent
        if params['batch_size'] * params['max_concurrent'] > 500:
            params['batch_size'] = min(params['batch_size'], 500 // params['max_concurrent'])
        
        # Ensure timeout is sufficient for batch processing
        min_timeout = params['batch_size'] * 5  # 5 seconds per item minimum
        params['timeout_seconds'] = max(params['timeout_seconds'], min_timeout)
        
        # Ensure memory is sufficient for concurrent processing
        min_memory = params['max_concurrent'] * 256  # 256MB per concurrent task
        params['max_memory_mb'] = max(params['max_memory_mb'], min_memory)
        
        return params
    
    def validate_parameters(self, params: Dict[str, Any]) -> bool:
        """Validate parameter values are within acceptable ranges."""
        for param_name, value in params.items():
            if param_name not in self.parameter_ranges:
                continue
                
            min_val, max_val = self.parameter_ranges[param_name]
            if not (min_val <= value <= max_val):
                return False
        
        return True


class MLParameterOptimizer:
    """ML-based parameter optimizer using historical performance data."""
    
    def __init__(self, model_dir: str = "models"):
        self.model_dir = Path(model_dir)
        self.model_dir.mkdir(parents=True, exist_ok=True)
        
        self.parameter_space = ParameterSpace()
        self.scaler = StandardScaler()
        
        # Models for different objectives
        self.models = {
            'success_rate': RandomForestRegressor(n_estimators=100, random_state=42),
            'results_quality': RandomForestRegressor(n_estimators=100, random_state=42),
            'processing_time': GradientBoostingRegressor(n_estimators=100, random_state=42),
            'cost_per_result': RandomForestRegressor(n_estimators=100, random_state=42),
            'throughput': GradientBoostingRegressor(n_estimators=100, random_state=42)
        }
        
        # Training data
        self.optimization_history: List[OptimizationHistory] = []
        self.models_trained = False
        
        # Optimization objectives weights
        self.objective_weights = {
            'success_rate': 0.3,
            'results_quality': 0.25,
            'processing_time': -0.15,  # Lower is better
            'cost_per_result': -0.15,  # Lower is better
            'throughput': 0.15
        }
    
    def add_performance_record(
        self,
        parameters: Dict[str, Any],
        performance: PerformanceMetrics,
        context: Dict[str, Any],
        scraper_type: str = "general"
    ):
        """Add a performance record for learning."""
        
        record = OptimizationHistory(
            timestamp=pd.Timestamp.now().isoformat(),
            parameters=parameters.copy(),
            performance=performance,
            context=context.copy(),
            scraper_type=scraper_type
        )
        
        self.optimization_history.append(record)
        
        # Retrain models if we have enough new data
        if len(self.optimization_history) % 20 == 0:
            logger.info("Retraining optimization models with new data")
            self.train_models()
    
    def train_models(self) -> Dict[str, float]:
        """Train ML models to predict performance from parameters."""
        
        if len(self.optimization_history) < 10:
            logger.warning(f"Insufficient data for training: {len(self.optimization_history)} records")
            return {}
        
        logger.info(f"Training optimization models with {len(self.optimization_history)} records")
        
        # Prepare training data
        X_features = []
        y_targets = {key: [] for key in self.models.keys()}
        
        for record in self.optimization_history:
            # Feature vector from parameters and context
            feature_vector = self._parameters_to_features(record.parameters, record.context)
            X_features.append(feature_vector)
            
            # Target values
            perf = record.performance
            y_targets['success_rate'].append(perf.success_rate)
            y_targets['results_quality'].append(perf.results_quality)
            y_targets['processing_time'].append(perf.processing_time)
            y_targets['cost_per_result'].append(perf.cost_per_result)
            y_targets['throughput'].append(perf.throughput)
        
        X = np.array(X_features)
        X_scaled = self.scaler.fit_transform(X)
        
        # Train models
        model_scores = {}
        
        for objective, model in self.models.items():
            y = np.array(y_targets[objective])
            
            if len(set(y)) < 2:  # Not enough variance
                logger.warning(f"Insufficient variance in {objective} for training")
                continue
            
            try:
                # Split data
                X_train, X_test, y_train, y_test = train_test_split(
                    X_scaled, y, test_size=0.2, random_state=42
                )
                
                # Train model
                model.fit(X_train, y_train)
                
                # Evaluate
                y_pred = model.predict(X_test)
                score = r2_score(y_test, y_pred)
                model_scores[objective] = score
                
                logger.info(f"Model {objective} R² score: {score:.3f}")
                
            except Exception as e:
                logger.error(f"Failed to train model for {objective}: {e}")
        
        self.models_trained = len(model_scores) > 0
        
        # Save models
        self._save_models()
        
        return model_scores
    
    def optimize_parameters(
        self,
        context: Dict[str, Any],
        current_params: Optional[Dict[str, Any]] = None,
        optimization_method: str = "ml_guided"
    ) -> OptimizationResult:
        """Optimize parameters for given context."""
        
        if optimization_method == "ml_guided" and self.models_trained:
            return self._ml_guided_optimization(context, current_params)
        elif optimization_method == "grid_search":
            return self._grid_search_optimization(context, current_params)
        elif optimization_method == "random_search":
            return self._random_search_optimization(context, current_params)
        else:
            return self._heuristic_optimization(context, current_params)
    
    def _ml_guided_optimization(
        self,
        context: Dict[str, Any],
        current_params: Optional[Dict[str, Any]] = None
    ) -> OptimizationResult:
        """Use ML models to guide parameter optimization."""
        
        logger.info("Starting ML-guided parameter optimization")
        
        best_params = current_params or self.parameter_space.get_random_parameters()
        best_score = self._predict_performance_score(best_params, context)
        
        # Optimization using scipy.minimize
        def objective_function(param_values):
            params = self._array_to_parameters(param_values)
            score = self._predict_performance_score(params, context)
            return -score  # Minimize negative score (maximize score)
        
        # Convert current params to array
        param_array = self._parameters_to_array(best_params)
        
        # Define bounds
        bounds = []
        param_names = list(self.parameter_space.parameter_ranges.keys())
        for param_name in param_names:
            if param_name in best_params:
                min_val, max_val = self.parameter_space.parameter_ranges[param_name]
                bounds.append((min_val, max_val))
        
        try:
            # Optimize
            result = minimize(
                objective_function,
                param_array,
                method='L-BFGS-B',
                bounds=bounds,
                options={'maxiter': 100}
            )
            
            if result.success:
                optimal_params = self._array_to_parameters(result.x)
                optimal_score = -result.fun
                
                # Validate and apply constraints
                optimal_params = self.parameter_space._apply_parameter_constraints(optimal_params)
                
                # Predict performance
                predicted_performance = self._predict_detailed_performance(optimal_params, context)
                
                confidence = self._calculate_optimization_confidence(optimal_params, context)
                improvement_ratio = (optimal_score - best_score) / max(abs(best_score), 1e-6)
                
                return OptimizationResult(
                    optimal_params=optimal_params,
                    expected_performance=predicted_performance,
                    confidence=confidence,
                    optimization_method="ml_guided",
                    iterations=result.nit,
                    improvement_ratio=improvement_ratio
                )
            
        except Exception as e:
            logger.error(f"ML-guided optimization failed: {e}")
        
        # Fallback to random search
        return self._random_search_optimization(context, current_params)
    
    def _random_search_optimization(
        self,
        context: Dict[str, Any],
        current_params: Optional[Dict[str, Any]] = None,
        n_iterations: int = 50
    ) -> OptimizationResult:
        """Random search optimization."""
        
        logger.info(f"Starting random search optimization with {n_iterations} iterations")
        
        best_params = current_params or self.parameter_space.get_random_parameters()
        best_score = self._evaluate_parameters(best_params, context) if current_params else -np.inf
        
        for i in range(n_iterations):
            # Generate random parameters
            candidate_params = self.parameter_space.get_random_parameters()
            
            # Evaluate
            score = self._evaluate_parameters(candidate_params, context)
            
            if score > best_score:
                best_params = candidate_params
                best_score = score
        
        # Predict performance
        predicted_performance = self._predict_detailed_performance(best_params, context)
        
        improvement_ratio = 0.0
        if current_params:
            current_score = self._evaluate_parameters(current_params, context)
            improvement_ratio = (best_score - current_score) / max(abs(current_score), 1e-6)
        
        return OptimizationResult(
            optimal_params=best_params,
            expected_performance=predicted_performance,
            confidence=0.6,  # Medium confidence for random search
            optimization_method="random_search",
            iterations=n_iterations,
            improvement_ratio=improvement_ratio
        )
    
    def _heuristic_optimization(
        self,
        context: Dict[str, Any],
        current_params: Optional[Dict[str, Any]] = None
    ) -> OptimizationResult:
        """Heuristic-based optimization using domain knowledge."""
        
        logger.info("Starting heuristic optimization")
        
        data_size = context.get('data_size', 100)
        complexity = context.get('complexity', 'medium')
        priority = context.get('priority', 'balanced')
        
        # Base parameters
        params = {
            'batch_size': min(50, max(10, data_size // 10)),
            'max_concurrent': 3,
            'rate_limit_delay': 1.0,
            'timeout_seconds': 300,
            'max_places_per_search': 10,
            'max_linkedin_searches': 20,
            'max_profiles_per_company': 5,
            'max_contact_enrichments': 50,
            'min_confidence_threshold': 0.5,
            'quality_score_threshold': 0.7,
            'max_retries': 2,
            'retry_delay': 2.0,
            'max_memory_mb': 2048,
            'chunk_size': 500
        }
        
        # Adjust based on data size
        if data_size > 1000:
            params['batch_size'] = min(100, params['batch_size'] * 2)
            params['max_concurrent'] = min(5, params['max_concurrent'] + 1)
            params['max_memory_mb'] = min(4096, params['max_memory_mb'] * 2)
        elif data_size < 50:
            params['batch_size'] = max(10, params['batch_size'] // 2)
            params['max_concurrent'] = max(1, params['max_concurrent'] - 1)
        
        # Adjust based on complexity
        if complexity == 'high':
            params['timeout_seconds'] *= 2
            params['max_retries'] += 1
            params['rate_limit_delay'] *= 1.5
        elif complexity == 'low':
            params['timeout_seconds'] = max(60, params['timeout_seconds'] // 2)
            params['rate_limit_delay'] *= 0.7
        
        # Adjust based on priority
        if priority == 'speed':
            params['max_concurrent'] = min(8, params['max_concurrent'] + 2)
            params['rate_limit_delay'] *= 0.5
            params['min_confidence_threshold'] = max(0.3, params['min_confidence_threshold'] - 0.1)
        elif priority == 'quality':
            params['max_concurrent'] = max(1, params['max_concurrent'] - 1)
            params['rate_limit_delay'] *= 1.5
            params['min_confidence_threshold'] = min(0.8, params['min_confidence_threshold'] + 0.1)
        elif priority == 'cost':
            params['max_places_per_search'] = max(5, params['max_places_per_search'] // 2)
            params['max_linkedin_searches'] = max(5, params['max_linkedin_searches'] // 2)
            params['max_contact_enrichments'] = max(10, params['max_contact_enrichments'] // 2)
        
        # Apply constraints
        params = self.parameter_space._apply_parameter_constraints(params)
        
        # Predict performance
        predicted_performance = self._predict_detailed_performance(params, context)
        
        return OptimizationResult(
            optimal_params=params,
            expected_performance=predicted_performance,
            confidence=0.7,  # Good confidence for heuristics
            optimization_method="heuristic",
            iterations=1,
            improvement_ratio=0.0  # Unknown without baseline
        )
    
    def _parameters_to_features(self, parameters: Dict[str, Any], context: Dict[str, Any]) -> List[float]:
        """Convert parameters and context to feature vector."""
        features = []
        
        # Parameter features
        param_names = list(self.parameter_space.parameter_ranges.keys())
        for param_name in param_names:
            value = parameters.get(param_name, 0)
            features.append(float(value))
        
        # Context features
        features.extend([
            float(context.get('data_size', 100)),
            float(context.get('complexity_score', 0.5)),
            float(context.get('has_phone_data', False)),
            float(context.get('has_email_data', False)),
            float(context.get('has_address_data', True)),
            float(context.get('time_budget_minutes', 60))
        ])
        
        return features
    
    def _parameters_to_array(self, parameters: Dict[str, Any]) -> np.ndarray:
        """Convert parameters dict to numpy array."""
        param_names = list(self.parameter_space.parameter_ranges.keys())
        return np.array([parameters.get(name, 0) for name in param_names])
    
    def _array_to_parameters(self, param_array: np.ndarray) -> Dict[str, Any]:
        """Convert numpy array to parameters dict."""
        param_names = list(self.parameter_space.parameter_ranges.keys())
        params = {}
        
        for i, param_name in enumerate(param_names):
            value = param_array[i]
            param_type = self.parameter_space.parameter_types[param_name]
            
            if param_type == 'int':
                params[param_name] = int(round(value))
            else:
                params[param_name] = float(value)
        
        return params
    
    def _predict_performance_score(self, parameters: Dict[str, Any], context: Dict[str, Any]) -> float:
        """Predict overall performance score for parameters."""
        if not self.models_trained:
            return 0.5  # Neutral score
        
        detailed_performance = self._predict_detailed_performance(parameters, context)
        
        # Calculate weighted score
        score = 0.0
        for objective, weight in self.objective_weights.items():
            if objective in detailed_performance:
                value = detailed_performance[objective]
                # Normalize values to [0, 1] range
                if objective in ['processing_time', 'cost_per_result']:
                    # Lower is better for these metrics
                    normalized_value = max(0, 1 - value / 100)  # Assuming reasonable ranges
                else:
                    normalized_value = min(1, max(0, value))
                
                score += weight * normalized_value
        
        return score
    
    def _predict_detailed_performance(self, parameters: Dict[str, Any], context: Dict[str, Any]) -> Dict[str, float]:
        """Predict detailed performance metrics."""
        if not self.models_trained:
            # Return heuristic estimates
            return {
                'success_rate': 0.7,
                'results_quality': 0.6,
                'processing_time': 300.0,
                'cost_per_result': 0.1,
                'throughput': 5.0
            }
        
        features = self._parameters_to_features(parameters, context)
        features_scaled = self.scaler.transform([features])
        
        predictions = {}
        for objective, model in self.models.items():
            try:
                prediction = model.predict(features_scaled)[0]
                predictions[objective] = max(0, prediction)  # Ensure non-negative
            except Exception as e:
                logger.error(f"Failed to predict {objective}: {e}")
                predictions[objective] = 0.5  # Default value
        
        return predictions
    
    def _evaluate_parameters(self, parameters: Dict[str, Any], context: Dict[str, Any]) -> float:
        """Evaluate parameters using available models or heuristics."""
        if self.models_trained:
            return self._predict_performance_score(parameters, context)
        else:
            # Simple heuristic evaluation
            score = 0.5
            
            # Prefer moderate batch sizes
            batch_size = parameters.get('batch_size', 50)
            if 20 <= batch_size <= 80:
                score += 0.1
            
            # Prefer reasonable concurrency
            concurrent = parameters.get('max_concurrent', 3)
            if 2 <= concurrent <= 6:
                score += 0.1
            
            # Prefer balanced timeouts
            timeout = parameters.get('timeout_seconds', 300)
            if 120 <= timeout <= 600:
                score += 0.1
            
            return score
    
    def _calculate_optimization_confidence(self, parameters: Dict[str, Any], context: Dict[str, Any]) -> float:
        """Calculate confidence in optimization result."""
        if not self.models_trained:
            return 0.3
        
        # Base confidence on amount of training data
        base_confidence = min(0.9, len(self.optimization_history) / 100)
        
        # Adjust based on parameter similarity to training data
        similarity_score = self._calculate_parameter_similarity(parameters)
        
        # Combine scores
        confidence = (base_confidence + similarity_score) / 2
        
        return confidence
    
    def _calculate_parameter_similarity(self, parameters: Dict[str, Any]) -> float:
        """Calculate similarity to historical parameter configurations."""
        if not self.optimization_history:
            return 0.3
        
        similarities = []
        
        for record in self.optimization_history[-20:]:  # Check last 20 records
            similarity = 0.0
            count = 0
            
            for param_name, value in parameters.items():
                if param_name in record.parameters:
                    historical_value = record.parameters[param_name]
                    
                    # Normalize difference
                    if param_name in self.parameter_space.parameter_ranges:
                        min_val, max_val = self.parameter_space.parameter_ranges[param_name]
                        range_size = max_val - min_val
                        diff = abs(value - historical_value) / range_size
                        similarity += 1 - diff
                        count += 1
            
            if count > 0:
                similarities.append(similarity / count)
        
        return np.mean(similarities) if similarities else 0.3
    
    def _save_models(self):
        """Save trained models and history."""
        data_to_save = {
            'models': self.models,
            'scaler': self.scaler,
            'optimization_history': [asdict(record) for record in self.optimization_history],
            'objective_weights': self.objective_weights,
            'models_trained': self.models_trained
        }
        
        model_file = self.model_dir / "ml_optimizer.pkl"
        
        try:
            with open(model_file, 'wb') as f:
                pickle.dump(data_to_save, f)
            logger.info(f"Optimizer models saved to {model_file}")
        except Exception as e:
            logger.error(f"Failed to save optimizer models: {e}")
    
    def load_models(self) -> bool:
        """Load trained models and history."""
        model_file = self.model_dir / "ml_optimizer.pkl"
        
        if not model_file.exists():
            logger.info("No saved optimizer models found")
            return False
        
        try:
            with open(model_file, 'rb') as f:
                data = pickle.load(f)
            
            self.models = data.get('models', self.models)
            self.scaler = data.get('scaler', StandardScaler())
            
            # Restore optimization history
            history_data = data.get('optimization_history', [])
            self.optimization_history = []
            for record_dict in history_data:
                # Convert dict back to OptimizationHistory
                record = OptimizationHistory(
                    timestamp=record_dict['timestamp'],
                    parameters=record_dict['parameters'],
                    performance=PerformanceMetrics(**record_dict['performance']),
                    context=record_dict['context'],
                    scraper_type=record_dict['scraper_type']
                )
                self.optimization_history.append(record)
            
            self.objective_weights = data.get('objective_weights', self.objective_weights)
            self.models_trained = data.get('models_trained', False)
            
            logger.info(f"Optimizer models loaded from {model_file}")
            logger.info(f"Loaded {len(self.optimization_history)} historical records")
            return True
            
        except Exception as e:
            logger.error(f"Failed to load optimizer models: {e}")
            return False


def create_ml_optimizer(model_dir: str = "models") -> MLParameterOptimizer:
    """Create and initialize an ML parameter optimizer."""
    optimizer = MLParameterOptimizer(model_dir)
    optimizer.load_models()
    return optimizer


if __name__ == "__main__":
    # Example usage
    
    # Create optimizer
    optimizer = create_ml_optimizer("/tmp/ml_optimizer")
    
    # Add some example performance records
    example_params = {
        'batch_size': 50,
        'max_concurrent': 3,
        'rate_limit_delay': 1.0,
        'timeout_seconds': 300,
        'max_places_per_search': 10
    }
    
    example_performance = PerformanceMetrics(
        success_rate=0.8,
        results_quality=0.75,
        processing_time=250.0,
        cost_per_result=0.05,
        throughput=8.0
    )
    
    example_context = {
        'data_size': 500,
        'complexity_score': 0.6,
        'has_phone_data': True,
        'has_address_data': True,
        'time_budget_minutes': 60
    }
    
    optimizer.add_performance_record(
        example_params, example_performance, example_context, "google_places"
    )
    
    # Add more examples for training
    for i in range(20):
        params = optimizer.parameter_space.get_random_parameters()
        
        # Simulate performance based on parameters
        performance = PerformanceMetrics(
            success_rate=np.random.uniform(0.5, 0.9),
            results_quality=np.random.uniform(0.4, 0.8),
            processing_time=np.random.uniform(100, 500),
            cost_per_result=np.random.uniform(0.02, 0.1),
            throughput=np.random.uniform(3, 12)
        )
        
        context = {
            'data_size': np.random.randint(50, 1000),
            'complexity_score': np.random.uniform(0.3, 0.8),
            'has_phone_data': np.random.choice([True, False]),
            'has_address_data': True,
            'time_budget_minutes': np.random.randint(30, 120)
        }
        
        optimizer.add_performance_record(params, performance, context)
    
    # Train models
    scores = optimizer.train_models()
    print(f"Model training scores: {scores}")
    
    # Optimize parameters
    test_context = {
        'data_size': 200,
        'complexity_score': 0.7,
        'has_phone_data': True,
        'has_address_data': True,
        'time_budget_minutes': 45
    }
    
    result = optimizer.optimize_parameters(test_context, optimization_method="ml_guided")
    
    print(f"\nOptimization result:")
    print(f"Method: {result.optimization_method}")
    print(f"Confidence: {result.confidence:.3f}")
    print(f"Improvement ratio: {result.improvement_ratio:.3f}")
    print(f"Optimal parameters: {result.optimal_params}")
    print(f"Expected performance: {result.expected_performance}")