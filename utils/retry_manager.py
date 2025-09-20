#!/usr/bin/env python3
"""
Smart retry logic for Apify scrapers with cost-aware limits.

This module provides intelligent retry strategies for failed searches,
particularly focused on LinkedIn and Google Maps with cost optimization.
"""

import re
import time
import logging
from typing import Dict, List, Optional, Any, Callable
from dataclasses import dataclass, field
from enum import Enum

from apify_client import ApifyClient

logger = logging.getLogger(__name__)


class RetryStrategy(Enum):
    """Available retry strategies."""
    SIMPLIFIED_NAME = "simplified_name"
    ALTERNATIVE_POSITIONS = "alternative_positions"
    BROADER_SEARCH = "broader_search"
    REDUCED_BATCH = "reduced_batch"
    INCREASED_TIMEOUT = "increased_timeout"


@dataclass
class RetryAttempt:
    """Represents a single retry attempt."""
    strategy: RetryStrategy
    attempt_number: int
    original_input: Dict[str, Any]
    modified_input: Dict[str, Any]
    timestamp: float = field(default_factory=time.time)
    cost_estimate: float = 0.0
    success: bool = False
    error_message: str = ""


@dataclass
class RetryConfig:
    """Configuration for retry behavior."""
    max_retries: int = 3
    max_cost_per_search: float = 150.0  # Maximum credits to spend on retries
    cost_escalation_factor: float = 1.5  # Cost multiplier for each retry
    timeout_escalation_factor: float = 1.2  # Timeout multiplier for each retry
    base_delay_seconds: float = 5.0
    exponential_backoff: bool = True
    enable_cost_tracking: bool = True
    
    # Strategy-specific settings
    strategies: List[RetryStrategy] = field(default_factory=lambda: [
        RetryStrategy.SIMPLIFIED_NAME,
        RetryStrategy.ALTERNATIVE_POSITIONS,
        RetryStrategy.BROADER_SEARCH
    ])


class LinkedInRetryManager:
    """Manage retries for failed LinkedIn searches."""
    
    def __init__(self, config: Optional[RetryConfig] = None):
        self.config = config or RetryConfig()
        self.retry_history: List[RetryAttempt] = []
        self.total_retry_cost = 0.0
        
        # Strategy mapping
        self.strategies = {
            RetryStrategy.SIMPLIFIED_NAME: self._retry_with_simplified_name,
            RetryStrategy.ALTERNATIVE_POSITIONS: self._retry_with_alternative_positions,
            RetryStrategy.BROADER_SEARCH: self._retry_with_broader_search,
            RetryStrategy.REDUCED_BATCH: self._retry_with_reduced_batch,
            RetryStrategy.INCREASED_TIMEOUT: self._retry_with_increased_timeout,
        }
        
        logger.info(
            f"LinkedInRetryManager initialized with max_retries={self.config.max_retries}, "
            f"max_cost={self.config.max_cost_per_search}"
        )
    
    def retry_failed_searches(
        self, 
        failed_searches: List[Dict], 
        client: ApifyClient,
        executor_func: Callable[[Dict, ApifyClient], List[Dict]]
    ) -> List[Dict]:
        """Retry failed LinkedIn searches with different strategies."""
        
        recovered_results = []
        
        for search in failed_searches:
            search_results = self._retry_single_search(search, client, executor_func)
            if search_results:
                recovered_results.extend(search_results)
        
        self._log_retry_summary()
        return recovered_results
    
    def _retry_single_search(
        self,
        search: Dict,
        client: ApifyClient,
        executor_func: Callable[[Dict, ApifyClient], List[Dict]]
    ) -> List[Dict]:
        """Retry a single search with multiple strategies."""
        
        company_name = search.get('company_name', 'Unknown')
        logger.info(f"Starting retry attempts for {company_name}")
        
        for attempt_num, strategy in enumerate(self.config.strategies, 1):
            if attempt_num > self.config.max_retries:
                logger.warning(f"Max retries ({self.config.max_retries}) exceeded for {company_name}")
                break
            
            # Check cost limits
            estimated_cost = self._estimate_retry_cost(attempt_num)
            if self.total_retry_cost + estimated_cost > self.config.max_cost_per_search:
                logger.warning(f"Cost limit would be exceeded for {company_name}, skipping retry")
                break
            
            try:
                logger.info(f"Retry attempt {attempt_num} for {company_name} using {strategy.value}")
                
                # Apply strategy to modify input
                strategy_func = self.strategies[strategy]
                modified_input = strategy_func(search, attempt_num)
                
                if not modified_input:
                    logger.info(f"Strategy {strategy.value} not applicable for {company_name}")
                    continue
                
                # Create retry attempt record
                retry_attempt = RetryAttempt(
                    strategy=strategy,
                    attempt_number=attempt_num,
                    original_input=search.copy(),
                    modified_input=modified_input,
                    cost_estimate=estimated_cost
                )
                
                # Add delay before retry
                self._apply_retry_delay(attempt_num)
                
                # Execute the search
                results = executor_func(modified_input, client)
                
                if results:
                    retry_attempt.success = True
                    self.total_retry_cost += estimated_cost
                    self.retry_history.append(retry_attempt)
                    
                    logger.info(
                        f"Retry successful for {company_name} using {strategy.value}, "
                        f"found {len(results)} results"
                    )
                    return results
                else:
                    retry_attempt.error_message = "No results returned"
                    self.retry_history.append(retry_attempt)
                    
            except Exception as e:
                logger.error(f"Retry attempt {attempt_num} failed for {company_name}: {e}")
                if self.retry_history:
                    self.retry_history[-1].error_message = str(e)
                continue
        
        logger.warning(f"All retry attempts failed for {company_name}")
        return []
    
    def _retry_with_simplified_name(self, search: Dict, attempt_num: int) -> Optional[Dict]:
        """Retry with simplified company name."""
        original_name = search.get('company_name', '')
        if not original_name:
            return None
        
        # Remove common company suffixes and legal forms
        simplified = re.sub(r'\b(SAS|SARL|SA|EURL|LTD|LLC|INC|CORP)\b', '', original_name, flags=re.IGNORECASE)
        simplified = re.sub(r'[^\w\s]', ' ', simplified)  # Remove special characters
        simplified = ' '.join(simplified.split())  # Normalize whitespace
        
        if simplified != original_name and len(simplified) > 3:
            search_copy = search.copy()
            search_copy['company_name'] = simplified
            if 'searchTerms' in search_copy:
                search_copy['searchTerms'] = [simplified]
            
            logger.debug(f"Simplified name: '{original_name}' -> '{simplified}'")
            return search_copy
        
        return None
    
    def _retry_with_alternative_positions(self, search: Dict, attempt_num: int) -> Optional[Dict]:
        """Retry with different executive positions."""
        alternative_positions = [
            ['CEO', 'Manager', 'Fondateur', 'Directeur'],
            ['CFO', 'Responsable', 'Comptable', 'Finance'],
            ['CTO', 'Technique', 'IT', 'Informatique'],
            ['President', 'Dirigeant', 'Chef']
        ]
        
        if attempt_num <= len(alternative_positions):
            positions = alternative_positions[attempt_num - 1]
            search_copy = search.copy()
            
            if 'filters' not in search_copy:
                search_copy['filters'] = {}
            
            search_copy['filters']['positions'] = positions
            logger.debug(f"Alternative positions: {positions}")
            return search_copy
        
        return None
    
    def _retry_with_broader_search(self, search: Dict, attempt_num: int) -> Optional[Dict]:
        """Retry with broader search parameters."""
        search_copy = search.copy()
        
        # Increase profile limit
        current_max = search_copy.get('maxProfiles', 5)
        search_copy['maxProfiles'] = min(current_max + (attempt_num * 2), 15)
        
        # Remove strict filters gradually
        if 'filters' in search_copy:
            if attempt_num >= 2:
                search_copy['filters'].pop('positions', None)
            if attempt_num >= 3:
                search_copy['filters'].pop('locations', None)
        
        logger.debug(f"Broader search: maxProfiles={search_copy['maxProfiles']}")
        return search_copy
    
    def _retry_with_reduced_batch(self, search: Dict, attempt_num: int) -> Optional[Dict]:
        """Retry with reduced batch size for better reliability."""
        search_copy = search.copy()
        
        # Reduce the number of search terms or companies
        if 'searchTerms' in search_copy and isinstance(search_copy['searchTerms'], list):
            original_terms = search_copy['searchTerms']
            reduced_size = max(1, len(original_terms) // (attempt_num + 1))
            search_copy['searchTerms'] = original_terms[:reduced_size]
            
            logger.debug(f"Reduced batch size: {len(original_terms)} -> {reduced_size}")
            return search_copy
        
        return None
    
    def _retry_with_increased_timeout(self, search: Dict, attempt_num: int) -> Optional[Dict]:
        """Retry with increased timeout."""
        search_copy = search.copy()
        
        # Increase timeout progressively
        timeout_multiplier = self.config.timeout_escalation_factor ** attempt_num
        base_timeout = search_copy.get('timeout', 300)
        new_timeout = int(base_timeout * timeout_multiplier)
        
        search_copy['timeout'] = min(new_timeout, 1800)  # Cap at 30 minutes
        
        logger.debug(f"Increased timeout: {base_timeout} -> {new_timeout}")
        return search_copy
    
    def _estimate_retry_cost(self, attempt_num: int) -> float:
        """Estimate the cost of a retry attempt."""
        base_cost = 30.0  # Base LinkedIn search cost
        escalated_cost = base_cost * (self.config.cost_escalation_factor ** (attempt_num - 1))
        return min(escalated_cost, 100.0)  # Cap per attempt
    
    def _apply_retry_delay(self, attempt_num: int) -> None:
        """Apply delay before retry attempt."""
        if self.config.exponential_backoff:
            delay = self.config.base_delay_seconds * (2 ** (attempt_num - 1))
        else:
            delay = self.config.base_delay_seconds
        
        # Cap delay at 60 seconds
        delay = min(delay, 60.0)
        
        if delay > 0:
            logger.debug(f"Applying retry delay: {delay:.1f} seconds")
            time.sleep(delay)
    
    def _log_retry_summary(self) -> None:
        """Log summary of retry attempts."""
        if not self.retry_history:
            return
        
        successful_retries = sum(1 for r in self.retry_history if r.success)
        total_attempts = len(self.retry_history)
        
        logger.info(
            f"Retry summary: {successful_retries}/{total_attempts} successful, "
            f"total cost: {self.total_retry_cost:.1f} credits"
        )
        
        # Log strategy effectiveness
        strategy_stats = {}
        for retry in self.retry_history:
            strategy = retry.strategy.value
            if strategy not in strategy_stats:
                strategy_stats[strategy] = {'attempts': 0, 'successes': 0}
            
            strategy_stats[strategy]['attempts'] += 1
            if retry.success:
                strategy_stats[strategy]['successes'] += 1
        
        for strategy, stats in strategy_stats.items():
            success_rate = (stats['successes'] / stats['attempts']) * 100
            logger.info(f"Strategy {strategy}: {success_rate:.1f}% success rate")
    
    def get_retry_stats(self) -> Dict[str, Any]:
        """Get detailed retry statistics."""
        if not self.retry_history:
            return {"total_attempts": 0, "total_cost": 0.0}
        
        successful_retries = sum(1 for r in self.retry_history if r.success)
        strategy_breakdown = {}
        
        for retry in self.retry_history:
            strategy = retry.strategy.value
            if strategy not in strategy_breakdown:
                strategy_breakdown[strategy] = {
                    'attempts': 0, 
                    'successes': 0, 
                    'total_cost': 0.0
                }
            
            strategy_breakdown[strategy]['attempts'] += 1
            strategy_breakdown[strategy]['total_cost'] += retry.cost_estimate
            if retry.success:
                strategy_breakdown[strategy]['successes'] += 1
        
        return {
            "total_attempts": len(self.retry_history),
            "successful_attempts": successful_retries,
            "success_rate": (successful_retries / len(self.retry_history)) * 100,
            "total_cost": self.total_retry_cost,
            "strategy_breakdown": strategy_breakdown,
            "cost_efficiency": successful_retries / max(self.total_retry_cost, 1)
        }


class GoogleMapsRetryManager:
    """Manage retries for failed Google Maps searches."""
    
    def __init__(self, config: Optional[RetryConfig] = None):
        self.config = config or RetryConfig()
        self.config.strategies = [
            RetryStrategy.SIMPLIFIED_NAME,
            RetryStrategy.BROADER_SEARCH,
            RetryStrategy.REDUCED_BATCH
        ]
        self.retry_history: List[RetryAttempt] = []
        
    def retry_failed_searches(
        self,
        failed_searches: List[Dict],
        client: ApifyClient,
        executor_func: Callable[[Dict, ApifyClient], List[Dict]]
    ) -> List[Dict]:
        """Retry failed Google Maps searches."""
        
        recovered_results = []
        
        for search in failed_searches:
            # Apply Google Maps specific retry strategies
            for attempt_num, strategy in enumerate(self.config.strategies, 1):
                if attempt_num > self.config.max_retries:
                    break
                
                try:
                    modified_input = self._apply_maps_strategy(search, strategy, attempt_num)
                    if modified_input:
                        results = executor_func(modified_input, client)
                        if results:
                            recovered_results.extend(results)
                            break
                except Exception as e:
                    logger.error(f"Google Maps retry failed: {e}")
                    continue
        
        return recovered_results
    
    def _apply_maps_strategy(self, search: Dict, strategy: RetryStrategy, attempt_num: int) -> Optional[Dict]:
        """Apply Google Maps specific retry strategy."""
        
        if strategy == RetryStrategy.SIMPLIFIED_NAME:
            return self._simplify_address(search)
        elif strategy == RetryStrategy.BROADER_SEARCH:
            return self._broaden_location_search(search, attempt_num)
        elif strategy == RetryStrategy.REDUCED_BATCH:
            return self._reduce_search_precision(search, attempt_num)
        
        return None
    
    def _simplify_address(self, search: Dict) -> Optional[Dict]:
        """Simplify address for better matching."""
        address = search.get('address', '')
        if not address:
            return None
        
        # Remove apartment numbers, building details  
        simplified = re.sub(r'\b(apt|apartment|suite|unit|floor)\s*\d+[a-z]?\b', '', address, flags=re.IGNORECASE)
        simplified = re.sub(r'\d+[a-z]\b', '', simplified)  # Remove "123a" style
        simplified = ' '.join(simplified.split())
        
        if simplified != address:
            search_copy = search.copy()
            search_copy['address'] = simplified
            return search_copy
        
        return None
    
    def _broaden_location_search(self, search: Dict, attempt_num: int) -> Optional[Dict]:
        """Broaden the location search scope."""
        search_copy = search.copy()
        
        # Increase search radius
        current_radius = search_copy.get('searchRadius', 1000)
        search_copy['searchRadius'] = current_radius * (1.5 ** attempt_num)
        
        # Increase max results
        current_max = search_copy.get('maxResults', 10)
        search_copy['maxResults'] = min(current_max + (attempt_num * 5), 50)
        
        return search_copy
    
    def _reduce_search_precision(self, search: Dict, attempt_num: int) -> Optional[Dict]:
        """Reduce search precision for broader matching."""
        search_copy = search.copy()
        
        # Remove specific filters
        if 'categories' in search_copy and attempt_num >= 2:
            # Keep only primary category
            categories = search_copy['categories']
            if isinstance(categories, list) and len(categories) > 1:
                search_copy['categories'] = categories[:1]
        
        return search_copy