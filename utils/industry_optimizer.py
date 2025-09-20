#!/usr/bin/env python3
"""
Industry-specific optimization for Apify scrapers.

This module provides industry-specific parameter optimization and
intelligent scraper selection based on business context and industry patterns.
"""

import re
import logging
from typing import Dict, List, Optional, Any, Tuple, Set
from dataclasses import dataclass, field
from enum import Enum
from collections import defaultdict

logger = logging.getLogger(__name__)


class IndustryCategory(Enum):
    """Major industry categories for optimization."""
    TECHNOLOGY = "technology"
    HEALTHCARE = "healthcare"
    FINANCE = "finance"
    RETAIL = "retail"
    MANUFACTURING = "manufacturing"
    PROFESSIONAL_SERVICES = "professional_services"
    CONSTRUCTION = "construction"
    HOSPITALITY = "hospitality"
    EDUCATION = "education"
    REAL_ESTATE = "real_estate"
    AUTOMOTIVE = "automotive"
    FOOD_BEVERAGE = "food_beverage"
    TRANSPORTATION = "transportation"
    ENERGY = "energy"
    MEDIA = "media"
    AGRICULTURE = "agriculture"
    NON_PROFIT = "non_profit"
    GOVERNMENT = "government"
    OTHER = "other"


class OptimizationStrategy(Enum):
    """Optimization strategies for different objectives."""
    CONTACT_FOCUSED = "contact_focused"
    EXECUTIVE_FOCUSED = "executive_focused"
    LOCATION_FOCUSED = "location_focused"
    COMPREHENSIVE = "comprehensive"
    COST_EFFICIENT = "cost_efficient"


@dataclass
class IndustryProfile:
    """Profile containing industry-specific optimization parameters."""
    industry: IndustryCategory
    
    # Scraper priorities (0.0 to 1.0)
    google_places_priority: float = 0.7
    google_maps_contacts_priority: float = 0.6
    linkedin_premium_priority: float = 0.5
    
    # LinkedIn-specific settings
    linkedin_positions: List[str] = field(default_factory=list)
    linkedin_keywords: List[str] = field(default_factory=list)
    linkedin_company_size_preference: Optional[str] = None
    
    # Google Places categories and keywords
    google_places_categories: List[str] = field(default_factory=list)
    google_places_keywords: List[str] = field(default_factory=list)
    
    # Contact preferences
    preferred_contact_types: List[str] = field(default_factory=lambda: ["email", "phone", "website"])
    contact_validation_strictness: float = 0.7  # 0.0 to 1.0
    
    # Quality thresholds
    min_data_completeness: float = 0.6
    require_verified_info: bool = False
    
    # Cost optimization factors
    cost_efficiency_weight: float = 0.5  # Balance between cost and quality
    timeout_tolerance: float = 1.0  # Multiplier for standard timeouts
    
    # Common company name patterns
    company_name_patterns: List[str] = field(default_factory=list)
    company_name_exclusions: List[str] = field(default_factory=list)


# Industry-specific profiles
INDUSTRY_PROFILES = {
    IndustryCategory.TECHNOLOGY: IndustryProfile(
        industry=IndustryCategory.TECHNOLOGY,
        google_places_priority=0.8,
        google_maps_contacts_priority=0.9,
        linkedin_premium_priority=0.9,
        linkedin_positions=["CTO", "CEO", "VP Engineering", "Software Engineer", "Developer", "Tech Lead", "Architect"],
        linkedin_keywords=["software", "tech", "development", "engineering", "IT", "digital"],
        linkedin_company_size_preference="startup_to_large",
        google_places_categories=["software_company", "computer_service", "electronics_store"],
        google_places_keywords=["software", "technology", "IT services", "development", "digital"],
        preferred_contact_types=["email", "website", "linkedin"],
        contact_validation_strictness=0.8,
        min_data_completeness=0.8,
        cost_efficiency_weight=0.3,  # Willing to pay more for quality
        timeout_tolerance=1.2,
        company_name_patterns=[r".*tech.*", r".*soft.*", r".*digital.*", r".*systems.*", r".*solutions.*"],
        company_name_exclusions=["hardware store", "tech repair"]
    ),
    
    IndustryCategory.HEALTHCARE: IndustryProfile(
        industry=IndustryCategory.HEALTHCARE,
        google_places_priority=0.9,
        google_maps_contacts_priority=0.8,
        linkedin_premium_priority=0.6,
        linkedin_positions=["MD", "Doctor", "Physician", "Administrator", "Director", "Manager", "RN"],
        linkedin_keywords=["healthcare", "medical", "hospital", "clinic", "physician", "nurse"],
        google_places_categories=["hospital", "doctor", "medical_center", "pharmacy", "dentist"],
        google_places_keywords=["medical", "healthcare", "clinic", "hospital", "doctor"],
        preferred_contact_types=["phone", "email", "website"],
        contact_validation_strictness=0.9,  # High strictness for healthcare
        min_data_completeness=0.7,
        require_verified_info=True,
        cost_efficiency_weight=0.6,
        timeout_tolerance=1.5,  # Healthcare searches can take longer
        company_name_patterns=[r".*medical.*", r".*health.*", r".*clinic.*", r".*hospital.*"],
        company_name_exclusions=["veterinary", "animal"]
    ),
    
    IndustryCategory.FINANCE: IndustryProfile(
        industry=IndustryCategory.FINANCE,
        google_places_priority=0.7,
        google_maps_contacts_priority=0.8,
        linkedin_premium_priority=0.9,
        linkedin_positions=["CFO", "CEO", "VP Finance", "Financial Advisor", "Analyst", "Manager", "Director"],
        linkedin_keywords=["finance", "banking", "investment", "accounting", "financial"],
        google_places_categories=["bank", "atm", "accounting", "financial_institution"],
        google_places_keywords=["bank", "finance", "investment", "accounting", "credit"],
        preferred_contact_types=["email", "phone", "linkedin"],
        contact_validation_strictness=0.9,
        min_data_completeness=0.8,
        require_verified_info=True,
        cost_efficiency_weight=0.4,  # Finance values quality
        timeout_tolerance=1.1,
        company_name_patterns=[r".*bank.*", r".*finance.*", r".*capital.*", r".*investment.*"],
        company_name_exclusions=["food bank", "river bank"]
    ),
    
    IndustryCategory.RETAIL: IndustryProfile(
        industry=IndustryCategory.RETAIL,
        google_places_priority=0.9,
        google_maps_contacts_priority=0.7,
        linkedin_premium_priority=0.4,
        linkedin_positions=["Manager", "Director", "CEO", "Owner", "Supervisor"],
        linkedin_keywords=["retail", "store", "sales", "merchandise", "customer"],
        google_places_categories=["store", "shopping_mall", "clothing_store", "electronics_store"],
        google_places_keywords=["store", "shop", "retail", "boutique", "market"],
        preferred_contact_types=["phone", "website", "email"],
        contact_validation_strictness=0.6,
        min_data_completeness=0.6,
        cost_efficiency_weight=0.7,  # Cost-conscious industry
        timeout_tolerance=0.9,
        company_name_patterns=[r".*store.*", r".*shop.*", r".*boutique.*", r".*retail.*"],
        company_name_exclusions=["wholesale", "warehouse"]
    ),
    
    IndustryCategory.PROFESSIONAL_SERVICES: IndustryProfile(
        industry=IndustryCategory.PROFESSIONAL_SERVICES,
        google_places_priority=0.6,
        google_maps_contacts_priority=0.8,
        linkedin_premium_priority=0.8,
        linkedin_positions=["Partner", "Principal", "Director", "Manager", "Consultant", "Advisor"],
        linkedin_keywords=["consulting", "advisory", "professional", "services", "strategy"],
        google_places_categories=["lawyer", "accounting", "consultant"],
        google_places_keywords=["consulting", "advisory", "professional", "services", "law"],
        preferred_contact_types=["email", "linkedin", "phone"],
        contact_validation_strictness=0.8,
        min_data_completeness=0.7,
        cost_efficiency_weight=0.4,
        timeout_tolerance=1.0,
        company_name_patterns=[r".*consulting.*", r".*advisory.*", r".*services.*", r".*associates.*"],
        company_name_exclusions=["repair services", "cleaning services"]
    ),
    
    IndustryCategory.CONSTRUCTION: IndustryProfile(
        industry=IndustryCategory.CONSTRUCTION,
        google_places_priority=0.8,
        google_maps_contacts_priority=0.6,
        linkedin_premium_priority=0.3,
        linkedin_positions=["Owner", "Manager", "Supervisor", "Foreman", "Director", "CEO"],
        linkedin_keywords=["construction", "building", "contractor", "engineering"],
        google_places_categories=["general_contractor", "electrician", "plumber"],
        google_places_keywords=["construction", "contractor", "building", "renovation"],
        preferred_contact_types=["phone", "email", "website"],
        contact_validation_strictness=0.6,
        min_data_completeness=0.5,
        cost_efficiency_weight=0.8,  # Very cost-conscious
        timeout_tolerance=0.8,
        company_name_patterns=[r".*construction.*", r".*building.*", r".*contractor.*"],
        company_name_exclusions=["building supplies", "construction equipment rental"]
    )
}

# Add remaining industry profiles with similar patterns...
for industry in IndustryCategory:
    if industry not in INDUSTRY_PROFILES:
        INDUSTRY_PROFILES[industry] = IndustryProfile(
            industry=industry,
            linkedin_positions=["CEO", "Manager", "Director", "Owner"],
            linkedin_keywords=[industry.value.replace("_", " ")],
            google_places_keywords=[industry.value.replace("_", " ")],
        )


class IndustryOptimizer:
    """Optimizes scraper configuration based on industry characteristics."""
    
    def __init__(self):
        self.industry_profiles = INDUSTRY_PROFILES.copy()
        self.optimization_history: List[Dict[str, Any]] = []
        
        # NAF code to industry mapping (French business classification)
        self.naf_to_industry = self._build_naf_mapping()
        
        logger.info("IndustryOptimizer initialized with {} industry profiles".format(
            len(self.industry_profiles)
        ))
    
    def _build_naf_mapping(self) -> Dict[str, IndustryCategory]:
        """Build mapping from NAF codes to industry categories."""
        
        # French NAF codes mapped to our industry categories
        naf_mapping = {
            # Technology & Software
            "6201Z": IndustryCategory.TECHNOLOGY,  # Computer programming
            "6202A": IndustryCategory.TECHNOLOGY,  # Computer consulting
            "6209Z": IndustryCategory.TECHNOLOGY,  # Other IT services
            "6311Z": IndustryCategory.TECHNOLOGY,  # Data processing
            "6312Z": IndustryCategory.TECHNOLOGY,  # Web portals
            
            # Healthcare
            "8610Z": IndustryCategory.HEALTHCARE,  # Hospital activities
            "8621Z": IndustryCategory.HEALTHCARE,  # General medical practice
            "8622A": IndustryCategory.HEALTHCARE,  # Specialist medical practice
            "8690A": IndustryCategory.HEALTHCARE,  # Ambulances
            "8690B": IndustryCategory.HEALTHCARE,  # Medical laboratories
            
            # Finance & Banking
            "6411Z": IndustryCategory.FINANCE,  # Central banking
            "6419Z": IndustryCategory.FINANCE,  # Other monetary intermediation
            "6420Z": IndustryCategory.FINANCE,  # Activities of holding companies
            "6430Z": IndustryCategory.FINANCE,  # Trusts, funds
            "6492Z": IndustryCategory.FINANCE,  # Other credit granting
            
            # Retail
            "4711A": IndustryCategory.RETAIL,  # Retail sale in non-specialized stores
            "4719A": IndustryCategory.RETAIL,  # Other retail sale in non-specialized stores
            "4771Z": IndustryCategory.RETAIL,  # Retail sale of clothing
            "4772A": IndustryCategory.RETAIL,  # Retail sale of footwear
            "4774Z": IndustryCategory.RETAIL,  # Retail sale of medical goods
            
            # Construction
            "4120A": IndustryCategory.CONSTRUCTION,  # Construction of residential buildings
            "4120B": IndustryCategory.CONSTRUCTION,  # Construction of non-residential buildings
            "4211Z": IndustryCategory.CONSTRUCTION,  # Construction of roads
            "4212Z": IndustryCategory.CONSTRUCTION,  # Construction of railways
            "4213A": IndustryCategory.CONSTRUCTION,  # Construction of bridges
            
            # Professional Services
            "6920Z": IndustryCategory.PROFESSIONAL_SERVICES,  # Accounting
            "6910Z": IndustryCategory.PROFESSIONAL_SERVICES,  # Legal activities
            "7022Z": IndustryCategory.PROFESSIONAL_SERVICES,  # Business consulting
            "7111Z": IndustryCategory.PROFESSIONAL_SERVICES,  # Architectural activities
            "7112A": IndustryCategory.PROFESSIONAL_SERVICES,  # Engineering activities
        }
        
        return naf_mapping
    
    def detect_industry(self, business_data: Dict[str, Any]) -> Tuple[IndustryCategory, float]:
        """Detect industry category from business data."""
        
        confidence_scores = defaultdict(float)
        
        # Check NAF code if available
        naf_code = business_data.get("naf_code", "").strip()
        if naf_code in self.naf_to_industry:
            industry = self.naf_to_industry[naf_code]
            confidence_scores[industry] += 0.8
            logger.debug(f"Industry detected from NAF code {naf_code}: {industry.value}")
        
        # Analyze company name
        company_name = business_data.get("company_name", "").lower()
        if company_name:
            for industry, profile in self.industry_profiles.items():
                name_score = self._score_company_name(company_name, profile)
                confidence_scores[industry] += name_score * 0.3
        
        # Analyze business description/activity
        description = business_data.get("description", "").lower()
        activity = business_data.get("activity", "").lower()
        combined_text = f"{description} {activity}".strip()
        
        if combined_text:
            for industry, profile in self.industry_profiles.items():
                text_score = self._score_business_text(combined_text, profile)
                confidence_scores[industry] += text_score * 0.4
        
        # Analyze address/location context
        address = business_data.get("address", "").lower()
        if address:
            location_score = self._score_location_context(address)
            for industry, score in location_score.items():
                confidence_scores[industry] += score * 0.1
        
        # Return industry with highest confidence
        if confidence_scores:
            best_industry = max(confidence_scores.keys(), key=lambda k: confidence_scores[k])
            best_confidence = confidence_scores[best_industry]
            
            # Normalize confidence to 0-1 range
            normalized_confidence = min(1.0, best_confidence)
            
            logger.debug(f"Industry detection result: {best_industry.value} (confidence: {normalized_confidence:.2f})")
            return best_industry, normalized_confidence
        
        return IndustryCategory.OTHER, 0.0
    
    def _score_company_name(self, company_name: str, profile: IndustryProfile) -> float:
        """Score company name against industry profile."""
        
        score = 0.0
        
        # Check positive patterns
        for pattern in profile.company_name_patterns:
            if re.search(pattern, company_name, re.IGNORECASE):
                score += 0.3
        
        # Check negative patterns (exclusions)
        for exclusion in profile.company_name_exclusions:
            if exclusion.lower() in company_name:
                score -= 0.5
        
        # Check industry keywords
        for keyword in profile.linkedin_keywords + profile.google_places_keywords:
            if keyword.lower() in company_name:
                score += 0.2
        
        return max(0.0, score)
    
    def _score_business_text(self, text: str, profile: IndustryProfile) -> float:
        """Score business description text against industry profile."""
        
        score = 0.0
        words = text.split()
        
        # Count keyword matches
        keyword_matches = 0
        total_keywords = len(profile.linkedin_keywords + profile.google_places_keywords)
        
        for keyword in profile.linkedin_keywords + profile.google_places_keywords:
            if keyword.lower() in text:
                keyword_matches += 1
        
        if total_keywords > 0:
            score = keyword_matches / total_keywords
        
        return min(1.0, score)
    
    def _score_location_context(self, address: str) -> Dict[IndustryCategory, float]:
        """Score location context for industry hints."""
        
        scores = defaultdict(float)
        
        # Business district indicators
        if any(term in address for term in ["business park", "industrial", "tech park", "innovation"]):
            scores[IndustryCategory.TECHNOLOGY] += 0.3
            scores[IndustryCategory.PROFESSIONAL_SERVICES] += 0.2
        
        # Medical district indicators
        if any(term in address for term in ["medical center", "hospital", "clinic"]):
            scores[IndustryCategory.HEALTHCARE] += 0.4
        
        # Financial district indicators
        if any(term in address for term in ["financial district", "banking center", "wall street"]):
            scores[IndustryCategory.FINANCE] += 0.3
        
        # Retail/commercial indicators
        if any(term in address for term in ["shopping", "mall", "commercial", "retail"]):
            scores[IndustryCategory.RETAIL] += 0.3
        
        return dict(scores)
    
    def optimize_configuration(
        self,
        base_config: Dict[str, Any],
        industry: IndustryCategory,
        optimization_strategy: OptimizationStrategy,
        context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Optimize configuration for specific industry and strategy."""
        
        profile = self.industry_profiles.get(industry, self.industry_profiles[IndustryCategory.OTHER])
        optimized_config = self._apply_industry_optimizations(base_config, profile, optimization_strategy)
        
        # Apply strategy-specific optimizations
        optimized_config = self._apply_strategy_optimizations(optimized_config, optimization_strategy, profile)
        
        # Record optimization decision
        self.optimization_history.append({
            "timestamp": time.time(),
            "industry": industry.value,
            "strategy": optimization_strategy.value,
            "profile_used": profile.industry.value,
            "context": context
        })
        
        logger.info(f"Configuration optimized for {industry.value} industry using {optimization_strategy.value} strategy")
        
        return optimized_config
    
    def _apply_industry_optimizations(
        self,
        base_config: Dict[str, Any],
        profile: IndustryProfile,
        strategy: OptimizationStrategy
    ) -> Dict[str, Any]:
        """Apply industry-specific optimizations to configuration."""
        
        import copy
        config = copy.deepcopy(base_config)
        apify_config = config.setdefault("apify", {})
        
        # Adjust scraper priorities
        self._adjust_scraper_priorities(apify_config, profile)
        
        # Configure LinkedIn Premium for industry
        self._configure_linkedin_for_industry(apify_config, profile)
        
        # Configure Google Places for industry
        self._configure_google_places_for_industry(apify_config, profile)
        
        # Configure contact enrichment for industry
        self._configure_contacts_for_industry(apify_config, profile)
        
        # Adjust timeouts based on industry tolerance
        self._adjust_timeouts_for_industry(apify_config, profile)
        
        # Configure quality thresholds
        self._configure_quality_for_industry(apify_config, profile)
        
        return config
    
    def _adjust_scraper_priorities(self, apify_config: Dict[str, Any], profile: IndustryProfile) -> None:
        """Adjust scraper enable/disable based on industry priorities."""
        
        # Google Places
        google_places = apify_config.setdefault("google_places", {})
        if profile.google_places_priority < 0.3:
            google_places["enabled"] = False
        else:
            google_places["enabled"] = True
            # Adjust search intensity based on priority
            base_places = google_places.get("max_places_per_search", 10)
            google_places["max_places_per_search"] = int(base_places * profile.google_places_priority)
        
        # Google Maps Contacts
        contacts = apify_config.setdefault("google_maps_contacts", {})
        if profile.google_maps_contacts_priority < 0.3:
            contacts["enabled"] = False
        else:
            contacts["enabled"] = True
            base_enrichments = contacts.get("max_contact_enrichments", 25)
            contacts["max_contact_enrichments"] = int(base_enrichments * profile.google_maps_contacts_priority)
        
        # LinkedIn Premium
        linkedin = apify_config.setdefault("linkedin_premium", {})
        if profile.linkedin_premium_priority < 0.3:
            linkedin["enabled"] = False
        else:
            linkedin["enabled"] = True
            base_searches = linkedin.get("max_linkedin_searches", 10)
            linkedin["max_linkedin_searches"] = int(base_searches * profile.linkedin_premium_priority)
    
    def _configure_linkedin_for_industry(self, apify_config: Dict[str, Any], profile: IndustryProfile) -> None:
        """Configure LinkedIn Premium settings for industry."""
        
        linkedin = apify_config.setdefault("linkedin_premium", {})
        
        if linkedin.get("enabled", True) and profile.linkedin_positions:
            # Set industry-specific positions
            linkedin.setdefault("filters", {})["positions"] = profile.linkedin_positions[:10]  # Limit to top 10
            
            # Add industry keywords for search refinement
            if profile.linkedin_keywords:
                linkedin["keywords"] = profile.linkedin_keywords[:5]  # Top 5 keywords
            
            # Adjust profile count based on industry (some industries have fewer executives)
            if profile.industry in [IndustryCategory.CONSTRUCTION, IndustryCategory.RETAIL]:
                # Industries with typically fewer LinkedIn profiles
                linkedin["max_profiles_per_company"] = max(2, linkedin.get("max_profiles_per_company", 5) - 2)
            elif profile.industry in [IndustryCategory.TECHNOLOGY, IndustryCategory.FINANCE]:
                # Industries with more LinkedIn presence
                linkedin["max_profiles_per_company"] = min(8, linkedin.get("max_profiles_per_company", 5) + 2)
    
    def _configure_google_places_for_industry(self, apify_config: Dict[str, Any], profile: IndustryProfile) -> None:
        """Configure Google Places settings for industry."""
        
        google_places = apify_config.setdefault("google_places", {})
        
        if google_places.get("enabled", True):
            # Set industry-specific categories
            if profile.google_places_categories:
                google_places["categories"] = profile.google_places_categories
            
            # Set industry-specific keywords
            if profile.google_places_keywords:
                google_places["keywords"] = profile.google_places_keywords[:10]
            
            # Adjust search radius based on industry (some are more location-specific)
            if profile.industry in [IndustryCategory.HEALTHCARE, IndustryCategory.RETAIL, IndustryCategory.HOSPITALITY]:
                # Location-specific industries - smaller radius for precision
                google_places["search_radius"] = 1000  # 1km
            elif profile.industry in [IndustryCategory.PROFESSIONAL_SERVICES, IndustryCategory.TECHNOLOGY]:
                # Service industries - larger radius
                google_places["search_radius"] = 5000  # 5km
    
    def _configure_contacts_for_industry(self, apify_config: Dict[str, Any], profile: IndustryProfile) -> None:
        """Configure contact enrichment for industry."""
        
        contacts = apify_config.setdefault("google_maps_contacts", {})
        
        if contacts.get("enabled", True):
            # Set preferred contact types
            contacts["preferred_contact_types"] = profile.preferred_contact_types
            
            # Adjust validation strictness
            contacts["validation_strictness"] = profile.contact_validation_strictness
            
            # Configure required verification for sensitive industries
            if profile.require_verified_info:
                contacts["require_verification"] = True
                contacts["skip_unverified"] = True
    
    def _adjust_timeouts_for_industry(self, apify_config: Dict[str, Any], profile: IndustryProfile) -> None:
        """Adjust timeouts based on industry tolerance."""
        
        base_timeout_multiplier = profile.timeout_tolerance
        
        for scraper in ["google_places", "google_maps_contacts", "linkedin_premium"]:
            if scraper in apify_config:
                current_timeout = apify_config[scraper].get("timeout_seconds", 300)
                adjusted_timeout = int(current_timeout * base_timeout_multiplier)
                apify_config[scraper]["timeout_seconds"] = max(60, min(1200, adjusted_timeout))  # Cap between 1-20 minutes
    
    def _configure_quality_for_industry(self, apify_config: Dict[str, Any], profile: IndustryProfile) -> None:
        """Configure quality thresholds for industry."""
        
        quality_control = apify_config.setdefault("quality_control", {})
        
        # Set minimum data completeness
        quality_control["min_data_completeness"] = profile.min_data_completeness
        
        # Configure verification requirements
        quality_control["require_verified_info"] = profile.require_verified_info
        
        # Set contact validation strictness
        quality_control["contact_validation_strictness"] = profile.contact_validation_strictness
    
    def _apply_strategy_optimizations(
        self,
        config: Dict[str, Any],
        strategy: OptimizationStrategy,
        profile: IndustryProfile
    ) -> Dict[str, Any]:
        """Apply strategy-specific optimizations."""
        
        apify_config = config.get("apify", {})
        
        if strategy == OptimizationStrategy.CONTACT_FOCUSED:
            # Maximize contact enrichment
            apify_config["google_maps_contacts"]["enabled"] = True
            current_enrichments = apify_config["google_maps_contacts"].get("max_contact_enrichments", 25)
            apify_config["google_maps_contacts"]["max_contact_enrichments"] = min(100, int(current_enrichments * 1.5))
            
            # Reduce LinkedIn to save budget for contacts
            if apify_config.get("linkedin_premium", {}).get("enabled"):
                linkedin_searches = apify_config["linkedin_premium"].get("max_linkedin_searches", 10)
                apify_config["linkedin_premium"]["max_linkedin_searches"] = max(3, int(linkedin_searches * 0.6))
        
        elif strategy == OptimizationStrategy.EXECUTIVE_FOCUSED:
            # Maximize LinkedIn Premium
            apify_config["linkedin_premium"]["enabled"] = True
            current_searches = apify_config["linkedin_premium"].get("max_linkedin_searches", 10)
            apify_config["linkedin_premium"]["max_linkedin_searches"] = min(50, int(current_searches * 1.8))
            
            profiles_per_company = apify_config["linkedin_premium"].get("max_profiles_per_company", 5)
            apify_config["linkedin_premium"]["max_profiles_per_company"] = min(10, profiles_per_company + 3)
            
            # Reduce other scrapers to save budget
            if apify_config.get("google_maps_contacts", {}).get("enabled"):
                contacts = apify_config["google_maps_contacts"].get("max_contact_enrichments", 25)
                apify_config["google_maps_contacts"]["max_contact_enrichments"] = max(5, int(contacts * 0.5))
        
        elif strategy == OptimizationStrategy.LOCATION_FOCUSED:
            # Maximize Google Places
            apify_config["google_places"]["enabled"] = True
            current_places = apify_config["google_places"].get("max_places_per_search", 10)
            apify_config["google_places"]["max_places_per_search"] = min(25, int(current_places * 1.5))
            
            # Enhance location precision
            apify_config["google_places"]["search_radius"] = 500  # Very precise
            apify_config["google_places"]["require_exact_match"] = True
        
        elif strategy == OptimizationStrategy.COST_EFFICIENT:
            # Minimize costs while maintaining basic functionality
            apify_config["max_addresses"] = max(5, int(apify_config.get("max_addresses", 10) * 0.7))
            
            # Reduce expensive operations
            if apify_config.get("linkedin_premium", {}).get("enabled"):
                linkedin_searches = apify_config["linkedin_premium"].get("max_linkedin_searches", 10)
                apify_config["linkedin_premium"]["max_linkedin_searches"] = max(2, int(linkedin_searches * 0.4))
            
            contacts = apify_config.get("google_maps_contacts", {}).get("max_contact_enrichments", 25)
            apify_config["google_maps_contacts"]["max_contact_enrichments"] = max(5, int(contacts * 0.5))
            
            # Reduce retry attempts
            retry_settings = apify_config.setdefault("retry_settings", {})
            retry_settings["max_retries"] = min(2, retry_settings.get("max_retries", 3))
            retry_settings["max_cost_per_search"] = retry_settings.get("max_cost_per_search", 100) * 0.6
        
        elif strategy == OptimizationStrategy.COMPREHENSIVE:
            # Enable all scrapers with balanced settings
            for scraper in ["google_places", "google_maps_contacts", "linkedin_premium"]:
                if scraper in apify_config:
                    apify_config[scraper]["enabled"] = True
            
            # Increase retry attempts for better coverage
            retry_settings = apify_config.setdefault("retry_settings", {})
            retry_settings["max_retries"] = min(5, retry_settings.get("max_retries", 3) + 2)
        
        return config
    
    def suggest_optimization_strategy(
        self,
        industry: IndustryCategory,
        business_context: Dict[str, Any],
        available_budget: float
    ) -> Tuple[OptimizationStrategy, str]:
        """Suggest the best optimization strategy for given context."""
        
        profile = self.industry_profiles.get(industry, self.industry_profiles[IndustryCategory.OTHER])
        
        # Analyze business context
        requires_contacts = business_context.get("requires_contacts", False)
        requires_executives = business_context.get("requires_executives", False)
        location_critical = business_context.get("location_critical", False)
        budget_constrained = available_budget < 300
        comprehensive_needed = business_context.get("comprehensive_data", False)
        
        # Decision logic
        if budget_constrained:
            return OptimizationStrategy.COST_EFFICIENT, "Budget constraints require cost-efficient approach"
        
        if comprehensive_needed and available_budget > 800:
            return OptimizationStrategy.COMPREHENSIVE, "Sufficient budget for comprehensive data enrichment"
        
        if requires_executives or profile.linkedin_premium_priority > 0.8:
            return OptimizationStrategy.EXECUTIVE_FOCUSED, "Executive data is priority for this industry"
        
        if requires_contacts or profile.google_maps_contacts_priority > 0.8:
            return OptimizationStrategy.CONTACT_FOCUSED, "Contact information is priority"
        
        if location_critical or profile.google_places_priority > 0.8:
            return OptimizationStrategy.LOCATION_FOCUSED, "Location accuracy is critical"
        
        # Default to contact-focused for most business use cases
        return OptimizationStrategy.CONTACT_FOCUSED, "Contact-focused approach suitable for general business needs"
    
    def get_industry_insights(self, industry: IndustryCategory) -> Dict[str, Any]:
        """Get insights and recommendations for an industry."""
        
        profile = self.industry_profiles.get(industry, self.industry_profiles[IndustryCategory.OTHER])
        
        return {
            "industry": industry.value,
            "scraper_priorities": {
                "google_places": profile.google_places_priority,
                "google_maps_contacts": profile.google_maps_contacts_priority,
                "linkedin_premium": profile.linkedin_premium_priority
            },
            "recommended_linkedin_positions": profile.linkedin_positions[:5],
            "key_search_keywords": profile.linkedin_keywords + profile.google_places_keywords,
            "data_quality_requirements": {
                "min_completeness": profile.min_data_completeness,
                "requires_verification": profile.require_verified_info,
                "contact_strictness": profile.contact_validation_strictness
            },
            "cost_characteristics": {
                "cost_efficiency_weight": profile.cost_efficiency_weight,
                "timeout_tolerance": profile.timeout_tolerance
            },
            "optimization_suggestions": self._get_industry_optimization_suggestions(profile)
        }
    
    def _get_industry_optimization_suggestions(self, profile: IndustryProfile) -> List[str]:
        """Get optimization suggestions for an industry profile."""
        
        suggestions = []
        
        if profile.linkedin_premium_priority > 0.8:
            suggestions.append("Focus on LinkedIn Premium for executive contacts")
        
        if profile.google_maps_contacts_priority > 0.8:
            suggestions.append("Prioritize contact enrichment for this industry")
        
        if profile.cost_efficiency_weight > 0.7:
            suggestions.append("This industry is cost-sensitive - consider conservative configurations")
        
        if profile.require_verified_info:
            suggestions.append("Enable strict verification for regulatory compliance")
        
        if profile.timeout_tolerance < 0.9:
            suggestions.append("Use shorter timeouts for faster processing in this industry")
        
        return suggestions
    
    def analyze_optimization_effectiveness(self, results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze the effectiveness of industry-specific optimizations."""
        
        if not results:
            return {"status": "no_data"}
        
        # Group results by industry
        industry_results = defaultdict(list)
        for result in results:
            industry = result.get("detected_industry", "other")
            industry_results[industry].append(result)
        
        analysis = {}
        
        for industry, industry_data in industry_results.items():
            # Calculate metrics
            total_cost = sum(r.get("total_cost", 0) for r in industry_data)
            total_results = sum(r.get("results_count", 0) for r in industry_data)
            avg_success_rate = sum(r.get("success_rate", 0) for r in industry_data) / len(industry_data)
            
            cost_per_result = total_cost / max(total_results, 1)
            
            analysis[industry] = {
                "operations_count": len(industry_data),
                "total_cost": total_cost,
                "total_results": total_results,
                "avg_success_rate": avg_success_rate,
                "cost_per_result": cost_per_result,
                "efficiency_score": avg_success_rate / max(cost_per_result, 1)
            }
        
        # Overall analysis
        best_industry = max(analysis.keys(), key=lambda k: analysis[k]["efficiency_score"]) if analysis else None
        worst_industry = min(analysis.keys(), key=lambda k: analysis[k]["efficiency_score"]) if analysis else None
        
        return {
            "industry_breakdown": analysis,
            "best_performing_industry": best_industry,
            "worst_performing_industry": worst_industry,
            "optimization_recommendations": self._generate_optimization_recommendations(analysis)
        }
    
    def _generate_optimization_recommendations(self, analysis: Dict[str, Any]) -> List[str]:
        """Generate recommendations based on optimization analysis."""
        
        recommendations = []
        
        for industry, metrics in analysis.items():
            efficiency = metrics["efficiency_score"]
            cost_per_result = metrics["cost_per_result"]
            success_rate = metrics["avg_success_rate"]
            
            if efficiency < 0.1:
                recommendations.append(f"Review {industry} optimization - low efficiency detected")
            
            if cost_per_result > 15:
                recommendations.append(f"Reduce costs for {industry} - high cost per result")
            
            if success_rate < 0.7:
                recommendations.append(f"Improve {industry} configuration - low success rate")
        
        return recommendations


def create_industry_optimizer() -> IndustryOptimizer:
    """Create an industry optimizer instance."""
    return IndustryOptimizer()


import time  # Add missing import