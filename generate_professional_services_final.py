#!/usr/bin/env python3
"""Generate comprehensive professional services job templates using the existing system."""

from pathlib import Path
from typing import Dict, List
import create_job

# Comprehensive NAF codes for professional services with relevant websites
PROFESSIONAL_SERVICES = {
    # Accounting and Financial Services
    "6920Z": {
        "name": "Activités comptables", 
        "seeds": [
            "https://www.experts-comptables.fr",
            "https://www.ifec.fr", 
            "https://www.cs.experts-comptables.org",
            "https://annuaire.experts-comptables.org",
            "https://www.compta-online.com",
            "https://www.cabinet-comptable.com"
        ],
        "domains": ["experts-comptables.fr", "ifec.fr", "cs.experts-comptables.org", "compta-online.com"]
    },
    
    # Legal Services
    "6910Z": {
        "name": "Activités juridiques",
        "seeds": [
            "https://www.cnb.avocat.fr",
            "https://www.avocat.fr", 
            "https://www.notaires.fr",
            "https://www.conseil-national.notaires.fr",
            "https://www.village-justice.com",
            "https://www.dalloz.fr"
        ],
        "domains": ["cnb.avocat.fr", "avocat.fr", "notaires.fr", "village-justice.com"]
    },
    
    "6922Z": {
        "name": "Conseils pour les affaires et autres conseils de gestion",
        "seeds": [
            "https://www.syntec.fr",
            "https://www.conseil-en-organisation.com",
            "https://www.expert-comptable.net",
            "https://www.cabinet-conseil.fr"
        ],
        "domains": ["syntec.fr", "conseil-en-organisation.com"]
    },
    
    # IT and Web Development  
    "6201Z": {
        "name": "Programmation informatique",
        "seeds": [
            "https://www.syntec-numerique.fr",
            "https://www.malt.fr",
            "https://www.freelance.com",
            "https://www.agence-web.fr",
            "https://www.dev.fr"
        ],
        "domains": ["syntec-numerique.fr", "malt.fr", "freelance.com"]
    },
    
    "6202A": {
        "name": "Conseil en systèmes et logiciels informatiques",
        "seeds": [
            "https://www.syntec-numerique.fr",
            "https://www.conseil-informatique.fr",
            "https://www.tech.fr",
            "https://www.ssii.fr"
        ],
        "domains": ["syntec-numerique.fr", "conseil-informatique.fr"]
    },
    
    "6202B": {
        "name": "Tierce maintenance de systèmes et d'applications informatiques",
        "seeds": [
            "https://www.syntec-numerique.fr",
            "https://www.maintenance-informatique.fr",
            "https://www.infogérance.fr"
        ],
        "domains": ["syntec-numerique.fr", "maintenance-informatique.fr"]
    },
    
    # Marketing and Advertising
    "7311Z": {
        "name": "Activités des agences de publicité",
        "seeds": [
            "https://www.aacc.fr",
            "https://www.communication.fr",
            "https://www.agence-publicite.fr",
            "https://www.marketing.fr",
            "https://www.strategies.fr"
        ],
        "domains": ["aacc.fr", "communication.fr", "strategies.fr"]
    },
    
    "7312Z": {
        "name": "Régie publicitaire de médias",
        "seeds": [
            "https://www.aacc.fr",
            "https://www.regie-publicitaire.fr",
            "https://www.medias.fr"
        ],
        "domains": ["aacc.fr", "regie-publicitaire.fr"]
    },
    
    "7320Z": {
        "name": "études de marché et sondages",
        "seeds": [
            "https://www.syntec-etudes.com",
            "https://www.etudes-marketing.fr",
            "https://www.sondages.fr"
        ],
        "domains": ["syntec-etudes.com", "etudes-marketing.fr"]
    },
    
    # Architecture and Engineering
    "7111Z": {
        "name": "Activités d'architecture",
        "seeds": [
            "https://www.architectes.org",
            "https://www.cnoa.com",
            "https://www.architecture.fr",
            "https://www.ordre-architectes.fr"
        ],
        "domains": ["architectes.org", "cnoa.com", "ordre-architectes.fr"]
    },
    
    "7112A": {
        "name": "Activité des géomètres",
        "seeds": [
            "https://www.geometre-expert.fr",
            "https://www.ordre-geometres-experts.fr",
            "https://www.geometres.org"
        ],
        "domains": ["geometre-expert.fr", "ordre-geometres-experts.fr"]
    },
    
    "7112B": {
        "name": "Ingénierie, études techniques",
        "seeds": [
            "https://www.syntec-ingenierie.fr",
            "https://www.ingenierie.fr",
            "https://www.bureau-etudes.fr",
            "https://www.cnisf.org"
        ],
        "domains": ["syntec-ingenierie.fr", "ingenierie.fr", "cnisf.org"]
    },
    
    # Real Estate
    "6831Z": {
        "name": "Agences immobilières",
        "seeds": [
            "https://www.fnaim.fr",
            "https://www.century21.fr",
            "https://www.orpi.com",
            "https://www.laforet.com",
            "https://www.immobilier.fr"
        ],
        "domains": ["fnaim.fr", "century21.fr", "orpi.com", "laforet.com"]
    },
    
    "6832A": {
        "name": "Administration d'immeubles et autres biens immobiliers",
        "seeds": [
            "https://www.fnaim.fr",
            "https://www.administration-biens.fr",
            "https://www.syndic.fr"
        ],
        "domains": ["fnaim.fr", "administration-biens.fr"]
    },
    
    "6832B": {
        "name": "Supports juridiques de gestion de patrimoine mobilier",
        "seeds": [
            "https://www.patrimoine.fr",
            "https://www.cgp.fr",
            "https://www.gestion-patrimoine.fr"
        ],
        "domains": ["patrimoine.fr", "cgp.fr"]
    },
    
    # Consulting and Business Services
    "7022Z": {
        "name": "Conseil pour les affaires et autres conseils de gestion",
        "seeds": [
            "https://www.syntec.fr",
            "https://www.cabinet-conseil.fr",
            "https://www.conseil-entreprises.fr",
            "https://www.management.fr"
        ],
        "domains": ["syntec.fr", "cabinet-conseil.fr"]
    },
    
    "7021Z": {
        "name": "Conseil en relations publiques et communication",
        "seeds": [
            "https://www.conseil-en-communication.fr",
            "https://www.relations-publiques.fr",
            "https://www.communication-corporate.fr"
        ],
        "domains": ["conseil-en-communication.fr", "relations-publiques.fr"]
    },
    
    # Design and Creative Services
    "7410Z": {
        "name": "Activités spécialisées de design",
        "seeds": [
            "https://www.alliance-francaise-des-designers.org",
            "https://www.design.fr",
            "https://www.graphisme.fr"
        ],
        "domains": ["alliance-francaise-des-designers.org", "design.fr"]
    },
    
    "7420Z": {
        "name": "Activités photographiques",
        "seeds": [
            "https://www.photographes.fr",
            "https://www.federation-photo.fr",
            "https://www.photo.fr"
        ],
        "domains": ["photographes.fr", "federation-photo.fr"]
    },
    
    # Translation and Interpretation
    "7430Z": {
        "name": "Traduction et interprétation",
        "seeds": [
            "https://www.sft.fr",
            "https://www.traducteurs.fr",
            "https://www.interpretation.fr"
        ],
        "domains": ["sft.fr", "traducteurs.fr"]
    },
    
    # Medical and Health Services
    "8621Z": {
        "name": "Activité des médecins généralistes",
        "seeds": [
            "https://www.conseil-national.medecin.fr",
            "https://www.ordre-medecins.fr",
            "https://www.medecins.fr"
        ],
        "domains": ["conseil-national.medecin.fr", "ordre-medecins.fr"]
    },
    
    "8622A": {
        "name": "Activités de radiodiagnostic et de radiothérapie",
        "seeds": [
            "https://www.radiologie.fr",
            "https://www.radiotherapie.fr"
        ],
        "domains": ["radiologie.fr", "radiotherapie.fr"]
    },
    
    "8622B": {
        "name": "Activités chirurgicales",
        "seeds": [
            "https://www.chirurgie.fr",
            "https://www.chirurgiens.fr"
        ],
        "domains": ["chirurgie.fr", "chirurgiens.fr"]
    },
    
    "8622C": {
        "name": "Autres activités des médecins spécialistes",
        "seeds": [
            "https://www.medecins-specialistes.fr",
            "https://www.specialistes.fr"
        ],
        "domains": ["medecins-specialistes.fr", "specialistes.fr"]
    },
    
    "8623Z": {
        "name": "Pratique dentaire",
        "seeds": [
            "https://www.ordre-chirurgiens-dentistes.fr",
            "https://www.dentistes.fr",
            "https://www.chirurgiens-dentistes.fr"
        ],
        "domains": ["ordre-chirurgiens-dentistes.fr", "dentistes.fr"]
    },
    
    # Insurance and Financial Intermediation
    "6622Z": {
        "name": "Activités des agents et courtiers d'assurances",
        "seeds": [
            "https://www.agea.fr",
            "https://www.courtiers-assurance.fr",
            "https://www.assurance.fr"
        ],
        "domains": ["agea.fr", "courtiers-assurance.fr"]
    },
    
    "6619A": {
        "name": "Supports juridiques de gestion de patrimoine mobilier",
        "seeds": [
            "https://www.patrimoine.fr",
            "https://www.conseillers-financiers.fr"
        ],
        "domains": ["patrimoine.fr", "conseillers-financiers.fr"]
    }
}


def create_professional_yaml_content(naf_code: str, service_info: Dict) -> str:
    """Create properly formatted YAML content for a professional service."""
    niche_name = f"naf_{naf_code.replace('.', '').replace(' ', '').upper()}"
    
    # Create proper YAML format for seeds
    seeds_yaml = "\n".join(f'    - "{seed}"' for seed in service_info["seeds"])
    
    # Create proper YAML format for domains  
    domains_yaml = "\n".join(f'    - "{domain}"' for domain in service_info["domains"])
    
    return f'''niche: "{niche_name}"

filters:
  naf_include: ["{naf_code}"]
  active_only: false
  # regions: ["75","92","93","94"]   # (optionnel) filtre par préfixe CP

profile: "standard"
steps_order: []               # laisse le CLI gérer l'ordre

http:
  seeds:
{seeds_yaml}
  per_domain_rps: 0.5

sitemap:
  domains:
{domains_yaml}
  allow_patterns: ["contact","mentions","about","equipe","team","avocats","experts","services","cabinet"]
  max_urls: 500

feeds:
  urls: []
pdf:
  urls: []
api:
  endpoints: []

enrich:
  directory_csv: ""
  email_formats_priority: ["contact@{{d}}","info@{{d}}","bonjour@{{d}}","cabinet@{{d}}","secretariat@{{d}}"]

dedupe:
  keys: ["siren","domain_root","best_email","telephone_norm"]
  fuzzy: false

scoring:
  weights: {{contactability:50, unicity:20, completeness:20, freshness:10}}

output:
  dir: "out/{niche_name}"
  lang: "fr"

kpi_targets:
  min_quality_score: 80
  max_dup_pct: 1.5
  min_url_valid_pct: 85
  min_domain_resolved_pct: 80
  min_email_plausible_pct: 60
  min_lines_per_s: 50

budgets:
  max_http_bytes: 52428800   # 50MB - supports Google Maps enrichment
  max_http_requests: 2000    # sufficient for web scraping and Google Maps searches
  time_budget_min: 90        # extended time for comprehensive enrichment
  ram_mb: 4096

retention_days: 30
'''


def generate_all_professional_jobs():
    """Generate all professional services job files using proper YAML."""
    jobs_dir = Path(__file__).parent / "jobs"
    
    print(f"Generating {len(PROFESSIONAL_SERVICES)} professional services job templates...")
    
    for naf_code, service_info in PROFESSIONAL_SERVICES.items():
        niche_name = f"naf_{naf_code.replace('.', '').replace(' ', '').upper()}"
        job_file = jobs_dir / f"{niche_name}.yaml"
        
        content = create_professional_yaml_content(naf_code, service_info)
        
        with open(job_file, 'w', encoding='utf-8') as f:
            f.write(content)
        
        print(f"✓ Created {job_file.name} for {service_info['name']} ({naf_code})")
    
    print(f"\nSuccessfully generated {len(PROFESSIONAL_SERVICES)} job templates in {jobs_dir}")
    
    # Create summary file
    summary_file = jobs_dir / "professional_services_summary.md"
    with open(summary_file, 'w', encoding='utf-8') as f:
        f.write("# Professional Services NAF Codes Summary\n\n")
        f.write("This document lists all generated professional services job templates.\n\n")
        
        for naf_code, service_info in PROFESSIONAL_SERVICES.items():
            niche_name = f"naf_{naf_code.replace('.', '').replace(' ', '').upper()}"
            f.write(f"## {naf_code} - {service_info['name']}\n")
            f.write(f"- **File**: `{niche_name}.yaml`\n")
            f.write(f"- **Seeds**: {len(service_info['seeds'])} websites\n")
            f.write(f"- **Domains**: {len(service_info['domains'])} domains\n")
            f.write(f"- **Example websites**: {', '.join(service_info['seeds'][:3])}\n")
            f.write("\n")
    
    print(f"✓ Created summary file: {summary_file.name}")


if __name__ == "__main__":
    generate_all_professional_jobs()