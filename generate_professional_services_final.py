#!/usr/bin/env python3
"""Generate comprehensive professional services job templates using the existing system."""

import argparse
from copy import deepcopy
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional
import unicodedata

import yaml

NAF_LABELS: Dict[str, str] = {
    "6920Z": "Activités comptables",
    "6910Z": "Activités juridiques",
    "6922Z": "Conseils pour les affaires et autres conseils de gestion",
    "6201Z": "Programmation informatique",
    "6202A": "Conseil en systèmes et logiciels informatiques",
    "6202B": "Tierce maintenance de systèmes et d'applications informatiques",
    "7311Z": "Activités des agences de publicité",
    "7312Z": "Régie publicitaire de médias",
    "7320Z": "Études de marché et sondages",
    "7111Z": "Activités d'architecture",
    "7112A": "Activité des géomètres",
    "7112B": "Ingénierie, études techniques",
    "6831Z": "Agences immobilières",
    "6832A": "Administration d'immeubles et autres biens immobiliers",
    "6832B": "Supports juridiques de gestion de patrimoine immobilier",
    "7022Z": "Conseil pour les affaires et autres conseils de gestion",
    "7021Z": "Conseil en relations publiques et communication",
    "7410Z": "Activités spécialisées de design",
    "7420Z": "Activités photographiques",
    "7430Z": "Traduction et interprétation",
    "8621Z": "Activité des médecins généralistes",
    "8622A": "Activités de radiodiagnostic et de radiothérapie",
    "8622B": "Activités chirurgicales",
    "8622C": "Autres activités des médecins spécialistes",
    "8623Z": "Pratique dentaire",
    "6622Z": "Activités des agents et courtiers d'assurances",
    "6619A": "Supports juridiques de gestion de patrimoine mobilier",
}


def _canonical_naf_label(naf_code: str, raw_label: str) -> str:
    label = NAF_LABELS.get(naf_code, raw_label)
    normalized = unicodedata.normalize("NFC", label)
    if "\uFFFD" in normalized:
        raise ValueError(f"Replacement character found in NAF label {naf_code}: {raw_label!r}")
    return normalized


def generate_niche_name(naf_code: str) -> str:
    """Generate a consistent niche identifier from a NAF code."""
    naf_clean = naf_code.replace('.', '').replace(' ', '').upper()
    return f"naf_{naf_clean}"


DEFAULT_EMAIL_FORMATS = [
    'contact@{{d}}',
    'info@{{d}}',
    'bonjour@{{d}}',
    'cabinet@{{d}}',
    'secretariat@{{d}}',
]

DEFAULT_ALLOW_PATTERNS = [
    'contact',
    'mentions',
    'about',
    'equipe',
    'team',
    'avocats',
    'experts',
    'services',
    'cabinet',
]

DEFAULT_JOB_SPEC: Dict[str, Any] = {
    'niche': '',
    'filters': {
        'naf_include': [],
        'active_only': False,
    },
    'profile': 'standard',
    'steps_order': [],
    'http': {
        'seeds': [],
        'per_domain_rps': 0.5,
    },
    'sitemap': {
        'domains': [],
        'allow_patterns': DEFAULT_ALLOW_PATTERNS,
        'max_urls': 500,
    },
    'feeds': {'urls': []},
    'pdf': {'urls': []},
    'api': {'endpoints': []},
    'enrich': {
        'directory_csv': '',
        'email_formats_priority': DEFAULT_EMAIL_FORMATS,
    },
    'dedupe': {
        'keys': ['siren', 'domain_root', 'best_email', 'telephone_norm'],
        'fuzzy': False,
    },
    'scoring': {
        'weights': {
            'contactability': 50,
            'unicity': 20,
            'completeness': 20,
            'freshness': 10,
        },
    },
    'output': {
        'dir': '',
        'lang': 'fr',
    },
    'kpi_targets': {
        'min_quality_score': 80,
        'max_dup_pct': 1.5,
        'min_url_valid_pct': 85,
        'min_domain_resolved_pct': 80,
        'min_email_plausible_pct': 60,
        'min_lines_per_s': 50,
    },
    'budgets': {
        'max_http_bytes': 52_428_800,
        'max_http_requests': 2000,
        'time_budget_min': 90,
        'ram_mb': 4096,
    },
    'retention_days': 30,
}

COMMENT_AFTER_RULES = {
    'filters:': '  # regions: ["75", "92", "93", "94"]   (optionnel) filtre par préfixe CP',
}

COMMENT_BEFORE_RULES = [
    ('max_http_bytes:', '50MB - supports Google Maps enrichment'),
    ('max_http_requests:', '2000 requests for web scraping and Google Maps searches'),
    ('time_budget_min:', '90 minutes to allow comprehensive enrichment'),
]


def _ensure_list_of_strings(values: Iterable[str], field_name: str) -> List[str]:
    if isinstance(values, str):
        raise TypeError(f"{field_name} must be a sequence of strings, not a single string")
    try:
        items = list(values)
    except TypeError as exc:
        raise TypeError(f"{field_name} must be an iterable of strings (got {type(values).__name__})") from exc
    if not items:
        raise ValueError(f"{field_name} must contain at least one string")
    for entry in items:
        if not isinstance(entry, str):
            raise TypeError(f"{field_name} entries must be strings: {entry!r}")
        if not entry:
            raise ValueError(f"{field_name} entries must be non-empty strings")
    return items


def _merge_dict(target: Dict[str, Any], overrides: Dict[str, Any]) -> None:
    for key, value in overrides.items():
        if isinstance(value, dict) and isinstance(target.get(key), dict):
            _merge_dict(target[key], value)
        else:
            target[key] = value


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
            "https://www.infogerance.fr"
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

for _code, _info in PROFESSIONAL_SERVICES.items():
    _info["name"] = _canonical_naf_label(_code, _info.get("name", _code))


def create_professional_yaml_content(naf_code: str, service_info: Dict) -> str:
    """Create properly formatted YAML content for a professional service using structured data."""
    niche_name = generate_niche_name(naf_code)

    if 'seeds' not in service_info or 'domains' not in service_info:
        raise KeyError(f"Configuration incohérente pour {naf_code}: 'seeds' et 'domains' sont requis")

    seeds = _ensure_list_of_strings(service_info['seeds'], 'seeds')
    domains = _ensure_list_of_strings(service_info['domains'], 'domains')

    job_spec = deepcopy(DEFAULT_JOB_SPEC)
    job_spec['niche'] = niche_name
    job_spec['filters']['naf_include'] = [naf_code]
    job_spec['http']['seeds'] = seeds
    job_spec['sitemap']['domains'] = domains
    job_spec['output']['dir'] = f'out/{niche_name}'

    recognized_keys = {'name', 'seeds', 'domains'}
    for key, value in service_info.items():
        if key in recognized_keys:
            continue
        if isinstance(value, dict) and isinstance(job_spec.get(key), dict):
            _merge_dict(job_spec[key], value)
        else:
            job_spec[key] = value

    yaml_content = yaml.safe_dump(job_spec, sort_keys=False, allow_unicode=True)
    yaml_lines = yaml_content.rstrip('\n').split('\n')

    for prefix, comment_line in COMMENT_AFTER_RULES.items():
        for index, line in enumerate(yaml_lines):
            if line.startswith(prefix):
                yaml_lines.insert(index + 1, comment_line)
                break

    for prefix, comment in COMMENT_BEFORE_RULES:
        for idx in range(len(yaml_lines) - 1, -1, -1):
            line = yaml_lines[idx]
            if line.lstrip().startswith(prefix):
                indent = line[: len(line) - len(line.lstrip())]
                yaml_lines.insert(idx, f"{indent}# {comment}")
                break

    return '\n'.join(yaml_lines) + '\n'


def generate_all_professional_jobs(output_dir: Path, target_naf: Optional[Iterable[str]] = None) -> None:
    """Generate all professional services job files using proper YAML."""
    output_dir = output_dir.expanduser()

    available_codes = list(PROFESSIONAL_SERVICES.keys())
    selected_codes: List[str]
    missing_codes: List[str] = []

    if target_naf:
        normalized_target = {code.strip().upper() for code in target_naf if code and code.strip()}
        missing_codes = sorted(code for code in normalized_target if code not in PROFESSIONAL_SERVICES)
        selected_codes = [code for code in available_codes if code in normalized_target]
        if not selected_codes:
            print("[WARN] Aucun code NAF valide; aucun fichier genere.")
            if missing_codes:
                print(f"[WARN] Codes NAF inconnus: {', '.join(missing_codes)}")
            return
    else:
        selected_codes = available_codes

    try:
        output_dir.mkdir(parents=True, exist_ok=True)
    except OSError as exc:
        print(f"[ERROR] Impossible de creer le dossier de sortie {output_dir}: {exc}")
        return

    print(f"Generating {len(selected_codes)} professional services job templates in {output_dir}...")
    if missing_codes:
        print(f"[WARN] Codes NAF ignores: {', '.join(missing_codes)}")

    generated_files = 0
    for naf_code in selected_codes:
        service_info = PROFESSIONAL_SERVICES[naf_code]
        niche_name = generate_niche_name(naf_code)
        job_file = output_dir / f"{niche_name}.yaml"

        content = create_professional_yaml_content(naf_code, service_info)

        try:
            with open(job_file, "w", encoding="utf-8") as f:
                f.write(content)
        except OSError as exc:
            print(f"[ERROR] Echec ecriture du fichier {job_file}: {exc}")
            continue

        generated_files += 1
        print(f"[OK] Fichier cree {job_file.name} pour {service_info['name']} ({naf_code})")

    if generated_files == 0:
        print("[WARN] Aucun job genere.")
        return

    summary_file = output_dir / "professional_services_summary.md"
    try:
        with open(summary_file, "w", encoding="utf-8") as f:
            f.write("# Professional Services NAF Codes Summary\n\n")
            f.write("This document lists generated professional services job templates.\n\n")
            for naf_code in selected_codes:
                service_info = PROFESSIONAL_SERVICES[naf_code]
                niche_name = generate_niche_name(naf_code)
                f.write(f"## {naf_code} - {service_info['name']}\n")
                f.write(f"- **File**: {niche_name}.yaml\n")
                f.write(f"- **Seeds**: {len(service_info['seeds'])} websites\n")
                f.write(f"- **Domains**: {len(service_info['domains'])} domains\n")
                f.write(f"- **Example websites**: {', '.join(service_info['seeds'][:3])}\n")
                f.write("\n")
    except OSError as exc:
        print(f"[WARN] Echec ecriture du fichier de synthese {summary_file}: {exc}")
    else:
        print(f"[OK] Fichier de synthese cree: {summary_file.name}")




def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate professional services job templates."
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(__file__).parent / "jobs",
        help="Directory to write generated job files (default: %(default)s)",
    )
    parser.add_argument(
        "--naf",
        nargs="+",
        metavar="CODE",
        help="Optional NAF codes to generate; defaults to all supported codes.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    generate_all_professional_jobs(args.output_dir, args.naf)


if __name__ == "__main__":
    main()