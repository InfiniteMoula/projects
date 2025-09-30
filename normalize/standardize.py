
# FILE: normalize/standardize.py
import re
import time
from pathlib import Path
from typing import Sequence

import pandas as pd
import pyarrow as pa

from utils import io
from utils.parquet import ArrowCsvWriter, ParquetBatchWriter, iter_batches

# --- config -------------------------------------------------------------
TEL_RE = re.compile(r"\D+")

# Placeholder values that should be treated as missing data
PLACEHOLDER_VALUES = {
    "TELEPHONE NON RENSEIGNE",
    "ADRESSE NON RENSEIGNEE", 
    "DENOMINATION NON RENSEIGNEE"
}

# Column grouping patterns for merging similar columns
COLUMN_GROUPS = {
    # Core identifiers - keep separate as they are unique
    'siren': ['siren', 'SIREN'],
    'siret': ['siret', 'SIRET'], 
    'nic': ['nic', 'NIC'],
    
    # Company names and denominations - merge similar variations
    'denomination': [
        'denominationUniteLegale', 'denominationunitelegale', 'DENOMINATIONUNITELEGALE',
        'denomination', 'DENOMINATION', 'company_name', 'COMPANY_NAME',
        'raison_sociale', 'RAISON_SOCIALE'
    ],
    'denomination_usuelle': [
        'denominationUsuelleEtablissement', 'denominationusuelleetablissement', 'DENOMINATIONUSUELLEETABLISSEMENT'
    ],
    'enseigne': [
        'enseigne1Etablissement', 'enseigne1etablissement', 'ENSEIGNE1ETABLISSEMENT',
        'enseigne', 'ENSEIGNE', 'nom_commercial', 'NOM_COMMERCIAL'
    ],
    
    # Geographic information
    'commune': [
        'libelleCommuneEtablissement', 'libellecommuneetablissement', 'LIBELLECOMMUNEETABLISSEMENT',
        'commune', 'COMMUNE', 'ville', 'VILLE'
    ],
    'code_postal': [
        'codePostalEtablissement', 'codepostaletablissement', 'CODEPOSTALETABLISSEMENT',
        'code_postal', 'CODE_POSTAL', 'cp', 'CP'
    ],
    
    # Address information - merge all address variations
    'adresse': [
        'adresseEtablissement', 'adresseetablissement', 'ADRESSEETABLISSEMENT',
        'adresse', 'ADRESSE', 'address', 'ADDRESS',
        'adresse_complete', 'ADRESSE_COMPLETE', 'adresse_ligne_1', 'ADRESSE_LIGNE_1'
    ],
    'numero_voie': [
        'numeroVoieEtablissement', 'numerovoieetablissement', 'NUMEROVOIEETABLISSEMENT'
    ],
    'type_voie': [
        'typeVoieEtablissement', 'typevoieetablissement', 'TYPEVOIEETABLISSEMENT'
    ],
    'libelle_voie': [
        'libelleVoieEtablissement', 'libellevoieetablissement', 'LIBELLEVOIEETABLISSEMENT'
    ],
    'complement_adresse': [
        'complementAdresseEtablissement', 'complementadresseetablissement', 'COMPLEMENTADRESSEETABLISSEMENT'
    ],
    
    # Activity codes
    'naf': [
        'activitePrincipaleEtablissement', 'activiteprincipaleetablissement', 'ACTIVITEPRINCIPALEESTABLISSEMENT',
        'activitePrincipaleUniteLegale', 'activiteprincipaleunitelegale', 'ACTIVITEPRINCIPALEUNITELLEGALE',
        'naf', 'NAF', 'code_naf', 'CODE_NAF'
    ],
    
    # Dates
    'date_creation': [
        'dateCreationEtablissement', 'datecreationetablissement', 'DATECREATIONETABLISSEMENT',
        'date_creation', 'DATE_CREATION'
    ],
    
    # Personal names - merge all name variations
    'nom': [
        'nomUniteLegale', 'nomunitelegale', 'NOMUNITELEGALE',
        'nom', 'NOM', 'name', 'NAME', 'noms', 'NOMS'
    ],
    'prenom': [
        'prenomsUniteLegale', 'prenomsunitelegale', 'PRENOMSUNITELEGALE',
        'prenom', 'PRENOM', 'prenoms', 'PRENOMS'
    ],
    
    # Contact information - merge all variations
    'telephone': [
        'telephone', 'TELEPHONE', 'phone', 'PHONE', 'tel', 'TEL'
    ],
    'telephone_mobile': [
        'tel_mobile', 'TEL_MOBILE', 'mobile', 'MOBILE', 'portable', 'PORTABLE'
    ],
    'fax': [
        'fax', 'FAX', 'telecopie', 'TELECOPIE'
    ],
    'email': [
        'email', 'EMAIL', 'mail', 'MAIL', 'e_mail', 'E_MAIL'
    ],
    'website': [
        'siteweb', 'SITEWEB', 'site_web', 'SITE_WEB', 'website', 'WEBSITE', 'url', 'URL'
    ],
    
    # Administrative info
    'etat_administratif': [
        'etatAdministratifEtablissement', 'etatadministratifetablissement', 'ETATADMINISTRATIFETABLISSEMENT'
    ],
    'effectif': [
        'trancheEffectifsEtablissement', 'trancheeffectifsetablissement', 'TRANCHEEFFECTIFSETABLISSEMENT',
        'trancheEffectifsUniteLegale', 'trancheeffectifsunitelegale', 'TRANCHEEFFECTIFSUNITELEGALE',
        'effectif', 'EFFECTIF', 'effectif_salarie', 'EFFECTIF_SALARIE', 'tranche_effectif', 'TRANCHE_EFFECTIF'
    ],
    
    # Additional business information
    'secteur_activite': [
        'secteur_activite', 'SECTEUR_ACTIVITE', 'secteur', 'SECTEUR'
    ],
    'forme_juridique': [
        'forme_juridique', 'FORME_JURIDIQUE', 'forme', 'FORME',
        'categorieJuridiqueUniteLegale', 'categoriejuridiqueunitelegale', 'CATEGORIEJURIDIQUEUNITELEGALE',
        'formeJuridique', 'formejuridique', 'FORMEJURIDIQUE'
    ],
    'capital_social': [
        'capital_social', 'CAPITAL_SOCIAL', 'capital', 'CAPITAL'
    ],
    'date_immatriculation': [
        'date_immatriculation', 'DATE_IMMATRICULATION', 'dateImmatriculation', 'dateimmatriculation', 'DATEIMMATRICULATION',
        'dateCreationUniteLegale', 'datecreationunitelegale', 'DATECREATIONUNITELEGALE',
        'date_creation', 'DATE_CREATION', 'date_debut_activite', 'DATE_DEBUT_ACTIVITE'
    ],
    'dirigeant_nom': [
        'dirigeant_nom', 'DIRIGEANT_NOM', 'dirigeant', 'DIRIGEANT',
        'nomUsageUniteLegale', 'nomusageunitelegale', 'NOMUSAGEUNITELEGALE',
        'denominationUsuelle1UniteLegale', 'denominationusuelle1unitelegale', 'DENOMINATIONUSUELLE1UNITELEGALE'
    ],
    'dirigeant_prenom': [
        'dirigeant_prenom', 'DIRIGEANT_PRENOM',
        'prenomUsuelUniteLegale', 'prenomusuelunitelegale', 'PRENOMUSUELUNITELEGALE'
    ]
}

# --- helpers ------------------------------------------------------------
def _is_placeholder(s: pd.Series) -> pd.Series:
    """Check if values in a Series are placeholder values that should be treated as missing data."""
    if s is None:
        return pd.Series(False, dtype=bool)
    return s.astype("string").isin(PLACEHOLDER_VALUES)


def _to_str(s: pd.Series | None) -> pd.Series:
    if s is None:
        return pd.Series(pd.NA, dtype="string")
    if str(s.dtype) != "string":
        return s.astype("string", copy=False)
    return s


def _to_str_filtered(s: pd.Series | None) -> pd.Series:
    """Convert to string and filter out placeholder values."""
    if s is None:
        return pd.Series(pd.NA, dtype="string")
    
    str_series = _to_str(s)
    # Replace placeholder values with NA
    is_placeholder_mask = str_series.isin(PLACEHOLDER_VALUES)
    str_series = str_series.where(~is_placeholder_mask, pd.NA)
    
    # Replace empty strings with NA
    str_series = str_series.replace("", pd.NA)
    
    return str_series


def _extract_departement(postal_code_series: pd.Series | None) -> pd.Series:
    """Extract département code from postal code (first 2 digits)."""
    if postal_code_series is None:
        return pd.Series(pd.NA, dtype="string")
    
    str_series = _to_str(postal_code_series)
    
    def extract_dept(postal_code):
        if pd.isna(postal_code) or postal_code == '':
            return pd.NA
        
        postal_str = str(postal_code).strip()
        
        # Handle 4-digit postal codes which are missing a leading zero
        if len(postal_str) == 4 and postal_str.isdigit():
            # For 4-digit codes, we need to determine if it's missing a leading zero
            # Codes like 1000-1999, 2000-2999, etc. are departments 01, 02, etc.
            first_digit = postal_str[0]
            if first_digit in '123456789':
                # These are departments 01-09, so add leading zero
                dept_code = '0' + first_digit
            else:
                # This shouldn't happen for valid French postal codes
                dept_code = postal_str[:2]
        elif len(postal_str) >= 5 and postal_str.isdigit():
            # Normal 5-digit postal code
            dept_code = postal_str[:2]
        elif len(postal_str) >= 2:
            # Extract first 2 characters
            dept_code = postal_str[:2]
        else:
            return pd.NA
        
        # Ensure we always return a 2-digit department code
        if len(dept_code) == 1:
            dept_code = '0' + dept_code
        
        return dept_code
    
    departement = str_series.apply(extract_dept)
    return departement.astype("string")


def _fr_tel_norm(s: pd.Series | None) -> pd.Series:
    if s is None:
        return pd.Series(pd.NA, dtype="string")
    
    # First filter out placeholder values
    str_series = _to_str(s)
    is_placeholder_mask = str_series.isin(PLACEHOLDER_VALUES)
    str_series = str_series.where(~is_placeholder_mask, "")
    
    # Now normalize phone numbers
    x = str_series.fillna("").str.replace(TEL_RE, "", regex=True)

    def _fmt(v: str) -> str:
        if not v:
            return ""
        if len(v) == 10 and v.startswith("0"):
            return "+33" + v[1:]
        if len(v) == 9:
            return "+33" + v
        return v

    x = x.map(_fmt).replace({"": pd.NA})
    return x.astype("string")


def _pick_first(df: pd.DataFrame, names: list[str]) -> pd.Series | None:
    """Pick the first available column from a list of names, supporting case-insensitive matching."""
    # First try exact match (for performance)
    for name in names:
        if name in df.columns:
            return df[name]
    
    # If no exact match, try case-insensitive matching
    df_columns_lower = {col.lower(): col for col in df.columns}
    for name in names:
        name_lower = name.lower()
        if name_lower in df_columns_lower:
            return df[df_columns_lower[name_lower]]
    
    return None


def _merge_columns(df: pd.DataFrame, column_names: list[str], prefer_non_empty: bool = True) -> pd.Series:
    """Merge multiple columns by combining their values, prioritizing non-empty values."""
    result = pd.Series(pd.NA, index=df.index, dtype="string")
    
    for col_name in column_names:
        if col_name in df.columns:
            col_data = _to_str_filtered(df[col_name])
            if prefer_non_empty:
                # Fill empty values in result with values from this column
                result = result.fillna(col_data)
            else:
                # Simple concatenation (could be used for other merge strategies)
                result = result.combine_first(col_data)
    
    return result


def _extract_all_columns(df: pd.DataFrame) -> dict:
    """Extract and merge all available columns based on column groups."""
    result = {}
    used_columns = set()
    
    # Process each column group
    for group_name, column_patterns in COLUMN_GROUPS.items():
        available_cols = [col for col in column_patterns if col in df.columns]
        if available_cols:
            # Mark these columns as used
            used_columns.update(available_cols)
            # Merge the columns
            result[group_name] = _merge_columns(df, available_cols)
    
    # Special processing for département extraction from postal code
    if 'code_postal' in result or any('postal' in col.lower() for col in df.columns):
        postal_code_series = result.get('code_postal')
        if postal_code_series is None:
            # Try to find postal code in remaining columns
            postal_cols = [col for col in df.columns if 'postal' in col.lower() or 'cp' in col.lower()]
            if postal_cols:
                postal_code_series = _merge_columns(df, postal_cols)
        
        if postal_code_series is not None:
            result['departement'] = _extract_departement(postal_code_series)
    
    # Add any remaining columns that weren't matched to groups
    remaining_cols = set(df.columns) - used_columns
    for col in remaining_cols:
        # Clean column name for output (lowercase, replace spaces/special chars)
        clean_name = col.lower().replace(' ', '_').replace('-', '_').replace('.', '_')
        # Avoid name conflicts
        if clean_name in result:
            clean_name = f"{clean_name}_extra"
        result[clean_name] = _to_str_filtered(df[col])
    
    return result




STRUCTURED_OUTPUT_COLUMNS = [
    "Nom entreprise",
    "Identifiant (SIREN/SIRET/DUNS)",
    "Forme juridique",
    "Date de création",
    "Secteur (NAF/APE)",
    "Adresse complète",
    "Région / Département",
    "Coordonnées GPS",
    "Téléphone standard",
    "Email générique",
    "Site web",
    "Effectif",
    "Chiffre d'affaires (fourchette)",
    "Nom + fonction du dirigeant",
    "Email pro vérifié du dirigeant",
    "Technologies web",
    "Chiffre d'affaires exact et/ou Résultat net",
    "Croissance CA (N vs N-1)",
    "Score de solvabilité",
]

DEPARTMENT_REGION_ENTRIES = [
    ("01", "Auvergne-Rhône-Alpes", "Ain"),
    ("02", "Hauts-de-France", "Aisne"),
    ("03", "Auvergne-Rhône-Alpes", "Allier"),
    ("04", "Provence-Alpes-Côte d'Azur", "Alpes-de-Haute-Provence"),
    ("05", "Provence-Alpes-Côte d'Azur", "Hautes-Alpes"),
    ("06", "Provence-Alpes-Côte d'Azur", "Alpes-Maritimes"),
    ("07", "Auvergne-Rhône-Alpes", "Ardèche"),
    ("08", "Grand Est", "Ardennes"),
    ("09", "Occitanie", "Ariège"),
    ("10", "Grand Est", "Aube"),
    ("11", "Occitanie", "Aude"),
    ("12", "Occitanie", "Aveyron"),
    ("13", "Provence-Alpes-Côte d'Azur", "Bouches-du-Rhône"),
    ("14", "Normandie", "Calvados"),
    ("15", "Auvergne-Rhône-Alpes", "Cantal"),
    ("16", "Nouvelle-Aquitaine", "Charente"),
    ("17", "Nouvelle-Aquitaine", "Charente-Maritime"),
    ("18", "Centre-Val de Loire", "Cher"),
    ("19", "Nouvelle-Aquitaine", "Corrèze"),
    ("2A", "Corse", "Corse-du-Sud"),
    ("2B", "Corse", "Haute-Corse"),
    ("21", "Bourgogne-Franche-Comté", "Côte-d'Or"),
    ("22", "Bretagne", "Côtes-d'Armor"),
    ("23", "Nouvelle-Aquitaine", "Creuse"),
    ("24", "Nouvelle-Aquitaine", "Dordogne"),
    ("25", "Bourgogne-Franche-Comté", "Doubs"),
    ("26", "Auvergne-Rhône-Alpes", "Drôme"),
    ("27", "Normandie", "Eure"),
    ("28", "Centre-Val de Loire", "Eure-et-Loir"),
    ("29", "Bretagne", "Finistère"),
    ("30", "Occitanie", "Gard"),
    ("31", "Occitanie", "Haute-Garonne"),
    ("32", "Occitanie", "Gers"),
    ("33", "Nouvelle-Aquitaine", "Gironde"),
    ("34", "Occitanie", "Hérault"),
    ("35", "Bretagne", "Ille-et-Vilaine"),
    ("36", "Centre-Val de Loire", "Indre"),
    ("37", "Centre-Val de Loire", "Indre-et-Loire"),
    ("38", "Auvergne-Rhône-Alpes", "Isère"),
    ("39", "Bourgogne-Franche-Comté", "Jura"),
    ("40", "Nouvelle-Aquitaine", "Landes"),
    ("41", "Centre-Val de Loire", "Loir-et-Cher"),
    ("42", "Auvergne-Rhône-Alpes", "Loire"),
    ("43", "Auvergne-Rhône-Alpes", "Haute-Loire"),
    ("44", "Pays de la Loire", "Loire-Atlantique"),
    ("45", "Centre-Val de Loire", "Loiret"),
    ("46", "Occitanie", "Lot"),
    ("47", "Nouvelle-Aquitaine", "Lot-et-Garonne"),
    ("48", "Occitanie", "Lozère"),
    ("49", "Pays de la Loire", "Maine-et-Loire"),
    ("50", "Normandie", "Manche"),
    ("51", "Grand Est", "Marne"),
    ("52", "Grand Est", "Haute-Marne"),
    ("53", "Pays de la Loire", "Mayenne"),
    ("54", "Grand Est", "Meurthe-et-Moselle"),
    ("55", "Grand Est", "Meuse"),
    ("56", "Bretagne", "Morbihan"),
    ("57", "Grand Est", "Moselle"),
    ("58", "Bourgogne-Franche-Comté", "Nièvre"),
    ("59", "Hauts-de-France", "Nord"),
    ("60", "Hauts-de-France", "Oise"),
    ("61", "Normandie", "Orne"),
    ("62", "Hauts-de-France", "Pas-de-Calais"),
    ("63", "Auvergne-Rhône-Alpes", "Puy-de-Dôme"),
    ("64", "Nouvelle-Aquitaine", "Pyrénées-Atlantiques"),
    ("65", "Occitanie", "Hautes-Pyrénées"),
    ("66", "Occitanie", "Pyrénées-Orientales"),
    ("67", "Grand Est", "Bas-Rhin"),
    ("68", "Grand Est", "Haut-Rhin"),
    ("69", "Auvergne-Rhône-Alpes", "Rhône"),
    ("70", "Bourgogne-Franche-Comté", "Haute-Saône"),
    ("71", "Bourgogne-Franche-Comté", "Saône-et-Loire"),
    ("72", "Pays de la Loire", "Sarthe"),
    ("73", "Auvergne-Rhône-Alpes", "Savoie"),
    ("74", "Auvergne-Rhône-Alpes", "Haute-Savoie"),
    ("75", "Île-de-France", "Paris"),
    ("76", "Normandie", "Seine-Maritime"),
    ("77", "Île-de-France", "Seine-et-Marne"),
    ("78", "Île-de-France", "Yvelines"),
    ("79", "Nouvelle-Aquitaine", "Deux-Sèvres"),
    ("80", "Hauts-de-France", "Somme"),
    ("81", "Occitanie", "Tarn"),
    ("82", "Occitanie", "Tarn-et-Garonne"),
    ("83", "Provence-Alpes-Côte d'Azur", "Var"),
    ("84", "Provence-Alpes-Côte d'Azur", "Vaucluse"),
    ("85", "Pays de la Loire", "Vendée"),
    ("86", "Nouvelle-Aquitaine", "Vienne"),
    ("87", "Nouvelle-Aquitaine", "Haute-Vienne"),
    ("88", "Grand Est", "Vosges"),
    ("89", "Bourgogne-Franche-Comté", "Yonne"),
    ("90", "Bourgogne-Franche-Comté", "Territoire de Belfort"),
    ("91", "Île-de-France", "Essonne"),
    ("92", "Île-de-France", "Hauts-de-Seine"),
    ("93", "Île-de-France", "Seine-Saint-Denis"),
    ("94", "Île-de-France", "Val-de-Marne"),
    ("95", "Île-de-France", "Val-d'Oise"),
    ("971", "Guadeloupe", "Guadeloupe"),
    ("972", "Martinique", "Martinique"),
    ("973", "Guyane", "Guyane"),
    ("974", "La Réunion", "La Réunion"),
    ("975", "Saint-Pierre-et-Miquelon", "Saint-Pierre-et-Miquelon"),
    ("976", "Mayotte", "Mayotte"),
    ("977", "Saint-Barthélemy", "Saint-Barthélemy"),
    ("978", "Saint-Martin", "Saint-Martin"),
    ("984", "Terres australes et antarctiques françaises", "Terres australes et antarctiques françaises"),
    ("986", "Wallis-et-Futuna", "Wallis-et-Futuna"),
    ("987", "Polynésie française", "Polynésie française"),
    ("988", "Nouvelle-Calédonie", "Nouvelle-Calédonie"),
    ("989", "Îles Éparses", "Îles Éparses"),
]

DEPARTMENT_REGION_MAP = {code: (region, department) for code, region, department in DEPARTMENT_REGION_ENTRIES}


def _clean_value(value):
    if value is None:
        return None
    if value is pd.NA:
        return None
    if isinstance(value, (list, tuple, set)):
        items: list[str] = []
        for item in value:
            cleaned = _clean_value(item)
            if cleaned:
                items.append(cleaned)
        return '; '.join(items) if items else None
    if isinstance(value, str):
        value_str = value.strip()
        if not value_str:
            return None
        if value_str.lower() == 'nan':
            return None
        return value_str
    try:
        if pd.isna(value):
            return None
    except TypeError:
        pass
    value_str = str(value).strip()
    if not value_str or value_str.lower() == 'nan':
        return None
    return value_str


def _select_first_non_empty(row: pd.Series, candidates: Sequence[str]):
    for col in candidates:
        if col in row.index:
            value = _clean_value(row.get(col))
            if value:
                return value
    return None


def _compose_sector(row: pd.Series):
    code = _select_first_non_empty(row, ['naf', 'naf_code', 'code_naf', 'activiteprincipaleregistremetiersetablissement'])
    label = _select_first_non_empty(row, ['libelle_naf', 'libelleactiviteprincipaleetablissement', 'libelle_activite', 'libelleactiviteprincipaleunitelegale'])
    if code and label:
        normalized_label = label.strip()
        if normalized_label.lower().startswith(code.lower()):
            return code
        return f"{code} - {normalized_label}"
    return code or label


def _compose_full_address(row: pd.Series):
    street_parts = []
    for col in ['numero_voie', 'type_voie', 'libelle_voie']:
        part = _clean_value(row.get(col))
        if part:
            street_parts.append(part)
    street = ' '.join(street_parts).strip()
    complement = _select_first_non_empty(row, ['complement_adresse', 'complementadresse2etablissement'])
    postal = _select_first_non_empty(row, ['code_postal', 'cp', 'codepostal2etablissement'])
    city = _select_first_non_empty(row, ['commune', 'ville', 'libellecommune2etablissement'])
    country = _select_first_non_empty(row, ['libellepaysetrangeretablissement', 'libellepaysetranger2etablissement', 'pays'])
    components = []
    if street:
        components.append(street)
    if complement:
        components.append(complement)
    locality_parts = [postal, city]
    locality = ' '.join(part for part in locality_parts if part)
    if locality:
        components.append(locality)
    if country:
        components.append(country)
    return ', '.join(components) if components else None


def _extract_department_code(row: pd.Series):
    dept = _clean_value(row.get('departement'))
    postal = _clean_value(row.get('code_postal')) or _clean_value(row.get('cp'))
    dept_code = None
    if dept:
        dept_code = dept.upper()
        if dept_code == '97' and postal:
            dept_code = postal[:3]
        elif dept_code.isdigit() and len(dept_code) == 1:
            dept_code = dept_code.zfill(2)
    if not dept_code and postal:
        if postal.startswith('97') and len(postal) >= 3:
            dept_code = postal[:3]
        else:
            dept_code = postal[:2]
    return dept_code


def _compose_region_departement(row: pd.Series):
    dept_code = _extract_department_code(row)
    if not dept_code:
        return None
    info = DEPARTMENT_REGION_MAP.get(dept_code)
    if info:
        region, department = info
        return f"{region} / {department}"
    return f"Département {dept_code}"


def _compose_coordinates(row: pd.Series):
    lat = None
    for col in ['latitude', 'lat', 'latitude_wgs84', 'coordonnees_gps_latitude']:
        lat = _clean_value(row.get(col))
        if lat:
            break
    lon = None
    for col in ['longitude', 'lon', 'longitude_wgs84', 'coordonnees_gps_longitude']:
        lon = _clean_value(row.get(col))
        if lon:
            break
    if lat and lon:
        return f"{lat}, {lon}"
    combined = _clean_value(row.get('coordonnees_gps'))
    return combined


def _compose_executive(row: pd.Series):
    prenom = _select_first_non_empty(row, ['dirigeant_prenom', 'prenom', 'prenoms'])
    nom = _select_first_non_empty(row, ['dirigeant_nom', 'nom'])
    fonction = _select_first_non_empty(row, ['dirigeant_fonction', 'fonction_dirigeant', 'qualite'])
    name_parts = [part for part in [prenom, nom] if part]
    full_name = ' '.join(name_parts) if name_parts else None
    if full_name and fonction:
        return f"{full_name} ({fonction})"
    return full_name


def _compose_financials(row: pd.Series):
    chiffre = _select_first_non_empty(row, ['chiffre_affaires', 'ca_exact', 'ca'])
    resultat = _select_first_non_empty(row, ['resultat_net', 'benefice_net', 'resultat'])
    if chiffre and resultat:
        return f"CA {chiffre}; Résultat net {resultat}"
    if chiffre:
        return f"CA {chiffre}"
    if resultat:
        return f"Résultat net {resultat}"
    return None


def _build_series(df: pd.DataFrame, func):
    if df.empty:
        return pd.Series([], index=df.index, dtype='string')
    values = df.apply(func, axis=1)
    series = values.astype('string')
    series = series.str.strip()
    series = series.replace('', pd.NA)
    return series


def _compute_structured_columns(df: pd.DataFrame) -> dict[str, pd.Series]:
    return {
        'Nom entreprise': _build_series(df, lambda row: _select_first_non_empty(row, ['denomination', 'denomination_usuelle', 'enseigne', 'raison_sociale', 'nom'])),
        'Identifiant (SIREN/SIRET/DUNS)': _build_series(df, lambda row: _select_first_non_empty(row, ['siret', 'siren', 'duns'])),
        'Forme juridique': _build_series(df, lambda row: _select_first_non_empty(row, ['forme_juridique', 'formejuridique', 'categorie_juridique', 'categorie_juridique_unite_legale'])),
        'Date de création': _build_series(df, lambda row: _select_first_non_empty(row, ['date_creation', 'date_immatriculation', 'datedebut', 'date_creation_unite_legale'])),
        'Secteur (NAF/APE)': _build_series(df, _compose_sector),
        'Adresse complète': _build_series(df, _compose_full_address),
        'Région / Département': _build_series(df, _compose_region_departement),
        'Coordonnées GPS': _build_series(df, _compose_coordinates),
        'Téléphone standard': _build_series(df, lambda row: _select_first_non_empty(row, ['telephone_norm', 'telephone', 'tel', 'phone'])),
        'Email générique': _build_series(df, lambda row: _select_first_non_empty(row, ['email', 'mail', 'email_contact'])),
        'Site web': _build_series(df, lambda row: _select_first_non_empty(row, ['siteweb', 'website', 'url'])),
        'Effectif': _build_series(df, lambda row: _select_first_non_empty(row, ['effectif', 'tranche_effectif', 'effectif_salarie', 'anneeeffectifsetablissement'])),
        "Chiffre d'affaires (fourchette)": _build_series(df, lambda row: _select_first_non_empty(row, ['ca_tranche', 'chiffre_affaires_fourchette', 'chiffre_affaires_tranche'])),
        'Nom + fonction du dirigeant': _build_series(df, _compose_executive),
        'Email pro vérifié du dirigeant': _build_series(df, lambda row: _select_first_non_empty(row, ['dirigeant_email', 'email_dirigeant', 'email_dirigeant_verifie'])),
        'Technologies web': _build_series(df, lambda row: _select_first_non_empty(row, ['technologies_web', 'web_technologies', 'technologies'])),
        "Chiffre d'affaires exact et/ou Résultat net": _build_series(df, _compose_financials),
        'Croissance CA (N vs N-1)': _build_series(df, lambda row: _select_first_non_empty(row, ['croissance_ca', 'variation_ca', 'evolution_ca'])),
        'Score de solvabilité': _build_series(df, lambda row: _select_first_non_empty(row, ['score_solvabilite', 'solvabilite_score', 'score_risque'])),
    }


def _create_dynamic_schema(columns: Sequence[str]) -> pa.Schema:
    """Create a PyArrow schema preserving column order."""
    fields = [pa.field(col_name, pa.string()) for col_name in columns]
    return pa.schema(fields)


# --- main ---------------------------------------------------------------
def run(cfg: dict, ctx: dict) -> dict:
    t0 = time.time()
    input_path = ctx.get("input_path") or ctx.get("input")
    input_path = Path(input_path) if input_path else None
    if input_path is None or not input_path.exists():
        return {"status": "FAIL", "error": f"Input parquet not found: {input_path}"}

    job = ctx.get("job", {}) or {}
    filters = (job.get("filters") or {})
    # Keep original case and just clean spaces/dots for better matching
    naf_include_raw = [x.replace(".", "").replace(" ", "") for x in (filters.get("naf_include") or [])]
    naf_prefixes = tuple(filter(None, naf_include_raw))
    active_only = bool(filters.get("active_only", False))

    # Remove usecols restriction to extract all columns
    batch_rows = int(job.get("standardize_batch_rows", 200_000))

    outdir = Path(ctx.get("outdir_path") or ctx.get("outdir"))
    out_parquet = outdir / "normalized.parquet"
    out_csv = outdir / "normalized.csv"

    for target in (out_parquet, out_csv):
        if target.exists():
            try:
                target.unlink()
            except OSError:
                pass

    total = 0
    raw_rows_total = 0
    filtered_rows_total = 0
    batches = 0
    logger = ctx.get("logger")
    dynamic_schema = None

    try:
        # Initialize writers without fixed schema - will be created dynamically
        pq_writer = None
        csv_writer = None
        
        for pdf in iter_batches(input_path, columns=None, batch_size=batch_rows):  # Extract all columns
            batches += 1
            raw_rows = len(pdf)
            raw_rows_total += raw_rows
            if raw_rows == 0:
                continue

            object_cols = [col for col in pdf.columns if str(pdf[col].dtype) == "object"]
            if object_cols:
                pdf[object_cols] = pdf[object_cols].astype("string", copy=False)

            # Apply NAF filtering
            naf_col_series = _pick_first(pdf, [
                "activitePrincipaleEtablissement","activiteprincipaleetablissement","ACTIVITEPRINCIPALEESTABLISSEMENT",
                "activitePrincipaleUniteLegale","activiteprincipaleunitelegale","ACTIVITEPRINCIPALEUNITELLEGALE"
            ])

            if naf_prefixes and naf_col_series is not None:
                # Improved NAF filtering
                naf_norm = _to_str(naf_col_series).str.replace(r"[\s\.]", "", regex=True).str.upper()
                combined_mask = pd.Series([False] * len(pdf))
                
                for prefix in naf_prefixes:
                    # Clean the prefix the same way as the data
                    prefix_clean = prefix.replace(".", "").replace(" ", "").upper()
                    
                    # Apply smart matching for 4+ digit codes ending with letters
                    if (len(prefix_clean) >= 4 and 
                        prefix_clean and prefix_clean[-1].isalpha() and
                        prefix_clean[:-1].isdigit() and 
                        not prefix_clean.startswith(('01', '02', '03'))):  # Exclude agriculture/forestry
                        
                        base_code = prefix_clean[:-1]  # Remove letter suffix
                        # Match either the specific code or the base category
                        mask = (naf_norm.fillna("").str.startswith(base_code) | 
                               naf_norm.fillna("").str.startswith(prefix_clean))
                    else:
                        # Standard prefix matching for other codes
                        mask = naf_norm.fillna("").str.startswith(prefix_clean)
                    
                    combined_mask = combined_mask | mask
                
                pdf = pdf[combined_mask.fillna(False)]

            if active_only:
                state_series = _pick_first(pdf, [
                    "etatAdministratifEtablissement","etatadministratifetablissement","ETATADMINISTRATIFETABLISSEMENT"
                ])
                if state_series is not None:
                    pdf = pdf[_to_str(state_series).eq("A")]

            filtered_rows = len(pdf)
            filtered_rows_total += filtered_rows
            if filtered_rows == 0:
                continue

            # Extract all columns using the comprehensive approach
            extracted_data = _extract_all_columns(pdf)
            
            # Apply special processing for phone numbers
            if 'telephone' in extracted_data:
                extracted_data['telephone_norm'] = _fr_tel_norm(extracted_data['telephone'])
            if 'telephone_mobile' in extracted_data:
                extracted_data['telephone_mobile_norm'] = _fr_tel_norm(extracted_data['telephone_mobile'])
            if 'fax' in extracted_data:
                extracted_data['fax_norm'] = _fr_tel_norm(extracted_data['fax'])
            
            # Clean postal codes
            if 'code_postal' in extracted_data:
                extracted_data['code_postal'] = extracted_data['code_postal'].str.extract(r"(\d{5})", expand=False).astype("string")
                # Add legacy alias
                extracted_data['cp'] = extracted_data['code_postal']
            
            # Clean NAF codes
            if 'naf' in extracted_data:
                extracted_data['naf'] = extracted_data['naf'].str.replace(r"\s", "", regex=True).astype("string")
                # Add legacy alias
                extracted_data['naf_code'] = extracted_data['naf']
            
            # Add backward compatibility aliases
            if 'denomination' in extracted_data:
                extracted_data['raison_sociale'] = extracted_data['denomination']
            if 'commune' in extracted_data:
                extracted_data['ville'] = extracted_data['commune'] 
            if 'adresse' in extracted_data:
                extracted_data['adresse_complete'] = extracted_data['adresse']
            if 'website' in extracted_data:
                extracted_data['siteweb'] = extracted_data['website']

            # Create DataFrame with all extracted data
            res = pd.DataFrame(extracted_data)

            # Ensure we have at least the basic required columns for compatibility
            required_cols = ['siren', 'siret']
            for col in required_cols:
                if col not in res.columns:
                    res[col] = pd.Series(pd.NA, index=res.index, dtype="string")

            structured_columns = _compute_structured_columns(res)
            for col_name, series in structured_columns.items():
                res[col_name] = series

            column_order = STRUCTURED_OUTPUT_COLUMNS + [col for col in res.columns if col not in STRUCTURED_OUTPUT_COLUMNS]

            rows_written = len(res)
            total += rows_written
            if rows_written == 0:
                continue

            if dynamic_schema is None:
                dynamic_schema = _create_dynamic_schema(column_order)
                pq_writer = ParquetBatchWriter(out_parquet, schema=dynamic_schema)
                csv_writer = ArrowCsvWriter(out_csv)

            schema_columns = [field.name for field in dynamic_schema]
            res = res.reindex(columns=schema_columns, fill_value=pd.NA)

            table = pa.Table.from_pandas(res, preserve_index=False).cast(dynamic_schema)
            pq_writer.write_table(table)
            csv_writer.write_table(table)

        # Close writers
        if pq_writer:
            pq_writer.close()
        if csv_writer:
            csv_writer.close()

        elapsed = time.time() - t0
        duration = round(elapsed, 3)
        rows_per_s = total / elapsed if elapsed > 0 else 0.0
        drop_pct = ((raw_rows_total - total) / raw_rows_total * 100) if raw_rows_total else 0.0

        kpi_targets = (job.get("kpi_targets") or {})
        kpi_evaluations: list[dict[str, object]] = []
        min_lines_target_raw = kpi_targets.get("min_lines_per_s")
        min_lines_target = float(min_lines_target_raw) if min_lines_target_raw is not None else None
        if min_lines_target is not None:
            kpi_evaluations.append({
                "name": "lines_per_s",
                "actual": rows_per_s,
                "target": min_lines_target,
                "met": rows_per_s >= min_lines_target,
            })

        summary = {
            "status": "OK",
            "step": "normalize.standardize",
            "batches": batches,
            "rows_raw": raw_rows_total,
            "rows_after_filters": filtered_rows_total,
            "rows_written": total,
            "rows_dropped": max(raw_rows_total - total, 0),
            "drop_pct": round(drop_pct, 2),
            "duration_s": duration,
            "rows_per_s": round(rows_per_s, 3),
            "filters": {
                "active_only": active_only,
                "naf_prefixes": list(naf_prefixes),
                "comprehensive_extraction": True,  # Indicate new approach
            },
            "files": {
                "parquet": str(out_parquet),
                "csv": str(out_csv),
            },
        }
        if kpi_evaluations:
            summary["kpi_evaluations"] = kpi_evaluations
            summary["kpi_status"] = "MET" if all(item["met"] for item in kpi_evaluations) else "WARN"
            summary["kpi_targets"] = {"min_lines_per_s": min_lines_target}

        reports_dir = outdir / "reports"
        report_path = io.write_json(reports_dir / "standardize_summary.json", summary)
        summary["report_path"] = str(report_path)

        log_path = ctx.get("logs")
        if log_path:
            io.log_json(log_path, {
                "step": "normalize.standardize",
                "event": "summary",
                "status": summary.get("kpi_status", "OK"),
                "rows": total,
                "rows_per_s": round(rows_per_s, 3),
                "drop_pct": round(drop_pct, 2),
                "report_path": str(report_path),
            })

        if logger:
            kpi_phrase = ""
            if kpi_evaluations:
                details = "; ".join(
                    f"{item['name']} {'OK' if item['met'] else 'WARN'} ({item['actual']:.2f}/{item['target']:.2f})"
                    for item in kpi_evaluations
                )
                kpi_phrase = f" | KPI {details}"
            logger.info(
                "normalize.standardize summary | rows=%d | drop_pct=%.1f%% | rows_per_s=%.2f%s",
                total,
                drop_pct,
                rows_per_s,
                kpi_phrase,
            )

        files = [str(out_parquet), str(out_csv), str(report_path)]
        return {
            "status": "OK",
            "files": files,
            "rows": total,
            "duration_s": duration,
            "rows_per_s": round(rows_per_s, 3),
            "summary": summary,
            "report_path": str(report_path),
        }

    except Exception as exc:
        if logger:
            logger.exception("standardize failed: %s", exc)
        return {"status": "FAIL", "error": str(exc), "duration_s": round(time.time() - t0, 3)}

