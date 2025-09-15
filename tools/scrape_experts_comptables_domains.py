# -*- coding: utf-8 -*-
"""
Scrape annuaire experts-comptables pour construire un mapping siren -> domain_root.

Usage (exemples) :
    python tools/scrape_experts_comptables_domains.py ^
        --input out/ec_std/normalized.csv ^
        --out output/experts_comptables_domains_clean.csv ^
        --base-url https://annuaire.experts-comptables.org/annuaire ^
        --dry-run

    # ensuite pour la vraie collecte (sans dry-run) :
    python tools/scrape_experts_comptables_domains.py --input out/ec_std/normalized.csv --out output/experts_comptables_domains_clean.csv

Ajuste ci-dessous la section SELECTORS si besoin (nom des classes HTML de l’annuaire).
"""

import argparse, csv, os, re, sys, time, random, json
from urllib.parse import urljoin, urlparse
import requests
from bs4 import BeautifulSoup
import pandas as pd
from rapidfuzz import fuzz, process as rf_process
import tldextract
from tenacity import retry, stop_after_attempt, wait_exponential

# --------------- Configuration scraping (à ajuster si besoin) -----------------
CONFIG = {
    # URL de base (page liste/pagination)
    "BASE_URL": "https://annuaire.experts-comptables.org/annuaire",
    # Paramètres de pagination (si c'est ?page=2, ?page=3, etc.)
    "PAGINATION_PARAM": "page",
    "START_PAGE": 1,
    "MAX_PAGES": 200,        # borne haute de sécurité
    "PAGE_SLEEP_SEC": (0.8, 1.6),  # délai (min,max) secs entre pages
    # Sélecteurs HTML : à vérifier/ajuster après un premier essai en --dry-run
    "SELECTORS": {
        "card": ".annuaire-card, .result-card, .directory-card",  # conteneur d’un cabinet
        "name": ".card-title, .name, .cabinet-name",
        "site": "a[href*='http']:not([href*='mailto'])",  # lien externe (site)
        "city": ".city, .ville, .locality",
    },
    # Si la page d’annuaire a des sous-listes A/B/C… tu peux configurer des suffixes
    "LIST_SUFFIXES": [],  # ex: ["?lettre=A", "?lettre=B"] si pertinent
    # UA / timeout
    "USER_AGENT": "Mozilla/5.0 (compatible; EC-DomainBot/1.0; +https://example.org/bot)",
    "TIMEOUT": 15,
    "RETRY_ATTEMPTS": 3,
    "RETRY_BACKOFF": (0.7, 2.0),
    # Score minimal d’appariement nom-annuaire <-> nom-normalisé (0..100)
    "MIN_NAME_SCORE": 85,
}

# ------------------------- Helpers --------------------------------------------

def log(msg):
    print(f"[EC-SCRAPER] {msg}", flush=True)

def rand_sleep(a, b):
    time.sleep(random.uniform(a, b))

def normalize_domain(url: str) -> str | None:
    if not url:
        return None
    u = url.strip()
    u = re.sub(r"^https?://", "", u, flags=re.I)
    u = re.sub(r"^www\.", "", u, flags=re.I)
    # enlève ancre/params
    u = u.split("/")[0]
    u = u.lower()
    # filtrage minimal (exclure emails collés, javascipt:void etc.)
    if "@" in u or "javascript" in u:
        return None
    # valider extraction tld
    ext = tldextract.extract(u)
    if not ext.domain or not ext.suffix:
        return None
    return f"{ext.domain}.{ext.suffix}"

def normalize_name(s: str) -> str:
    if s is None:
        return ""
    x = s.upper()
    x = re.sub(r"[^A-Z0-9\s\-\&]", " ", x)
    x = re.sub(r"\s+", " ", x).strip()
    # quelques mots vides typiques
    x = re.sub(r"\b(SARL|SAS|SASU|SCI|SCP|SELARL|SELAS|EURL|SARLU|SOCIETE|CABINET|ET|DE|DES|DU|LA|LE)\b", " ", x)
    x = re.sub(r"\s+", " ", x).strip()
    return x

def load_input_entities(csv_path: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path, dtype=str, keep_default_na=False)
    # colonnes attendues pour aider l’appariement
    for col in ["siren","siret","raison_sociale","commune","cp"]:
        if col not in df.columns:
            df[col] = ""
    df["name_norm"] = df["raison_sociale"].map(normalize_name)
    return df

# ------------------------- HTTP avec retry ------------------------------------

session = requests.Session()
session.headers.update({
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
                  "(KHTML, like Gecko) Chrome/124.0 Safari/537.36",
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    "Accept-Language": "fr-FR,fr;q=0.9,en;q=0.8",
    "Referer": "https://annuaire.experts-comptables.org/",
    "Upgrade-Insecure-Requests": "1",
})

@retry(stop=stop_after_attempt(CONFIG["RETRY_ATTEMPTS"]),
       wait=wait_exponential(multiplier=CONFIG["RETRY_BACKOFF"][0], max=CONFIG["RETRY_BACKOFF"][1]))
def get_html(url: str) -> str:
    r = session.get(url, timeout=CONFIG["TIMEOUT"])
    r.raise_for_status()
    r.encoding = r.apparent_encoding or "utf-8"
    return r.text

# -------------------------- Scraper annuaire ----------------------------------

def extract_cards(html: str, base_url: str) -> list[dict]:
    sel = CONFIG["SELECTORS"]
    soup = BeautifulSoup(html, "html.parser")
    cards = []
    for card in soup.select(sel["card"]):
        # nom
        name_el = card.select_one(sel["name"])
        name = name_el.get_text(strip=True) if name_el else ""
        # ville
        city_el = card.select_one(sel["city"])
        city = city_el.get_text(strip=True) if city_el else ""
        # site : on prend le 1er lien externe “probable site”
        site = None
        for a in card.select(sel["site"]):
            href = a.get("href") or ""
            if href.startswith("mailto:"):
                continue
            site = urljoin(base_url, href)
            break
        if name or site:
            cards.append({"name": name, "city": city, "site": site})
    return cards

def iterate_pages(base_url: str) -> list[dict]:
    all_cards = []
    suffixes = CONFIG["LIST_SUFFIXES"] or [""]
    for suf in suffixes:
        for p in range(CONFIG["START_PAGE"], CONFIG["MAX_PAGES"]+1):
            page_url = base_url
            if suf:
                if "?" in suf:
                    # on concatène proprement
                    if "?" in base_url:
                        page_url = base_url + "&" + suf.lstrip("?")
                    else:
                        page_url = base_url + suf
                else:
                    page_url = base_url + suf
            # pagination
            if "?" in page_url:
                page_url = f"{page_url}&{CONFIG['PAGINATION_PARAM']}={p}"
            else:
                page_url = f"{page_url}?{CONFIG['PAGINATION_PARAM']}={p}"

            log(f"GET page {p}: {page_url}")
            try:
                html = get_html(page_url)
            except Exception as e:
                log(f"Stop pagination (HTTP {e}).")
                break  # on stoppe ce suffixe
            cards = extract_cards(html, base_url)
            log(f"  -> {len(cards)} cartes")
            if not cards:
                # Heuristique d’arrêt si page vide
                break
            all_cards.extend(cards)
            rand_sleep(*CONFIG["PAGE_SLEEP_SEC"])
    return all_cards

# -------------------------- Matching noms -> siren -----------------------------

def fuzzy_match(to_match: list[dict], df_entities: pd.DataFrame) -> list[dict]:
    """
    Apparie cards[name] sur df_entities[name_norm] avec RapidFuzz.
    Optionnel : si city présente, on la favorise (bonus de score si commune contient le morceau).
    """
    name_index = df_entities["name_norm"].tolist()
    result = []
    for c in to_match:
        name_norm = normalize_name(c["name"])
        if not name_norm:
            continue
        # rapide : prend les 5 meilleurs candidats
        matches = rf_process.extract(name_norm, name_index, scorer=fuzz.WRatio, limit=5)
        best = None
        best_score = -1
        for (cand, score, idx) in matches:
            # petit bonus si city “matche” la commune (ou CP)
            if c.get("city"):
                cc = c["city"].upper()
                row = df_entities.iloc[idx]
                bonus = 0
                if row.get("commune","") and cc in row["commune"].upper():
                    bonus += 5
                if row.get("cp","") and row["cp"] in cc:
                    bonus += 5
                score = min(100, score + bonus)
            if score > best_score:
                best_score = score
                best = (idx, score)
        if best and best_score >= CONFIG["MIN_NAME_SCORE"]:
            row = df_entities.iloc[best[0]].to_dict()
            domain_root = normalize_domain(c.get("site",""))
            if domain_root:
                result.append({
                    "siren": row.get("siren",""),
                    "siret": row.get("siret",""),
                    "raison_sociale": row.get("raison_sociale",""),
                    "commune": row.get("commune",""),
                    "cp": row.get("cp",""),
                    "name_annuaire": c.get("name",""),
                    "city_annuaire": c.get("city",""),
                    "domain_root": domain_root,
                    "match_score": int(best_score),
                })
    return result

# ------------------------------- Main -----------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True, help="CSV d’entrée (ex: out/ec_std/normalized.csv)")
    ap.add_argument("--out", required=True, help="CSV de sortie (mapping siren,domain_root)")
    ap.add_argument("--base-url", default=CONFIG["BASE_URL"], help="URL base de l’annuaire")
    ap.add_argument("--dry-run", action="store_true", help="Ne fait qu’un petit nombre de pages et montre des exemples")
    args = ap.parse_args()

    df = load_input_entities(args.input)
    log(f"Entrée: {args.input} -> {len(df)} entités")

    # si dry-run, bornons
    max_pages_backup = CONFIG["MAX_PAGES"]
    if args.dry_run:
        CONFIG["MAX_PAGES"] = min(CONFIG["MAX_PAGES"], 2)
        log(f"DRY-RUN : on limite à {CONFIG['MAX_PAGES']} pages pour tester les sélecteurs")

    cards = iterate_pages(args.base_url)
    log(f"Total cartes collectées: {len(cards)}")

    # aperçu si dry-run
    if args.dry_run:
        log("Aperçu 5 cartes :")
        for z in cards[:5]:
            log(f"  - name={z.get('name')} | city={z.get('city')} | site={z.get('site')}")

    matched = fuzzy_match(cards, df)
    log(f"Appariés & avec domaine valide: {len(matched)}")

    # Dédup par siren -> on garde le meilleur score
    if matched:
        md = pd.DataFrame(matched)
        md = md.sort_values(["siren","match_score"], ascending=[True, False])
        md = md.drop_duplicates(subset=["siren"], keep="first")

        # Export propre pour enrich.domain (colonnes utiles)
        out_df = md[["siren","domain_root"]].copy()
        out_df = out_df.dropna(subset=["siren","domain_root"])
        out_df = out_df.drop_duplicates()
        os.makedirs(os.path.dirname(args.out), exist_ok=True)
        out_df.to_csv(args.out, index=False, quoting=csv.QUOTE_MINIMAL, encoding="utf-8")
        log(f"Écrit: {args.out} ({len(out_df)} lignes)")
    else:
        log("Aucun appariement valide -> rien écrit")

    # restore config
    CONFIG["MAX_PAGES"] = max_pages_backup


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        log("Interrompu par l’utilisateur.")
        sys.exit(130)
