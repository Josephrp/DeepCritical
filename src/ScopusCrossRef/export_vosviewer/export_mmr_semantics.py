#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os, re, glob
import pandas as pd
from typing import Dict, Set, List
from dotenv import load_dotenv
from neo4j import GraphDatabase, exceptions as neo_err

# ============== CONFIG ==============
load_dotenv()  # usa tu .env tal cual

BASE_DIR   = os.path.dirname(os.path.abspath(__file__))
IN_ROOT    = os.path.join(BASE_DIR, "data_checkpoints_plus")  # donde están */<method>_representatives.csv
OUT_DIR    = os.path.join(BASE_DIR, "vosviewer_exports")
os.makedirs(OUT_DIR, exist_ok=True)

NEO4J_URI  = os.getenv("NEO4J_URI", "bolt://localhost:7687")
NEO4J_USER = os.getenv("NEO4J_USER", "neo4j")
NEO4J_PWD  = os.getenv("NEO4J_PASSWORD", "neo4j")
NEO4J_DB   = os.getenv("NEO4J_DATABASE", "neo4j")

# >>> Elige aquí el método de representantes que quieres exportar: "mmr" o "fps"
REP_METHOD   = os.getenv("REP_METHOD", "mmr").lower().strip()
REP_FILENAME = f"{REP_METHOD}_representatives.csv"   # p.ej., "mmr_representatives.csv"
OUT_SUFFIX   = REP_METHOD                             # sufijo en los CSV de salida

BATCH = 200  # tamaño de lote para DOIs

# ============== LIMPIEZA DE KEYWORDS (VOSviewer-friendly) ==============
_SPLIT_RX  = re.compile(r"\s*(?:;|,|\||/|·|•|–|—|&|\band\b|\+)\s*", re.I)
_WS_RX     = re.compile(r"\s+")
_MAX_WORDS = 7

_ALIAS = {
    "polyethylene terephthalate":"PET","poly(ethylene terephthalate)":"PET","pet":"PET",
    "high density polyethylene":"HDPE","low density polyethylene":"LDPE",
    "polypropylene":"PP","polystyrene":"PS","polyvinyl chloride":"PVC","polylactic acid":"PLA",
    "life cycle assessment":"LCA","lca (life cycle assessment)":"LCA",
    "co 2":"CO2","co2":"CO2","greenhouse gas":"GHG","greenhouse gases":"GHG",
    "ghg emissions":"GHG emissions",
    "non intentionally added substances":"NIAS","non-intentionally added substances":"NIAS",
    "food contact materials":"FCM","post consumer resin":"PCR","post-consumer resin":"PCR",
    "circular economy":"Circular economy","mechanical recycling":"Mechanical recycling",
    "chemical recycling":"Chemical recycling","pyrolysis":"Pyrolysis",
}

_STOPLIKE = {"article","review","paper","study","research","introduction","conclusion",
             "methods","results","dataset","analysis"}

_KEEP_SHORT = {"LCA","PET","PP","PE","PS","PVC","PLA","CE","EU","NIAS","PCR","GHG",
               "FCM","HDPE","LDPE"}

def _tidy_token(tok: str) -> str:
    s = tok.replace("_"," ").strip(" .,:;|/–—•·()[]{}")
    s = _WS_RX.sub(" ", s).strip()
    if not s:
        return ""
    low = s.lower()
    if low in _ALIAS:
        return _ALIAS[low]
    parts = s.split()
    return " ".join([p if p.upper() in _KEEP_SHORT else p.capitalize() for p in parts])

def clean_keywords(val) -> str:
    """
    Acepta str o list[str] y devuelve 'kw1; kw2; ...' sin '|' ni separadores problemáticos,
    aplicando alias, capitalización y filtrado básico.
    """
    raw = []
    if isinstance(val, list):
        for v in val:
            if isinstance(v, str):
                raw += _SPLIT_RX.split(v)
    elif isinstance(val, str):
        raw = _SPLIT_RX.split(val)
    else:
        raw = []

    seen, out = set(), []
    for tok in raw:
        t = _tidy_token(tok)
        if not t:
            continue
        if t.lower() in _STOPLIKE:
            continue
        if len(t.split()) > _MAX_WORDS and t.upper() not in _KEEP_SHORT:
            continue
        key = t.casefold()
        if key in seen:
            continue
        seen.add(key)
        out.append(t)
    return "; ".join(out)

# ============== UTILS ==============
def norm_doi(x: str) -> str:
    if not isinstance(x, str): return ""
    s = x.strip().lower()
    for p in ("https://doi.org/","http://doi.org/","doi:"):
        if s.startswith(p): s = s[len(p):]
    return s.strip()

def read_rep_dois(root: str, rep_filename: str) -> Dict[str, Set[str]]:
    """
    Busca */<rep_filename> y extrae los DOIs por carpeta (cluster).
    """
    out: Dict[str, Set[str]] = {}
    pattern = os.path.join(root, "*", rep_filename)
    files = glob.glob(pattern)
    if not files:
        print(f"❌ No encontré ningún '{rep_filename}' en {root}/*")
        return out

    for fp in sorted(files):
        cluster = os.path.basename(os.path.dirname(fp))
        try:
            df = pd.read_csv(fp)
        except Exception:
            print(f"⚠️  No pude leer {fp}")
            continue
        if "doi" not in df.columns:
            print(f"⚠️  {cluster}: 'doi' no está en {rep_filename}")
            continue
        dois = {norm_doi(d) for d in df["doi"].astype(str).tolist() if norm_doi(d)}
        out[cluster] = dois
        print(f"✅ {cluster}: {len(dois)} DOIs {REP_METHOD.upper()}")
    return out

# ============== CYPHER (campos estilo Scopus/VOSviewer) ==============
CYPHER = """
UNWIND $dois AS d
MATCH (p:Publication {doi: d})
OPTIONAL MATCH (a:Author)-[:AUTHORED]->(p)
OPTIONAL MATCH (a)-[:AFFILIATED_WITH]->(ai:Institution)
OPTIONAL MATCH (a)-[:AFFILIATED_WITH]->(ac:Country)
OPTIONAL MATCH (p)-[:HAS_KEYWORD]->(k:Keyword)
OPTIONAL MATCH (p)-[:AFFILIATED_WITH]->(pc:Country)
OPTIONAL MATCH (p)-[:PUBLISHED_IN]->(j:Journal)
OPTIONAL MATCH (p)-[:FUNDED_BY]->(fa:FundingAgency)
OPTIONAL MATCH (p)-[:FUNDED_BY]->(g:Grant)
RETURN
  p.doi                              AS doi,
  p.eid                              AS eid,
  p.title                            AS title,
  p.year                             AS year,
  COALESCE(toInteger(p.citedBy),0)   AS citedBy,
  p.abstract                         AS abstract,
  j.name                             AS source_title,
  collect(DISTINCT a.name)           AS authors,
  collect(DISTINCT ai.name)          AS institutions,
  collect(DISTINCT ac.name)          AS author_countries,
  collect(DISTINCT pc.name)          AS pub_countries,
  collect(DISTINCT k.name)           AS author_keywords,
  collect(DISTINCT fa.name)          AS funding_agencies,
  p.funding_text                     AS funding_text
"""

# ============== QUERY ==============
def query_by_dois(driver, dois: List[str]) -> pd.DataFrame:
    rows: List[dict] = []
    with driver.session(database=NEO4J_DB) as s:
        for i in range(0, len(dois), BATCH):
            chunk = dois[i:i+BATCH]
            rows.extend([r.data() for r in s.run(CYPHER, dois=chunk)])

    if not rows:
        return pd.DataFrame()

    df = pd.DataFrame(rows)

    # 1) Limpiar keywords ANTES de aplanar (clave para VOSviewer)
    if "author_keywords" in df.columns:
        df["author_keywords"] = df["author_keywords"].apply(clean_keywords)

    # 2) Aplanar listas (autores, afiliaciones, países…)
    def j(x):
        if isinstance(x, list): return "; ".join([str(z) for z in x if z])
        return x

    for col in ["authors","institutions","author_countries","pub_countries","funding_agencies"]:
        if col in df.columns:
            df[col] = df[col].apply(j)

    # 3) Renombrar a estilo Scopus/VOSviewer
    rename = {
        "title":"Title","authors":"Authors","year":"Year","source_title":"Source title",
        "institutions":"Affiliations","author_countries":"Author Countries",
        "pub_countries":"Publication Countries","author_keywords":"Author Keywords",
        "funding_agencies":"Funding Sponsors","funding_text":"Funding Text",
        "citedBy":"Cited by","doi":"DOI","eid":"EID","abstract":"Abstract",
    }
    df = df.rename(columns=rename)

    # 4) Orden canónico
    cols = ["Title","Authors","Year","Source title","Affiliations","Author Countries",
            "Publication Countries","Author Keywords","Funding Sponsors","Funding Text",
            "Cited by","DOI","EID","Abstract"]
    for c in cols:
        if c not in df.columns: df[c] = ""
    return df[cols]

# ============== MAIN ==============
def main():
    # 1) DOIs por cluster (lee MMR o FPS según REP_METHOD)
    doi_map = read_rep_dois(IN_ROOT, REP_FILENAME)
    if not doi_map:
        return

    # 2) Driver Neo4j
    try:
        driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PWD))
        with driver.session(database=NEO4J_DB) as s:
            s.run("RETURN 1").consume()
    except neo_err.AuthError as e:
        print("❌ Auth Neo4j: revisa NEO4J_USER/NEO4J_PASSWORD/NEO4J_URI/NEO4J_DATABASE")
        print(e)
        return

    # 3) Export por cluster + merge
    merged = []
    method_dir = os.path.join(OUT_DIR, OUT_SUFFIX)
    os.makedirs(method_dir, exist_ok=True)

    for cluster, dois in sorted(doi_map.items()):
        if not dois:
            continue
        df = query_by_dois(driver, sorted(list(dois)))
        # traZabilidad: añade columna Cluster
        df.insert(0, "Cluster", cluster)
        out_path = os.path.join(method_dir, f"vosviewer_{cluster}_{OUT_SUFFIX}.csv")
        df.to_csv(out_path, index=False)
        merged.append(df)
        print(f"📄 {cluster}: {len(df)} filas → {out_path}")

    if merged:
        all_df = pd.concat(merged, ignore_index=True)
        out_all = os.path.join(method_dir, f"vosviewer_ALL_clusters_{OUT_SUFFIX}.csv")
        all_df.to_csv(out_all, index=False)
        print(f"🎯 Merge total: {len(all_df)} filas → {out_all}")

    driver.close()

if __name__ == "__main__":
    main()
