#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Exporta metadatos desde Neo4j para DOIs seleccionados por MMR/FPS y deja
un CSV estilo Scopus listo para VOSviewer (sin keywords con '|').

Uso:
  python export_vosviewer.py --method fps
  python export_vosviewer.py --method mmr
Opcionales:
  --in-root ./data_checkpoints_plus
  --out-dir ./vosviewer_exports/mmr
"""

import os, re, glob, argparse
import pandas as pd
from typing import Dict, Set, List
from dotenv import load_dotenv
from neo4j import GraphDatabase, exceptions as neo_err

# ------------------ CLI & ENV ------------------

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--method", choices=["fps","mmr"], default="fps",
                   help="Escoge representantes: fps o mmr")
    p.add_argument("--in-root", default=None,
                   help="Raíz con */{fps,mmr}_representatives.csv")
    p.add_argument("--out-dir", default=None,
                   help="Directorio de salida (por defecto: ./vosviewer_exports/<method>/)")
    return p.parse_args()

load_dotenv()  # usa tu .env tal cual

NEO4J_URI  = os.getenv("NEO4J_URI", "neo4j://localhost:7687")
NEO4J_USER = os.getenv("NEO4J_USER", "neo4j")
NEO4J_PWD  = os.getenv("NEO4J_PASSWORD", "neo4j")
NEO4J_DB   = os.getenv("NEO4J_DATABASE", "neo4j")
BATCH = 200

# ------------------ Limpieza de keywords ------------------

_SPLIT_RX = re.compile(r"\s*(?:;|,|\||/|·|•|–|—|&|\band\b|\+)\s*", re.I)
_WS_RX = re.compile(r"\s+")
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
    if not s: return ""
    low = s.lower()
    if low in _ALIAS: return _ALIAS[low]
    parts = s.split()
    return " ".join([p if p.upper() in _KEEP_SHORT else p.capitalize() for p in parts])

def clean_keywords(val) -> str:
    # Acepta str o list[str]; devuelve 'kw1; kw2; ...'
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
        if not t: continue
        if t.lower() in _STOPLIKE: continue
        if len(t.split()) > _MAX_WORDS and t.upper() not in _KEEP_SHORT: continue
        key = t.casefold()
        if key in seen: continue
        seen.add(key)
        out.append(t)
    return "; ".join(out)

# ------------------ Utilidades ------------------

def norm_doi(x: str) -> str:
    if not isinstance(x, str): return ""
    s = x.strip().lower()
    for p in ("https://doi.org/","http://doi.org/","doi:"):
        if s.startswith(p): s = s[len(p):]
    return s.strip()

def read_seed_dois(root: str, method: str) -> Dict[str, Set[str]]:
    pattern = os.path.join(root, "*", f"{method}_representatives.csv")
    out: Dict[str, Set[str]] = {}
    for fp in sorted(glob.glob(pattern)):
        cluster = os.path.basename(os.path.dirname(fp))
        try:
            df = pd.read_csv(fp)
        except Exception:
            print(f"⚠️  No pude leer {fp}"); continue
        if "doi" not in df.columns:
            print(f"⚠️  {cluster}: falta columna 'doi'"); continue
        dois = {norm_doi(d) for d in df["doi"].astype(str) if norm_doi(d)}
        out[cluster] = dois
        print(f"✅ {cluster}: {len(dois)} DOIs {method.upper()}")
    if not out:
        print(f"❌ No encontré {pattern}")
    return out

# ------------------ Cypher ------------------

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
  p.doi                             AS doi,
  p.eid                             AS eid,
  p.title                           AS title,
  p.year                            AS year,
  COALESCE(toInteger(p.citedBy),0)  AS citedBy,
  p.abstract                        AS abstract,
  j.name                            AS source_title,
  collect(DISTINCT a.name)          AS authors,
  collect(DISTINCT ai.name)         AS institutions,
  collect(DISTINCT ac.name)         AS author_countries,
  collect(DISTINCT pc.name)         AS pub_countries,
  collect(DISTINCT k.name)          AS author_keywords,
  collect(DISTINCT fa.name)         AS funding_agencies,
  p.funding_text                    AS funding_text
"""

def query_by_dois(driver, dois: List[str]) -> pd.DataFrame:
    rows: List[dict] = []
    with driver.session(database=NEO4J_DB) as s:
        for i in range(0, len(dois), BATCH):
            chunk = dois[i:i+BATCH]
            rows.extend([r.data() for r in s.run(CYPHER, dois=chunk)])
    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame(rows)

    # 1) Limpiar keywords ANTES de aplanar
    if "author_keywords" in df.columns:
        df["author_keywords"] = df["author_keywords"].apply(clean_keywords)

    # 2) Aplanar listas (autores, afiliaciones, países…)
    def j(x):
        if isinstance(x, list): return "; ".join([str(z) for z in x if z])
        return x
    for col in ["authors","institutions","author_countries","pub_countries","funding_agencies"]:
        if col in df.columns: df[col] = df[col].apply(j)

    # 3) Renombrar a estilo Scopus
    rename = {
        "title":"Title","authors":"Authors","year":"Year","source_title":"Source title",
        "institutions":"Affiliations","author_countries":"Author Countries",
        "pub_countries":"Publication Countries","author_keywords":"Author Keywords",
        "funding_agencies":"Funding Sponsors","funding_text":"Funding Text",
        "citedBy":"Cited by","doi":"DOI","eid":"EID","abstract":"Abstract",
    }
    df = df.rename(columns=rename)

    # 4) Orden canónico (sin columna “raw” para que VOSviewer no se confunda)
    cols = ["Title","Authors","Year","Source title","Affiliations","Author Countries",
            "Publication Countries","Author Keywords","Funding Sponsors","Funding Text",
            "Cited by","DOI","EID","Abstract"]
    for c in cols:
        if c not in df.columns: df[c] = ""
    return df[cols]

# ------------------ Main ------------------

def main():
    args = parse_args()
    base = os.path.dirname(os.path.abspath(__file__))
    in_root = args.in_root or os.path.join(base, "data_checkpoints_plus")
    out_dir = args.out_dir or os.path.join(base, "vosviewer_exports", args.method)
    os.makedirs(out_dir, exist_ok=True)

    doi_map = read_seed_dois(in_root, args.method)
    if not doi_map: return

    try:
        driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PWD))
        with driver.session(database=NEO4J_DB) as s: s.run("RETURN 1").consume()
    except neo_err.AuthError as e:
        print("❌ Auth Neo4j: revisa NEO4J_USER/NEO4J_PASSWORD/NEO4J_URI/NEO4J_DATABASE"); print(e); return

    merged = []
    for cluster, dois in sorted(doi_map.items()):
        if not dois: continue
        df = query_by_dois(driver, sorted(dois))
        out_path = os.path.join(out_dir, f"vosviewer_{cluster}_{args.method}.csv")
        df.to_csv(out_path, index=False)
        merged.append(df)
        print(f"📄 {cluster}: {len(df)} filas → {out_path}")

    if merged:
        all_df = pd.concat(merged, ignore_index=True)
        out_all = os.path.join(out_dir, f"vosviewer_ALL_clusters_{args.method}.csv")
        all_df.to_csv(out_all, index=False)
        print(f"🎯 Merge total: {len(all_df)} filas → {out_all}")

if __name__ == "__main__":
    main()
