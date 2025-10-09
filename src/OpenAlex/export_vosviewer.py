"""
OpenAlex → VOSviewer exporter
- CSV “Scopus-like” compatible con VOSviewer (pestaña Scopus)
- MMR avanzado con SBERT o TF-IDF (fallback)

Requisitos pip (instalar dentro del venv):
pip install pandas numpy python-dotenv neo4j scikit-learn sentence-transformers

Si no quieres SBERT: omite sentence-transformers y el script usa TF-IDF.
"""

import os
import sys
import re
import math
import csv
import time
from datetime import datetime
from typing import List, Dict

import numpy as np
import pandas as pd
from dotenv import load_dotenv
from neo4j import GraphDatabase
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# SBERT opcional
try:
    from sentence_transformers import SentenceTransformer
except Exception:
    SentenceTransformer = None

# ================== CONFIG ==================
load_dotenv()

NEO4J_URI = os.getenv("NEO4J_URI", "neo4j://localhost:7687")
NEO4J_USER = os.getenv("NEO4J_USER", "neo4j")
NEO4J_PWD  = os.getenv("NEO4J_PASSWORD", "")
NEO4J_DB   = os.getenv("NEO4J_DATABASE", "alzheimerdb")

# Labels (por si tu grafo usa otros)
PAPER_LABEL   = os.getenv("PAPER_LABEL", "Paper")
AUTHOR_LABEL  = os.getenv("AUTHOR_LABEL", "Author")
JOURNAL_LABEL = os.getenv("JOURNAL_LABEL", "Journal")
CONCEPT_LABEL = os.getenv("CONCEPT_LABEL", "Concept")

# Relación alternativas (si tu esquema usa otros nombres)
# Se usarán con type(r) IN [...]
AUTH_RELS   = [r.strip() for r in os.getenv("AUTHORED_RELS", "AUTHORED;AUTHORED_BY").split(";") if r.strip()]
JOUR_RELS   = [r.strip() for r in os.getenv("PUBLISHED_IN_RELS", "PUBLISHED_IN;APPEARS_IN").split(";") if r.strip()]
CONC_RELS   = [r.strip() for r in os.getenv("HAS_CONCEPT_RELS", "HAS_CONCEPT;HAS_FIELD_OF_STUDY").split(";") if r.strip()]

# Tamaños
PAGE_SIZE   = int(os.getenv("EXPORT_BATCH_SIZE", "300"))   # reduce si ves OOM
MMR_LIMIT   = int(os.getenv("MMR_MAX_ROWS", "3000"))       # máximo papers a considerar para MMR

BASE_DIR   = os.path.dirname(os.path.abspath(__file__))
OUT_DIR    = os.path.join(BASE_DIR, "vosviewer_exports")
os.makedirs(OUT_DIR, exist_ok=True)

# ================== LIMPIEZA / UTIL ==================

_ALIAS = {
    "alzheimer disease": "Alzheimer's Disease",
    "alzheimer's disease": "Alzheimer's Disease",
    "dementia": "Dementia",
    "mild cognitive impairment": "Mild Cognitive Impairment",
    "amyloid beta": "Amyloid Beta",
    "tau protein": "Tau Protein",
    "neurodegeneration": "Neurodegeneration",
    "brain imaging": "Brain Imaging",
    "cognitive function": "Cognitive Function",
    "memory": "Memory",
    "biomarker": "Biomarker",
    "neuroinflammation": "Neuroinflammation",
}
_STOPLIKE = {
    "article","review","paper","study","research","introduction","conclusion",
    "methods","results","dataset","analysis","human","humans","medicine",
    "medical","clinical"
}

def clean_concepts(concepts_list) -> str:
    if not concepts_list or not isinstance(concepts_list, list):
        return ""
    seen, out = set(), []
    for c in concepts_list:
        if not isinstance(c, str):
            continue
        w = c.strip().lower()
        if not w or w in _STOPLIKE:
            continue
        if w in _ALIAS:
            w2 = _ALIAS[w]
        else:
            w2 = " ".join(s.capitalize() for s in w.split())
        if len(w2.split()) > 6:
            continue
        k = w2.lower()
        if k not in seen:
            seen.add(k)
            out.append(w2)
    return "; ".join(out)

def clean_abstract(text: str) -> str:
    if not text:
        return ""
    patt = [
        r"©\s*\d{4}.*?rights reserved\.?",
        r"all rights reserved\.?",
        r"this is an open access article.*?license\.",
        r"creative commons.*?license",
        r"supplementary material.*",
        r"https?://\S+|doi:\s*\S+|10\.\d{4,9}/\S+",
        r"conflict of interest.*",
    ]
    s = re.sub(r"\s+", " ", text).strip()
    for p in patt:
        s = re.sub(p, " ", s, flags=re.I)
    s = re.sub(r"\s+", " ", s).strip()
    return s

def join_semi(items: List[str]) -> str:
    items = [x for x in (items or []) if isinstance(x, str) and x.strip()]
    return "; ".join(dict.fromkeys([x.strip() for x in items]))  # dedup preservando orden

def now_stamp() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")

# ================== QUERIES ==================

COUNT_QUERY = f"""
MATCH (p:{PAPER_LABEL})
WHERE p.title IS NOT NULL
RETURN count(p) AS total
"""

# Usamos type(r) IN [...] para compatibilidad (evita errores de sintaxis con pipes)
PAGED_QUERY = f"""
MATCH (p:{PAPER_LABEL})
WHERE p.title IS NOT NULL
OPTIONAL MATCH (p)-[ra]-(a:{AUTHOR_LABEL})
WHERE ra IS NULL OR type(ra) IN {AUTH_RELS}
OPTIONAL MATCH (p)-[rj]->(j:{JOURNAL_LABEL})
WHERE rj IS NULL OR type(rj) IN {JOUR_RELS}
OPTIONAL MATCH (p)-[rc]->(c:{CONCEPT_LABEL})
WHERE rc IS NULL OR type(rc) IN {CONC_RELS}
WITH p, j,
     collect(DISTINCT a.display_name) AS a_names,
     collect(DISTINCT coalesce(a.openalex_id, a.author_id)) AS a_ids,
     collect(DISTINCT c.display_name) AS concepts
ORDER BY p.cited_by_count DESC
SKIP $offset
LIMIT $limit
RETURN
  p.openalex_id           AS openalex_id,
  p.doi                   AS doi,
  p.title                 AS title,
  p.publication_year      AS year,
  p.cited_by_count        AS cited_by,
  p.abstract              AS abstract,
  p.type                  AS document_type,
  p.language              AS language,
  j.display_name          AS journal,
  a_names                 AS authors_raw,
  a_ids                   AS author_ids_raw,
  concepts                AS concepts_raw
"""

MMR_QUERY = f"""
MATCH (p:{PAPER_LABEL})
WHERE p.title IS NOT NULL
  AND p.abstract IS NOT NULL
  AND p.abstract <> ''
OPTIONAL MATCH (p)-[rj]->(j:{JOURNAL_LABEL})
WHERE rj IS NULL OR type(rj) IN {JOUR_RELS}
OPTIONAL MATCH (p)-[ra]-(a:{AUTHOR_LABEL})
WHERE ra IS NULL OR type(ra) IN {AUTH_RELS}
RETURN
  p.openalex_id      AS openalex_id,
  p.title            AS title,
  p.publication_year AS year,
  p.cited_by_count   AS cited_by,
  p.abstract         AS abstract,
  j.display_name     AS journal,
  collect(DISTINCT a.display_name) AS authors
ORDER BY p.cited_by_count DESC
LIMIT $limit
"""

# ================== NEO4J ==================

def connect_driver():
    drv = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PWD))
    with drv.session(database=NEO4J_DB) as s:
        s.run("RETURN 1").consume()
    return drv

def db_stats(drv) -> Dict[str, int]:
    stats = {}
    with drv.session(database=NEO4J_DB) as s:
        for L in [PAPER_LABEL, AUTHOR_LABEL, JOURNAL_LABEL, "Institution", CONCEPT_LABEL, "Country"]:
            try:
                c = s.run(f"MATCH (n:{L}) RETURN count(n) AS c").single()["c"]
            except Exception:
                c = 0
            stats[L] = c
        try:
            cabs = s.run(f"MATCH (p:{PAPER_LABEL}) WHERE p.abstract IS NOT NULL AND p.abstract<>'' RETURN count(p) AS c").single()["c"]
        except Exception:
            cabs = 0
        try:
            cites = s.run("MATCH ()-[r:CITES]->() RETURN count(r) AS c").single()["c"]
        except Exception:
            cites = 0
    stats["with_abstract"] = cabs
    stats["Citations"] = cites
    return stats

# ================== EXPORT SCOPUS-LIKE ==================

def to_scopus_row(rec: dict) -> dict:
    authors = join_semi(rec.get("authors_raw") or [])
    author_ids = join_semi([str(x) for x in (rec.get("author_ids_raw") or []) if x])
    concepts = clean_concepts(rec.get("concepts_raw") or [])
    return {
        # Campos principales que VOSviewer (Scopus) entiende muy bien:
        "Authors": authors,                        # "apellido, iniciales; …"
        "Author(s) ID": author_ids,                # ids separados por ;
        "Title": rec.get("title") or "",
        "Year": rec.get("year") or "",
        "Source title": rec.get("journal") or "",
        "Abstract": rec.get("abstract") or "",
        "Cited by": rec.get("cited_by") or 0,
        "DOI": rec.get("doi") or "",
        "Author Keywords": concepts,               # usamos conceptos como “keywords”
        # Extras opcionales (vacíos si no hay):
        "Affiliations": "",
        "Volume": "",
        "Issue": "",
        "Page start": "",
        "Page end": "",
        "EID": rec.get("openalex_id") or "",
    }

def export_scopus_csv(drv, hard_limit: int | None = None) -> str:
    with drv.session(database=NEO4J_DB) as ses:
        total = ses.run(COUNT_QUERY).single()["total"]
    if hard_limit:
        total = min(total, hard_limit)

    ts = now_stamp()
    out_path = os.path.join(OUT_DIR, f"scopus_vosviewer_{ts}.csv")

    exported = 0
    offset = 0

    # Usamos writer de csv para líneas robustas y rápidas
    fieldnames = [
        "Authors","Author(s) ID","Title","Year","Source title","Abstract",
        "Cited by","DOI","Author Keywords","Affiliations","Volume","Issue",
        "Page start","Page end","EID"
    ]
    print(f"Total registros a exportar: {total:,}")
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        with drv.session(database=NEO4J_DB) as ses:
            while exported < total:
                lim = min(PAGE_SIZE, total - exported)
                res = ses.run(PAGED_QUERY, offset=offset, limit=lim)
                batch = res.data()
                if not batch:
                    break
                for r in batch:
                    w.writerow(to_scopus_row(r))
                exported += len(batch)
                offset += len(batch)
                if exported % (PAGE_SIZE * 2) == 0 or exported == total:
                    print(f"  → Exportados: {exported:,}/{total:,}")
    print(f"\n✅ CSV Scopus listo: {out_path}\n   Registros: {exported:,}")
    return out_path

# ================== MMR ==================

def create_embeddings(texts: List[str], method: str = "sbert") -> np.ndarray:
    if method == "sbert" and SentenceTransformer is not None:
        model_name = os.getenv("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
        model = SentenceTransformer(model_name)
        emb = model.encode(texts, show_progress_bar=True, convert_to_numpy=True, normalize_embeddings=True)
        return emb
    # TF-IDF fallback
    vec = TfidfVectorizer(
        max_features=5000, ngram_range=(1,2), min_df=2, max_df=0.8, stop_words="english"
    )
    X = vec.fit_transform(texts).astype(np.float32)
    X = X / (np.linalg.norm(X.toarray(), axis=1, keepdims=True) + 1e-12)
    return X

def relevance_meta(papers: pd.DataFrame, query_terms: List[str] | None = None) -> np.ndarray:
    if not query_terms:
        query_terms = ["alzheimer","dementia","cognitive","neurodegeneration","amyloid","tau","memory"]
    scores = []
    for _, r in papers.iterrows():
        cites = int(r.get("cited_by", 0) or 0)
        sc_cit = min(np.log1p(cites) / np.log1p(1000), 1.0)
        year = int(r.get("year", 1990) or 1990)
        sc_year = (year - 1990) / (2024 - 1990) if year >= 1990 else 0.0
        txt = f"{str(r.get('title','')).lower()} {str(r.get('abstract','')).lower()}"
        sc_terms = min(sum(1 for t in query_terms if t in txt) / len(query_terms), 1.0)
        abs_len = len(str(r.get("abstract","")).split())
        sc_abs = min(abs_len / 200, 1.0)
        scores.append(0.4*sc_cit + 0.2*sc_year + 0.3*sc_terms + 0.1*sc_abs)
    return np.array(scores, dtype=np.float32)

def mmr_select(df: pd.DataFrame, Z: np.ndarray, k: int = 50, lam: float = 0.6,
               qterms: List[str] | None = None) -> pd.DataFrame:
    n = len(df)
    if n == 0 or k == 0:
        return df.head(0)
    k = min(k, n)
    meta = relevance_meta(df, qterms)
    q = Z.mean(axis=0, keepdims=True)
    def norm(X): 
        nv = np.linalg.norm(X, axis=1, keepdims=True)
        return X / (nv + 1e-12)
    Z = norm(Z); Q = norm(q)
    rel_q = cosine_similarity(Z, Q).ravel()
    rel = 0.7*meta + 0.3*rel_q

    chosen = [int(np.argmax(rel))]
    pool = set(range(n)) - set(chosen)

    while len(chosen) < k and pool:
        best_idx, best_score = None, -1e9
        Ze = Z[chosen]
        for i in list(pool):
            div = np.max(cosine_similarity(Z[[i]], Ze))
            score = lam*rel[i] - (1-lam)*div
            if score > best_score:
                best_score, best_idx = score, i
        chosen.append(best_idx)
        pool.remove(best_idx)

    out = df.iloc[chosen].copy().reset_index(drop=True)
    out["mmr_rank"] = np.arange(1, len(out)+1)
    out["relevance_score"] = rel[chosen]
    out["query_similarity"] = rel_q[chosen]
    out["metadata_relevance"] = meta[chosen]
    return out

def diversity_metrics(Z: np.ndarray) -> Dict[str, float]:
    if Z.shape[0] < 2:
        return {"n": int(Z.shape[0]), "diameter_cos": 0.0, "mean_distance": 0.0,
                "std_distance": 0.0, "p90_distance": 0.0, "p95_distance": 0.0,
                "spectral_entropy": 0.0, "participation_ratio": 1.0}
    S = cosine_similarity(Z)
    D = 1.0 - S
    iu = np.triu_indices(D.shape[0], 1)
    d = D[iu]
    try:
        C = Z - Z.mean(axis=0, keepdims=True)
        cov = (C.T @ C) / max(1, Z.shape[0]-1)
        ev = np.linalg.eigvalsh(cov)
        ev = np.clip(ev, 1e-12, None)
        p = ev/ev.sum()
        H = float(-np.sum(p*np.log(p)))
        PR = float((ev.sum()**2)/np.sum(ev**2))
    except Exception:
        H, PR = 0.0, 1.0
    return {
        "n": int(Z.shape[0]),
        "diameter_cos": float(d.max()),
        "mean_distance": float(d.mean()),
        "std_distance": float(d.std()),
        "p90_distance": float(np.percentile(d, 90)),
        "p95_distance": float(np.percentile(d, 95)),
        "spectral_entropy": H,
        "participation_ratio": PR,
    }

def run_mmr(drv, k: int = 50, method: str = "sbert", lam: float = 0.6) -> str:
    print(f"MMR → método={method}, k={k}, λ={lam}")
    with drv.session(database=NEO4J_DB) as s:
        rows = s.run(MMR_QUERY, limit=MMR_LIMIT).data()
    if not rows:
        print("No hay rows elegibles para MMR.")
        return ""
    df = pd.DataFrame([{
        "openalex_id": r["openalex_id"], "title": r["title"], "year": r["year"],
        "cited_by": r["cited_by"] or 0, "abstract": clean_abstract(r["abstract"] or ""),
        "journal": r["journal"] or "", "authors": join_semi(r["authors"] or [])
    } for r in rows])

    # filtra abstracts muy cortos
    df["abs_words"] = df["abstract"].str.split().str.len()
    df = df[df["abs_words"] >= 20].reset_index(drop=True)
    texts = [f"{t}. {t}. {a}" for t,a in zip(df["title"], df["abstract"])]
    Z = create_embeddings(texts, method=method)
    reps = mmr_select(df, Z, k=k, lam=lam, qterms=["alzheimer","dementia","cognitive","neurodegeneration","amyloid","tau"])
    Zsel = Z[reps.index.values]
    metrics = diversity_metrics(Zsel)

    ts = now_stamp()
    out_reps = os.path.join(OUT_DIR, f"mmr_{method}_{ts}.csv")
    reps.to_csv(out_reps, index=False, encoding="utf-8")
    out_div = os.path.join(OUT_DIR, f"mmr_diversity_{ts}.csv")
    pd.DataFrame([metrics]).to_csv(out_div, index=False)

    print(f"\nMMR guardado:")
    print(f"  • Representantes: {out_reps}")
    print(f"  • Métricas:       {out_div}")
    print(f"  • n={len(reps)}, años {reps['year'].min()}–{reps['year'].max()}, citas medias {reps['cited_by'].mean():.1f}")
    return out_reps

# ================== MAIN ==================

def main():
    print("\nEXPORTADOR OPENALEX → VOSviewer")
    print("======================================================")
    try:
        drv = connect_driver()
        print(f"Conectado a Neo4j DB: {NEO4J_DB}")
    except Exception as e:
        print(f"Error conectando a Neo4j: {e}")
        sys.exit(1)

    stats = db_stats(drv)
    print("\nESTADÍSTICAS:")
    for k, v in [
        (PAPER_LABEL, stats.get(PAPER_LABEL, 0)),
        (AUTHOR_LABEL, stats.get(AUTHOR_LABEL, 0)),
        (JOURNAL_LABEL, stats.get(JOURNAL_LABEL, 0)),
        ("Institution", stats.get("Institution", 0)),
        (CONCEPT_LABEL, stats.get(CONCEPT_LABEL, 0)),
        ("Country", stats.get("Country", 0)),
        ("Papers_with_abstract", stats.get("with_abstract", 0)),
        ("Citations", stats.get("Citations", 0)),
    ]:
        print(f"  • {k}: {v:,}")

    print(f"\nSalida: {OUT_DIR}\n")
    print("Opciones:")
    print("1) Exportar CSV formato Scopus (VOSviewer)")
    print("2) MMR avanzado (SBERT/TF-IDF) - representantes")
    print("3) Ambos\n")

    try:
        opt = input("Opción (1-3): ").strip()
        if opt == "1":
            lim = input("Límite opcional de registros (ENTER = todo): ").strip()
            lim = int(lim) if lim else None
            export_scopus_csv(drv, lim)
        elif opt == "2":
            k = input("Número de representantes (default 50): ").strip()
            k = int(k) if k else 50
            method = input("Embeddings [sbert|tfidf] (default sbert): ").strip().lower() or "sbert"
            lam = input("λ (relevancia vs. diversidad, 0..1, default 0.6): ").strip()
            lam = float(lam) if lam else 0.6
            run_mmr(drv, k=k, method=method, lam=lam)
        elif opt == "3":
            export_scopus_csv(drv, None)
            k = 50
            method = "sbert"
            lam = 0.6
            run_mmr(drv, k=k, method=method, lam=lam)
        else:
            print("Opción no válida.")
    except KeyboardInterrupt:
        print("\nInterrumpido por usuario.")
    finally:
        drv.close()
    print(f"\nDone. Archivos en: {OUT_DIR}")

if __name__ == "__main__":
    main()
