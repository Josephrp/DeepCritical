#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
complex_semantic.py — Abstracts-only semantic clustering + MMR/FPS + diversity + contextual term stats
- Dedupe robusto, control de sesgo por términos de query (strip|downweight|keep), SBERT/TF-IDF, métricas internas + bootstrap ARI (90%).
- Etiquetado c-TF-IDF (términos por clúster), representantes, timelines con análisis temporal avanzado, PCA 2D.
- Diversidad: diámetro coseno, percentiles, PR espectral, entropía + MMR/FPS + cobertura adaptativa.
- Distancias entre términos (intra-clúster) con embeddings contextuales (media de oraciones donde aparece cada término).
- Cluster ensemble (k-means + agglomerative) con consenso y selección automática por ranking (ARI, silhouette, CH, DB).
- Sensibilidad cross-cutting más amplia (1..5) con matriz de Jaccard.
- (Opcional) Validación externa multi-fuente en Neo4j: modularidad de citas, coautoría, co-ocurrencia de journals/keywords.
"""

import os, re, json, argparse, logging, math, warnings
from dataclasses import dataclass
from typing import List, Dict, Any, Tuple, Optional

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import numpy as np
import pandas as pd

from dotenv import load_dotenv
from neo4j import GraphDatabase

from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics import (
    silhouette_score, silhouette_samples, davies_bouldin_score,
    calinski_harabasz_score, adjusted_rand_score,
)
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.decomposition import PCA

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Backends opcionales
try:
    from sentence_transformers import SentenceTransformer
except Exception:
    SentenceTransformer = None
try:
    import networkx as nx  # solo si usas validaciones de grafo
except Exception:
    nx = None

logging.basicConfig(level=logging.INFO, format='[%(levelname)s] %(message)s')
log = logging.getLogger("complex_semantic")
warnings.filterwarnings("ignore", category=FutureWarning)

# ========= helpers JSON-safe =========
def _to_py_scalar(x):
    import numpy as _np
    if isinstance(x, (_np.integer,)):
        return int(x)
    if isinstance(x, (_np.floating,)):
        return float(x)
    if isinstance(x, (_np.bool_,)):
        return bool(x)
    return x

def to_py(obj):
    """Convierte recursivamente numpy/keys no serializables a tipos Python nativos (y keys a str si hace falta)."""
    import numpy as _np
    if isinstance(obj, dict):
        out = {}
        for k, v in obj.items():
            if not isinstance(k, (str, int, float, bool, type(None))):
                try:
                    k = str(_to_py_scalar(k))
                except Exception:
                    k = str(k)
            out[str(k)] = to_py(v)
        return out
    elif isinstance(obj, (list, tuple)):
        return [to_py(x) for x in obj]
    elif isinstance(obj, (_np.ndarray,)):
        return to_py(obj.tolist())
    else:
        return _to_py_scalar(obj)

def _cluster_sizes(labels: np.ndarray) -> Dict[int, int]:
    """Asegura claves y valores Python puros (evita numpy.int32 como key)."""
    uniq = np.unique(labels).tolist()
    return {int(c): int(np.sum(labels == c)) for c in uniq}

# ========= Frases & stopwords (dominio) =========
PHRASES = [
    "life cycle","life cycle assessment","circular economy",
    "mechanical recycling","chemical recycling","plastics recycling",
    "polyethylene terephthalate","high density polyethylene","low density polyethylene",
    "non intentionally added substances","non-intentionally added substances",
    "food contact materials","post consumer resin","post-consumer resin",
    "closed loop recycling","open loop recycling","solid state polymerization",
    "mass balance","carbon footprint","greenhouse gas","global warming potential",
]
STOPWORDS_EN = {
    "the","and","or","for","of","to","in","on","by","with","a","an","is","are","as","that","this","these","those",
    "we","our","their","its","from","at","be","been","it","into","such","using","used","use","can","may","could","should",
    "however","therefore","thus","also","between","among","across","within","over","under","more","less","most","least","both",
    "results","methods","introduction","conclusion","study","paper","research",
    "was","were","has","have","had","which","not","new","two","three","based","among","using","used","use",
    "authors","author","rights","reserved","copyright","publisher","preprint","peer","reviewed","license",
    "creative","commons","open","access","article","version","supplementary","material","graphical","abstract",
    "statement","competing","interests","conflict","role","funding","acknowledgements","permission","figure","table",
    "note","received","accepted","revised","issue","volume","pages","doi","elsevier","springer","wiley","mdpi","taylor","francis",
}
TOKEN_CHEM_WHITELIST = {
    "pet","pe","hdpe","ldpe","pp","ps","pla","pbat","pbt","pvc","pa","abs","pc","pmma","psf","pbs","pvoh","pva",
    "tio2","zno","nias","nist","uv","rpet","ldpe/hdpe","microplastics","nanoplastics","lca","gwp","ghg"
}
TOKEN_PATTERN = re.compile(r"[A-Za-z][A-Za-z0-9_\-.()]+|\d+(?:\.\d+)?%?")
BOILERPLATE_PATTERNS = [
    r"©\s*\d{4}.*?rights reserved\.?",
    r"all rights reserved\.?",
    r"this is an open access article.*?license\.",
    r"the authors? \d{4}",
    r"authors? declare .*?",
    r"publisher.*?not responsible.*?",
    r"preprint.*?not peer reviewed",
    r"supplementary material.*",
    r"graphical abstract.*",
    r"creative commons.*?license",
    r"conflict of interest.*",
    r"competing interests.*",
    r"acknowledg(e)?ments?.*",
]

def clean_abstract(t: str) -> str:
    s = re.sub(r"\s+", " ", t or " ").strip()
    for pat in BOILERPLATE_PATTERNS:
        s = re.sub(pat, " ", s, flags=re.IGNORECASE)
    s = re.sub(r"https?://\S+|doi:\s*\S+|10\.\d{4,9}/\S+", " ", s, flags=re.IGNORECASE)
    s = re.sub(r"\b\d{4}\s+authors?\b", " ", s, flags=re.IGNORECASE)
    s = re.sub(r"©\s*\d{4}", " ", s)
    return re.sub(r"\s+", " ", s).strip()

def protect_phrases(txt: str) -> str:
    s = txt
    for ph in sorted(PHRASES, key=len, reverse=True):
        s = re.sub(re.escape(ph), ph.replace(" ", "_"), s, flags=re.IGNORECASE)
    return s

def tokenize(text: str) -> List[str]:
    text = protect_phrases(text)
    toks = TOKEN_PATTERN.findall(text)
    out: List[str] = []
    for t in toks:
        low = t.lower().strip(".,;:()[]{}\"'")
        if not low: continue
        if low in TOKEN_CHEM_WHITELIST: out.append(low); continue
        if low in STOPWORDS_EN: continue
        if len(low) <= 2 and low not in TOKEN_CHEM_WHITELIST: continue
        out.append(low)
    return out

# ========= Definiciones de clúster (para construir Cypher) =========
BUSINESS_CLUSTER_DEFS: Dict[str, List[str]] = {
    "recycling_processes": ["mechanical recycling","chemical recycling","plastics recycling","pyrolysis"],
    "materials_polymers": ["recycled plastic","plastic packaging","polyethylene","terephthalate","pet",
                           "nias","nist","non-intentionally added substances","non intentionally added substances",
                           "contaminant","migration","decontamination"],
    "environmental_assessment": ["circular economy","life cycle","life cycle assessment","co 2","carbon footprint","environmental impact"],
    "social_perception": [
        "social attitude","public perception","user acceptance","consumer acceptance","social acceptance",
        "public acceptance","behavioral change","environmental behavior","pro-environmental behavior",
        "risk perception","health concern","consumer behavior","consumer attitude","willingness to pay","purchase intention",
        "behavioral intention","technology acceptance model","stakeholder engagement","public participation","social license",
        "environmental concern","green consumption","sustainable behavior","intention to recycle","perceived risk",
        "perceived usefulness","perceived ease of use"
    ],
    "regulatory_economic": ["legislation","policy","regulation","recycling target","recycling rate","quality standard",
                            "economic viability","cost analysis","business model","supply chain","post-consumer","energy recovery"],
}

# ========= Neo4j client =========
class Neo4jClient:
    def __init__(self):
        load_dotenv()
        self.uri = os.getenv("NEO4J_URI", "neo4j://localhost:7687")
        self.user = os.getenv("NEO4J_USER", "neo4j")
        self.pwd  = os.getenv("NEO4J_PASSWORD", "neo4j")
        self.db   = os.getenv("NEO4J_DATABASE", "neo4j")
        self.driver = GraphDatabase.driver(self.uri, auth=(self.user, self.pwd))
        log.info(f"Neo4j: {self.uri} | DB: {self.db}")

    def close(self):
        try: self.driver.close()
        except Exception: pass

    def fetch(self, cypher: str, params: Optional[Dict[str,Any]] = None) -> pd.DataFrame:
        params = params or {}
        with self.driver.session(database=self.db) as s:
            rows = [r.data() for r in s.run(cypher, **params)]
        df = pd.DataFrame(rows)
        if df.empty: return df
        for c in ["doi","eid","title","abstract","year","citedBy"]:
            if c not in df.columns: df[c] = None
        df = df[["doi","eid","title","abstract","year","citedBy"]]
        # Dedupe robusto
        df["doi_norm"] = df["doi"].astype(str).str.strip().str.lower()
        mask = df["doi_norm"].notna() & (df["doi_norm"]!="") & (df["doi_norm"]!="nan")
        df_valid = df.loc[mask].drop_duplicates(subset=["doi_norm"])
        df_null  = df.loc[~mask].copy()
        df_null["title_norm"] = (df_null["title"].fillna("").astype(str)
                                 .str.lower().str.replace(r"\s+"," ",regex=True).str.strip())
        subset_null = ["title_norm","year"] if "year" in df_null.columns else ["title_norm"]
        df_null = df_null.drop_duplicates(subset=subset_null)
        df = pd.concat([df_valid.drop(columns=["doi_norm"],errors="ignore"),
                        df_null.drop(columns=["doi_norm","title_norm"],errors="ignore")], ignore_index=True)
        # Limpieza de abstracts + filtro por longitud
        df["abstract"] = df["abstract"].fillna("").astype(str).apply(clean_abstract)
        df["abstract_len_words"] = df["abstract"].str.split().apply(len)
        min_words = int(os.getenv("MIN_ABS_WORDS", 20))
        kept = df[df["abstract_len_words"]>=min_words].drop(columns=["abstract_len_words"]).reset_index(drop=True)
        try:
            yy = pd.to_numeric(kept["year"], errors="coerce")
            if yy.notna().any():
                log.info(f"Años en corpus: {int(yy.min()) if yy.notna().any() else 'NA'}..{int(yy.max()) if yy.notna().any() else 'NA'} | n={len(kept)}")
        except Exception:
            pass
        return kept

# ========= Cypher builders =========
def cypher_for_keyword_cluster(terms: List[str]) -> str:
    or_block = " OR ".join([f"toLower(k.name) CONTAINS '{t.lower()}'" for t in terms])
    return f"""
    MATCH (p:Publication)-[:HAS_KEYWORD]->(k:Keyword)
    WHERE p.abstract IS NOT NULL AND p.abstract <> ''
      AND p.year IS NOT NULL AND p.year <> ''
      AND ({or_block})
    WITH DISTINCT p
    RETURN p.doi AS doi, p.eid AS eid, p.title AS title,
           p.abstract AS abstract, p.year AS year,
           COALESCE(toInteger(p.citedBy), 0) AS citedBy
    ORDER BY citedBy DESC
    """

def cypher_for_cross_cutting(threshold:int=2) -> str:
    c1 = BUSINESS_CLUSTER_DEFS["recycling_processes"]
    c2 = BUSINESS_CLUSTER_DEFS["materials_polymers"]
    c3 = BUSINESS_CLUSTER_DEFS["environmental_assessment"]
    c4 = BUSINESS_CLUSTER_DEFS["social_perception"]
    c5 = BUSINESS_CLUSTER_DEFS["regulatory_economic"]
    # Agregación por publicación (no uses k tras el último WITH)
    return f"""
    WITH {json.dumps(c1)} AS c1_terms,
         {json.dumps(c2)} AS c2_terms,
         {json.dumps(c3)} AS c3_terms,
         {json.dumps(c4)} AS c4_terms,
         {json.dumps(c5)} AS c5_terms
    MATCH (p:Publication)-[:HAS_KEYWORD]->(k:Keyword)
    WHERE p.abstract IS NOT NULL AND p.abstract <> ''
      AND p.year IS NOT NULL AND p.year <> ''
    WITH p,
         SUM(CASE WHEN ANY(t IN c1_terms WHERE toLower(k.name) CONTAINS toLower(t)) THEN 1 ELSE 0 END) AS in1,
         SUM(CASE WHEN ANY(t IN c2_terms WHERE toLower(k.name) CONTAINS toLower(t)) THEN 1 ELSE 0 END) AS in2,
         SUM(CASE WHEN ANY(t IN c3_terms WHERE toLower(k.name) CONTAINS toLower(t)) THEN 1 ELSE 0 END) AS in3,
         SUM(CASE WHEN ANY(t IN c4_terms WHERE toLower(k.name) CONTAINS toLower(t)) THEN 1 ELSE 0 END) AS in4,
         SUM(CASE WHEN ANY(t IN c5_terms WHERE toLower(k.name) CONTAINS toLower(t)) THEN 1 ELSE 0 END) AS in5
    WITH p,
         (CASE WHEN in1>0 THEN 1 ELSE 0 END) +
         (CASE WHEN in2>0 THEN 1 ELSE 0 END) +
         (CASE WHEN in3>0 THEN 1 ELSE 0 END) +
         (CASE WHEN in4>0 THEN 1 ELSE 0 END) +
         (CASE WHEN in5>0 THEN 1 ELSE 0 END) AS clusters_count
    WHERE clusters_count >= {int(threshold)}
    RETURN p.doi AS doi, p.eid AS eid, p.title AS title,
           p.abstract AS abstract, p.year AS year,
           COALESCE(toInteger(p.citedBy), 0) AS citedBy
    ORDER BY citedBy DESC
    """

# ========= Control de sesgo por términos de consulta =========
def _normalize_term(term: str) -> List[str]:
    t = term.strip().lower()
    if not t: return []
    variants = {t, t.replace(" ","_"), re.sub(r"[^a-z0-9_ ]+","",t)}
    return sorted(v for v in variants if v)

def filter_query_terms(texts: List[str], query_terms: List[str]) -> List[str]:
    """Elimina (hard strip) los términos de la query."""
    if not query_terms: return texts
    variants = set()
    for term in query_terms: variants.update(_normalize_term(term))
    if not variants: return texts
    pat = r"\b(" + "|".join([re.escape(v) for v in sorted(variants)]) + r")\b"
    rx = re.compile(pat, flags=re.IGNORECASE)
    out = []
    for text in texts:
        s = rx.sub(" ", text)
        s = re.sub(r"\s+"," ",s).strip()
        out.append(s)
    return out

def downweight_query_terms(texts: List[str], query_terms: List[str], penalty: float = 0.5) -> List[str]:
    """
    Atenúa (no elimina) los términos de la query. Heurística simple:
    - Mantiene solo una ocurrencia por término/abstract.
    - Reemplaza ocurrencias extra por una marca neutra 'topic' para preservar cohesión sin dominar.
    """
    if not query_terms: return texts
    variants = set()
    for term in query_terms: variants.update(_normalize_term(term))
    if not variants: return texts
    rx = re.compile(r"\b(" + "|".join([re.escape(v) for v in sorted(variants)]) + r")\b", flags=re.IGNORECASE)
    out = []
    for text in texts:
        seen = set()
        def repl(m):
            key = m.group(1).lower()
            if key not in seen:
                seen.add(key); return m.group(1)
            return "topic" if penalty < 1.0 else m.group(1)
        s = rx.sub(repl, text)
        s = re.sub(r"\s+"," ", s).strip()
        out.append(s)
    return out

# ========= Prepro + embeddings =========
class Analyzer:
    def __init__(self, backend="sbert", random_state=42):
        self.backend = backend
        self.random_state = random_state
        self.df: Optional[pd.DataFrame] = None
        self.proc: Optional[List[str]] = None
        self.X = None
        self.model = None
        self.knn = None

    def set_df(self, df: pd.DataFrame):
        self.df = df.reset_index(drop=True)

    def preprocess(self, filter_terms: Optional[List[str]] = None, query_bias:str="strip"):
        texts = (self.df["abstract"].fillna("").astype(str)).tolist()
        texts = [clean_abstract(t) for t in texts]
        if filter_terms:
            if query_bias == "strip":
                texts = filter_query_terms(texts, filter_terms)
            elif query_bias == "downweight":
                texts = downweight_query_terms(texts, filter_terms, penalty=0.5)
            # if "keep": no tocar
        self.proc = [" ".join(tokenize(t)) for t in texts]

    def embed(self):
        backend = self.backend.lower()
        if backend == "tfidf":
            vec = TfidfVectorizer(ngram_range=(1,2), min_df=2, max_df=0.95, token_pattern=r"[A-Za-z0-9_\-.]+")
            self.X = vec.fit_transform(self.proc)
            self.model = vec
            return
        if backend in {"sbert","specter2","chemberta"}:
            if SentenceTransformer is None:
                raise RuntimeError("Install sentence-transformers for transformer backends")
            if backend == "sbert":
                name = os.getenv("EMBEDDING_MODEL","sentence-transformers/all-MiniLM-L6-v2")
            elif backend == "specter2":
                name = os.getenv("SPECTER2_MODEL","allenai/specter2_base")
            else:
                name = os.getenv("CHEMBERT_MODEL","DeepChem/ChemBERTa-77M-MTR")
            log.info(f"Embedding model: {name}")
            st = SentenceTransformer(name)
            self.model = st
            self.X = st.encode(self.proc, show_progress_bar=False, convert_to_numpy=True, normalize_embeddings=True)
            return
        raise ValueError(f"Unknown backend: {self.backend}")

    def fit_knn(self, k=10):
        if self.X is None: raise RuntimeError("Call embed() first")
        n = len(self.df)
        if n < 2: return
        self.knn = NearestNeighbors(n_neighbors=min(k+1, max(2, n//2)), metric="cosine", algorithm="brute")
        Xe = self.X if isinstance(self.X, np.ndarray) else self.X.toarray()
        self.knn.fit(Xe)

# ========= Clustering & selección de modelo =========
def _to_dense(X):
    if isinstance(X, np.ndarray): return X
    if hasattr(X, 'toarray'): return X.toarray()
    return np.asarray(X)

def auto_k_grid(n:int, kmin:int, kmax:int) -> List[int]:
    k_cap = max(2, min(kmax, int(math.sqrt(max(10, n))) + 2, n//10 + 2))
    kmin = max(2, min(kmin, k_cap))
    return list(range(kmin, k_cap+1))

@dataclass
class ModelScores:
    k: int
    algo: str
    labels: np.ndarray
    silhouette: float
    db: float
    ch: float
    ari_median: float
    ari_iqr: Tuple[float,float]
    cluster_sizes: Dict[int,int]

def eval_internal(X, labels, metric="cosine") -> Tuple[float,float,float]:
    Xe = _to_dense(X)
    try: s = silhouette_score(Xe, labels, metric=metric)
    except Exception: s = float("nan")
    try:
        ch = calinski_harabasz_score(Xe, labels)
        db = davies_bouldin_score(Xe, labels)
    except Exception:
        ch, db = float("nan"), float("nan")
    return s, db, ch

def _choice_sorted(rng: np.random.Generator, n: int, frac: float = 0.9, min_n: int = 20) -> np.ndarray:
    m = max(min_n, int(frac*n)); m = min(m, n)
    return np.sort(rng.choice(n, size=m, replace=False))

def bootstrap_stability(X, clusterer_fn, k:int, B:int=20, seed:int=42) -> Tuple[float,Tuple[float,float]]:
    """
    Bootstrap más conservador: 90% del conjunto, y tamaño mínimo ~3k por seguridad (si se puede).
    """
    Xe = _to_dense(X); n = Xe.shape[0]
    if k < 2 or n < 10 or B <= 1:
        return float("nan"), (float("nan"), float("nan"))
    rng = np.random.default_rng(seed)
    labels_list = []
    min_n = max(20, k*3)
    for _ in range(B):
        idx = _choice_sorted(rng, n, frac=0.9, min_n=min_n)
        Xb = Xe[idx]; lb = clusterer_fn(Xb, k)
        labels_list.append((idx, lb))
    aris = []
    for i in range(len(labels_list)):
        idx_i, li = labels_list[i]
        for j in range(i+1, len(labels_list)):
            idx_j, lj = labels_list[j]
            inter, ia, ja = np.intersect1d(idx_i, idx_j, return_indices=True)
            if len(inter) < max(20, k*3): continue
            aris.append(adjusted_rand_score(li[ia], lj[ja]))
    if not aris: return float("nan"), (float("nan"), float("nan"))
    ar = np.array(aris)
    return float(np.median(ar)), (float(np.quantile(ar,0.25)), float(np.quantile(ar,0.75)))

def cluster_kmeans(X, k:int, random_state:int=42) -> np.ndarray:
    Xe = _to_dense(X); km = KMeans(n_clusters=k, n_init=20, random_state=random_state)
    return km.fit_predict(Xe)

def cluster_agglo(X, k:int) -> np.ndarray:
    Xe = _to_dense(X)
    try: ac = AgglomerativeClustering(n_clusters=k, metric='cosine', linkage='average')
    except TypeError: ac = AgglomerativeClustering(n_clusters=k, affinity='cosine', linkage='average')
    return ac.fit_predict(Xe)

def consensus_from_label_sets(label_sets: List[np.ndarray]) -> np.ndarray:
    """
    Consenso simple vía co-asociación: prob. de co-ocurrir en mismo clúster.
    Luego clusteriza la matriz (1 - coassoc) con agglomerative average.
    """
    L = len(label_sets)
    n = len(label_sets[0])
    for l in label_sets:
        if len(l) != n: raise ValueError("Label sets size mismatch")
    co = np.zeros((n,n), dtype=float)
    for lab in label_sets:
        same = (lab[:,None] == lab[None,:]).astype(float)
        co += same
    co /= L
    D = 1.0 - co
    ks = [len(np.unique(l)) for l in label_sets]
    k = int(np.median(ks)); k = max(2, k)
    try:
        ac = AgglomerativeClustering(n_clusters=k, affinity='precomputed', linkage='average')
    except TypeError:
        ac = AgglomerativeClustering(n_clusters=k, metric='precomputed', linkage='average')
    labels = ac.fit_predict(D)
    return labels

def select_model_auto(X, kmin:int, kmax:int, bootstrap:int, seed:int=42, allow_ensemble:bool=True) -> ModelScores:
    Xe = _to_dense(X); n = Xe.shape[0]; grid = auto_k_grid(n, kmin, kmax)
    best: Optional[ModelScores] = None
    for k in grid:
        # KMeans
        labs_km = cluster_kmeans(Xe, k, random_state=seed)
        km_sil, km_db, km_ch = eval_internal(Xe, labs_km)
        km_ari, km_iqr = bootstrap_stability(Xe, lambda Xb, kk=k: cluster_kmeans(Xb, kk, random_state=seed), k, bootstrap, seed)
        cand = ModelScores(k, "kmeans", labs_km, km_sil, km_db, km_ch, km_ari, km_iqr, _cluster_sizes(labs_km))
        best = cand if best is None else (cand if (
            (km_ari, km_sil, km_ch, -km_db) > (best.ari_median, best.silhouette, best.ch, -best.db)
        ) else best)
        # Agglo
        labs_ag = cluster_agglo(Xe, k)
        ag_sil, ag_db, ag_ch = eval_internal(Xe, labs_ag)
        ag_ari, ag_iqr = bootstrap_stability(Xe, lambda Xb, kk=k: cluster_agglo(Xb, kk), k, bootstrap, seed)
        cand = ModelScores(k, "agglo", labs_ag, ag_sil, ag_db, ag_ch, ag_ari, ag_iqr, _cluster_sizes(labs_ag))
        best = cand if ( (ag_ari, ag_sil, ag_ch, -ag_db) > (best.ari_median, best.silhouette, best.ch, -best.db) ) else best
        # Ensemble
        if allow_ensemble:
            labs_cons = consensus_from_label_sets([labs_km, labs_ag])
            en_sil, en_db, en_ch = eval_internal(Xe, labs_cons)
            def clusterer_cons(Xb, kk=k):
                lk = cluster_kmeans(Xb, kk, random_state=seed)
                la = cluster_agglo(Xb, kk)
                return consensus_from_label_sets([lk, la])
            en_ari, en_iqr = bootstrap_stability(Xe, clusterer_cons, k, bootstrap, seed)
            cand = ModelScores(k, "ensemble", labs_cons, en_sil, en_db, en_ch, en_ari, en_iqr, _cluster_sizes(labs_cons))
            best = cand if ( (en_ari, en_sil, en_ch, -en_db) > (best.ari_median, best.silhouette, best.ch, -best.db) ) else best
    if best is None:
        labs = cluster_kmeans(Xe, 2, random_state=seed)
        sil, db, ch = eval_internal(Xe, labs)
        best = ModelScores(2, "kmeans", labs, sil, db, ch, float('nan'), (float('nan'), float('nan')), _cluster_sizes(labs))
    return best

def select_model(X, algo:"kmeans|agglo|ensemble|auto", kmin:int, kmax:int, bootstrap:int, seed:int=42) -> ModelScores:
    if algo == "auto":
        return select_model_auto(X, kmin, kmax, bootstrap, seed, allow_ensemble=True)
    Xe = _to_dense(X); n = Xe.shape[0]; grid = auto_k_grid(n, kmin, kmax)
    best: Optional[ModelScores] = None
    for k in grid:
        if algo == 'kmeans':
            labels = cluster_kmeans(Xe, k, random_state=seed)
            clusterer_fn = lambda Xb, kk=k: cluster_kmeans(Xb, kk, random_state=seed)
        elif algo == 'agglo':
            labels = cluster_agglo(Xe, k)
            clusterer_fn = lambda Xb, kk=k: cluster_agglo(Xb, kk)
        else:  # ensemble
            lk = cluster_kmeans(Xe, k, random_state=seed)
            la = cluster_agglo(Xe, k)
            labels = consensus_from_label_sets([lk, la])
            def clusterer_fn(Xb, kk=k):
                _lk = cluster_kmeans(Xb, kk, random_state=seed)
                _la = cluster_agglo(Xb, kk)
                return consensus_from_label_sets([_lk, _la])
        sizes = _cluster_sizes(labels)
        if min(sizes.values()) < max(3, int(0.01*n)):
            log.info(f"[select] k={k} rejected: tiny cluster"); continue
        sil, db, ch = eval_internal(Xe, labels, metric="cosine")
        ari_med, ari_iqr = bootstrap_stability(Xe, clusterer_fn, k, B=bootstrap, seed=seed)
        ms = ModelScores(k=k, algo=algo, labels=labels, silhouette=float(sil), db=float(db), ch=float(ch),
                         ari_median=float(ari_med), ari_iqr=ari_iqr, cluster_sizes=sizes)
        def rank_tuple(m:ModelScores):
            db_inv = -m.db if not math.isnan(m.db) else float('-inf')
            return ((m.ari_median if not math.isnan(m.ari_median) else -1.0),
                    (m.silhouette if not math.isnan(m.silhouette) else -1.0),
                    (m.ch if not math.isnan(m.ch) else -1.0),
                    db_inv)
        if (best is None) or (rank_tuple(ms) > rank_tuple(best)): best = ms
    if best is None:
        labels = cluster_kmeans(Xe, 2, random_state=seed)
        sizes = _cluster_sizes(labels)
        sil, db, ch = eval_internal(Xe, labels, metric="cosine")
        best = ModelScores(k=2, algo=algo, labels=labels, silhouette=float(sil), db=float(db), ch=float(ch),
                           ari_median=float('nan'), ari_iqr=(float('nan'), float('nan')), cluster_sizes=sizes)
    return best

# ========= Etiquetado (c-TF-IDF) =========
def label_topics_c_tfidf(texts: List[str], labels: np.ndarray, top_n=12, ngram=(2,3), min_df=2) -> Dict[int, List[str]]:
    df = pd.DataFrame({"text": texts, "label": labels})
    clusters = [c for c in sorted(df.label.unique()) if c != -1]
    docs = [" ".join(df[df.label==c].text.tolist()) for c in clusters]
    if not docs: return {}
    cv = CountVectorizer(ngram_range=ngram, min_df=min_df, token_pattern=r"[A-Za-z0-9_\-]+")
    X = cv.fit_transform(docs)
    if X.shape[1] < top_n:
        cv = CountVectorizer(ngram_range=(1,ngram[1]), min_df=min_df, token_pattern=r"[A-Za-z0-9_\-]+")
        X = cv.fit_transform(docs)
    tf = X.astype(float)
    row_sums = np.asarray(tf.sum(axis=1)).ravel() + 1e-12
    tf_norm = tf.multiply(1.0/row_sums.reshape(-1,1))
    df_counts = np.asarray((X > 0).sum(axis=0)).ravel()
    n_cls = X.shape[0]
    idf = np.log(1 + (n_cls / (df_counts + 1)))
    ctfidf = tf_norm.toarray() * idf
    terms = np.array(cv.get_feature_names_out())
    out: Dict[int, List[str]] = {}
    for i, c in enumerate(clusters):
        idx = np.argsort(-ctfidf[i])[:top_n]
        out[int(c)] = [terms[j].replace("_"," ") for j in idx]
    return out

# ========= Representantes, timeline =========
def representative_docs(X, df: pd.DataFrame, labels: np.ndarray, top_m=5) -> Dict[int, pd.DataFrame]:
    df2 = df.copy(); df2["cluster"] = labels
    Xe = _to_dense(X)
    out: Dict[int, pd.DataFrame] = {}
    for c in sorted(df2.cluster.unique()):
        if c == -1: continue
        rows = np.where(df2.cluster.values==c)[0]
        if rows.size == 0: continue
        centroid = Xe[rows].mean(axis=0, keepdims=True)
        sims = cosine_similarity(Xe[rows], centroid).ravel()
        order = np.argsort(-sims)[:top_m]
        rep = df2.iloc[rows[order]][["doi","title","year"]].copy()
        rep["rep_similarity"] = sims[order]
        out[int(c)] = rep
    return out

def cluster_timeline(df: pd.DataFrame, labels: np.ndarray) -> pd.DataFrame:
    d = df.copy(); d["cluster"] = labels
    d["year"] = pd.to_numeric(d["year"], errors="coerce").astype("Int64")
    return d[d["year"].notna()].groupby(["cluster","year"]).size().reset_index(name="count")

def _fit_linear(x, y):
    A = np.vstack([x, np.ones_like(x)]).T
    beta, intercept = np.linalg.lstsq(A, y, rcond=None)[0]
    yhat = beta*x + intercept
    ss_res = float(np.sum((y-yhat)**2))
    ss_tot = float(np.sum((y - np.mean(y))**2)) + 1e-12
    r2 = 1.0 - ss_res/ss_tot
    return beta, intercept, yhat, r2

def _fit_exponential(x, y):
    # y ≈ a * exp(bx) -> ln(y+1) ≈ ln(a) + b x
    y_log = np.log(y + 1.0)
    b, ln_a, yhat_log, r2 = _fit_linear(x, y_log)
    a = math.exp(ln_a)
    yhat = a * np.exp(b * x) - 1.0
    return a, b, yhat, r2

def _best_one_break(x, y):
    """Modelo piecewise lineal con 1 cambio de régimen. Devuelve idx*, RSS, parámetros."""
    n = len(x)
    if n < 5: return None
    best = None
    for br in range(2, n-2):
        beta1, c1, y1, _ = _fit_linear(x[:br], y[:br])
        beta2, c2, y2, _ = _fit_linear(x[br:], y[br:])
        yhat = np.concatenate([y1, y2])
        rss = float(np.sum((y - yhat)**2))
        if (best is None) or (rss < best[0]):
            best = (rss, br, (beta1, c1, beta2, c2))
    return best

def _autocorr(y, lag):
    if lag >= len(y): return 0.0
    y = np.asarray(y, dtype=float)
    y = y - y.mean()
    return float(np.dot(y[:-lag], y[lag:]) / (np.sqrt(np.dot(y[:-lag], y[:-lag])*np.dot(y[lag:], y[lag:])) + 1e-12))

def enhanced_timeline_trends(tl: pd.DataFrame) -> pd.DataFrame:
    """
    Para cada clúster calcula:
    - pendiente lineal + t-like
    - comparación lineal vs exponencial (R2)
    - 1 cambio de régimen (índice relativo) si mejora mucho el RSS
    - señal de estacionalidad (autocorr a lag=1..3)
    """
    rows = []
    for c in sorted(tl.cluster.unique()):
        sub = tl[tl.cluster==c].sort_values("year")
        x = sub["year"].astype(float).values
        y = sub["count"].astype(float).values
        if len(x) < 3 or np.std(x)==0:
            rows.append({"cluster": int(c), "slope": float('nan'), "t_like": float('nan'),
                         "model": "NA", "r2_linear": float('nan'), "r2_exp": float('nan'),
                         "break_at": None, "season_lag1": float('nan'), "season_lag2": float('nan'), "season_lag3": float('nan')})
            continue
        beta, intercept, yhat_lin, r2_lin = _fit_linear(x, y)
        resid = y - yhat_lin
        s2 = np.sum(resid**2) / max(1, len(x)-2); s = math.sqrt(max(0.0, s2))
        sxx = np.sum((x - x.mean())**2)
        t_like = beta * math.sqrt(sxx) / (s + 1e-9) if sxx>0 else float('nan')
        a, b, yhat_exp, r2_exp = _fit_exponential(x, y)
        model = "exponential" if (r2_exp > r2_lin + 0.1) else "linear"
        br = _best_one_break(x, y)
        break_at = None
        if br is not None:
            rss_base = float(np.sum((y - yhat_lin)**2))
            rss_br = br[0]
            if rss_br < 0.8 * rss_base:
                break_at = int(sub["year"].iloc[br[1]])
        rows.append({"cluster": int(c), "slope": float(beta), "t_like": float(t_like),
                     "model": model, "r2_linear": float(r2_lin), "r2_exp": float(r2_exp),
                     "break_at": break_at,
                     "season_lag1": _autocorr(y,1), "season_lag2": _autocorr(y,2), "season_lag3": _autocorr(y,3)})
    return pd.DataFrame(rows)

def centroid_cosine_matrix(X, labels: np.ndarray) -> pd.DataFrame:
    Xe = _to_dense(X); clus = sorted(np.unique(labels))
    cents = []
    for c in clus:
        rows = np.where(labels==c)[0]
        cents.append(Xe[rows].mean(axis=0, keepdims=True))
    C = np.vstack(cents); M = cosine_similarity(C)
    return pd.DataFrame(M, index=[f"C{c}" for c in clus], columns=[f"C{c}" for c in clus])

# ========= Diversidad + MMR/FPS + Cobertura adaptativa =========
def _cosine_dist(A, B=None):
    S = cosine_similarity(A, B) if B is not None else cosine_similarity(A)
    return 1.0 - S

def diversity_metrics(X):
    Xe = _to_dense(X); n = Xe.shape[0]
    if n <= 5000:
        D = _cosine_dist(Xe); iu = np.triu_indices(n, k=1); dvals = D[iu]
        diam = float(dvals.max()) if dvals.size else 0.0
        mean_d = float(dvals.mean()) if dvals.size else 0.0
        p90 = float(np.quantile(dvals, 0.90)) if dvals.size else 0.0
        p95 = float(np.quantile(dvals, 0.95)) if dvals.size else 0.0
    else:
        rng = np.random.default_rng(42); idx = rng.choice(n, size=min(5000, n), replace=False)
        D = _cosine_dist(Xe[idx]); iu = np.triu_indices(len(idx), k=1); dvals = D[iu]
        diam, mean_d = float(dvals.max()), float(dvals.mean()); p90 = float(np.quantile(dvals,0.90)); p95=float(np.quantile(dvals,0.95))
    Xc = Xe - Xe.mean(axis=0, keepdims=True)
    C = (Xc.T @ Xc) / max(1, n - 1)
    w = np.linalg.eigvalsh(C); w = np.clip(w, 0, None)
    tr = float(w.sum()) + 1e-12
    participation_ratio = float((tr**2) / (np.sum(w**2) + 1e-12))
    spectral_entropy = float(-np.sum((w/tr) * np.log((w/tr) + 1e-12)))
    return {"n": int(n), "diameter_cos": diam, "mean_pairwise_cos": mean_d,
            "p90_pairwise_cos": p90, "p95_pairwise_cos": p95,
            "participation_ratio": participation_ratio, "spectral_entropy": spectral_entropy}

def mmr_select(X, k=10, lam=0.5, query_vec=None):
    Xe = _to_dense(X); n = Xe.shape[0]
    if n == 0 or k <= 0: return np.array([], dtype=int)
    if query_vec is None: query_vec = Xe.mean(axis=0, keepdims=True)
    def _norm(A): nrm = np.linalg.norm(A, axis=1, keepdims=True) + 1e-12; return A / nrm
    Q = _norm(query_vec); Z = _norm(Xe)
    rel = cosine_similarity(Z, Q).ravel()
    selected = []; candidates = set(range(n))
    i0 = int(np.argmax(rel)); selected.append(i0); candidates.remove(i0)
    while len(selected) < min(k, n):
        sel_mat = Z[selected]
        max_sim_to_S = cosine_similarity(Z[list(candidates)], sel_mat).max(axis=1)
        cand_list = np.array(list(candidates))
        scores = lam * rel[cand_list] - (1.0 - lam) * max_sim_to_S
        pick = int(cand_list[np.argmax(scores)])
        selected.append(pick); candidates.remove(pick)
    return np.array(selected, dtype=int)

def farthest_point_sampling(X, k=10):
    Xe = _to_dense(X); n = Xe.shape[0]
    if n == 0 or k <= 0: return np.array([], dtype=int)
    rng = np.random.default_rng(42)
    first = int(rng.integers(0, n)); centers = [first]
    dmin = _cosine_dist(Xe, Xe[[first]]).ravel()
    for _ in range(1, min(k, n)):
        nxt = int(np.argmax(dmin)); centers.append(nxt)
        dmin = np.minimum(dmin, _cosine_dist(Xe, Xe[[nxt]]).ravel())
    return np.array(centers, dtype=int)

def adaptive_coverage_analysis(X, seeds_idx, percentiles=(0.10, 0.25, 0.50, 0.75)):
    """
    Calcula radios adaptativos por clúster en función de la densidad (percentiles de distancias NN).
    """
    Xe = _to_dense(X)
    if len(seeds_idx) == 0:
        return [{"radius_cos": float('nan'), "covered_frac": 0.0, "p": float(p)} for p in percentiles]
    S = Xe[seeds_idx]; D = _cosine_dist(Xe, S); d_nn = D.min(axis=1)
    cov = []
    base = np.quantile(d_nn, percentiles)
    for p, r in zip(percentiles, base):
        cov.append({"radius_cos": float(r), "covered_frac": float((d_nn <= r).mean()), "p": float(p)})
    return cov

# ========= Embeddings de términos (contextuales) =========
SENT_SPLIT = re.compile(r"(?<=[.!?])\s+")

def _embed_terms_raw(terms: List[str], analyzer: Analyzer) -> np.ndarray:
    """Fallback: embed de cadenas aisladas (no contextual)."""
    if analyzer.model is not None and hasattr(analyzer.model, "encode"):
        vecs = analyzer.model.encode(terms, show_progress_bar=False, convert_to_numpy=True, normalize_embeddings=True)
        return vecs
    tfv = TfidfVectorizer(analyzer='char', ngram_range=(3,5), min_df=1)
    X = tfv.fit_transform(terms).toarray().astype(float)
    X /= (np.linalg.norm(X, axis=1, keepdims=True) + 1e-12)
    return X

def contextual_term_embeddings(terms: List[str], documents: List[str], analyzer: Analyzer, max_sents_per_term:int=2000) -> np.ndarray:
    """
    Para cada término, promedia embeddings de oraciones donde aparece (si hay modelo transformer).
    Si no hay oraciones (o no hay modelo), cae al embedding aislado.
    """
    if analyzer.model is None or not hasattr(analyzer.model, "encode"):
        return _embed_terms_raw(terms, analyzer)
    vecs = []
    for t in terms:
        rx = re.compile(rf"\b{re.escape(t)}\b", flags=re.IGNORECASE)
        sents = []
        for doc in documents:
            for s in SENT_SPLIT.split(doc):
                if rx.search(s):
                    sents.append(s.strip())
                    if len(sents) >= max_sents_per_term: break
            if len(sents) >= max_sents_per_term: break
        if not sents:
            v = analyzer.model.encode([t], show_progress_bar=False, convert_to_numpy=True, normalize_embeddings=True)[0]
        else:
            E = analyzer.model.encode(sents, show_progress_bar=False, convert_to_numpy=True, normalize_embeddings=True)
            v = E.mean(axis=0)
            v /= (np.linalg.norm(v) + 1e-12)
        vecs.append(v)
    return np.vstack(vecs)

def term_distance_stats_by_cluster(topics: Dict[int, List[str]], analyzer: Analyzer, documents: Optional[List[str]]=None) -> pd.DataFrame:
    rows = []
    docs = documents if documents is not None else []
    for c, term_list in sorted(topics.items()):
        terms = [t.strip() for t in term_list if isinstance(t, str) and t.strip()]
        terms = list(dict.fromkeys(terms))
        if len(terms) < 2:
            rows.append({"cluster": int(c), "n_terms": len(terms), "pairs": 0,
                         "mean_sim": float('nan'), "mean_dist": float('nan'),
                         "p25_sim": float('nan'), "p50_sim": float('nan'), "p75_sim": float('nan'),
                         "min_sim": float('nan'), "max_sim": float('nan')})
            continue
        V = contextual_term_embeddings(terms, docs, analyzer) if docs else _embed_terms_raw(terms, analyzer)
        S = cosine_similarity(V)
        iu = np.triu_indices(len(terms), k=1)
        sims = S[iu]
        dists = 1.0 - sims
        rows.append({
            "cluster": int(c),
            "n_terms": len(terms),
            "pairs": int(len(sims)),
            "mean_sim": float(np.mean(sims)),
            "mean_dist": float(np.mean(dists)),
            "p25_sim": float(np.quantile(sims, 0.25)),
            "p50_sim": float(np.quantile(sims, 0.50)),
            "p75_sim": float(np.quantile(sims, 0.75)),
            "min_sim": float(np.min(sims)),
            "max_sim": float(np.max(sims)),
        })
    return pd.DataFrame(rows)

# ========= Exports & reports =========
def export_tables(name: str, df: pd.DataFrame, labels: np.ndarray, topics: Dict[int,List[str]], reps: Dict[int,pd.DataFrame], tl: pd.DataFrame, outdir: str):
    os.makedirs(outdir, exist_ok=True)
    short = {c: ", ".join(v[:4]) for c, v in topics.items()}
    df_map = df.copy(); df_map["cluster"] = labels; df_map["topic_label"] = df_map["cluster"].map(short)
    df_map[["doi","title","year","cluster","topic_label"]].to_csv(os.path.join(outdir,"paper_topics.csv"), index=False)
    rows = [{"cluster": c, "top_terms": "; ".join(topics.get(c, []))} for c in sorted(set(labels)) if c != -1]
    pd.DataFrame(rows).to_csv(os.path.join(outdir,"semantic_topics.csv"), index=False)
    tl.to_csv(os.path.join(outdir,"cluster_timeline.csv"), index=False)
    rep_tables = []
    for c, t in reps.items():
        t2 = t.copy(); t2.insert(0, "cluster", c); rep_tables.append(t2)
    if rep_tables:
        pd.concat(rep_tables, ignore_index=True).to_csv(os.path.join(outdir,"cluster_representatives.csv"), index=False)

def export_diversity(name: str, df: pd.DataFrame, X, labels: np.ndarray, outdir: str, mmr_k:int=10, lam:float=0.55):
    os.makedirs(outdir, exist_ok=True)
    Xe = _to_dense(X)
    all_rows, mmr_rows, fps_rows, cov_rows = [], [], [], []
    for c in sorted(np.unique(labels)):
        idx = np.where(labels==c)[0]
        if len(idx)==0: continue
        Xc = Xe[idx]; dmet = diversity_metrics(Xc); dmet["cluster"] = int(c); all_rows.append(dmet)
        q = Xc.mean(axis=0, keepdims=True)
        # MMR
        mmr_idx_local = mmr_select(Xc, k=min(mmr_k, len(idx)), lam=lam, query_vec=q)
        mmr_idx_global = idx[mmr_idx_local]
        for j, gi in enumerate(mmr_idx_global):
            r = df.iloc[gi]
            mmr_rows.append({"cluster": int(c), "rank": j+1, "global_idx": int(gi),
                             "doi": r.get("doi"), "title": r.get("title"), "year": r.get("year")})
        # FPS
        fps_idx_local = farthest_point_sampling(Xc, k=min(mmr_k, len(idx)))
        fps_idx_global = idx[fps_idx_local]
        for j, gi in enumerate(fps_idx_global):
            r = df.iloc[gi]
            fps_rows.append({"cluster": int(c), "rank": j+1, "global_idx": int(gi),
                             "doi": r.get("doi"), "title": r.get("title"), "year": r.get("year")})
        # Cobertura: fija + adaptativa
        for method, loc_idx in [("MMR", mmr_idx_local), ("FPS", fps_idx_local)]:
            if len(loc_idx):
                d_nn = _cosine_dist(Xc, Xc[loc_idx]).min(axis=1)
                for rad in [0.05, 0.10, 0.20, 0.30]:
                    cov_rows.append({"cluster": int(c), "method": method, "k": int(len(loc_idx)),
                                     "radius_cos": rad, "covered_frac": float((d_nn <= rad).mean())})
            for cc in adaptive_coverage_analysis(Xc, loc_idx, percentiles=(0.10,0.25,0.50,0.75)):
                cov_rows.append({"cluster": int(c), "method": method+"_adaptive", "k": int(len(loc_idx)), **cc})
    if all_rows: pd.DataFrame(all_rows).to_csv(os.path.join(outdir,"diversity_metrics.csv"), index=False)
    if mmr_rows: pd.DataFrame(mmr_rows).to_csv(os.path.join(outdir,"mmr_representatives.csv"), index=False)
    if fps_rows: pd.DataFrame(fps_rows).to_csv(os.path.join(outdir,"fps_representatives.csv"), index=False)
    if cov_rows: pd.DataFrame(cov_rows).to_csv(os.path.join(outdir,"coverage_curves.csv"), index=False)

def write_semantic_report(name: str, df: pd.DataFrame, labels: np.ndarray, topics: Dict[int,List[str]], reps: Dict[int,pd.DataFrame], outdir:str, method_str:str):
    path = os.path.join(outdir, "semantic_report.md")
    with open(path, "w", encoding="utf-8") as f:
        f.write("# Semantic Topic Report\n\n")
        f.write(f"Cluster set: {name}\n\n")
        f.write(f"Total papers: {len(df)}\n\n")
        f.write(f"**Clustering method:** {method_str}\n\n")
        for c in sorted(set(labels)):
            if c == -1: continue
            terms = ", ".join(topics.get(c, [])[:10])
            f.write(f"## Cluster {c} — {terms}\n\n")
            tbl = reps.get(c)
            if tbl is not None:
                for _, r in tbl.iterrows():
                    f.write(f"- **{r['title']}** ({r['year']}), DOI: {r['doi']} — rep_sim={r['rep_similarity']:.3f}\n")
            f.write("\n")

def write_validation_report(name:str, scores: ModelScores, linkM: pd.DataFrame, trends: pd.DataFrame,
                            outdir:str, modQ: Optional[float]=None, external: Optional[Dict[str,Any]]=None):
    path = os.path.join(outdir, "validation_report.md")
    with open(path, "w", encoding="utf-8") as f:
        f.write("# Validation & Diagnostics\n\n")
        f.write(f"Cluster set: {name}\n\n")
        f.write("## Internal metrics\n\n")
        f.write(f"- Algorithm: {scores.algo}\n")
        f.write(f"- k: {scores.k}\n")
        f.write(f"- Silhouette (cosine): {scores.silhouette:.3f}\n")
        f.write(f"- Davies–Bouldin: {scores.db:.3f}\n")
        f.write(f"- Calinski–Harabasz: {scores.ch:.1f}\n")
        f.write(f"- Bootstrap ARI (median): {scores.ari_median if not math.isnan(scores.ari_median) else 'NA'}\n")
        f.write(f"- Bootstrap ARI IQR: {scores.ari_iqr}\n")
        f.write(f"- Cluster sizes: {scores.cluster_sizes}\n\n")
        f.write("## Centroid cosine links\n\n")
        f.write(linkM.to_csv(index=True)); f.write("\n")
        f.write("## Timeline trends (model, slope, t-like)\n\n")
        f.write(trends.to_csv(index=False)); f.write("\n")
        if modQ is not None:
            f.write("## External (citation graph)\n\n")
            f.write(f"- Modularity Q: {modQ}\n")
        if external:
            f.write("## External validation suite\n\n")
            for k,v in external.items():
                f.write(f"- {k}: {v}\n")

def write_business_insights(name: str, insights_df: pd.DataFrame, outdir: str):
    insights_df.to_csv(os.path.join(outdir, "cluster_insights.csv"), index=False)
    path = os.path.join(outdir, "business_insights.md")
    with open(path, "w", encoding="utf-8") as f:
        f.write("# Business-Oriented Insights per Cluster\n\n")
        f.write(f"Cluster set: {name}\n\n")
        f.write(f"- ARI_global: {insights_df['ari_global'].iloc[0]:.3f} | baseline_slope_pos: {insights_df['explosive_baseline_slope'].iloc[0]:.4f}\n\n")
        for _, r in insights_df.sort_values("cluster").iterrows():
            f.write(f"## Cluster {int(r['cluster'])} — {r['insight_label']}\n\n")
            f.write(f"- n: {int(r['n'])}\n")
            f.write(f"- citas/paper: {r['citations_per_paper']:.2f}\n")
            f.write(f"- silhouette_mean: {r['silhouette_mean']:.3f}\n")
            f.write(f"- cohesion (centroid cosine): {r['centroid_cohesion']:.3f}\n")
            f.write(f"- slope: {r['slope']:.4f} | t-like: {r['t_like']:.2f}\n")
            f.write(f"- Rationale: {r['insight_reason']}\n\n")

def write_config(outdir:str, config:Dict[str,Any]):
    os.makedirs(outdir, exist_ok=True)
    with open(os.path.join(outdir, "config.json"), "w", encoding="utf-8") as f:
        json.dump(to_py(config), f, indent=2, ensure_ascii=False)

# ========= Insights por clúster =========
def per_cluster_metrics(X, df: pd.DataFrame, labels: np.ndarray, tl_trends: pd.DataFrame) -> pd.DataFrame:
    Xe = _to_dense(X); d = df.copy(); d["cluster"] = labels
    d["citedBy"] = pd.to_numeric(d["citedBy"], errors="coerce").fillna(0.0)
    try: s_samples = silhouette_samples(Xe, labels, metric="cosine")
    except Exception: s_samples = np.full(len(labels), np.nan)
    d["sil_sample"] = s_samples
    rows = []
    for c in sorted(d.cluster.unique()):
        sub = d[d.cluster==c]; idx = np.where(labels==c)[0]
        centroid = Xe[idx].mean(axis=0, keepdims=True)
        sims = cosine_similarity(Xe[idx], centroid).ravel()
        slope = float(tl_trends.loc[tl_trends.cluster==c, "slope"].values[0]) if ((tl_trends is not None) and (tl_trends.cluster==c).any()) else float('nan')
        t_like = float(tl_trends.loc[tl_trends.cluster==c, "t_like"].values[0]) if ((tl_trends is not None) and (tl_trends.cluster==c).any()) else float('nan')
        rows.append({"cluster": int(c), "n": int(len(sub)),
                     "citations_per_paper": float(sub["citedBy"].mean()) if len(sub) else float('nan'),
                     "silhouette_mean": float(sub["sil_sample"].mean()) if len(sub) else float('nan'),
                     "centroid_cohesion": float(np.mean(sims)) if len(sims) else float('nan'),
                     "slope": slope, "t_like": t_like})
    return pd.DataFrame(rows)

def classify_clusters(df_metrics: pd.DataFrame, ari_global: float) -> pd.DataFrame:
    pos_slopes = df_metrics["slope"].dropna()
    base_slope = float(pos_slopes[pos_slopes > 0].mean()) if (pos_slopes > 0).any() else float('nan')
    labels, reasons = [], []
    for _, r in df_metrics.iterrows():
        lbl = "—"; reason = []
        if (not math.isnan(ari_global)) and (ari_global >= 0.6) and (r["citations_per_paper"] < 3.0):
            lbl = "Nichos Académicos Maduros"
            reason = [f"ARI_global={ari_global:.2f} (≥0.6)", f"citas/paper={r['citations_per_paper']:.2f} (<3)"]
        if (not math.isnan(ari_global)) and (ari_global < 0.4) and (not math.isnan(base_slope)) and (r["slope"] > 2.0*base_slope):
            lbl = "Campos Emergentes Sin Consenso"
            reason = [f"slope={r['slope']:.3f} (>2× {base_slope:.3f})", f"ARI_global={ari_global:.2f} (<0.4)"]
        if lbl == "—":
            lbl = "Mixto / Indeterminado"; reason.append("Sin señales claras; revisar términos/topicos y t_like")
        labels.append(lbl); reasons.append("; ".join(reason))
    out = df_metrics.copy()
    out["insight_label"] = labels; out["insight_reason"] = reasons
    out["ari_global"] = ari_global; out["explosive_baseline_slope"] = base_slope
    return out

# ========= (Opcional) Validaciones externas desde Neo4j =========
def fetch_citation_edges(neo: Neo4jClient, dois: List[str]) -> Optional[pd.DataFrame]:
    dois = [d for d in dois if isinstance(d, str) and len(d)>0]
    if not dois: return None
    cy = """
        UNWIND $dois AS d
        MATCH (p:Publication {doi:d})-[:CITES]->(q:Publication)
        WHERE q.doi IS NOT NULL
        RETURN p.doi AS src, q.doi AS dst
    """
    try:
        with neo.driver.session(database=neo.db) as s:
            rows = [r.data() for r in s.run(cy, dois=dois)]
        df = pd.DataFrame(rows)
        return df if not df.empty else None
    except Exception as e:
        log.warning(f"Citation fetch failed: {e}"); return None

def fetch_coauthorship(neo: Neo4jClient, dois: List[str]) -> Optional[pd.DataFrame]:
    cy = """
        UNWIND $dois AS d
        MATCH (a:Author)-[:AUTHORED]->(p:Publication {doi:d})<-[:AUTHORED]-(b:Author)
        WHERE id(a) < id(b)
        RETURN a.name AS a, b.name AS b
    """
    try:
        with neo.driver.session(database=neo.db) as s:
            rows = [r.data() for r in s.run(cy, dois=dois)]
        df = pd.DataFrame(rows)
        return df if not df.empty else None
    except Exception as e:
        log.info(f"Coauthor fetch skipped/failed: {e}"); return None

def fetch_journal_pairs(neo: Neo4jClient, dois: List[str]) -> Optional[pd.DataFrame]:
    cy = """
        UNWIND $dois AS d
        MATCH (p:Publication {doi:d})-[:PUBLISHED_IN]->(j:Journal)
        RETURN d AS doi, j.name AS journal
    """
    try:
        with neo.driver.session(database=neo.db) as s:
            rows = [r.data() for r in s.run(cy, dois=dois)]
        return pd.DataFrame(rows)
    except Exception as e:
        log.info(f"Journal fetch skipped/failed: {e}"); return None

def multi_external_validation(neo: Neo4jClient, df: pd.DataFrame, labels: np.ndarray) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    labels_by_doi = {d:int(c) for d,c in zip(df['doi'].tolist(), labels)}
    # modularidad de citas
    edges = fetch_citation_edges(neo, df['doi'].tolist())
    modQ = modularity_on_clusters(edges, labels_by_doi)
    if modQ is not None: out["citation_modularity_Q"] = round(modQ, 4)
    # coautoría
    co = fetch_coauthorship(neo, df['doi'].tolist())
    if nx is not None and co is not None and not co.empty:
        G = nx.from_pandas_edgelist(co, 'a', 'b', create_using=nx.Graph())
        out["coauth_components"] = int(nx.number_connected_components(G)) if G.number_of_nodes() else 0
        out["coauth_avg_degree"] = float(np.mean([d for n,d in G.degree()])) if G.number_of_nodes() else float('nan')
    # journals
    jp = fetch_journal_pairs(neo, df['doi'].tolist())
    if jp is not None and not jp.empty:
        ent = []
        for c in np.unique(labels):
            sub = jp[jp['doi'].isin(df.loc[labels==c,'doi'])]
            if sub.empty: continue
            p = sub['journal'].value_counts(normalize=True).values + 1e-12
            ent.append(float(-(p*np.log(p)).sum()))
        out["journal_entropy_mean"] = float(np.mean(ent)) if ent else float('nan')
    return out

def modularity_on_clusters(edges: Optional[pd.DataFrame], labels_by_doi: Dict[str,int]) -> Optional[float]:
    if nx is None or edges is None or edges.empty: return None
    G = nx.from_pandas_edgelist(edges, 'src', 'dst', create_using=nx.Graph())
    if G.number_of_nodes() < 10: return None
    nodes = [n for n in G.nodes if n in labels_by_doi and labels_by_doi[n] != -1]
    if len(nodes) < 10: return None
    part: Dict[int, List[str]] = {}
    for n in nodes:
        c = labels_by_doi[n]; part.setdefault(c, []).append(n)
    try:
        import networkx.algorithms.community.quality as q
        comms = [set(v) for v in part.values() if len(v) > 0]
        return float(q.modularity(G.subgraph(nodes), comms))
    except Exception:
        return None

# ========= Sensibilidad cross-cutting =========
def run_crosscut_sensitivity_full(neo: Neo4jClient, backend: str, kmin:int, kmax:int, bootstrap:int, outroot:str, query_bias:str):
    results = {}
    for thr in [1,2,3,4,5]:
        name = f"cross_cutting_t{thr}"
        cy = cypher_for_cross_cutting(threshold=thr)
        df = neo.fetch(cy)
        if df.empty: continue
        q_terms = sorted({t for terms in BUSINESS_CLUSTER_DEFS.values() for t in terms})
        an = Analyzer(backend=backend); an.set_df(df)
        an.preprocess(filter_terms=q_terms, query_bias=query_bias); an.embed()
        scores = select_model(an.X, algo="auto", kmin=kmin, kmax=kmax, bootstrap=bootstrap)
        outdir = os.path.join(outroot, name); os.makedirs(outdir, exist_ok=True)
        topics = label_topics_c_tfidf(an.proc or [], scores.labels, top_n=12)
        reps = representative_docs(an.X, df, scores.labels, top_m=5)
        tl = cluster_timeline(df, scores.labels)
        export_tables(name, df, scores.labels, topics, reps, tl, outdir)
        linkM = centroid_cosine_matrix(an.X, scores.labels)
        trends = enhanced_timeline_trends(tl)
        write_validation_report(name, scores, linkM, trends, outdir)
        save_plot_2d(an.X, scores.labels, os.path.join(outdir,"clusters_pca.png"), title=f"{name} — PCA")
        results[thr] = set(df['doi'].dropna())
    if len(results) >= 2:
        sens_dir = os.path.join(outroot,"crosscut_sensitivity_full"); os.makedirs(sens_dir, exist_ok=True)
        thrs = sorted(results.keys())
        J = np.zeros((len(thrs), len(thrs)), dtype=float)
        for i,a in enumerate(thrs):
            for j,b in enumerate(thrs):
                A, B = results[a], results[b]
                inter = len(A & B); union = len(A | B) if (A or B) else 0
                J[i,j] = inter/union if union else float('nan')
        pd.DataFrame(J, index=[f"t={t}" for t in thrs], columns=[f"t={t}" for t in thrs]).to_csv(os.path.join(sens_dir,"jaccard_matrix.csv"))

# ========= Plot =========
def save_plot_2d(X, labels: np.ndarray, outpath: str, title: str = "PCA 2D"):
    try: P = PCA(n_components=2, random_state=42).fit_transform(_to_dense(X))
    except Exception as e: log.warning(f"PCA plot skipped: {e}"); return
    try:
        os.makedirs(os.path.dirname(outpath), exist_ok=True)
        plt.figure(figsize=(7,5))
        for c in sorted(np.unique(labels)):
            idx = np.where(labels==c)[0]
            if idx.size == 0: continue
            plt.scatter(P[idx,0], P[idx,1], s=12, alpha=0.7, label=f"C{c}")
        plt.title(title); plt.xlabel("PC1"); plt.ylabel("PC2")
        plt.legend(markerscale=1, fontsize=8)
        plt.tight_layout(); plt.savefig(outpath, dpi=160); plt.close()
    except Exception as e:
        log.warning(f"Plot save failed: {e}")

# ========= Runner =========
def run_by_cypher(which: List[str], backend: str, kmin: int, kmax: int, outroot: str,
                  bootstrap:int=20, compare_baselines:bool=False, crosscut_sens:bool=False,
                  crosscut_thr:int=2, query_bias:str="strip", graph_diagnostics: bool=False,
                  algo_choice:str="auto"):
    neo = Neo4jClient()
    try:
        if crosscut_sens:
            run_crosscut_sensitivity_full(neo, backend, kmin, kmax, bootstrap, outroot, query_bias)

        for name in which:
            if name == "cross_cutting":
                cypher = cypher_for_cross_cutting(threshold=crosscut_thr)
                query_terms = sorted({t for terms in BUSINESS_CLUSTER_DEFS.values() for t in terms})
            else:
                terms = BUSINESS_CLUSTER_DEFS.get(name)
                if not terms:
                    log.warning(f"Unknown cluster '{name}', skipping."); continue
                cypher = cypher_for_keyword_cluster(terms); query_terms = terms

            df = neo.fetch(cypher)
            if df.empty:
                log.warning(f"[{name}] no results"); continue
            log.info(f"[{name}] papers: {len(df)}")

            an = Analyzer(backend=backend); an.set_df(df)
            an.preprocess(filter_terms=query_terms, query_bias=query_bias)
            an.embed(); an.fit_knn(k=10)

            scores = select_model(an.X, algo=algo_choice, kmin=kmin, kmax=kmax, bootstrap=bootstrap)
            method_str = f"{scores.algo}_auto(k={scores.k}, sil={scores.silhouette:.3f}, DB={scores.db:.3f}, CH={scores.ch:.1f}, ARI_med={scores.ari_median})"
            log.info(f"[{name}] {method_str}")

            topics = label_topics_c_tfidf(an.proc or [], scores.labels, top_n=12)
            reps   = representative_docs(an.X, df, scores.labels, top_m=5)
            tl     = cluster_timeline(df, scores.labels)
            outdir = os.path.join(outroot, name)
            export_tables(name, df, scores.labels, topics, reps, tl, outdir)
            write_semantic_report(name, df, scores.labels, topics, reps, outdir, method_str)
            linkM  = centroid_cosine_matrix(an.X, scores.labels)
            trends = enhanced_timeline_trends(tl)

            # Insights
            clus_metrics = per_cluster_metrics(an.X, df, scores.labels, trends)
            insights_df  = classify_clusters(clus_metrics, ari_global=scores.ari_median)
            write_business_insights(name, insights_df, outdir)

            # Diversidad + MMR/FPS + cobertura
            export_diversity(name, df, an.X, scores.labels, outdir, mmr_k=10, lam=0.55)

            # Distancia entre términos de cada clúster (contextual si posible)
            term_stats = term_distance_stats_by_cluster(topics, an, documents=df['abstract'].tolist())
            if term_stats is not None and not term_stats.empty:
                term_stats.to_csv(os.path.join(outdir, "term_distance_stats.csv"), index=False)

            # Plot
            save_plot_2d(an.X, scores.labels, os.path.join(outdir,"clusters_pca.png"), title=f"{name} — PCA")

            # Baseline TF-IDF (opcional) — JSON seguro
            if compare_baselines:
                base_an = Analyzer(backend="tfidf"); base_an.set_df(df)
                base_an.preprocess(filter_terms=query_terms, query_bias=query_bias)
                base_an.embed()
                base_scores = select_model(base_an.X, algo="auto", kmin=kmin, kmax=kmax, bootstrap=max(5, bootstrap//2))
                base_dir = os.path.join(outdir, "baseline_tfidf"); os.makedirs(base_dir, exist_ok=True)
                with open(os.path.join(base_dir, "validation_summary.json"), "w", encoding="utf-8") as f:
                    json.dump(to_py({
                        "algo": base_scores.algo, "k": base_scores.k,
                        "silhouette": base_scores.silhouette, "db": base_scores.db, "ch": base_scores.ch,
                        "ari_median": base_scores.ari_median, "ari_iqr": base_scores.ari_iqr,
                        "cluster_sizes": _cluster_sizes(base_scores.labels)
                    }), f, indent=2, ensure_ascii=False)

            # (Opcional) Validación de grafo + suite externa
            modQ = None
            external = None
            if graph_diagnostics:
                edges = fetch_citation_edges(neo, df['doi'].tolist())
                labels_by_doi = {d:int(c) for d,c in zip(df['doi'].tolist(), scores.labels)}
                modQ = modularity_on_clusters(edges, labels_by_doi)
                external = multi_external_validation(neo, df, scores.labels)

            write_validation_report(name, scores, linkM, trends, outdir, modQ=modQ, external=external)

            # Guardar configuración (JSON-safe)
            write_config(outdir, {
                "cluster_name": name, "backend": backend, "algo": scores.algo, "k": scores.k,
                "k_grid": auto_k_grid(len(df), kmin, kmax), "bootstrap_runs": bootstrap,
                "query_bias": query_bias,
                "graph_diagnostics": graph_diagnostics,
                "env": {"NEO4J_URI": os.getenv("NEO4J_URI"), "NEO4J_DATABASE": os.getenv("NEO4J_DATABASE"),
                        "EMBEDDING_MODEL": os.getenv("EMBEDDING_MODEL","sentence-transformers/all-MiniLM-L6-v2")}
            })
    finally:
        neo.close()

# ========= CLI =========
def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--by-cypher", action="store_true", help="Run per business cluster via Cypher filters")
    p.add_argument("--clusters", type=str, default="", help="Comma list of cluster ids (...,cross_cutting)")
    p.add_argument("--all-clusters", action="store_true", help="Ejecuta todos los clústeres predefinidos + cross_cutting")
    p.add_argument("--backend", type=str, default="sbert", choices=["sbert","tfidf","specter2","chemberta"])
    p.add_argument("--kmin", type=int, default=2); p.add_argument("--kmax", type=int, default=12)
    p.add_argument("--outdir", type=str, default=os.getenv("DATA_DIR","./data_checkpoints_plus"))
    p.add_argument("--bootstrap", type=int, default=20, help="Bootstrap runs for ARI stability (default 20, 90% sample)")
    p.add_argument("--compare-baselines", action="store_true", help="Also run TF-IDF baseline")
    p.add_argument("--crosscut-sensitivity", action="store_true", help="Run extended sensitivity for cross_cutting (t=1..5)")
    p.add_argument("--crosscut-threshold", type=int, default=2, help="Threshold (>=t clusters) for cross_cutting corpus (used only when selecting cross_cutting directly)")
    p.add_argument("--query-bias", type=str, default="strip", choices=["strip","downweight","keep"], help="How to handle query terms before embedding")
    p.add_argument("--graph-diagnostics", action="store_true", help="Compute optional external validations (citations, coauth, journals)")
    p.add_argument("--algo", type=str, default="auto", choices=["auto","kmeans","agglo","ensemble"], help="Clustering strategy")
    return p.parse_args()

if __name__ == "__main__":
    args = parse_args()
    if not args.by_cypher:
        log.error("Use --by-cypher y --clusters para ejecutar por clúster."); raise SystemExit(1)

    if args.all_clusters:
        clusters = list(BUSINESS_CLUSTER_DEFS.keys()) + ["cross_cutting"]
    else:
        clusters = [c.strip() for c in args.clusters.split(",") if c.strip()]

    if not clusters:
        log.error("Proporciona --all-clusters o --clusters (p.ej., materials_polymers,environmental_assessment,...,cross_cutting)")
        raise SystemExit(1)

    os.makedirs(args.outdir, exist_ok=True)
    run_by_cypher(
        clusters, backend=args.backend, kmin=args.kmin, kmax=args.kmax,
        outroot=args.outdir, bootstrap=args.bootstrap,
        compare_baselines=args.compare_baselines,
        crosscut_sens=args.crosscut_sensitivity,
        crosscut_thr=args.crosscut_threshold,
        query_bias=args.query_bias,
        graph_diagnostics=args.graph_diagnostics,
        algo_choice=args.algo
    )
