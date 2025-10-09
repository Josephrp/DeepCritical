#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
complex_semantic.py — Abstracts-only semantic clustering + MMR/FPS + diversity + term-distance stats
- Dedupe robusto, filtrado anti-sesgo (default), SBERT/TF-IDF, métricas internas + bootstrap ARI.
- c-TF-IDF (términos por clúster), representantes, timelines, PCA 2D.
- Diversidad: diámetro coseno, percentiles, PR espectral, entropía + MMR/FPS + cobertura.
- Distancias entre términos (intra-clúster): embeddings de términos (backend) → sims/dists.
- (Opcional) Diagnóstico de grafo con --graph-diagnostics (modularidad citas, coautoría, etc).
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
    import networkx as nx  # solo si usas --graph-diagnostics
except Exception:
    nx = None

logging.basicConfig(level=logging.INFO, format='[%(levelname)s] %(message)s')
log = logging.getLogger("complex_semantic")
warnings.filterwarnings("ignore", category=FutureWarning)

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
    "social_perception": ["social attitude","public perception","user acceptance","consumer acceptance","social acceptance",
                          "public acceptance","behavioral change","environmental behavior","pro-environmental behavior",
                          "risk perception","health concern","consumer behavior","consumer attitude","willingness to pay","purchase intention"],
    "regulatory_economic": ["legislation","policy","regulation","recycling target","recycling rate","quality standard",
                            "economic viability","cost analysis","business model","supply chain","post-consumer","energy recovery"],
}

# ========= Neo4j client (para selección del corpus y, opcional, grafo) =========
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

# ========= Filtro anti-sesgo (elimina términos de la query del texto para embeddings) =========
def _normalize_term(term: str) -> List[str]:
    t = term.strip().lower()
    if not t: return []
    variants = {t, t.replace(" ","_"), re.sub(r"[^a-z0-9_ ]+","",t)}
    return sorted(v for v in variants if v)

def filter_query_terms(texts: List[str], query_terms: List[str]) -> List[str]:
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

    def preprocess(self, filter_terms: Optional[List[str]] = None):
        texts = (self.df["abstract"].fillna("").astype(str)).tolist()
        texts = [clean_abstract(t) for t in texts]
        if filter_terms:
            texts = filter_query_terms(texts, filter_terms)
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

def _choice_sorted(rng: np.random.Generator, n: int, frac: float = 0.8, min_n: int = 10) -> np.ndarray:
    m = max(min_n, int(frac*n)); m = min(m, n)
    return np.sort(rng.choice(n, size=m, replace=False))

def bootstrap_stability(X, clusterer_fn, k:int, B:int=20, seed:int=42) -> Tuple[float,Tuple[float,float]]:
    Xe = _to_dense(X); n = Xe.shape[0]
    if k < 2 or n < 10 or B <= 1:
        return float("nan"), (float("nan"), float("nan"))
    rng = np.random.default_rng(seed)
    labels_list = []
    for _ in range(B):
        idx = _choice_sorted(rng, n, frac=0.8, min_n=10)
        Xb = Xe[idx]; lb = clusterer_fn(Xb, k)
        labels_list.append((idx, lb))
    aris = []
    for i in range(len(labels_list)):
        idx_i, li = labels_list[i]
        for j in range(i+1, len(labels_list)):
            idx_j, lj = labels_list[j]
            inter, ia, ja = np.intersect1d(idx_i, idx_j, return_indices=True)
            if len(inter) < 10: continue
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

def select_model(X, algo:"kmeans|agglo", kmin:int, kmax:int, bootstrap:int, seed:int=42) -> ModelScores:
    Xe = _to_dense(X); n = Xe.shape[0]; grid = auto_k_grid(n, kmin, kmax)
    best: Optional[ModelScores] = None
    for k in grid:
        if algo == 'kmeans':
            labels = cluster_kmeans(Xe, k, random_state=seed)
            clusterer_fn = lambda Xb, kk=k: cluster_kmeans(Xb, kk, random_state=seed)
        else:
            labels = cluster_agglo(Xe, k)
            clusterer_fn = lambda Xb, kk=k: cluster_agglo(Xb, kk)
        sizes = {c:int((labels==c).sum()) for c in np.unique(labels)}
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
        labels = cluster_kmeans(Xe, 2, random_state=seed) if algo=='kmeans' else cluster_agglo(Xe, 2)
        sizes = {c:int((labels==c).sum()) for c in np.unique(labels)}
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

# ========= Representantes, timeline, links de centroides =========
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

def timeline_trends(tl: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for c in sorted(tl.cluster.unique()):
        sub = tl[tl.cluster==c].sort_values("year")
        x = sub["year"].astype(float).values; y = sub["count"].astype(float).values
        if len(x) >= 3 and np.std(x) > 0:
            A = np.vstack([x, np.ones_like(x)]).T
            beta, intercept = np.linalg.lstsq(A, y, rcond=None)[0]
            yhat = beta*x + intercept; resid = y - yhat
            if len(x) > 2:
                s2 = np.sum(resid**2) / (len(x)-2); s = math.sqrt(s2) if s2>0 else 0.0
                sxx = np.sum((x - x.mean())**2)
                t_like = beta * math.sqrt(sxx) / (s + 1e-9) if sxx>0 else float('nan')
            else:
                t_like = float('nan')
        else:
            beta, t_like = float('nan'), float('nan')
        rows.append({"cluster": int(c), "slope": float(beta), "t_like": float(t_like)})
    return pd.DataFrame(rows)

def centroid_cosine_matrix(X, labels: np.ndarray) -> pd.DataFrame:
    Xe = _to_dense(X); clus = sorted(np.unique(labels))
    cents = []
    for c in clus:
        rows = np.where(labels==c)[0]
        cents.append(Xe[rows].mean(axis=0, keepdims=True))
    C = np.vstack(cents); M = cosine_similarity(C)
    return pd.DataFrame(M, index=[f"C{c}" for c in clus], columns=[f"C{c}" for c in clus])

# ========= Diversidad + MMR/FPS + Cobertura =========
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

def coverage_curve(X, seeds_idx, radii=(0.05, 0.1, 0.2, 0.3)):
    Xe = _to_dense(X)
    if len(seeds_idx) == 0:
        return [{"radius_cos": float(r), "covered_frac": 0.0} for r in radii]
    S = Xe[seeds_idx]; D = _cosine_dist(Xe, S); d_nn = D.min(axis=1)
    return [{"radius_cos": float(r), "covered_frac": float((d_nn <= r).mean())} for r in radii]

# ========= Distancias entre términos (intra-clúster) =========
def _embed_terms(terms: List[str], analyzer: Analyzer) -> np.ndarray:
    """Embebe términos con el backend disponible (preferible SentenceTransformer)."""
    if analyzer.model is not None and hasattr(analyzer.model, "encode"):
        vecs = analyzer.model.encode(terms, show_progress_bar=False, convert_to_numpy=True, normalize_embeddings=True)
        return vecs
    # Fallback: TF-IDF sobre los propios términos (caracter n-gram puede ser más robusto)
    tfv = TfidfVectorizer(analyzer='char', ngram_range=(3,5), min_df=1)
    X = tfv.fit_transform(terms).toarray().astype(float)
    # normalizar
    X /= (np.linalg.norm(X, axis=1, keepdims=True) + 1e-12)
    return X

def term_distance_stats_by_cluster(topics: Dict[int, List[str]], analyzer: Analyzer) -> pd.DataFrame:
    """Para cada clúster, calcula métricas de distancia entre los términos top_n."""
    rows = []
    for c, term_list in sorted(topics.items()):
        terms = [t.strip() for t in term_list if isinstance(t, str) and t.strip()]
        terms = list(dict.fromkeys(terms))  # unique, preserva orden
        if len(terms) < 2:
            rows.append({"cluster": int(c), "n_terms": len(terms), "pairs": 0,
                         "mean_sim": float('nan'), "mean_dist": float('nan'),
                         "p25_sim": float('nan'), "p50_sim": float('nan'), "p75_sim": float('nan'),
                         "min_sim": float('nan'), "max_sim": float('nan')})
            continue
        V = _embed_terms(terms, analyzer)
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
        # MMR
        q = Xc.mean(axis=0, keepdims=True)
        mmr_idx_local = mmr_select(Xc, k=min(mmr_k, len(idx)), lam=lam, query_vec=q)
        mmr_idx_global = idx[mmr_idx_local]
        cov = coverage_curve(Xc, mmr_idx_local, radii=(0.05,0.1,0.2,0.3))
        for cc in cov: cov_rows.append({"cluster": int(c), "method":"MMR", "k": len(mmr_idx_local), **cc})
        for j, gi in enumerate(mmr_idx_global):
            r = df.iloc[gi]
            mmr_rows.append({"cluster": int(c), "rank": j+1, "global_idx": int(gi),
                             "doi": r.get("doi"), "title": r.get("title"), "year": r.get("year")})
        # FPS
        fps_idx_local = farthest_point_sampling(Xc, k=min(mmr_k, len(idx)))
        fps_idx_global = idx[fps_idx_local]
        cov = coverage_curve(Xc, fps_idx_local, radii=(0.05,0.1,0.2,0.3))
        for cc in cov: cov_rows.append({"cluster": int(c), "method":"FPS", "k": len(fps_idx_local), **cc})
        for j, gi in enumerate(fps_idx_global):
            r = df.iloc[gi]
            fps_rows.append({"cluster": int(c), "rank": j+1, "global_idx": int(gi),
                             "doi": r.get("doi"), "title": r.get("title"), "year": r.get("year")})
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
                            outdir:str, modQ: Optional[float]=None):
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
        f.write("## Timeline trends (slope, t-like)\n\n")
        f.write(trends.to_csv(index=False)); f.write("\n")
        if modQ is not None:
            f.write("## External (citation graph)\n\n")
            f.write(f"- Modularity Q: {modQ}\n")

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
        json.dump(config, f, indent=2, ensure_ascii=False)

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

# ========= (Opcional) Grafo de citas (solo si --graph-diagnostics) =========
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
def run_crosscut_sensitivity(neo: Neo4jClient, backend: str, kmin:int, kmax:int, bootstrap:int, outroot:str, keep_query_terms: bool):
    results = []
    for thr in [2,3]:
        name = f"cross_cutting_t{thr}"
        cy = cypher_for_cross_cutting(threshold=thr)
        df = neo.fetch(cy)
        if df.empty: continue
        q_terms = sorted({t for terms in BUSINESS_CLUSTER_DEFS.values() for t in terms})
        an = Analyzer(backend=backend); an.set_df(df)
        an.preprocess(filter_terms=None if keep_query_terms else q_terms); an.embed()
        algo = 'agglo' if len(df) < 100 else 'kmeans'
        scores = select_model(an.X, algo=algo, kmin=kmin, kmax=kmax, bootstrap=bootstrap)
        outdir = os.path.join(outroot, name)
        topics = label_topics_c_tfidf(an.proc or [], scores.labels, top_n=12)
        reps = representative_docs(an.X, df, scores.labels, top_m=5)
        tl = cluster_timeline(df, scores.labels)
        export_tables(name, df, scores.labels, topics, reps, tl, outdir)
        linkM = centroid_cosine_matrix(an.X, scores.labels)
        trends = timeline_trends(tl)
        write_validation_report(name, scores, linkM, trends, outdir)
        save_plot_2d(an.X, scores.labels, os.path.join(outdir,"clusters_pca.png"), title=f"{name} — PCA")
        results.append({"thr":thr, "n":int(len(df)), "k":scores.k, "sil":scores.silhouette, "db":scores.db, "ch":scores.ch})
    if len(results) >= 2:
        cy2 = cypher_for_cross_cutting(threshold=2); cy3 = cypher_for_cross_cutting(threshold=3)
        df2 = neo.fetch(cy2); df3 = neo.fetch(cy3)
        s2, s3 = set(df2['doi'].dropna()), set(df3['doi'].dropna())
        inter = len(s2 & s3); union = len(s2 | s3)
        j = inter/union if union else float('nan')
        sens_dir = os.path.join(outroot,"crosscut_sensitivity"); os.makedirs(sens_dir, exist_ok=True)
        with open(os.path.join(sens_dir,"sensitivity_report.md"),"w",encoding="utf-8") as f:
            f.write("# Cross-cutting Sensitivity\n\n")
            f.write(f"Jaccard(t=2 vs t=3): {j:.3f}\n\n")
            f.write("Threshold summary (n, k, silhouette, DB, CH):\n\n")
            for r in results: f.write(str(r)+"\n")

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
                  bootstrap:int=10, compare_baselines:bool=False, crosscut_sens:bool=False,
                  crosscut_thr:int=2, keep_query_terms: bool=False, graph_diagnostics: bool=False):
    neo = Neo4jClient()
    try:
        if crosscut_sens:
            run_crosscut_sensitivity(neo, backend, kmin, kmax, bootstrap, outroot, keep_query_terms)

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
            an.preprocess(filter_terms=None if keep_query_terms else query_terms)
            an.embed(); an.fit_knn(k=10)

            algo = 'agglo' if len(df) < 100 else 'kmeans'
            scores = select_model(an.X, algo=algo, kmin=kmin, kmax=kmax, bootstrap=bootstrap)
            method_str = f"{scores.algo}_auto(k={scores.k}, sil={scores.silhouette:.3f}, DB={scores.db:.3f}, CH={scores.ch:.1f}, ARI_med={scores.ari_median})"
            log.info(f"[{name}] {method_str}")

            topics = label_topics_c_tfidf(an.proc or [], scores.labels, top_n=12)
            reps   = representative_docs(an.X, df, scores.labels, top_m=5)
            tl     = cluster_timeline(df, scores.labels)
            outdir = os.path.join(outroot, name)
            export_tables(name, df, scores.labels, topics, reps, tl, outdir)
            write_semantic_report(name, df, scores.labels, topics, reps, outdir, method_str)
            linkM  = centroid_cosine_matrix(an.X, scores.labels)
            trends = timeline_trends(tl)

            # Insights
            clus_metrics = per_cluster_metrics(an.X, df, scores.labels, trends)
            insights_df  = classify_clusters(clus_metrics, ari_global=scores.ari_median)
            write_business_insights(name, insights_df, outdir)

            # Diversidad + MMR/FPS + cobertura
            export_diversity(name, df, an.X, scores.labels, outdir, mmr_k=10, lam=0.55)

            # Distancia entre términos de cada clúster
            term_stats = term_distance_stats_by_cluster(topics, an)
            if term_stats is not None and not term_stats.empty:
                term_stats.to_csv(os.path.join(outdir, "term_distance_stats.csv"), index=False)

            # Plot
            save_plot_2d(an.X, scores.labels, os.path.join(outdir,"clusters_pca.png"), title=f"{name} — PCA")

            # Baseline TF-IDF (opcional)
            if compare_baselines:
                base_an = Analyzer(backend="tfidf"); base_an.set_df(df)
                base_an.preprocess(filter_terms=None if keep_query_terms else query_terms)
                base_an.embed()
                base_algo = 'agglo' if len(df) < 100 else 'kmeans'
                base_scores = select_model(base_an.X, algo=base_algo, kmin=kmin, kmax=kmax, bootstrap=max(5, bootstrap//2))
                base_dir = os.path.join(outdir, "baseline_tfidf"); os.makedirs(base_dir, exist_ok=True)
                with open(os.path.join(base_dir, "validation_summary.json"), "w", encoding="utf-8") as f:
                    json.dump({
                        "algo": base_scores.algo, "k": base_scores.k,
                        "silhouette": base_scores.silhouette, "db": base_scores.db, "ch": base_scores.ch,
                        "ari_median": base_scores.ari_median, "ari_iqr": base_scores.ari_iqr,
                        "cluster_sizes": base_scores.cluster_sizes
                    }, f, indent=2)

            # (Opcional) Diagnóstico de grafo
            modQ = None
            if graph_diagnostics:
                edges = fetch_citation_edges(neo, df['doi'].tolist())
                labels_by_doi = {d:int(c) for d,c in zip(df['doi'].tolist(), scores.labels)}
                modQ = modularity_on_clusters(edges, labels_by_doi)

            write_validation_report(name, scores, linkM, trends, outdir, modQ=modQ)

            # Guardar configuración
            write_config(outdir, {
                "cluster_name": name, "backend": backend, "algo": scores.algo, "k": scores.k,
                "k_grid": auto_k_grid(len(df), kmin, kmax), "bootstrap_runs": bootstrap,
                "keep_query_terms": keep_query_terms,
                "filtered_terms_count": 0 if keep_query_terms else len(query_terms),
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
    p.add_argument("--backend", type=str, default="sbert", choices=["sbert","tfidf","specter2","chemberta"])
    p.add_argument("--kmin", type=int, default=2); p.add_argument("--kmax", type=int, default=12)
    p.add_argument("--outdir", type=str, default=os.getenv("DATA_DIR","./data_checkpoints_plus"))
    p.add_argument("--bootstrap", type=int, default=10, help="Bootstrap runs for ARI stability")
    p.add_argument("--compare-baselines", action="store_true", help="Also run TF-IDF baseline")
    p.add_argument("--crosscut-sensitivity", action="store_true", help="Run sensitivity for cross_cutting threshold")
    p.add_argument("--crosscut-threshold", type=int, default=2, help="Threshold (>=t clusters) for cross_cutting corpus")
    p.add_argument("--keep-query-terms", action="store_true", help="Do NOT strip query terms before embedding")
    p.add_argument("--graph-diagnostics", action="store_true", help="Compute optional citation-graph diagnostics")
    return p.parse_args()

if __name__ == "__main__":
    args = parse_args()
    if not args.by_cypher:
        log.error("Use --by-cypher y --clusters para ejecutar por clúster."); raise SystemExit(1)
    clusters = [c.strip() for c in args.clusters.split(",") if c.strip()]
    if not clusters:
        log.error("Proporciona --clusters (p.ej., materials_polymers,environmental_assessment,...,cross_cutting)")
        raise SystemExit(1)
    os.makedirs(args.outdir, exist_ok=True)
    run_by_cypher(
        clusters, backend=args.backend, kmin=args.kmin, kmax=args.kmax,
        outroot=args.outdir, bootstrap=args.bootstrap,
        compare_baselines=args.compare_baselines,
        crosscut_sens=args.crosscut_sensitivity,
        crosscut_thr=args.crosscut_threshold,
        keep_query_terms=args.keep_query_terms,
        graph_diagnostics=args.graph_diagnostics,
    )
