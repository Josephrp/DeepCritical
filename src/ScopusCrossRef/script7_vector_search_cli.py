"""
Script 7: Vector search system for publications
- Simple vector search (semantic similarity)
- Hybrid search (vector + citation graph)
- Search by author
- Advanced filters (year, citations, journal)
"""

import os
import logging
from typing import List, Dict, Any, Optional
from neo4j import GraphDatabase
from config_manager import get_config

# Configuración
config = get_config()

# Logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("vector_search.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class VectorSearchSystem:
    def __init__(self):
        self.driver = None
        self.model = None
        
    def connect(self):
        """Conectar a Neo4j"""
        try:
            self.driver = GraphDatabase.driver(
                config.neo4j_uri,
                auth=(config.neo4j_user, config.neo4j_password)
            )
            with self.driver.session(database=config.neo4j_database) as session:
                session.run("RETURN 1")
            logger.info(f"Connected to Neo4j: {config.neo4j_database}")
            return True
        except Exception as e:
            logger.error(f"Connection error: {e}")
            return False
    
    def load_model(self):
        """Cargar modelo de embeddings"""
        try:
            from sentence_transformers import SentenceTransformer
            model_name = os.getenv("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
            self.model = SentenceTransformer(model_name)
            logger.info(f"Model loaded: {model_name}")
            return True
        except Exception as e:
            logger.error(f"Error loading model: {e}")
            return False
    
    def generate_query_embedding(self, query_text: str) -> List[float]:
        """Generar embedding para query"""
        try:
            embedding = self.model.encode(query_text, convert_to_numpy=True)
            return embedding.tolist()
        except Exception as e:
            logger.error(f"Error generating embedding: {e}")
            return []
    
    def vector_search(
        self, 
        query: str, 
        top_k: int = 10,
        min_year: Optional[int] = None,
        max_year: Optional[int] = None,
        min_citations: Optional[int] = None,
        search_in: str = "abstract"  # "abstract", "title", or "both"
    ) -> List[Dict[str, Any]]:
        """
        Búsqueda vectorial simple
        
        Args:
            query: Texto de búsqueda
            top_k: Número de resultados
            min_year: Año mínimo (opcional)
            max_year: Año máximo (opcional)
            min_citations: Citaciones mínimas (opcional)
            search_in: Dónde buscar ("abstract", "title", "both")
        """
        
        logger.info(f"Vector search: '{query[:50]}...' (top_k={top_k}, search_in={search_in})")
        
        # Generar embedding de la query
        query_embedding = self.generate_query_embedding(query)
        
        if not query_embedding:
            return []
        
        # Construir query Cypher
        with self.driver.session(database=config.neo4j_database) as session:
            
            # Índice a usar
            if search_in == "abstract":
                index_name = "publication_abstract_embeddings"
                embedding_field = "abstract_embedding"
            elif search_in == "title":
                index_name = "publication_title_embeddings"
                embedding_field = "title_embedding"
            else:  # both - buscar en abstract por defecto
                index_name = "publication_abstract_embeddings"
                embedding_field = "abstract_embedding"
            
            # Query base con vector search
            cypher_query = f"""
                CALL db.index.vector.queryNodes($index_name, $top_k, $query_vector)
                YIELD node AS p, score
                WHERE p.{embedding_field} IS NOT NULL
            """
            
            # Agregar filtros
            filters = []
            params = {
                "index_name": index_name,
                "top_k": top_k * 2,  # Pedir más para compensar filtros
                "query_vector": query_embedding
            }
            
            if min_year:
                filters.append("toInteger(p.year) >= $min_year")
                params["min_year"] = min_year
            
            if max_year:
                filters.append("toInteger(p.year) <= $max_year")
                params["max_year"] = max_year
            
            if min_citations:
                filters.append("toInteger(p.citedBy) >= $min_citations")
                params["min_citations"] = min_citations
            
            if filters:
                cypher_query += " AND " + " AND ".join(filters)
            
            # Obtener información adicional
            cypher_query += """
                OPTIONAL MATCH (p)-[:PUBLISHED_IN]->(j:Journal)
                OPTIONAL MATCH (a:Author)-[:AUTHORED]->(p)
                WITH p, score, j, collect(DISTINCT a.name)[0..5] AS authors
                RETURN p.eid AS eid,
                       p.doi AS doi,
                       p.title AS title,
                       p.abstract AS abstract,
                       p.year AS year,
                       p.citedBy AS citations,
                       j.name AS journal,
                       authors,
                       score
                ORDER BY score DESC
                LIMIT $top_k_final
            """
            
            params["top_k_final"] = top_k
            
            try:
                result = session.run(cypher_query, params)
                results = [dict(record) for record in result]
                logger.info(f"Found {len(results)} results")
                return results
            except Exception as e:
                logger.error(f"Search error: {e}")
                return []
    
    def hybrid_search(
        self,
        query: str,
        top_k: int = 10,
        citation_weight: float = 0.3,
        min_citations: int = 5
    ) -> List[Dict[str, Any]]:
        """
        Búsqueda híbrida: similitud vectorial + importancia en grafo de citaciones
        
        Args:
            query: Texto de búsqueda
            top_k: Número de resultados
            citation_weight: Peso del PageRank (0-1)
            min_citations: Citaciones mínimas
        """
        
        logger.info(f"Hybrid search: '{query[:50]}...' (citation_weight={citation_weight})")
        
        query_embedding = self.generate_query_embedding(query)
        
        if not query_embedding:
            return []
        
        with self.driver.session(database=config.neo4j_database) as session:
            cypher_query = """
                CALL db.index.vector.queryNodes('publication_abstract_embeddings', $top_k_multiplier, $query_vector)
                YIELD node AS p, score AS vector_score
                WHERE p.abstract_embedding IS NOT NULL
                  AND toInteger(p.citedBy) >= $min_citations
                
                // Calcular score de citaciones normalizado
                MATCH (p2:Publication)
                WITH p, vector_score, 
                     toFloat(p.citedBy) AS citations,
                     max(toFloat(p2.citedBy)) AS max_citations
                
                WITH p, vector_score, citations,
                     CASE WHEN max_citations > 0 
                          THEN citations / max_citations 
                          ELSE 0 END AS citation_score
                
                // Score híbrido
                WITH p, vector_score, citation_score,
                     ((1 - $citation_weight) * vector_score + $citation_weight * citation_score) AS hybrid_score
                
                // Información adicional
                OPTIONAL MATCH (p)-[:PUBLISHED_IN]->(j:Journal)
                OPTIONAL MATCH (a:Author)-[:AUTHORED]->(p)
                
                WITH p, vector_score, citation_score, hybrid_score, j,
                     collect(DISTINCT a.name)[0..5] AS authors
                
                RETURN p.eid AS eid,
                       p.doi AS doi,
                       p.title AS title,
                       p.abstract AS abstract,
                       p.year AS year,
                       p.citedBy AS citations,
                       j.name AS journal,
                       authors,
                       round(vector_score * 100) / 100 AS vector_score,
                       round(citation_score * 100) / 100 AS citation_score,
                       round(hybrid_score * 100) / 100 AS hybrid_score
                ORDER BY hybrid_score DESC
                LIMIT $top_k
            """
            
            try:
                result = session.run(
                    cypher_query,
                    query_vector=query_embedding,
                    top_k=top_k,
                    top_k_multiplier=top_k * 3,
                    citation_weight=citation_weight,
                    min_citations=min_citations
                )
                results = [dict(record) for record in result]
                logger.info(f"Found {len(results)} hybrid results")
                return results
            except Exception as e:
                logger.error(f"Hybrid search error: {e}")
                return []
    
    def search_by_author(
        self,
        author_name: str,
        query: Optional[str] = None,
        top_k: int = 10
    ) -> List[Dict[str, Any]]:
        """
        Buscar publicaciones de un autor específico
        
        Args:
            author_name: Nombre del autor (búsqueda parcial)
            query: Query opcional para filtrar por similitud
            top_k: Número de resultados
        """
        
        logger.info(f"Author search: '{author_name}' (query={query is not None})")
        
        with self.driver.session(database=config.neo4j_database) as session:
            
            if query:
                # Búsqueda vectorial dentro de publicaciones del autor
                query_embedding = self.generate_query_embedding(query)
                
                cypher_query = """
                    MATCH (a:Author)-[:AUTHORED]->(p:Publication)
                    WHERE toLower(a.name) CONTAINS toLower($author_name)
                      AND p.abstract_embedding IS NOT NULL
                    
                    WITH p, a, 
                         reduce(s = 0.0, i IN range(0, size(p.abstract_embedding)-1) | 
                                s + p.abstract_embedding[i] * $query_vector[i]) AS similarity
                    
                    OPTIONAL MATCH (p)-[:PUBLISHED_IN]->(j:Journal)
                    
                    RETURN p.eid AS eid,
                           p.doi AS doi,
                           p.title AS title,
                           p.abstract AS abstract,
                           p.year AS year,
                           p.citedBy AS citations,
                           j.name AS journal,
                           collect(DISTINCT a.name)[0..5] AS authors,
                           round(similarity * 100) / 100 AS similarity_score
                    ORDER BY similarity DESC
                    LIMIT $top_k
                """
                
                result = session.run(
                    cypher_query,
                    author_name=author_name,
                    query_vector=query_embedding,
                    top_k=top_k
                )
            else:
                # Búsqueda simple por autor
                cypher_query = """
                    MATCH (a:Author)-[:AUTHORED]->(p:Publication)
                    WHERE toLower(a.name) CONTAINS toLower($author_name)
                    
                    OPTIONAL MATCH (p)-[:PUBLISHED_IN]->(j:Journal)
                    OPTIONAL MATCH (a2:Author)-[:AUTHORED]->(p)
                    
                    WITH p, j, collect(DISTINCT a2.name)[0..5] AS authors
                    
                    RETURN p.eid AS eid,
                           p.doi AS doi,
                           p.title AS title,
                           p.abstract AS abstract,
                           p.year AS year,
                           p.citedBy AS citations,
                           j.name AS journal,
                           authors
                    ORDER BY p.citedBy DESC
                    LIMIT $top_k
                """
                
                result = session.run(
                    cypher_query,
                    author_name=author_name,
                    top_k=top_k
                )
            
            results = [dict(record) for record in result]
            logger.info(f"Found {len(results)} publications by author")
            return results
    
    def find_similar_papers(
        self,
        doi: str,
        top_k: int = 10
    ) -> List[Dict[str, Any]]:
        """
        Encontrar papers similares a uno dado
        
        Args:
            doi: DOI del paper de referencia
            top_k: Número de resultados
        """
        
        logger.info(f"Finding similar papers to: {doi}")
        
        with self.driver.session(database=config.neo4j_database) as session:
            cypher_query = """
                MATCH (ref:Publication {doi: $doi})
                WHERE ref.abstract_embedding IS NOT NULL
                
                CALL db.index.vector.queryNodes('publication_abstract_embeddings', $top_k_plus, ref.abstract_embedding)
                YIELD node AS p, score
                WHERE p.doi <> $doi
                
                OPTIONAL MATCH (p)-[:PUBLISHED_IN]->(j:Journal)
                OPTIONAL MATCH (a:Author)-[:AUTHORED]->(p)
                
                RETURN p.eid AS eid,
                       p.doi AS doi,
                       p.title AS title,
                       p.abstract AS abstract,
                       p.year AS year,
                       p.citedBy AS citations,
                       j.name AS journal,
                       collect(DISTINCT a.name)[0..5] AS authors,
                       score
                ORDER BY score DESC
                LIMIT $top_k
            """
            
            try:
                result = session.run(
                    cypher_query,
                    doi=doi,
                    top_k=top_k,
                    top_k_plus=top_k + 1
                )
                results = [dict(record) for record in result]
                logger.info(f"Found {len(results)} similar papers")
                return results
            except Exception as e:
                logger.error(f"Error finding similar papers: {e}")
                return []
    
    def close(self):
        """Cerrar conexión"""
        if self.driver:
            self.driver.close()
            logger.info("Connection closed")

def format_result(result: Dict[str, Any], index: int) -> str:
    """Formatear un resultado para display"""
    output = f"\n{'='*80}\n"
    output += f"[{index}] {result.get('title', 'No title')}\n"
    output += f"{'='*80}\n"
    
    authors = result.get('authors', [])
    if authors:
        output += f"Authors: {', '.join(authors[:3])}"
        if len(authors) > 3:
            output += f" et al. ({len(authors)} total)"
        output += "\n"
    
    output += f"Year: {result.get('year', 'N/A')} | "
    output += f"Citations: {result.get('citations', 0)} | "
    
    if result.get('journal'):
        output += f"Journal: {result['journal']}\n"
    
    if result.get('doi'):
        output += f"DOI: {result['doi']}\n"
    
    # Scores si existen
    if 'score' in result:
        output += f"Similarity: {result['score']:.3f}\n"
    if 'vector_score' in result:
        output += f"Vector: {result['vector_score']:.3f} | Citation: {result['citation_score']:.3f} | Hybrid: {result['hybrid_score']:.3f}\n"
    
    abstract = result.get('abstract', '')
    if abstract:
        preview = abstract[:300] + "..." if len(abstract) > 300 else abstract
        output += f"\nAbstract: {preview}\n"
    
    return output

def interactive_mode():
    """Modo interactivo"""
    print("\n" + "="*80)
    print("VECTOR SEARCH SYSTEM - Interactive Mode")
    print("="*80)
    
    search_system = VectorSearchSystem()
    
    if not search_system.connect():
        print("Failed to connect to Neo4j")
        return
    
    if not search_system.load_model():
        print("Failed to load embedding model")
        return
    
    print("\nCommands:")
    print("  1. search <query>          - Vector search")
    print("  2. hybrid <query>          - Hybrid search (vector + citations)")
    print("  3. author <name>           - Search by author")
    print("  4. similar <doi>           - Find similar papers")
    print("  5. quit                    - Exit")
    
    try:
        while True:
            print("\n" + "-"*80)
            command = input("\nEnter command: ").strip()
            
            if not command:
                continue
            
            if command.lower() in ['quit', 'exit', 'q']:
                break
            
            parts = command.split(maxsplit=1)
            cmd = parts[0].lower()
            
            if cmd == 'search' and len(parts) > 1:
                query = parts[1]
                results = search_system.vector_search(query, top_k=5)
                
                if results:
                    print(f"\nFound {len(results)} results:")
                    for i, result in enumerate(results, 1):
                        print(format_result(result, i))
                else:
                    print("No results found")
            
            elif cmd == 'hybrid' and len(parts) > 1:
                query = parts[1]
                results = search_system.hybrid_search(query, top_k=5)
                
                if results:
                    print(f"\nFound {len(results)} results:")
                    for i, result in enumerate(results, 1):
                        print(format_result(result, i))
                else:
                    print("No results found")
            
            elif cmd == 'author' and len(parts) > 1:
                author = parts[1]
                results = search_system.search_by_author(author, top_k=5)
                
                if results:
                    print(f"\nFound {len(results)} publications:")
                    for i, result in enumerate(results, 1):
                        print(format_result(result, i))
                else:
                    print("No results found")
            
            elif cmd == 'similar' and len(parts) > 1:
                doi = parts[1]
                results = search_system.find_similar_papers(doi, top_k=5)
                
                if results:
                    print(f"\nFound {len(results)} similar papers:")
                    for i, result in enumerate(results, 1):
                        print(format_result(result, i))
                else:
                    print("No results found")
            
            else:
                print("Invalid command. Use: search/hybrid/author/similar <query>")
    
    except KeyboardInterrupt:
        print("\n\nExiting...")
    finally:
        search_system.close()

if __name__ == "__main__":
    interactive_mode()