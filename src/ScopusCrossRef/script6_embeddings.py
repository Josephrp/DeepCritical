#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Script 6: Generate vector embeddings for publications
- Generates embeddings for abstracts and titles
- Stores them in Neo4j for vector search
- Processes in batches for efficiency
"""

import os
import logging
from typing import List, Dict, Any
from tqdm import tqdm
from neo4j import GraphDatabase
from config_manager import get_config

# Configuración
config = get_config()

# Logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("embedding_generation.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

def connect_to_neo4j():
    """Conectar a Neo4j"""
    try:
        driver = GraphDatabase.driver(
            config.neo4j_uri, 
            auth=(config.neo4j_user, config.neo4j_password)
        )
        with driver.session(database=config.neo4j_database) as session:
            session.run("RETURN 1")
        logger.info(f"Connection successful - Database: {config.neo4j_database}")
        return driver
    except Exception as e:
        logger.error(f"Error connecting to Neo4j: {e}")
        return None

def load_embedding_model():
    """Cargar modelo de embeddings"""
    try:
        from sentence_transformers import SentenceTransformer
        
        model_name = os.getenv("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
        logger.info(f"Loading embedding model: {model_name}")
        
        model = SentenceTransformer(model_name)
        logger.info(f"Model loaded successfully - Dimension: {model.get_sentence_embedding_dimension()}")
        
        return model
    except Exception as e:
        logger.error(f"Error loading model: {e}")
        return None

def get_publications_without_embeddings(driver, limit=None):
    """Obtener publicaciones sin embeddings"""
    with driver.session(database=config.neo4j_database) as session:
        query = """
            MATCH (p:Publication)
            WHERE p.abstract_embedding IS NULL 
               OR p.title_embedding IS NULL
            RETURN p.eid AS eid, 
                   p.doi AS doi,
                   p.title AS title, 
                   p.abstract AS abstract
            ORDER BY p.citedBy DESC
        """
        
        if limit:
            query += f" LIMIT {limit}"
        
        result = session.run(query)
        publications = [dict(record) for record in result]
    
    logger.info(f"Found {len(publications)} publications without embeddings")
    return publications

def clean_text(text: str) -> str:
    """Limpiar texto para embeddings"""
    if not text or text == "":
        return ""
    
    # Convertir a string si no lo es
    text = str(text)
    
    # Limpieza básica
    text = text.replace('\n', ' ').replace('\r', ' ')
    text = ' '.join(text.split())  # Normalizar espacios
    
    # Truncar si es muy largo (límite de la mayoría de modelos: 512 tokens ≈ 2048 chars)
    if len(text) > 2000:
        text = text[:2000]
    
    return text.strip()

def generate_embeddings_batch(model, texts: List[str]) -> List[List[float]]:
    """Generar embeddings para un lote de textos"""
    try:
        # Limpiar textos
        cleaned_texts = [clean_text(text) if text else "" for text in texts]
        
        # Reemplazar vacíos con placeholder
        cleaned_texts = [text if text else "No content available" for text in cleaned_texts]
        
        # Generar embeddings
        embeddings = model.encode(
            cleaned_texts,
            show_progress_bar=False,
            convert_to_numpy=True
        )
        
        # Convertir a listas de Python (Neo4j no acepta numpy arrays directamente)
        return [embedding.tolist() for embedding in embeddings]
        
    except Exception as e:
        logger.error(f"Error generating embeddings: {e}")
        return [[] for _ in texts]

def update_embeddings_in_neo4j(driver, updates: List[Dict[str, Any]]):
    """Actualizar embeddings en Neo4j"""
    with driver.session(database=config.neo4j_database) as session:
        for update in updates:
            try:
                session.run("""
                    MATCH (p:Publication {eid: $eid})
                    SET p.abstract_embedding = $abstract_embedding,
                        p.title_embedding = $title_embedding,
                        p.embeddings_generated = datetime()
                """,
                eid=update['eid'],
                abstract_embedding=update['abstract_embedding'],
                title_embedding=update['title_embedding']
                )
            except Exception as e:
                logger.error(f"Error updating {update['eid']}: {e}")

def process_embeddings(driver, model, batch_size=50, limit=None):
    """Procesar embeddings en lotes"""
    
    # Obtener publicaciones
    publications = get_publications_without_embeddings(driver, limit)
    
    if not publications:
        logger.info("No publications to process")
        return
    
    total = len(publications)
    logger.info(f"Processing {total} publications in batches of {batch_size}")
    
    # Procesar en lotes
    for i in tqdm(range(0, total, batch_size), desc="Processing batches"):
        batch = publications[i:i+batch_size]
        
        # Extraer textos
        abstracts = [pub.get('abstract', '') for pub in batch]
        titles = [pub.get('title', '') for pub in batch]
        
        # Generar embeddings
        abstract_embeddings = generate_embeddings_batch(model, abstracts)
        title_embeddings = generate_embeddings_batch(model, titles)
        
        # Preparar updates
        updates = []
        for j, pub in enumerate(batch):
            updates.append({
                'eid': pub['eid'],
                'abstract_embedding': abstract_embeddings[j],
                'title_embedding': title_embeddings[j]
            })
        
        # Actualizar en Neo4j
        update_embeddings_in_neo4j(driver, updates)
        
        if (i + batch_size) % 200 == 0:
            logger.info(f"Processed {min(i + batch_size, total)}/{total} publications")
    
    logger.info(f"Completed processing {total} publications")

def verify_embeddings(driver):
    """Verificar embeddings generados"""
    with driver.session(database=config.neo4j_database) as session:
        # Contar totales
        total = session.run("MATCH (p:Publication) RETURN count(p) AS total").single()["total"]
        
        # Contar con embeddings
        with_embeddings = session.run("""
            MATCH (p:Publication)
            WHERE p.abstract_embedding IS NOT NULL 
              AND p.title_embedding IS NOT NULL
            RETURN count(p) AS count
        """).single()["count"]
        
        # Contar sin embeddings
        without_embeddings = session.run("""
            MATCH (p:Publication)
            WHERE p.abstract_embedding IS NULL 
               OR p.title_embedding IS NULL
            RETURN count(p) AS count
        """).single()["count"]
        
        logger.info("\n--- EMBEDDING STATISTICS ---")
        logger.info(f"Total publications: {total}")
        logger.info(f"With embeddings: {with_embeddings} ({100*with_embeddings/total:.1f}%)")
        logger.info(f"Without embeddings: {without_embeddings}")
        
        # Sample verificación
        sample = session.run("""
            MATCH (p:Publication)
            WHERE p.abstract_embedding IS NOT NULL
            RETURN p.title AS title, 
                   size(p.abstract_embedding) AS abstract_dim,
                   size(p.title_embedding) AS title_dim
            LIMIT 3
        """).data()
        
        if sample:
            logger.info("\n--- SAMPLE VERIFICATION ---")
            for i, s in enumerate(sample, 1):
                logger.info(f"{i}. {s['title'][:60]}...")
                logger.info(f"   Abstract dim: {s['abstract_dim']}, Title dim: {s['title_dim']}")

def run_script6():
    """Función principal"""
    logger.info("\n" + "="*80)
    logger.info("STARTING SCRIPT 6: GENERATE EMBEDDINGS")
    logger.info("="*80 + "\n")
    
    # Conectar a Neo4j
    driver = connect_to_neo4j()
    if not driver:
        return False
    
    try:
        # Cargar modelo
        model = load_embedding_model()
        if not model:
            return False
        
        # Procesar embeddings
        batch_size = int(os.getenv("BATCH_SIZE_EMBEDDING", "50"))
        
        # Opcional: descomentar para limitar durante pruebas
        # limit = 100  
        limit = None  # Procesar todo
        
        process_embeddings(driver, model, batch_size=batch_size, limit=limit)
        
        # Verificar
        verify_embeddings(driver)
        
        logger.info("\n✅ Script 6 completed successfully!")
        return True
        
    except Exception as e:
        logger.error(f"Error in script 6: {e}")
        import traceback
        logger.error(f"Traceback: {traceback.format_exc()}")
        return False
    finally:
        driver.close()
        logger.info("Neo4j connection closed")

if __name__ == "__main__":
    run_script6()