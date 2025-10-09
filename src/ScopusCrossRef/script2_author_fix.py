"""
Script 2: Verificar y corregir la importación de autores en Neo4j.
Este script es independiente y puede ejecutarse por separado para diagnosticar
y corregir problemas con los autores en la base de datos.
"""

import os
import json
import pandas as pd
from neo4j import GraphDatabase
import logging
import hashlib
from config_manager import get_config

# Configuración global
config = get_config()

# Configuración de logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("author_fix.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

def connect_to_neo4j():
    """Conectar a la base de datos Neo4j"""
    try:
        driver = GraphDatabase.driver(config.neo4j_uri, auth=(config.neo4j_user, config.neo4j_password))
        with driver.session() as session:
            session.run("RETURN 1")
        logger.info(f"Conexión exitosa a Neo4j - Base de datos: {config.neo4j_database}")
        return driver
    except Exception as e:
        logger.error(f"Error al conectar a Neo4j: {e}")
        return None

def check_database_statistics(driver):
    """Verificar las estadísticas de la base de datos"""
    logger.info("--- VERIFICANDO ESTADÍSTICAS DE LA BASE DE DATOS ---")
    
    with driver.session(database=config.neo4j_database) as session:
        node_types = ["Publication", "Author", "Keyword", "Journal", "Country"]
        for node_type in node_types:
            count = session.run(f"MATCH (n:{node_type}) RETURN COUNT(n) AS count").single()["count"]
            logger.info(f"Nodos {node_type}: {count}")
        
        rel_types = ["AUTHORED", "HAS_KEYWORD", "PUBLISHED_IN", "CITES", "COLLABORATES_WITH"]
        for rel_type in rel_types:
            count = session.run(f"MATCH ()-[r:{rel_type}]->() RETURN COUNT(r) AS count").single()["count"]
            logger.info(f"Relaciones {rel_type}: {count}")

def load_enriched_data(query_hash="default"):
    """Cargar los datos enriquecidos desde el checkpoint"""
    enriched_data_file = os.path.join(config.data_dir, f"enriched_data_{query_hash}.json")
    
    if not os.path.exists(enriched_data_file):
        logger.warning(f"No se encontró el archivo {enriched_data_file}")
        # Intentar buscar el archivo sin hash como fallback
        fallback_file = os.path.join(config.data_dir, "enriched_data.json")
        if os.path.exists(fallback_file):
            logger.info(f"Usando archivo fallback: {fallback_file}")
            enriched_data_file = fallback_file
        else:
            return None
    
    logger.info(f"Cargando datos desde: {enriched_data_file}")
    
    try:
        with open(enriched_data_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        if isinstance(data, dict):
            df = pd.DataFrame.from_dict(data, orient='index')
        elif isinstance(data, list):
            df = pd.DataFrame(data)
        else:
            logger.error(f"Formato de datos no reconocido: {type(data)}")
            return None
        
        logger.info(f"Cargados {len(df)} registros")
        return df
    except Exception as e:
        logger.error(f"Error al cargar datos: {e}")
        
        try:
            df = pd.read_json(enriched_data_file)
            logger.info(f"Cargados {len(df)} registros usando pandas")
            return df
        except Exception as e2:
            logger.error(f"Error al cargar con pandas: {e2}")
            return None

def check_author_fields(df):
    """Verificar los campos de autor en los datos"""
    logger.info("--- VERIFICANDO CAMPOS DE AUTOR EN LOS DATOS ---")
    
    if df is None:
        logger.warning("No hay datos para verificar")
        return
    
    for field in ['author', 'authors', 'author_ids']:
        has_field = field in df.columns
        logger.info(f"Campo '{field}' existe: {has_field}")
        
        if has_field:
            sample = df[field].iloc[0] if not df[field].empty else None
            logger.info(f"  - Ejemplo del campo '{field}': {type(sample)} - {sample}")
    
    if 'authors' in df.columns:
        has_authors = df['authors'].apply(lambda x: isinstance(x, list) and len(x) > 0)
        logger.info(f"Publicaciones con autores (campo 'authors'): {has_authors.sum()}/{len(df)}")
    
    if 'author_ids' in df.columns:
        has_author_ids = df['author_ids'].apply(lambda x: isinstance(x, list) and len(x) > 0)
        logger.info(f"Publicaciones con IDs de autor (campo 'author_ids'): {has_author_ids.sum()}/{len(df)}")
    
    if 'author' in df.columns:
        has_author = df['author'].apply(lambda x: isinstance(x, list) and len(x) > 0)
        logger.info(f"Publicaciones con autor (campo 'author'): {has_author.sum()}/{len(df)}")
        
        if has_author.any():
            sample_idx = has_author.idxmax()
            sample_author = df.loc[sample_idx, 'author'][0]
            logger.info("Estructura de un autor de ejemplo:")
            logger.info(json.dumps(sample_author, indent=2))

def fix_authors(driver, df, max_pubs=None):
    """Corregir autores en la base de datos"""
    logger.info("--- CORRIGIENDO AUTORES EN LA BASE DE DATOS ---")
    
    if df is None:
        logger.warning("No hay datos para corregir")
        return
    
    use_direct_author = 'author' in df.columns and df['author'].apply(lambda x: isinstance(x, list) and len(x) > 0).any()
    
    logger.info(f"Usando campo 'author' directo: {use_direct_author}")
    
    # Usar configuración si está disponible
    if max_pubs is None:
        max_pubs = config.max_papers_authors if config.max_papers_authors > 0 else len(df)
    
    if max_pubs > 0 and len(df) > max_pubs:
        logger.info(f"Limitando a {max_pubs} publicaciones para procesamiento")
        df = df.head(max_pubs)
    
    with driver.session(database=config.neo4j_database) as session:
        for i, pub in df.iterrows():
            eid = pub.get('eid', '')
            if not eid:
                continue
            
            author_data = []
            
            if use_direct_author and isinstance(pub.get('author'), list):
                for author in pub['author']:
                    if 'authid' in author and 'authname' in author:
                        author_data.append({
                            'id': author['authid'],
                            'name': author['authname']
                        })
            elif 'author_ids' in pub and 'authors' in pub:
                if isinstance(pub['author_ids'], list) and isinstance(pub['authors'], list):
                    min_len = min(len(pub['author_ids']), len(pub['authors']))
                    for j in range(min_len):
                        author_data.append({
                            'id': pub['author_ids'][j],
                            'name': pub['authors'][j]
                        })
            
            if not author_data:
                logger.warning(f"No se encontraron datos de autor para la publicación {eid}")
                continue
            
            logger.info(f"Procesando {len(author_data)} autores para la publicación {eid}")
            
            for author in author_data:
                author_id = str(author['id'])
                author_name = author['name']
                
                if not author_id or not author_name:
                    continue
                
                try:
                    session.run("""
                        MERGE (a:Author {id: $author_id})
                        SET a.name = $author_name
                    """,
                    author_id=author_id,
                    author_name=author_name)
                    
                    session.run("""
                        MATCH (a:Author {id: $author_id})
                        MATCH (p:Publication {eid: $eid})
                        MERGE (a)-[:AUTHORED]->(p)
                    """,
                    author_id=author_id,
                    eid=eid)
                    
                    logger.info(f"Autor {author_name} (ID: {author_id}) conectado a publicación {eid}")
                except Exception as e:
                    logger.error(f"Error procesando autor {author_name}: {e}")
    
    logger.info("Corrección de autores completada")

def create_coauthor_relationships(driver):
    """Crear relaciones de coautoría entre autores"""
    logger.info("--- CREANDO RELACIONES DE COAUTORÍA ---")
    
    with driver.session(database=config.neo4j_database) as session:
        result = session.run("MATCH (a:Author) RETURN COUNT(a) AS count")
        author_count = result.single()["count"]
        
        if author_count == 0:
            logger.warning("No hay autores en la base de datos")
            return
        
        logger.info(f"Encontrados {author_count} autores")
        
        coauthor_query = """
            MATCH (a1:Author)-[:AUTHORED]->(p:Publication)<-[:AUTHORED]-(a2:Author)
            WHERE a1.id < a2.id
            WITH a1, a2, COUNT(p) AS collaboration_count
            WHERE collaboration_count > 0
            RETURN a1.id AS author1_id, a2.id AS author2_id, collaboration_count AS weight
            ORDER BY weight DESC
        """
        
        result = session.run(coauthor_query)
        coauthor_data = list(result)
        
        if not coauthor_data:
            logger.warning("No se encontraron patrones de coautoría")
            return
        
        logger.info(f"Encontrados {len(coauthor_data)} pares de coautores")
        
        created_count = 0
        batch_size = config.batch_size_authors
        
        for i in range(0, len(coauthor_data), batch_size):
            batch = coauthor_data[i:i+batch_size]
            logger.info(f"Procesando lote {i//batch_size + 1}/{(len(coauthor_data) + batch_size - 1)//batch_size}")
            
            with session.begin_transaction() as tx:
                for record in batch:
                    try:
                        tx.run("""
                            MATCH (a1:Author {id: $author1_id})
                            MATCH (a2:Author {id: $author2_id})
                            MERGE (a1)-[r1:COLLABORATES_WITH]->(a2)
                            SET r1.weight = $weight
                            MERGE (a2)-[r2:COLLABORATES_WITH]->(a1)
                            SET r2.weight = $weight
                        """,
                        author1_id=record["author1_id"],
                        author2_id=record["author2_id"],
                        weight=record["weight"])
                        
                        created_count += 2
                    except Exception as e:
                        logger.error(f"Error creando relaciones entre {record['author1_id']} y {record['author2_id']}: {e}")
        
        logger.info(f"Creadas {created_count} relaciones COLLABORATES_WITH bidireccionales")

def run_script2(interactive: bool = False, query_hash: str = "default"):
    """Función principal del script 2"""
    logger.info("\n" + "="*80)
    logger.info("INICIANDO SCRIPT 2: VERIFICACIÓN Y CORRECCIÓN DE AUTORES")
    logger.info("="*80 + "\n")
    
    logger.info(f"Usando query hash: {query_hash}")
    
    driver = connect_to_neo4j()
    if not driver:
        return False
    
    try:
        # Verificar estadísticas actuales
        check_database_statistics(driver)
        
        # Cargar datos con el hash correcto
        df = load_enriched_data(query_hash)
        if df is None:
            logger.error("No se pudieron cargar los datos. Saliendo...")
            return False
        
        # Verificar campos
        check_author_fields(df)
        
        # Corregir autores
        fix_authors(driver, df)
        
        # Crear relaciones de coautoría
        create_coauthor_relationships(driver)
        
        # Verificar estadísticas finales
        logger.info("--- ESTADÍSTICAS FINALES ---")
        check_database_statistics(driver)
        
        logger.info("\n¡Script 2 completado con éxito!")
        return True
        
    except Exception as e:
        logger.error(f"Error no controlado en script 2: {e}")
        import traceback
        logger.error(f"Traceback: {traceback.format_exc()}")
        return False
    finally:
        driver.close()
        logger.info("Conexión a Neo4j cerrada")

if __name__ == "__main__":
    # Si se ejecuta directamente, intentar detectar el hash más reciente
    query = input("Ingrese la consulta de búsqueda (para generar hash): ").strip()
    if query:
        query_hash = hashlib.md5(query.encode()).hexdigest()[:8]
        logger.info(f"Hash generado: {query_hash}")
        run_script2(interactive=True, query_hash=query_hash)
    else:
        run_script2(interactive=True)