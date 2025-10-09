"""
Script 5: Setup Vector Indices en Neo4j
Usa procedimientos db.index.vector.createNodeIndex para Neo4j 5.11+
"""

import logging
from neo4j import GraphDatabase
from config_manager import get_config

config = get_config()

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("vector_setup.log"),
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
        logger.error(f"Connection error: {e}")
        return None

def check_neo4j_version(driver):
    """Verificar versión de Neo4j"""
    with driver.session(database=config.neo4j_database) as session:
        result = session.run("CALL dbms.components() YIELD versions RETURN versions[0] as version")
        version = result.single()["version"]
        logger.info(f"Neo4j Version: {version}")
        
        major_version = int(version.split('.')[0])
        minor_version = int(version.split('.')[1])
        
        if major_version < 5 or (major_version == 5 and minor_version < 11):
            logger.error(f"Neo4j {version} does not support vector indices. Need 5.11+")
            return False
        return True

def create_vector_indices(driver):
    """Crear índices vectoriales usando procedimientos"""
    logger.info("--- CREATING VECTOR INDICES ---")
    
    with driver.session(database=config.neo4j_database) as session:
        # Index for abstract embeddings
        try:
            session.run(f"""
                CALL db.index.vector.createNodeIndex(
                    'publication_abstract_embeddings',
                    'Publication',
                    'abstract_embedding',
                    {config.vector_dimension},
                    '{config.similarity_function}'
                )
            """)
            logger.info("Vector index for abstracts created successfully")
        except Exception as e:
            error_msg = str(e).lower()
            if "already exists" in error_msg or "equivalent index" in error_msg:
                logger.info("Abstract vector index already exists")
            else:
                logger.error(f"Failed to create abstract index: {e}")
        
        # Index for title embeddings
        try:
            session.run(f"""
                CALL db.index.vector.createNodeIndex(
                    'publication_title_embeddings',
                    'Publication',
                    'title_embedding',
                    {config.vector_dimension},
                    '{config.similarity_function}'
                )
            """)
            logger.info("Vector index for titles created successfully")
        except Exception as e:
            error_msg = str(e).lower()
            if "already exists" in error_msg or "equivalent index" in error_msg:
                logger.info("Title vector index already exists")
            else:
                logger.error(f"Failed to create title index: {e}")
        
        # Composite index for filtered searches
        try:
            session.run("""
                CREATE INDEX publication_year_citations IF NOT EXISTS
                FOR (p:Publication) ON (p.year, p.citedBy)
            """)
            logger.info("Composite index (year, citedBy) created")
        except Exception as e:
            error_msg = str(e).lower()
            if "already exists" in error_msg or "equivalent index" in error_msg:
                logger.info("Composite index already exists")
            else:
                logger.warning(f"Composite index warning: {e}")

def show_index_status(driver):
    """Mostrar estado de los índices"""
    logger.info("--- INDEX STATUS ---")
    
    with driver.session(database=config.neo4j_database) as session:
        # Check vector indices specifically
        try:
            result = session.run("CALL db.index.vector.list()")
            vector_indices = list(result)
            
            if vector_indices:
                logger.info(f"Vector indices found: {len(vector_indices)}")
                for idx in vector_indices:
                    name = idx.get("name", "N/A")
                    state = idx.get("state", "N/A")
                    logger.info(f"  - {name}: {state}")
            else:
                logger.warning("No vector indices found via db.index.vector.list()")
        except Exception as e:
            logger.warning(f"Could not list vector indices: {e}")
        
        # Check all indices
        result = session.run("SHOW INDEXES")
        vector_count = 0
        other_count = 0
        
        for record in result:
            name = record.get("name", "N/A")
            type_desc = str(record.get("type", ""))
            
            if "vector" in type_desc.lower() or "vector" in name.lower():
                vector_count += 1
            else:
                other_count += 1
        
        logger.info(f"\nTotal from SHOW INDEXES: {vector_count} vector, {other_count} other")

def run_script5():
    """Función principal del script 5"""
    logger.info("\n" + "="*80)
    logger.info("STARTING SCRIPT 5: SETUP VECTOR INDICES")
    logger.info("="*80 + "\n")
    
    if not config.vector_store_enabled:
        logger.warning("Vector Store disabled in configuration")
        return False
    
    driver = connect_to_neo4j()
    if not driver:
        return False
    
    try:
        # Check version
        if not check_neo4j_version(driver):
            return False
        
        # Create indices
        create_vector_indices(driver)
        
        # Show status
        show_index_status(driver)
        
        logger.info("\nScript 5 completed!")
        logger.info("If vector indices were created, you can now run script6_embeddings.py")
        return True
        
    except Exception as e:
        logger.error(f"Unhandled error in script 5: {e}")
        import traceback
        logger.error(f"Traceback: {traceback.format_exc()}")
        return False
    finally:
        driver.close()
        logger.info("Neo4j connection closed")

if __name__ == "__main__":
    run_script5()