"""
Script 3: Completar datos de publicaciones usando Scopus API
"""

import os
import requests
import time
from neo4j import GraphDatabase
import logging
from config_manager import get_config

# Configuración global
config = get_config()

# Configuración de logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("complete_data.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

def init_scopus():
    """Función para inicializar pybliometrics"""
    config_file = os.path.expanduser('~/.pybliometrics.cfg')
    with open(config_file, 'w') as f:
        f.write(f"""[Scopus]
APIKey = {config.scopus_api_key}
InstToken = 
Identifier = 

[Directories]
AbstractRetrieval = 
AffiliationRetrieval = 
AuthorRetrieval = 
CitationOverview = 
ContentAffiliationRetrieval = 

[Authentication]
APIKey = {config.scopus_api_key}
InstToken = 
""")
    logger.info(f"Archivo de configuración creado en: {config_file}")
    
    import pybliometrics
    pybliometrics.init()
    logger.info("Pybliometrics inicializado correctamente")

def connect_to_neo4j():
    """Conectar a Neo4j"""
    try:
        driver = GraphDatabase.driver(config.neo4j_uri, auth=(config.neo4j_user, config.neo4j_password))
        with driver.session() as session:
            session.run("RETURN 1")
        logger.info(f"Conexión exitosa a Neo4j - Base de datos: {config.neo4j_database}")
        return driver
    except Exception as e:
        logger.error(f"Error al conectar a Neo4j: {e}")
        return None

def debug_print_abstract(abstract_obj, doi):
    """Función de depuración para imprimir información detallada sobre el objeto abstract"""
    logger.info(f"\n--- INFO BÁSICA PARA {doi} ---")
    
    if hasattr(abstract_obj, 'abstract') and abstract_obj.abstract:
        logger.info(f"Abstract length: {len(str(abstract_obj.abstract))}")
        logger.info(f"Abstract preview: {str(abstract_obj.abstract)[:100]}...")
    else:
        logger.info(f"No abstract encontrado en el atributo 'abstract'")
        
    if hasattr(abstract_obj, 'description') and abstract_obj.description:
        logger.info(f"Description length: {len(str(abstract_obj.description))}")
        logger.info(f"Description preview: {str(abstract_obj.description)[:100]}...")
    else:
        logger.info(f"No abstract encontrado en el atributo 'description'")

def get_incomplete_publications(driver, limit=500):
    """Obtener publicaciones con datos incompletos"""
    logger.info("Buscando publicaciones con datos incompletos...")
    
    with driver.session(database=config.neo4j_database) as session:
        result = session.run("""
            MATCH (p:Publication)
            WHERE p.doi IS NOT NULL AND 
                (p.abstract IS NULL OR p.abstract = "" OR 
                 NOT EXISTS((p)-[:HAS_KEYWORD]->()) OR 
                 p.citedBy IS NULL)
            RETURN p.doi AS doi, p.title AS title, p.eid AS eid
            LIMIT $limit
        """, limit=limit)
        
        publications = [(record["doi"], record["title"], record["eid"]) for record in result]
    
    logger.info(f"Encontradas {len(publications)} publicaciones para completar")
    return publications

def complete_publication_data(driver, publications):
    """Completar datos de cada publicación con los datos de Scopus"""
    from pybliometrics.scopus import AbstractRetrieval
    
    for i, (doi, title, eid) in enumerate(publications):
        logger.info(f"\n[{i+1}/{len(publications)}] Completando: {title}")
        logger.info(f"DOI: {doi}")
        
        try:
            logger.info("Consultando Scopus API...")
            abstract = AbstractRetrieval(doi, view='FULL')
            logger.info("Datos obtenidos correctamente")
            
            # Debuggear el objeto abstract
            debug_print_abstract(abstract, doi)
            
            # Extraer el texto del abstract correctamente
            abstract_text = ""
            if hasattr(abstract, 'abstract') and abstract.abstract:
                abstract_text = abstract.abstract
            elif hasattr(abstract, 'description') and abstract.description:
                abstract_text = abstract.description
            
            if abstract_text:
                logger.info(f"✓ Abstract extraído correctamente ({len(abstract_text)} caracteres)")
                logger.info(f"Vista previa: {abstract_text[:100]}...")
            else:
                logger.warning("⚠️ No se pudo extraer el abstract")
            
            # Actualizar datos en Neo4j
            with driver.session(database=config.neo4j_database) as session:
                # 1. Actualizar abstract, citedBy y otros campos básicos
                session.run("""
                    MATCH (p:Publication {doi: $doi})
                    SET p.abstract = $abstract,
                        p.citedBy = $citedBy,
                        p.updated = datetime()
                """, 
                doi=doi, 
                abstract=abstract_text,
                citedBy=abstract.citedby_count if hasattr(abstract, 'citedby_count') else 0)
                
                if abstract_text:
                    logger.info("✓ Abstract actualizado")
                logger.info("✓ Citaciones actualizadas")
                
                # 2. Agregar keywords
                keywords = []
                if hasattr(abstract, 'authkeywords') and abstract.authkeywords:
                    keywords.extend(abstract.authkeywords)
                if hasattr(abstract, 'idxterms') and abstract.idxterms:
                    keywords.extend(abstract.idxterms)
                
                if keywords:
                    for keyword in keywords:
                        if keyword:
                            session.run("""
                                MERGE (k:Keyword {name: $keyword})
                                WITH k
                                MATCH (p:Publication {doi: $doi})
                                MERGE (p)-[:HAS_KEYWORD]->(k)
                            """, 
                            doi=doi, 
                            keyword=keyword.lower())
                    logger.info(f"✓ {len(keywords)} keywords añadidas")
                
                # 3. Agregar journal
                if hasattr(abstract, 'publicationName') and abstract.publicationName:
                    session.run("""
                        MERGE (j:Journal {name: $journal})
                        WITH j
                        MATCH (p:Publication {doi: $doi})
                        MERGE (p)-[:PUBLISHED_IN]->(j)
                    """, 
                    doi=doi, 
                    journal=abstract.publicationName)
                    logger.info("✓ Journal actualizado")
                    
                # 4. Agregar afiliaciones/países
                if hasattr(abstract, 'affiliation') and abstract.affiliation:
                    for affiliation in abstract.affiliation:
                        if hasattr(affiliation, 'country') and affiliation.country:
                            session.run("""
                                MERGE (c:Country {name: $country})
                                WITH c
                                MATCH (p:Publication {doi: $doi})
                                MERGE (p)-[:AFFILIATED_WITH]->(c)
                            """, 
                            doi=doi, 
                            country=affiliation.country)
                    logger.info("✓ Países/afiliaciones actualizados")
                
                logger.info(f"✅ Datos completados para: {doi}")
        
        except Exception as e:
            logger.error(f"❌ Error procesando {doi}: {str(e)}")
        
        # Pausa para respetar los límites de la API
        time.sleep(1)

def run_script3():
    """Función principal del script 3"""
    logger.info("\n" + "="*80)
    logger.info("INICIANDO SCRIPT 3: COMPLETAR DATOS DE PUBLICACIONES")
    logger.info("="*80 + "\n")
    
    try:
        # Inicializar pybliometrics correctamente
        logger.info("Inicializando pybliometrics...")
        init_scopus()

        # Importar módulos después de la inicialización
        from pybliometrics.scopus import AbstractRetrieval

        # Conectar a Neo4j
        driver = connect_to_neo4j()
        if not driver:
            return False

        try:
            # Obtener publicaciones incompletas
            publications = get_incomplete_publications(driver)
            
            if not publications:
                logger.info("No hay publicaciones incompletas para procesar")
                return True
            
            # Completar datos
            complete_publication_data(driver, publications)
            
            logger.info("\n✅ Script 3 completado con éxito!")
            return True
            
        finally:
            driver.close()
            logger.info("Conexión a Neo4j cerrada")

    except Exception as e:
        logger.error(f"Error no controlado en script 3: {e}")
        import traceback
        logger.error(f"Traceback: {traceback.format_exc()}")
        return False

if __name__ == "__main__":
    run_script3()