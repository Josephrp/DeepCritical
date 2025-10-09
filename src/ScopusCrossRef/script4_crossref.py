"""
Script 4: Enriquecer datos usando Crossref API
"""

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
        logging.FileHandler("crossref_enrichment.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

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

def obtener_dois(driver, limit=200):
    """Obtener todos los DOIs de la base de datos"""
    with driver.session(database=config.neo4j_database) as session:
        result = session.run("""
            MATCH (p:Publication)
            WHERE p.doi IS NOT NULL
            RETURN p.doi AS doi, p.title AS title
            LIMIT $limit
        """, limit=limit)
        return [(record["doi"], record["title"]) for record in result]

def obtener_info_crossref(doi):
    """Obtener información completa de Crossref para un DOI"""
    try:
        url = f"https://api.crossref.org/works/{doi}"
        headers = {'User-Agent': f'PythonScript/1.0 ({config.crossref_email})'}
        
        response = requests.get(url, headers=headers)
        if response.status_code == 200:
            return response.json()['message']
        else:
            logger.warning(f"Error consultando Crossref: {response.status_code}")
            return None
    except Exception as e:
        logger.error(f"Error en consulta Crossref: {e}")
        return None

def obtener_citas_entrantes(doi):
    """Obtener DOIs que citan a este trabajo"""
    try:
        # 1. Obtener conteo de citas
        info = obtener_info_crossref(doi)
        if not info:
            return {'count': 0, 'dois': []}
        
        citation_count = info.get('is-referenced-by-count', 0)
        
        # 2. Obtener DOIs que citan
        citing_dois = []
        if citation_count > 0:
            filter_query = f"references:{doi}"
            citing_url = f"https://api.crossref.org/works?filter={filter_query}&rows=1000"
            
            headers = {'User-Agent': f'PythonScript/1.0 ({config.crossref_email})'}
            citing_response = requests.get(citing_url, headers=headers)
            
            if citing_response.status_code == 200:
                citing_data = citing_response.json()
                if 'message' in citing_data and 'items' in citing_data['message']:
                    for item in citing_data['message']['items']:
                        if 'DOI' in item:
                            citing_dois.append(item['DOI'].lower())
        
        return {'count': citation_count, 'dois': citing_dois}
    except Exception as e:
        logger.error(f"Error obteniendo citas para {doi}: {e}")
        return {'count': 0, 'dois': []}

def obtener_referencias(info):
    """Extraer referencias de la información de Crossref"""
    try:
        ref_dois = []
        if 'reference' in info:
            for ref in info['reference']:
                if 'DOI' in ref:
                    ref_dois.append(ref['DOI'].lower())
        return ref_dois
    except Exception as e:
        logger.error(f"Error procesando referencias: {e}")
        return []

def actualizar_neo4j(driver, doi, info, citas_entrantes):
    """Actualizar Neo4j con la información de Crossref"""
    with driver.session(database=config.neo4j_database) as session:
        try:
            # 1. Actualizar información básica de la publicación
            url = info.get('URL', '')
            
            props = {
                'doi': doi,
                'crossrefCitedBy': citas_entrantes['count'],
                'url': url,
                'updated': True
            }
            
            # Agregar información de publicación si está disponible
            if 'published' in info and 'date-parts' in info['published']:
                if info['published']['date-parts'][0]:
                    year = info['published']['date-parts'][0][0]
                    if year:
                        props['year'] = str(year)
            
            # Actualizar propiedades
            props_string = ", ".join([f"p.{k} = ${k}" for k in props.keys()])
            session.run(f"""
                MATCH (p:Publication {{doi: $doi}})
                SET {props_string}
            """, **props)
            
            # 2. Procesar referencias (qué publicaciones cita este trabajo)
            if 'reference' in info:
                ref_dois = []
                for ref in info['reference']:
                    if 'DOI' in ref:
                        ref_dois.append(ref['DOI'].lower())
                
                if ref_dois:
                    result = session.run("""
                        MATCH (p:Publication)
                        WHERE p.doi IN $dois
                        RETURN p.doi AS doi
                    """, dois=ref_dois)
                    
                    existing_ref_dois = [record["doi"] for record in result]
                    
                    if existing_ref_dois:
                        session.run("""
                            MATCH (citing:Publication {doi: $doi})
                            MATCH (cited:Publication)
                            WHERE cited.doi IN $ref_dois
                            MERGE (citing)-[r:CITES]->(cited)
                        """, doi=doi, ref_dois=existing_ref_dois)
                        
                        logger.info(f"Creadas {len(existing_ref_dois)}/{len(ref_dois)} relaciones de referencias")
                    else:
                        logger.info(f"Ninguno de los {len(ref_dois)} DOIs de referencias existe en la base de datos")
            
            # 3. Procesar citas entrantes (qué publicaciones citan este trabajo)
            if citas_entrantes['dois']:
                result = session.run("""
                    MATCH (p:Publication)
                    WHERE p.doi IN $dois
                    RETURN p.doi AS doi
                """, dois=citas_entrantes['dois'])
                
                existing_citing_dois = [record["doi"] for record in result]
                
                if existing_citing_dois:
                    session.run("""
                        MATCH (cited:Publication {doi: $doi})
                        MATCH (citing:Publication)
                        WHERE citing.doi IN $citing_dois
                        MERGE (citing)-[r:CITES]->(cited)
                    """, doi=doi, citing_dois=existing_citing_dois)
                    
                    logger.info(f"Creadas {len(existing_citing_dois)}/{len(citas_entrantes['dois'])} relaciones de citas entrantes")
                else:
                    logger.info(f"Ninguno de los {len(citas_entrantes['dois'])} DOIs citantes existe en la base de datos")
            
            return True
        except Exception as e:
            logger.error(f"Error actualizando Neo4j para {doi}: {e}")
            return False

def run_script4():
    """Función principal del script 4"""
    logger.info("\n" + "="*80)
    logger.info("INICIANDO SCRIPT 4: ENRIQUECIMIENTO CON CROSSREF")
    logger.info("="*80 + "\n")
    
    # Conectar a Neo4j
    driver = connect_to_neo4j()
    if not driver:
        return False
    
    try:
        # Obtener DOIs de Neo4j
        logger.info("Obteniendo DOIs de la base de datos...")
        dois = obtener_dois(driver)
        logger.info(f"Encontrados {len(dois)} DOIs")
        
        # Procesar cada DOI
        for i, (doi, title) in enumerate(dois):
            logger.info(f"\n[{i+1}/{len(dois)}] Procesando: {title}")
            logger.info(f"DOI: {doi}")
            
            # 1. Obtener información de Crossref
            logger.info("Consultando información en Crossref...")
            info = obtener_info_crossref(doi)
            
            if not info:
                logger.warning(f"No se pudo obtener información de Crossref para {doi}")
                continue
            
            # 2. Obtener citas entrantes
            logger.info("Consultando citas entrantes...")
            citas_entrantes = obtener_citas_entrantes(doi)
            logger.info(f"Encontradas {citas_entrantes['count']} citas")
            
            # 3. Actualizar Neo4j
            logger.info("Actualizando Neo4j...")
            actualizar_neo4j(driver, doi, info, citas_entrantes)
            
            # Pausar para no sobrecargar la API
            if i < len(dois) - 1:
                logger.info("Pausa para no sobrecargar la API...")
                time.sleep(1)
        
        logger.info("\n✅ Script 4 completado con éxito!")
        return True
        
    except Exception as e:
        logger.error(f"Error no controlado en script 4: {e}")
        import traceback
        logger.error(f"Traceback: {traceback.format_exc()}")
        return False
    finally:
        driver.close()
        logger.info("Conexión a Neo4j cerrada")

if __name__ == "__main__":
    run_script4()