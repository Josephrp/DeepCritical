"""
Script de funding corregido - Soluciona el error de Neo4j con objetos complejos
"""

import os
import json
import time
import pandas as pd
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
        logging.FileHandler("funding_extraction_fixed.log"),
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

def initialize_pybliometrics():
    """Inicializar pybliometrics para usar la API de Scopus"""
    logger.info("--- INICIALIZANDO PYBLIOMETRICS ---")
    
    try:
        import pybliometrics
        pybliometrics.init()
        logger.info("Scopus API inicializado exitosamente")
        return True
    except Exception as e:
        logger.error(f"Error al inicializar Scopus API: {e}")
        return False

def extract_funding_data_corrected(abstract_obj, doi):
    """
    Extrae funding data de manera más exhaustiva del objeto AbstractRetrieval
    CORREGIDO: Convierte objetos a strings/listas simples para Neo4j
    """
    funding_data = {
        'funding_text': '',
        'grants_json': [],  # Como JSON string para Neo4j
        'funding_agencies': [],
        'funding_details_json': []  # Como JSON string para Neo4j
    }
    
    try:
        logger.info(f"Extrayendo funding para {doi}")
        
        # 1. Intentar obtener funding text
        funding_text_fields = ['funding_text', 'funding', 'acknowledgment', 'acknowledgments']
        for field in funding_text_fields:
            if hasattr(abstract_obj, field) and getattr(abstract_obj, field):
                funding_data['funding_text'] = str(getattr(abstract_obj, field))
                logger.info(f"✓ Funding text encontrado en campo '{field}'")
                break
        
        # 2. Intentar obtener funding structured data
        if hasattr(abstract_obj, 'funding') and abstract_obj.funding:
            logger.info(f"✓ Funding object encontrado con {len(abstract_obj.funding)} elementos")
            
            for funding_item in abstract_obj.funding:
                # CORREGIDO: Extraer como strings simples, no objetos
                funding_detail = {}
                
                # Extraer todos los campos disponibles del funding como strings
                funding_fields = ['agency', 'agency_id', 'string', 'acronym', 'id', 'funding_id']
                for field in funding_fields:
                    if hasattr(funding_item, field):
                        value = getattr(funding_item, field)
                        funding_detail[field] = str(value) if value is not None else ''
                
                if funding_detail:
                    # CORREGIDO: Guardar como JSON string
                    funding_data['funding_details_json'].append(json.dumps(funding_detail))
                    
                    # Extraer agency name para lista simple
                    agency_name = funding_detail.get('agency', funding_detail.get('string', ''))
                    if agency_name and agency_name not in funding_data['funding_agencies']:
                        funding_data['funding_agencies'].append(agency_name)
                    
                    # CORREGIDO: Crear grant info como JSON string
                    grant_info = {
                        'agency': funding_detail.get('agency', ''),
                        'string': funding_detail.get('string', funding_detail.get('id', '')),
                        'agency_id': funding_detail.get('agency_id', ''),
                        'acronym': funding_detail.get('acronym', '')
                    }
                    funding_data['grants_json'].append(json.dumps(grant_info))
        
        # Log resultado
        if funding_data['funding_text'] or funding_data['grants_json']:
            logger.info(f"✅ Funding data extraída exitosamente para {doi}")
            logger.info(f"  - Funding text: {len(funding_data['funding_text'])} chars")
            logger.info(f"  - Grants: {len(funding_data['grants_json'])}")
            logger.info(f"  - Agencies: {len(funding_data['funding_agencies'])}")
        else:
            logger.info(f"ℹ️  No funding data encontrada para {doi}")
        
        return funding_data
        
    except Exception as e:
        logger.error(f"Error extrayendo funding data para {doi}: {e}")
        return funding_data

def get_publications_for_funding_extraction(driver, limit=None):
    """Obtener publicaciones para extraer funding data"""
    logger.info("Obteniendo publicaciones para extracción de funding...")
    
    with driver.session(database=config.neo4j_database) as session:
        # CORREGIDO: Verificar cuáles ya tienen funding data para evitar reprocesar
        result = session.run("""
            MATCH (p:Publication)
            WHERE p.doi IS NOT NULL 
            AND (p.funding_extracted IS NULL OR p.funding_extracted = false)
            RETURN p.doi AS doi, p.title AS title, 
                   COALESCE(p.citedBy, 0) AS citations
            ORDER BY citations DESC
        """)
        
        publications = [(record["doi"], record["title"], record["citations"]) for record in result]
    
    # Si se especifica un límite, aplicarlo
    if limit and limit > 0:
        publications = publications[:limit]
        logger.info(f"Limitado a {len(publications)} publicaciones más citadas")
    else:
        logger.info(f"Procesando TODAS las {len(publications)} publicaciones disponibles")
    
    return publications

def extract_and_store_funding_corrected(driver, publications):
    """
    Extraer funding data y almacenar en Neo4j
    CORREGIDO: Maneja tipos de datos correctamente para Neo4j
    MEJORADO: Progreso y estadísticas en tiempo real
    """
    
    try:
        from pybliometrics.scopus import AbstractRetrieval
        
        total_publications = len(publications)
        processed_count = 0
        funding_found_count = 0
        error_count = 0
        
        logger.info(f"🚀 Iniciando procesamiento de {total_publications} publicaciones")
        
        for i, (doi, title, citations) in enumerate(publications):
            logger.info(f"\n[{i+1}/{total_publications}] Procesando: {title[:60]}...")
            logger.info(f"DOI: {doi}, Citas: {citations}")
            
            try:
                # Obtener datos de Scopus con vista FULL
                time.sleep(0.5)  # Rate limiting más agresivo para todas las publicaciones
                abstract = AbstractRetrieval(doi, view='FULL')
                
                # Extraer funding data corregido
                funding_data = extract_funding_data_corrected(abstract, doi)
                
                # CORREGIDO: Almacenar en Neo4j con tipos de datos correctos
                with driver.session(database=config.neo4j_database) as session:
                    # 1. Actualizar propiedades de funding en Publication
                    session.run("""
                        MATCH (p:Publication {doi: $doi})
                        SET p.funding_text = $funding_text,
                            p.funding_agencies = $funding_agencies,
                            p.grants_json = $grants_json,
                            p.funding_details_json = $funding_details_json,
                            p.funding_extracted = datetime(),
                            p.has_funding = CASE 
                                WHEN $funding_text <> "" OR size($funding_agencies) > 0 
                                THEN true 
                                ELSE false 
                            END
                    """, 
                    doi=doi,
                    funding_text=funding_data['funding_text'],
                    funding_agencies=funding_data['funding_agencies'],
                    grants_json=funding_data['grants_json'],
                    funding_details_json=funding_data['funding_details_json']
                    )
                    
                    # 2. Crear nodos FundingAgency si hay datos
                    for agency in funding_data['funding_agencies']:
                        if agency.strip():
                            session.run("""
                                MERGE (fa:FundingAgency {name: $agency})
                                WITH fa
                                MATCH (p:Publication {doi: $doi})
                                MERGE (p)-[:FUNDED_BY]->(fa)
                            """, agency=agency.strip(), doi=doi)
                    
                    # 3. Crear nodos Grant si hay datos detallados
                    for grant_json in funding_data['grants_json']:
                        try:
                            grant = json.loads(grant_json)
                            if grant.get('agency') and grant.get('string'):
                                session.run("""
                                    MERGE (g:Grant {agency: $agency, string: $string})
                                    SET g.agency_id = $agency_id,
                                        g.acronym = $acronym,
                                        g.updated = datetime()
                                    WITH g
                                    MATCH (p:Publication {doi: $doi})
                                    MERGE (p)-[:FUNDED_BY]->(g)
                                """, 
                                agency=grant['agency'],
                                string=grant['string'],
                                agency_id=grant.get('agency_id', ''),
                                acronym=grant.get('acronym', ''),
                                doi=doi)
                        except json.JSONDecodeError as e:
                            logger.warning(f"Error decodificando grant JSON: {e}")
                
                processed_count += 1
                
                if funding_data['funding_text'] or funding_data['grants_json']:
                    funding_found_count += 1
                    logger.info(f"✅ Funding data almacenada para: {doi}")
                else:
                    logger.info(f"ℹ️  No funding data para: {doi}")
                
                # Mostrar progreso cada 10 publicaciones
                if (i + 1) % 10 == 0:
                    progress_pct = ((i + 1) / total_publications) * 100
                    logger.info(f"📊 Progreso: {i+1}/{total_publications} ({progress_pct:.1f}%) - Funding encontrado: {funding_found_count}")
                    
            except Exception as e:
                error_count += 1
                logger.error(f"❌ Error procesando {doi}: {e}")
                
                # Marcar como procesada aunque haya error para no reprocesar
                with driver.session(database=config.neo4j_database) as session:
                    session.run("""
                        MATCH (p:Publication {doi: $doi})
                        SET p.funding_extracted = datetime(),
                            p.has_funding = false,
                            p.funding_error = $error
                    """, doi=doi, error=str(e))
                continue
        
        logger.info(f"\n📈 RESUMEN DE PROCESAMIENTO:")
        logger.info(f"  - Total procesadas: {processed_count}")
        logger.info(f"  - Con funding: {funding_found_count}")
        logger.info(f"  - Errores: {error_count}")
        logger.info(f"  - Tasa de éxito: {(funding_found_count/total_publications)*100:.1f}%")
        
        return True
                
    except ImportError as e:
        logger.error(f"Error importando pybliometrics: {e}")
        return False
    except Exception as e:
        logger.error(f"Error general: {e}")
        return False

def run_funding_extraction_corrected():
    """Función principal para extraer funding data - versión corregida"""
    logger.info("\n" + "="*80)
    logger.info("INICIANDO EXTRACCIÓN CORREGIDA DE FUNDING DATA")
    logger.info("="*80 + "\n")
    
    # Inicializar pybliometrics
    if not initialize_pybliometrics():
        logger.error("No se pudo inicializar pybliometrics. Saliendo...")
        return False
    
    # Conectar a Neo4j
    driver = connect_to_neo4j()
    if driver is None:
        logger.error("No se pudo conectar a Neo4j. Saliendo...")
        return False
    
    try:
        # CORREGIDO: Procesar TODAS las publicaciones disponibles
        # Cambiar limit=10 a limit=None para procesar todas
        publications = get_publications_for_funding_extraction(driver, limit=None)
        
        if not publications:
            logger.info("No se encontraron publicaciones para procesar (todas ya tienen funding data)")
            return True
        
        logger.info(f"📊 ESTADÍSTICAS ANTES DE PROCESAR:")
        with driver.session(database=config.neo4j_database) as session:
            # Ver cuántas ya tienen funding
            result = session.run("""
                MATCH (p:Publication)
                RETURN 
                    COUNT(p) AS total,
                    COUNT(CASE WHEN p.funding_extracted IS NOT NULL THEN 1 END) AS already_processed
            """)
            stats = result.single()
            logger.info(f"  - Total publicaciones: {stats['total']}")
            logger.info(f"  - Ya procesadas: {stats['already_processed']}")
            logger.info(f"  - Por procesar: {len(publications)}")
        
        # Extraer funding data para TODAS las publicaciones
        success = extract_and_store_funding_corrected(driver, publications)
        
        if success:
            # Verificar resultados finales
            with driver.session(database=config.neo4j_database) as session:
                result = session.run("""
                    MATCH (fa:FundingAgency)
                    RETURN COUNT(fa) AS funding_agencies
                """)
                fa_count = result.single()["funding_agencies"]
                
                result = session.run("""
                    MATCH (g:Grant)
                    RETURN COUNT(g) AS grants
                """)
                grant_count = result.single()["grants"]
                
                result = session.run("""
                    MATCH (p:Publication)
                    WHERE p.funding_text IS NOT NULL AND p.funding_text <> ""
                    RETURN COUNT(p) AS with_funding_text
                """)
                funding_text_count = result.single()["with_funding_text"]
                
                result = session.run("""
                    MATCH (p:Publication)
                    WHERE p.has_funding = true
                    RETURN COUNT(p) AS with_any_funding
                """)
                any_funding_count = result.single()["with_any_funding"]
                
                result = session.run("""
                    MATCH (p:Publication)
                    WHERE p.funding_extracted IS NOT NULL
                    RETURN COUNT(p) AS total_processed
                """)
                processed_count = result.single()["total_processed"]
                
                logger.info(f"\n📈 RESULTADOS FINALES:")
                logger.info(f"  - Total publicaciones procesadas: {processed_count}")
                logger.info(f"  - Funding Agencies creadas: {fa_count}")
                logger.info(f"  - Grants creados: {grant_count}")
                logger.info(f"  - Publicaciones con funding text: {funding_text_count}")
                logger.info(f"  - Publicaciones con ANY funding: {any_funding_count}")
                logger.info(f"  - Porcentaje con funding: {(any_funding_count/processed_count)*100:.1f}%")
        
        logger.info("\n🎉 Extracción de funding completada!")
        return True
        
    except Exception as e:
        logger.error(f"Error no controlado: {e}")
        import traceback
        logger.error(f"Traceback: {traceback.format_exc()}")
        return False
    finally:
        driver.close()
        logger.info("Conexión a Neo4j cerrada")

if __name__ == "__main__":
    run_funding_extraction_corrected()