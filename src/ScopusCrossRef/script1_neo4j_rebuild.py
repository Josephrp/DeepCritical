"""
Script 1: Reconstruir completamente la base de datos Neo4j con todas las relaciones
entre publicaciones, autores, revistas, abstracts, países, etc.
CORREGIDO: Extracción mejorada de autores, abstracts y afiliaciones usando ScopusSearch
"""

import os
import json
import time
import requests
import pandas as pd
import numpy as np
from tqdm import tqdm
from neo4j import GraphDatabase
from datetime import datetime
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
        logging.FileHandler("neo4j_rebuild.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

def connect_to_neo4j():
    """Conectar a la base de datos Neo4j"""
    try:
        driver = GraphDatabase.driver(config.neo4j_uri, auth=(config.neo4j_user, config.neo4j_password))
        # Verificamos la conexión
        with driver.session() as session:
            session.run("RETURN 1")
        logger.info(f"Conexión exitosa a Neo4j - Base de datos: {config.neo4j_database}")
        return driver
    except Exception as e:
        logger.error(f"Error al conectar a Neo4j: {e}")
        return None

def clear_database(driver):
    """Limpiar completamente la base de datos"""
    logger.info("--- LIMPIANDO BASE DE DATOS ---")
    
    with driver.session(database=config.neo4j_database) as session:
        try:
            session.run("MATCH (n) DETACH DELETE n")
            logger.info("Base de datos limpiada correctamente")
        except Exception as e:
            logger.error(f"Error al limpiar la base de datos: {e}")

def create_constraints(driver):
    """Crear restricciones e índices en Neo4j"""
    logger.info("--- CREANDO RESTRICCIONES E ÍNDICES ---")
    
    with driver.session(database=config.neo4j_database) as session:
        constraints = [
            "CREATE CONSTRAINT IF NOT EXISTS FOR (p:Publication) REQUIRE p.eid IS UNIQUE",
            "CREATE CONSTRAINT IF NOT EXISTS FOR (a:Author) REQUIRE a.id IS UNIQUE",
            "CREATE CONSTRAINT IF NOT EXISTS FOR (k:Keyword) REQUIRE k.name IS UNIQUE",
            "CREATE CONSTRAINT IF NOT EXISTS FOR (sk:SemanticKeyword) REQUIRE sk.name IS UNIQUE",
            "CREATE CONSTRAINT IF NOT EXISTS FOR (j:Journal) REQUIRE j.name IS UNIQUE",
            "CREATE CONSTRAINT IF NOT EXISTS FOR (c:Country) REQUIRE c.name IS UNIQUE",
            "CREATE CONSTRAINT IF NOT EXISTS FOR (i:Institution) REQUIRE i.name IS UNIQUE",
            "CREATE CONSTRAINT IF NOT EXISTS FOR (g:Grant) REQUIRE (g.agency, g.string) IS UNIQUE",
            "CREATE CONSTRAINT IF NOT EXISTS FOR (fa:FundingAgency) REQUIRE fa.name IS UNIQUE"
        ]
        
        indexes = [
            "CREATE INDEX IF NOT EXISTS FOR (p:Publication) ON (p.year)",
            "CREATE INDEX IF NOT EXISTS FOR (p:Publication) ON (p.citedBy)",
            "CREATE INDEX IF NOT EXISTS FOR (j:Journal) ON (j.name)",
            "CREATE INDEX IF NOT EXISTS FOR (p:Publication) ON (p.year, p.citedBy)",
            "CREATE INDEX IF NOT EXISTS FOR (k:Keyword) ON (k.name)",
            "CREATE INDEX IF NOT EXISTS FOR (i:Institution) ON (i.name)"
        ]
        
        for constraint in constraints:
            try:
                session.run(constraint)
                logger.info(f"Restricción creada: {constraint}")
            except Exception as e:
                logger.error(f"Error creando restricción: {e}")
        
        for index in indexes:
            try:
                session.run(index)
                logger.info(f"Índice creado: {index}")
            except Exception as e:
                logger.error(f"Error creando índice: {e}")

def extract_authors_and_affiliations_from_search(pub):
    """
    Extrae autores y afiliaciones correctamente del objeto ScopusSearch
    según la documentación de Stack Overflow
    """
    authors_data = []
    
    # Verificar si hay datos de autores
    if not hasattr(pub, 'author_ids') or not pub.author_ids:
        logger.warning("No se encontraron author_ids en la publicación")
        return [], [], [], []
    
    # Separar IDs de autores y afiliaciones como indica la documentación
    authors = pub.author_ids.split(";") if pub.author_ids else []
    affs = pub.author_afids.split(";") if hasattr(pub, 'author_afids') and pub.author_afids else []
    
    # Obtener nombres de autores si están disponibles
    author_names = []
    if hasattr(pub, 'author_names') and pub.author_names:
        author_names = pub.author_names.split(";")
    elif hasattr(pub, 'authors') and pub.authors:
        author_names = pub.authors.split(";")
    
    # Limpiar y procesar los datos
    authors = [a.strip() for a in authors if a.strip()]
    affs = [a.strip() for a in affs if a.strip()]
    author_names = [a.strip() for a in author_names if a.strip()]
    
    # Si no hay nombres, usar IDs como placeholder
    if not author_names:
        author_names = [f"Author_{aid}" for aid in authors]
    
    # Asegurar que las listas tengan la misma longitud
    max_len = max(len(authors), len(author_names))
    while len(authors) < max_len:
        authors.append("")
    while len(author_names) < max_len:
        author_names.append("")
    while len(affs) < max_len:
        affs.append("")
    
    # Crear lista de datos de autores con afiliaciones
    for i in range(max_len):
        if authors[i]:  # Solo procesar si hay ID de autor
            # Las afiliaciones múltiples están separadas por guión según la documentación
            author_affs = affs[i].split("-") if affs[i] else []
            author_affs = [aff.strip() for aff in author_affs if aff.strip()]
            
            authors_data.append({
                'id': authors[i],
                'name': author_names[i] if i < len(author_names) else f"Author_{authors[i]}",
                'affiliations': author_affs
            })
    
    return authors_data

def get_affiliation_details(affiliation_id):
    """
    Obtiene detalles de la afiliación usando la API de Scopus
    """
    try:
        from pybliometrics.scopus import AffiliationRetrieval
        
        if not affiliation_id or affiliation_id == "":
            return None
            
        aff = AffiliationRetrieval(affiliation_id)
        
        return {
            'id': affiliation_id,
            'name': aff.affiliation_name if hasattr(aff, 'affiliation_name') else '',
            'country': aff.country if hasattr(aff, 'country') else '',
            'city': aff.city if hasattr(aff, 'city') else '',
            'address': aff.address if hasattr(aff, 'address') else ''
        }
    except Exception as e:
        logger.warning(f"No se pudo obtener detalles de afiliación {affiliation_id}: {e}")
        return {
            'id': affiliation_id,
            'name': f"Institution_{affiliation_id}",
            'country': '',
            'city': '',
            'address': ''
        }

def initialize_search(query: str):
    """Realizar búsqueda inicial en Scopus y guardar resultados"""
    logger.info("--- REALIZANDO BÚSQUEDA INICIAL EN SCOPUS ---")
    
    # Crear nombre único basado en hash de la query
    query_hash = hashlib.md5(query.encode()).hexdigest()[:8]
    search_file = os.path.join(config.data_dir, f"search_results_{query_hash}.json")
    
    logger.info(f"Hash de query: {query_hash}")
    logger.info(f"Archivo de resultados: {search_file}")
    
    if os.path.exists(search_file):
        logger.info(f"Usando archivo de resultados existente: {search_file}")
        
        try:
            results_df = pd.read_json(search_file)
            logger.info(f"Cargados {len(results_df)} resultados de búsqueda")
            return results_df, query_hash
        except Exception as e:
            logger.error(f"Error al cargar resultados de búsqueda: {e}")
    
    try:
        from pybliometrics.scopus import ScopusSearch
        
        logger.info(f"Ejecutando búsqueda en Scopus: {query}")
        
        # CORREGIDO: Usar view='COMPLETE' para obtener más datos
        search_results = ScopusSearch(query, refresh=True, view='COMPLETE')
        
        if not hasattr(search_results, 'results'):
            logger.warning("La búsqueda retornó un objeto sin resultados")
            return None, query_hash
            
        if hasattr(search_results, 'get_results_size'):
            results_size = search_results.get_results_size()
            logger.info(f"Se encontraron {results_size} resultados")
        
        results_df = pd.DataFrame(search_results.results)
        
        if results_df is None or results_df.empty:
            logger.warning("La búsqueda retornó un DataFrame vacío")
            return None, query_hash
            
        results_df.to_json(search_file)
        logger.info(f"Resultados guardados en: {search_file}")
        
        return results_df, query_hash
        
    except Exception as e:
        logger.error(f"Error al realizar búsqueda en Scopus: {e}")
        logger.error(f"Detalles: {str(e)}")
        import traceback
        logger.error(f"Traceback: {traceback.format_exc()}")
        return None, None

def enrich_publication_data(df, max_papers=None, query_hash="default"):
    """
    Obtener datos adicionales para cada publicación
    CORREGIDO: Usar datos del ScopusSearch en lugar de AbstractRetrieval cuando sea posible
    """
    logger.info("--- ENRIQUECIENDO DATOS DE PUBLICACIONES ---")
    
    enriched_file = os.path.join(config.data_dir, f"enriched_data_{query_hash}.json")
    
    if os.path.exists(enriched_file):
        logger.info(f"Usando archivo de datos enriquecidos existente: {enriched_file}")
        
        try:
            enriched_df = pd.read_json(enriched_file)
            logger.info(f"Cargados {len(enriched_df)} registros enriquecidos")
            return enriched_df
        except Exception as e:
            logger.error(f"Error al cargar datos enriquecidos: {e}")
    
    if df is None or len(df) == 0:
        logger.error("No hay datos para enriquecer")
        return None
    
    try:
        from pybliometrics.scopus import AbstractRetrieval
        
        partial_files = [f for f in os.listdir(config.data_dir) 
                        if f.startswith(f"enriched_data_temp_{query_hash}_")]
        
        enriched_data = []
        start_idx = 0
        
        if partial_files:
            partial_files.sort(key=lambda x: int(x.split("_")[-1].split(".")[0]))
            latest_file = os.path.join(config.data_dir, partial_files[-1])
            
            try:
                temp_df = pd.read_json(latest_file)
                enriched_data = temp_df.to_dict('records')
                start_idx = len(enriched_data)
                logger.info(f"Continuando desde el checkpoint {latest_file} con {start_idx} registros ya procesados")
            except Exception as e:
                logger.error(f"Error al cargar checkpoint parcial: {e}")
                start_idx = 0
        
        papers_to_process = len(df) if max_papers is None else min(len(df), max_papers)
        logger.info(f"Enriqueciendo datos para {papers_to_process - start_idx} publicaciones adicionales...")
        
        for i, row in df.iloc[start_idx:papers_to_process].iterrows():
            try:
                # CORREGIDO: Primero extraer lo que se puede del ScopusSearch
                logger.info(f"Procesando {i+1}/{papers_to_process}: {row.get('title', 'Sin título')[:50]}...")
                
                # Extraer autores y afiliaciones del resultado de búsqueda
                authors_data = extract_authors_and_affiliations_from_search(row)
                
                # Extraer datos básicos del resultado de búsqueda
                keywords = []
                if hasattr(row, 'authkeywords') and row.authkeywords:
                    keywords.extend(row.authkeywords.split(";"))
                if hasattr(row, 'idxterms') and row.idxterms:
                    keywords.extend(row.idxterms.split(";"))
                
                # Limpiar keywords
                keywords = [k.strip().lower() for k in keywords if k and k.strip()]
                
                # Extraer afiliaciones detalladas
                institutions = []
                countries = []
                affiliations_detailed = []
                
                for author_data in authors_data:
                    for aff_id in author_data['affiliations']:
                        if aff_id:
                            aff_details = get_affiliation_details(aff_id)
                            if aff_details:
                                affiliations_detailed.append(aff_details)
                                if aff_details['name']:
                                    institutions.append(aff_details['name'])
                                if aff_details['country']:
                                    countries.append(aff_details['country'])
                
                # Remover duplicados
                institutions = list(set(institutions))
                countries = list(set(countries))
                
                # Intentar obtener abstract con AbstractRetrieval si está disponible
                abstract_text = ""
                identifier = row.get('doi', row.get('eid', None))
                
                if identifier:
                    try:
                        time.sleep(0.5)  # Rate limiting
                        ab = AbstractRetrieval(identifier, view='FULL')
                        
                        if hasattr(ab, 'abstract') and ab.abstract:
                            abstract_text = ab.abstract
                        elif hasattr(ab, 'description') and ab.description:
                            abstract_text = ab.description
                            
                        # Obtener funding si está disponible
                        grants = []
                        funding_agencies = []
                        if hasattr(ab, 'funding') and ab.funding:
                            for funding in ab.funding:
                                grant_info = {
                                    'agency': funding.agency if hasattr(funding, 'agency') and funding.agency else '',
                                    'agency_id': funding.agency_id if hasattr(funding, 'agency_id') and funding.agency_id else '',
                                    'string': funding.string if hasattr(funding, 'string') and funding.string else '',
                                    'acronym': funding.acronym if hasattr(funding, 'acronym') and funding.acronym else ''
                                }
                                grants.append(grant_info)
                                
                                if grant_info['agency']:
                                    funding_agencies.append(grant_info['agency'])
                                    
                    except Exception as e:
                        logger.warning(f"No se pudo obtener abstract para {identifier}: {e}")
                        grants = []
                        funding_agencies = []
                
                # Crear registro con datos corregidos
                record = {
                    'eid': row.get('eid', ''),
                    'doi': row.get('doi', ''),
                    'title': row.get('title', ''),
                    'authors': [author['name'] for author in authors_data],
                    'author_ids': [author['id'] for author in authors_data],
                    'year': row.get('coverDate', '')[:4] if row.get('coverDate') else '',
                    'source_title': row.get('publicationName', ''),
                    'cited_by': int(row.get('citedby_count', 0)) if row.get('citedby_count') else 0,
                    'abstract': abstract_text,
                    'keywords': keywords,
                    'affiliations': affiliations_detailed,
                    'institutions': institutions,
                    'countries': countries,
                    'grants': grants if 'grants' in locals() else [],
                    'funding_agencies': funding_agencies if 'funding_agencies' in locals() else [],
                    'affiliation': countries[0] if countries else '',  # Para compatibilidad
                    'source_id': row.get('source_id', ''),
                    'authors_with_affiliations': authors_data  # Datos completos de autores
                }
                
                enriched_data.append(record)
                
                # Debug: Log datos extraídos
                logger.info(f"✓ Título: {record['title'][:50]}...")
                logger.info(f"✓ Autores: {len(authors_data)} encontrados")
                logger.info(f"✓ Abstract: {'Sí' if abstract_text else 'No'} ({len(abstract_text)} chars)")
                logger.info(f"✓ Keywords: {len(keywords)} encontradas")
                logger.info(f"✓ Instituciones: {len(institutions)} encontradas: {institutions[:3] if institutions else []}")
                logger.info(f"✓ Países: {len(countries)} encontrados: {countries}")
                
                # Guardar checkpoint cada 5 registros
                if (len(enriched_data) % 5 == 0) or (i + 1 == papers_to_process):
                    temp_df = pd.DataFrame(enriched_data)
                    temp_file = os.path.join(config.data_dir, f"enriched_data_temp_{query_hash}_{len(enriched_data)}.json")
                    temp_df.to_json(temp_file)
                    logger.info(f"Checkpoint guardado: {temp_file}")
                
            except Exception as e:
                logger.error(f"Error procesando publicación {i}: {e}")
                import traceback
                logger.error(f"Traceback: {traceback.format_exc()}")
                continue
        
        if not enriched_data:
            logger.error("No se pudo enriquecer ninguna publicación")
            return None
            
        enriched_df = pd.DataFrame(enriched_data)
        enriched_df.to_json(enriched_file)
        logger.info(f"Datos enriquecidos guardados en: {enriched_file}")
        
        return enriched_df
        
    except ImportError as e:
        logger.error(f"Error de importación: {e}. Instalando pybliometrics...")
        try:
            import subprocess
            subprocess.check_call(["pip", "install", "pybliometrics"])
            logger.info("pybliometrics instalado, reintentando enriquecimiento...")
            return enrich_publication_data(df, max_papers, query_hash)
        except Exception as install_e:
            logger.error(f"Error al instalar pybliometrics: {install_e}")
            return None
    except Exception as e:
        logger.error(f"Error general en enriquecimiento de datos: {e}")
        import traceback
        logger.error(f"Traceback: {traceback.format_exc()}")
        return None

def import_data_to_neo4j(driver, data_df, query_hash="default"):
    """Importar datos a Neo4j usando transacciones explícitas con datos corregidos"""
    logger.info("--- IMPORTANDO DATOS A NEO4J ---")
    
    if data_df is None or len(data_df) == 0:
        logger.error("No hay datos para importar")
        return
    
    progress_file = os.path.join(config.data_dir, f"import_progress_{query_hash}.json")
    start_index = 0
    
    if os.path.exists(progress_file):
        try:
            with open(progress_file, 'r') as f:
                progress_data = json.load(f)
                start_index = progress_data.get('last_index', 0)
            logger.info(f"Continuando importación desde el índice {start_index}")
        except Exception as e:
            logger.error(f"Error al cargar progreso de importación: {e}")
    
    total_publications = len(data_df)
    batch_size = config.batch_size_import
    
    max_pubs = config.max_papers_import
    if max_pubs <= 0:
        max_pubs = total_publications - start_index
    
    end_index = min(start_index + max_pubs, total_publications)
    logger.info(f"Importando publicaciones {start_index+1}-{end_index} de {total_publications}")
    
    with driver.session(database=config.neo4j_database) as session:
        for i in range(start_index, end_index, batch_size):
            batch_end = min(i + batch_size, end_index)
            batch = data_df.iloc[i:batch_end]
            
            with session.begin_transaction() as tx:
                for _, pub in batch.iterrows():
                    eid = pub.get('eid', '')
                    if not eid:
                        continue
                        
                    # Crear publicación
                    tx.run("""
                        MERGE (p:Publication {eid: $eid})
                        SET p.title = $title,
                            p.year = $year,
                            p.doi = $doi,
                            p.citedBy = $cited_by,
                            p.abstract = $abstract
                    """, 
                    eid=eid,
                    title=pub.get('title', ''),
                    year=pub.get('year', ''),
                    doi=pub.get('doi', ''),
                    cited_by=int(pub.get('cited_by', 0)),
                    abstract=pub.get('abstract', '')
                    )
                    
                    # Crear Journal
                    journal_name = pub.get('source_title')
                    if journal_name:
                        tx.run("""
                            MERGE (j:Journal {name: $journal_name})
                            WITH j
                            MATCH (p:Publication {eid: $eid})
                            MERGE (p)-[:PUBLISHED_IN]->(j)
                        """,
                        journal_name=journal_name,
                        eid=eid
                        )
                    
                    # CORREGIDO: Crear Authors con datos mejorados
                    authors_with_affs = pub.get('authors_with_affiliations', [])
                    if authors_with_affs:
                        for author_data in authors_with_affs:
                            author_id = author_data.get('id')
                            author_name = author_data.get('name')
                            
                            if author_id and author_name:
                                # Crear autor
                                tx.run("""
                                    MERGE (a:Author {id: $author_id})
                                    SET a.name = $author_name
                                    WITH a
                                    MATCH (p:Publication {eid: $eid})
                                    MERGE (a)-[:AUTHORED]->(p)
                                """,
                                author_id=author_id,
                                author_name=author_name,
                                eid=eid
                                )
                                
                                # Crear afiliaciones del autor
                                for aff_id in author_data.get('affiliations', []):
                                    if aff_id:
                                        tx.run("""
                                            MERGE (a:Author {id: $author_id})
                                            MERGE (aff:Affiliation {id: $aff_id})
                                            MERGE (a)-[:AFFILIATED_WITH]->(aff)
                                        """,
                                        author_id=author_id,
                                        aff_id=aff_id
                                        )
                    
                    # Crear Keywords
                    keywords = pub.get('keywords', [])
                    if isinstance(keywords, list):
                        for keyword in keywords:
                            if keyword and isinstance(keyword, str):
                                tx.run("""
                                    MERGE (k:Keyword {name: $keyword})
                                    WITH k
                                    MATCH (p:Publication {eid: $eid})
                                    MERGE (p)-[:HAS_KEYWORD]->(k)
                                """,
                                keyword=keyword.lower(),
                                eid=eid
                                )
                    
                    # CORREGIDO: Crear Institutions con datos detallados
                    affiliations_detailed = pub.get('affiliations', [])
                    if isinstance(affiliations_detailed, list):
                        for aff in affiliations_detailed:
                            if isinstance(aff, dict) and aff.get('name'):
                                # Crear institución
                                tx.run("""
                                    MERGE (i:Institution {name: $institution})
                                    SET i.id = $aff_id,
                                        i.country = $country,
                                        i.city = $city,
                                        i.address = $address
                                    WITH i
                                    MATCH (p:Publication {eid: $eid})
                                    MERGE (p)-[:AFFILIATED_WITH]->(i)
                                """,
                                institution=aff['name'],
                                aff_id=aff.get('id', ''),
                                country=aff.get('country', ''),
                                city=aff.get('city', ''),
                                address=aff.get('address', ''),
                                eid=eid
                                )
                                
                                # Crear país si existe
                                if aff.get('country'):
                                    tx.run("""
                                        MERGE (c:Country {name: $country})
                                        MERGE (i:Institution {name: $institution})
                                        MERGE (i)-[:LOCATED_IN]->(c)
                                        WITH c
                                        MATCH (p:Publication {eid: $eid})
                                        MERGE (p)-[:AFFILIATED_WITH]->(c)
                                    """,
                                    country=aff['country'],
                                    institution=aff['name'],
                                    eid=eid
                                    )
            
            # Guardar progreso
            with open(progress_file, 'w') as f:
                json.dump({'last_index': batch_end}, f)
            
            logger.info(f"Importadas publicaciones {i+1}-{batch_end}/{end_index}")
            
    return end_index

def initialize_pybliometrics():
    """Inicializar pybliometrics para usar la API de Scopus"""
    logger.info("--- INICIALIZANDO PYBLIOMETRICS ---")
    
    try:
        import pybliometrics
        pybliometrics.init()
        logger.info("Scopus API inicializado exitosamente con pybliometrics.init()")
        return True
    except Exception as e:
        logger.error(f"Error al inicializar Scopus API: {e}")
        return False

def load_enriched_data(query_hash="default"):
    """Cargar los datos enriquecidos desde el checkpoint"""
    checkpoint_file = os.path.join(config.data_dir, f"enriched_data_{query_hash}.json")
    
    if not os.path.exists(checkpoint_file):
        logger.warning(f"No se encontró el archivo {checkpoint_file}")
        return None
    
    logger.info(f"Cargando datos desde: {checkpoint_file}")
    
    try:
        with open(checkpoint_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        if isinstance(data, dict):
            if all(isinstance(key, str) and key.isdigit() for key in data.keys()):
                df = pd.DataFrame.from_dict(data, orient='index')
                logger.info(f"Cargados {len(df)} registros (diccionario indexado)")
                return df
            else:
                df = pd.DataFrame([data])
                logger.info(f"Cargado 1 registro (diccionario de registro)")
                return df
                
        elif isinstance(data, list):
            df = pd.DataFrame(data)
            logger.info(f"Cargados {len(df)} registros (lista)")
            return df
        else:
            logger.error(f"Formato de datos no reconocido: {type(data)}")
            return None
    except Exception as e:
        logger.error(f"Error al cargar archivo como JSON: {e}")
        
        try:
            df = pd.read_json(checkpoint_file)
            logger.info(f"Cargados {len(df)} registros (pandas)")
            return df
        except Exception as e2:
            logger.error(f"Error al cargar archivo como pandas DataFrame: {e2}")
            return None

def run_script1(search_query: str, clear_db: bool = False, interactive: bool = False):
    """Función principal del script 1"""
    logger.info("\n" + "="*80)
    logger.info("INICIANDO SCRIPT 1: RECONSTRUCCIÓN DE LA BASE DE DATOS NEO4J")
    logger.info("="*80 + "\n")
    
    # Crear hash para esta query
    query_hash = hashlib.md5(search_query.encode()).hexdigest()[:8]
    logger.info(f"Hash de query: {query_hash}")
    
    # Asegurar directorio
    os.makedirs(config.data_dir, exist_ok=True)
    
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
        # Limpiar base de datos si se requiere
        if clear_db:
            clear_database(driver)
        
        # Crear restricciones e índices
        create_constraints(driver)
        
        # Realizar búsqueda en Scopus
        search_results, query_hash = initialize_search(search_query)
        
        if search_results is None:
            logger.error("No se pudieron obtener resultados de búsqueda. Saliendo...")
            return False
        
        # Enriquecer datos con hash único
        max_enrich = config.max_papers_enrich if config.max_papers_enrich > 0 else None
        enriched_df = enrich_publication_data(search_results, max_papers=max_enrich, query_hash=query_hash)
        
        if enriched_df is None:
            enriched_df = load_enriched_data(query_hash)
            if enriched_df is None:
                logger.error("No se pudieron obtener datos enriquecidos. Saliendo...")
                return False
        
        # Importar datos a Neo4j con hash único
        import_data_to_neo4j(driver, enriched_df, query_hash)
        
        logger.info(f"\n¡Script 1 completado con éxito para query {query_hash}!")
        return True
        
    except Exception as e:
        logger.error(f"Error no controlado en script 1: {e}")
        import traceback
        logger.error(f"Traceback: {traceback.format_exc()}")
        return False
    finally:
        if driver:
            driver.close()
            logger.info("Conexión a Neo4j cerrada")

if __name__ == "__main__":
    # Si se ejecuta directamente, pedir query de búsqueda
    query = input("Ingrese la consulta de búsqueda para Scopus: ")
    clear_db = input("¿Limpiar base de datos? (s/n): ").lower() == 's'
    run_script1(query, clear_db, interactive=True)