"""
OPENALEX DIRECT IMPORTER - CONTINUACIÓN AUTOMÁTICA (Robusto)
- Importa DIRECTAMENTE a Neo4j sin guardar en disco local.
- Continúa desde el último cursor (ImportProgress).
- Maneja duplicados verificando existencia ANTES de importar.
- Límite de importación configurable; <=0 significa ILIMITADO.
- BÚSQUEDA EN TÍTULO con filtro preciso (sin stop words).
"""

import os
import time
import requests
from typing import Dict, Any, Optional
from neo4j import GraphDatabase
from datetime import datetime
import logging
import html
import random

# Cargar variables de entorno
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

# Configuración de logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger("openalex_importer")

class Config:
    def __init__(self):
        self.neo4j_uri = os.getenv("NEO4J_URI")
        self.neo4j_user = os.getenv("NEO4J_USER")
        self.neo4j_password = os.getenv("NEO4J_PASSWORD")
        self.neo4j_database = os.getenv("NEO4J_DATABASE")
        self.openalex_email = os.getenv("OPENALEX_EMAIL")
        self.max_papers_import = int(os.getenv("MAX_PAPERS_IMPORT", "-1"))
        self.batch_size = int(os.getenv("BATCH_SIZE", "25"))
        self.backoff_base = float(os.getenv("BACKOFF_BASE", "1.0"))
        self.backoff_max = float(os.getenv("BACKOFF_MAX", "30.0"))

config = Config()

class OpenAlexDirectImporter:
    def __init__(self):
        self.driver = None
        self.headers = {
            # User-Agent con mailto recomendado por OpenAlex
            "User-Agent": f"openalex-importer/1.2 (mailto:{config.openalex_email})"
        }

    # --------------------------- Neo4j ---------------------------

    def connect_to_neo4j(self):
        """Conectar a Neo4j"""
        try:
            self.driver = GraphDatabase.driver(
                config.neo4j_uri, auth=(config.neo4j_user, config.neo4j_password)
            )
            with self.driver.session(database=config.neo4j_database) as session:
                session.run("RETURN 1")
            logger.info(f"✓ Conexión exitosa - DB: {config.neo4j_database}")
            return True
        except Exception as e:
            logger.error(f"Error conectando: {e}")
            return False

    def setup_database(self):
        """Setup inicial - solo constraints necesarios"""
        logger.info("🔧 Setup inicial...")
        with self.driver.session(database=config.neo4j_database) as session:
            constraints = [
                "CREATE CONSTRAINT IF NOT EXISTS FOR (p:Paper) REQUIRE p.openalex_id IS UNIQUE",
                "CREATE CONSTRAINT IF NOT EXISTS FOR (a:Author) REQUIRE a.openalex_id IS UNIQUE",
                "CREATE CONSTRAINT IF NOT EXISTS FOR (i:Institution) REQUIRE i.openalex_id IS UNIQUE",
                "CREATE CONSTRAINT IF NOT EXISTS FOR (j:Journal) REQUIRE j.openalex_id IS UNIQUE",
                "CREATE CONSTRAINT IF NOT EXISTS FOR (c:Concept) REQUIRE c.openalex_id IS UNIQUE",
                "CREATE CONSTRAINT IF NOT EXISTS FOR (co:Country) REQUIRE co.code IS UNIQUE"
            ]
            session.run("""
                MERGE (p:ImportProgress {id: 'main'})
                ON CREATE SET p.last_cursor='*', p.total_imported=0, p.query='', p.last_update=datetime()
            """)
            for stmt in constraints:
                try:
                    session.run(stmt)
                except Exception as e:
                    logger.warning(f"Constraint warning: {e}")
        logger.info("✓ Setup completado")

    def get_last_progress(self) -> Dict[str, Any]:
        with self.driver.session(database=config.neo4j_database) as session:
            rec = session.run("""
                MATCH (p:ImportProgress {id:'main'})
                RETURN p.last_cursor AS cursor, p.total_imported AS imported, p.query AS query, p.last_update AS last_update
            """).single()
            if rec:
                return {
                    "cursor": rec["cursor"],
                    "imported": rec["imported"],
                    "query": rec["query"],
                    "last_update": str(rec["last_update"])
                }
            return {"cursor": "*", "imported": 0, "query": "", "last_update": "never"}

    def save_progress(self, cursor: str, imported: int, search_query: str):
        with self.driver.session(database=config.neo4j_database) as session:
            session.run("""
                MERGE (p:ImportProgress {id:'main'})
                SET p.last_cursor=$cursor,
                    p.total_imported=$imported,
                    p.query=$search_query,
                    p.last_update=datetime()
            """, cursor=cursor, imported=imported, search_query=search_query)

    def paper_exists(self, openalex_id: str) -> bool:
        """Verifica si un paper ya existe en la BD"""
        with self.driver.session(database=config.neo4j_database) as session:
            result = session.run("""
                MATCH (p:Paper {openalex_id: $oid})
                RETURN count(p) > 0 AS exists
            """, oid=openalex_id).single()
            return result["exists"] if result else False

    # --------------------------- Utilidades ---------------------------

    def reconstruct_abstract(self, inverted_index: Optional[Dict[str, Any]]) -> str:
        """Reconstruir abstract desde inverted index de OpenAlex"""
        try:
            if not inverted_index:
                return ""
            max_pos = 0
            for positions in inverted_index.values():
                if positions:
                    max_pos = max(max_pos, max(positions))
            if max_pos == 0:
                return ""
            words = [""] * (max_pos + 1)
            for word, positions in inverted_index.items():
                for pos in positions:
                    if 0 <= pos <= max_pos:
                        words[pos] = word
            return " ".join(w for w in words if w).strip()
        except Exception:
            return ""

    def phrase_in_text(self, phrase: str, title: str, abstract: str) -> bool:
        """Verifica si la frase exacta está en título o abstract (case-insensitive)"""
        phrase_lower = phrase.lower().strip('"')
        title_lower = (title or "").lower()
        abstract_lower = (abstract or "").lower()
        
        return phrase_lower in title_lower or phrase_lower in abstract_lower

    def _safe_get_journal_from_locations(self, paper_data: Dict[str, Any]) -> Dict[str, Optional[str]]:
        """Devuelve {journal_id, journal_name, publisher} manejando None en primary_location/locations/source"""
        primary = paper_data.get("primary_location") or {}
        source = primary.get("source") or {}

        journal_id = source.get("id")
        journal_name = source.get("display_name")
        publisher = source.get("host_organization_name")

        if not journal_id:
            locations = paper_data.get("locations") or []
            for loc in locations:
                src = (loc or {}).get("source") or {}
                if src.get("id"):
                    journal_id = src.get("id")
                    journal_name = src.get("display_name")
                    publisher = src.get("host_organization_name")
                    break

        return {
            "journal_id": journal_id,
            "journal_name": journal_name,
            "publisher": publisher
        }

    # --------------------------- Importación ---------------------------

    def import_paper_safe(self, paper_data: Dict[str, Any], required_phrase: str = None) -> bool:
        """Importar/actualizar un paper y sus vínculos (robusto a None)"""
        openalex_id = paper_data.get("id")
        if not openalex_id:
            return False

        # VERIFICAR SI YA EXISTE
        if self.paper_exists(openalex_id):
            return False

        try:
            doi = paper_data.get("doi")
            if doi:
                doi = doi.replace("https://doi.org/", "").lower()
            title = paper_data.get("title") or paper_data.get("display_name") or ""
            title = html.unescape(title)
            abstract = self.reconstruct_abstract(paper_data.get("abstract_inverted_index"))

            # FILTRO LOCAL: verificar que la frase esté en título/abstract
            if required_phrase:
                if not self.phrase_in_text(required_phrase, title, abstract):
                    return False

            journal_info = self._safe_get_journal_from_locations(paper_data)

            authorships = paper_data.get("authorships") or []
            concepts = paper_data.get("concepts") or []
            referenced = paper_data.get("referenced_works") or []

            with self.driver.session(database=config.neo4j_database) as session:
                def _tx(tx):
                    # Paper
                    tx.run("""
                        MERGE (p:Paper {openalex_id:$openalex_id})
                        SET p.doi=$doi,
                            p.title=$title,
                            p.abstract=$abstract,
                            p.publication_year=$pub_year,
                            p.cited_by_count=$cited_by,
                            p.type=$type,
                            p.language=$lang,
                            p.updated_at=datetime()
                    """, openalex_id=openalex_id,
                         doi=doi,
                         title=title,
                         abstract=abstract,
                         pub_year=paper_data.get("publication_year"),
                         cited_by=paper_data.get("cited_by_count", 0),
                         type=paper_data.get("type") or "",
                         lang=paper_data.get("language"))

                    # Journal (si disponible)
                    if journal_info["journal_id"]:
                        tx.run("""
                            MERGE (j:Journal {openalex_id:$jid})
                            SET j.display_name=$jname,
                                j.publisher=$publisher
                            WITH j
                            MATCH (p:Paper {openalex_id:$pid})
                            MERGE (p)-[:PUBLISHED_IN]->(j)
                        """, jid=journal_info["journal_id"],
                             jname=journal_info["journal_name"] or "",
                             publisher=journal_info["publisher"],
                             pid=openalex_id)

                    # Autores y afiliaciones
                    for authorship in authorships:
                        author_info = (authorship or {}).get("author") or {}
                        author_id = author_info.get("id")
                        if not author_id:
                            continue
                        tx.run("""
                            MERGE (a:Author {openalex_id:$aid})
                            SET a.display_name=$aname,
                                a.orcid=$orcid
                            WITH a
                            MATCH (p:Paper {openalex_id:$pid})
                            MERGE (a)-[r:AUTHORED]->(p)
                            SET r.author_position=$apos
                        """, aid=author_id,
                             aname=author_info.get("display_name") or "",
                             orcid=author_info.get("orcid"),
                             pid=openalex_id,
                             apos=(authorship or {}).get("author_position"))

                        # Instituciones (afiliaciones)
                        for inst in (authorship.get("institutions") or []):
                            if not inst:
                                continue
                            inst_id = inst.get("id")
                            if not inst_id:
                                continue
                            tx.run("""
                                MERGE (i:Institution {openalex_id:$iid})
                                SET i.display_name=$iname,
                                    i.country_code=$cc
                                WITH i
                                MATCH (a:Author {openalex_id:$aid})
                                MERGE (a)-[:AFFILIATED_WITH]->(i)
                            """, iid=inst_id,
                                 iname=inst.get("display_name") or "",
                                 cc=inst.get("country_code"),
                                 aid=author_id)

                    # Conceptos
                    for concept in concepts:
                        if not concept:
                            continue
                        cid = concept.get("id")
                        if not cid:
                            continue
                        tx.run("""
                            MERGE (c:Concept {openalex_id:$cid})
                            SET c.display_name=$cname,
                                c.level=$clevel
                            WITH c
                            MATCH (p:Paper {openalex_id:$pid})
                            MERGE (p)-[r:HAS_CONCEPT]->(c)
                            SET r.score=$cscore
                        """, cid=cid,
                             cname=concept.get("display_name") or "",
                             clevel=concept.get("level", 0),
                             pid=openalex_id,
                             cscore=concept.get("score", 0.0))

                    # Citas (crear solo nodos referenciados y relación)
                    for ref_id in referenced[:10]:
                        if not ref_id:
                            continue
                        tx.run("""
                            MERGE (ref:Paper {openalex_id:$rid})
                            WITH ref
                            MATCH (p:Paper {openalex_id:$pid})
                            MERGE (p)-[:CITES]->(ref)
                        """, rid=ref_id, pid=openalex_id)

                session.execute_write(_tx)

            return True

        except Exception as e:
            logger.error(f"Error importando {openalex_id}: {e}")
            return False

    # --------------------------- Bucle principal ---------------------------

    def run_import(self, query: str, resume: bool = True):
        """Ejecutar importación directa con continuación"""
        print(f"\n🚀 IMPORTACIÓN DIRECTA: '{query}'")

        if not self.connect_to_neo4j():
            return False

        self.setup_database()

        progress = self.get_last_progress()
        start_cursor = "*"
        total_imported = 0

        if resume and progress["query"] == query:
            start_cursor = progress["cursor"]
            total_imported = progress["imported"]
            print(f"🔄 CONTINUANDO desde: imported={total_imported}, last_update={progress['last_update']}")
        else:
            print("🆕 IMPORTACIÓN NUEVA")

        q = query.strip().strip('"')  # Limpiar comillas del usuario

        url = "https://api.openalex.org/works"
        params = {
            "per_page": 200,
            "cursor": start_cursor,
            "mailto": config.openalex_email,
            "select": "id,doi,title,display_name,publication_year,type,abstract_inverted_index,authorships,primary_location,locations,concepts,referenced_works,cited_by_count,language"
        }

        # BÚSQUEDA MEJORADA: filtrar stop words antes de buscar
        STOP_WORDS = {'a', 'an', 'and', 'are', 'as', 'at', 'be', 'by', 'for', 'from', 
                      'has', 'he', 'in', 'is', 'it', 'its', 'of', 'on', 'that', 'the', 
                      'to', 'was', 'will', 'with'}

        words = [w for w in q.split() if w.lower() not in STOP_WORDS]

        if not words:
            print("❌ Error: la búsqueda solo contiene stop words")
            return False

        title_filters = ",".join([f"title.search:{word}" for word in words])
        base_filter = "type:article"

        params["filter"] = f"{base_filter},{title_filters}"
        
        print(f"🔍 Búsqueda: frase original '{q}'")
        print(f"🔍 Palabras clave (sin stop words): {words}")
        print(f"🔍 Filtro aplicado: {params['filter']}")
        print(f"⚠️  Se verificará localmente que la frase exacta esté en título/abstract")

        new_imported = 0
        fetched = 0
        skipped = 0
        max_papers = config.max_papers_import
        unlimited = (max_papers is None) or (max_papers <= 0)
        print(f"🔍 Límite de importación: {'ilimitado' if unlimited else max_papers}")

        try:
            page = 0
            while unlimited or (total_imported + new_imported) < max_papers:
                page += 1
                print(f"\n📡 Página {page} - Total: {total_imported + new_imported}, Nuevos: {new_imported}, Skipped: {skipped}")

                # Request con backoff
                attempt = 0
                while True:
                    try:
                        response = requests.get(url, params=params, headers=self.headers, timeout=60)

                        if response.status_code == 429:
                            attempt += 1
                            delay = min(config.backoff_max, config.backoff_base * (2 ** attempt)) * (0.5 + random.random()/2)
                            print(f"⏳ 429 rate-limited. Reintentando en {delay:.1f}s...")
                            time.sleep(delay)
                            continue

                        if response.status_code == 403:
                            attempt += 1
                            delay = min(config.backoff_max, config.backoff_base * (2 ** attempt))
                            print(f"🚫 403 Forbidden. Reintentando en {delay:.1f}s...")
                            time.sleep(delay)
                            if attempt >= 5:
                                response.raise_for_status()
                            continue

                        response.raise_for_status()
                        data = response.json()
                        break
                    except requests.RequestException as e:
                        attempt += 1
                        delay = min(config.backoff_max, config.backoff_base * (2 ** attempt))
                        print(f"❌ Error en request: {e} -> reintento en {delay:.1f}s")
                        time.sleep(delay)
                        if attempt >= 5:
                            print("❌ Fallaron demasiados intentos. Abortando.")
                            return False

                results = data.get("results") or []
                meta = data.get("meta") or {}

                print(f"📄 Resultados obtenidos: {len(results)}")
                if not results:
                    print("✓ No hay más resultados")
                    break

                # Procesar cada paper
                for item in results:
                    if (not unlimited) and ((total_imported + new_imported) >= max_papers):
                        break

                    openalex_id = (item or {}).get("id")
                    if not openalex_id:
                        continue

                    ok = self.import_paper_safe(item, required_phrase=q)
                    if ok:
                        new_imported += 1
                        t = (item.get("title") or item.get("display_name") or "Sin título")[:50]
                        print(f"✅ Paper {total_imported + new_imported}: {t}...")
                    else:
                        skipped += 1
                        if skipped % 50 == 0:
                            print(f"⏭️  Skipped {skipped} (duplicados o sin frase exacta)...")

                    fetched += 1

                    # Guardar progreso cada batch_size
                    if new_imported > 0 and (new_imported % max(1, config.batch_size) == 0):
                        current_cursor = meta.get("next_cursor", params.get("cursor"))
                        self.save_progress(current_cursor or params["cursor"], total_imported + new_imported, query)
                        print(f"💾 Progreso guardado: {total_imported + new_imported} total")

                # Siguiente página
                next_cursor = meta.get("next_cursor")
                if not next_cursor:
                    print("✓ No hay más páginas")
                    break

                params["cursor"] = next_cursor
                time.sleep(0.2)

            # Guardar progreso final
            if 'data' in locals() and data is not None:
                final_cursor = (data.get("meta") or {}).get("next_cursor", params.get("cursor", "*"))
                self.save_progress(final_cursor, total_imported + new_imported, query)

            print(f"\n✅ COMPLETADO - Total en DB: {total_imported + new_imported}")
            print(f"   Nuevos importados: {new_imported}")
            print(f"   Duplicados/filtrados: {skipped}")
            print(f"   Papers procesados: {fetched}")
            return True

        except KeyboardInterrupt:
            print(f"\n⚠️ INTERRUMPIDO - Progreso guardado: {total_imported + new_imported}")
            return True
        except Exception as e:
            print(f"❌ Error: {e}")
            import traceback
            traceback.print_exc()
            return False
        finally:
            if self.driver:
                self.driver.close()

    # --------------------------- Stats ---------------------------

    def get_stats(self):
        if not self.connect_to_neo4j():
            return

        with self.driver.session(database=config.neo4j_database) as session:
            stats = {}
            stats['Papers']  = session.run("MATCH (p:Paper) RETURN count(p) AS c").single()['c']
            stats['Authors'] = session.run("MATCH (a:Author) RETURN count(a) AS c").single()['c']
            stats['Citations'] = session.run("MATCH ()-[r:CITES]->() RETURN count(r) AS c").single()['c']
            progress = self.get_last_progress()

        print(f"\n📊 ESTADÍSTICAS:")
        for k, v in stats.items():
            print(f"  • {k}: {v:,}")

        print(f"\n📋 PROGRESO:")
        print(f"  • Último query: {progress['query']}")
        print(f"  • Total importado: {progress['imported']:,}")
        print(f"  • Última actualización: {progress['last_update']}")

        if self.driver:
            self.driver.close()

# --------------------------- CLI ---------------------------

def main():
    importer = OpenAlexDirectImporter()

    print("\n🔬 OPENALEX DIRECT IMPORTER")
    print("="*40)
    print("1. Importar (nuevo o continuar)")
    print("2. Ver estadísticas")

    option = input("\n👉 Opción (1-2): ").strip()

    if option == "1":
        query = input("👉 Query de búsqueda: ").strip()
        if query:
            importer.run_import(query, resume=True)
    elif option == "2":
        importer.get_stats()

if __name__ == "__main__":
    main()