"""
OPEN ORCHESTRATOR - Neo4j Knowledge Graph Builder
Incluye paso opcional de extracción de funding.
"""

import os
import sys
import logging
import importlib
from config_manager import get_config

# Configuración global
config = get_config()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - [ORCHESTRATOR] - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler("orchestrator_open.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("orchestrator_open")


# === Helpers ===
def run_script(module_name, func_name="main"):
    """Cargar dinámicamente un módulo y ejecutar su función principal"""
    try:
        logger.info(f"🚀 Iniciando {module_name}.{func_name}() ...")
        module = importlib.import_module(module_name)
        func = getattr(module, func_name)
        func()
        logger.info(f"✅ {module_name} completado")
        return True
    except Exception as e:
        logger.error(f"❌ Error en {module_name}: {e}")
        return False


def run_funding_step():
    """Ejecutar funding extraction corregido"""
    try:
        from funding import run_funding_extraction_corrected
        return run_funding_extraction_corrected()
    except Exception as e:
        logger.error(f"❌ Error en funding.py: {e}")
        return False


# === Orchestrator ===
def main():
    print("\n" + "="*80)
    print("OPEN ORCHESTRATOR - Neo4j Knowledge Graph Builder")
    print("="*80 + "\n")

    # 1. Input query
    query = input("Ingrese su consulta de búsqueda (Scopus TITLE-ABS-KEY...):\n> ").strip()
    if not query:
        print("No se proporcionó query. Saliendo...")
        sys.exit(1)

    print("\nEjemplo de queries válidas:")
    print('  TITLE-ABS-KEY("Alzheimer" OR "amyloid")')
    print('  TITLE-ABS-KEY("plastic recycling" AND "toxicity")\n')

    # 2. Confirmar limpiar base
    limpiar = input("¿Desea limpiar completamente la base de datos Neo4j? (s/n): ").lower() == "s"

    # 3. Paso funding
    funding_enabled = input("¿Desea ejecutar la extracción de funding (paso 4b)? (s/n): ").lower() == "s"

    # 4. Modelo de embeddings
    print("\nSeleccione modelo de embeddings:")
    print("  1. all-MiniLM-L6-v2 (rápido, general)")
    print("  2. scibert_scivocab_uncased (científico)")
    print("  3. S-PubMedBert-MLM (biomédico recomendado)")
    choice = input("> ").strip()
    if choice == "2":
        os.environ["EMBEDDING_MODEL"] = "allenai/scibert_scivocab_uncased"
    elif choice == "3":
        os.environ["EMBEDDING_MODEL"] = "pritamdeka/S-PubMedBert-MLM"
    else:
        os.environ["EMBEDDING_MODEL"] = "sentence-transformers/all-MiniLM-L6-v2"

    # 5. Vector search al final
    interactive = input("¿Ejecutar búsqueda vectorial interactiva al final? (s/n): ").lower() == "s"

    # === Resumen ===
    print("\n" + "="*80)
    print("📋 RESUMEN DE CONFIGURACIÓN")
    print("="*80)
    print(f"Query: {query}")
    print(f"Limpiar base: {limpiar}")
    print(f"Funding: {'Sí' if funding_enabled else 'No'}")
    print(f"Embeddings model: {os.environ['EMBEDDING_MODEL']}")
    print(f"Vector search final: {'Sí' if interactive else 'No'}")
    print("="*80 + "\n")

    # Confirmar
    if input("¿Desea continuar con esta configuración? (s/n): ").lower() != "s":
        print("Cancelado por el usuario")
        sys.exit(0)

    # === Ejecución de pasos ===
    # Script 1
    ok = run_script("script1_neo4j_rebuild", "main")
    if not ok:
        sys.exit(1)

    # Script 2
    run_script("script2_author_fix", "main")

    # Script 3 y 4 (completar datos y crossref)
    run_script("script3_complete_data", "main")
    run_script("script4_crossref", "main")

    # Paso 4b Funding (opcional)
    if funding_enabled:
        run_funding_step()

    # Script 5: vector setup
    run_script("script5_vector_setup", "main")

    # Script 6: embeddings
    run_script("script6_embeddings", "main")

    # Script 7: vector search
    if interactive:
        from script7_vector_search_cli import interactive_mode
        interactive_mode()

    logger.info("🎉 ORCHESTRATOR OPEN finalizado correctamente")


if __name__ == "__main__":
    main()
