#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import glob
import pandas as pd

# === RUTAS ===
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ECRII_DIR = os.path.join(BASE_DIR, "data_checkpoints", "ecrii_outputs")
CLUSTER_DIR = os.path.join(BASE_DIR, "cluster_exports")
OUTPUT_DIR = os.path.join(BASE_DIR, "cluster_exports_enriched")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# === 1) Seleccionar el CSV ECRII más reciente ===
ecrii_files = sorted(glob.glob(os.path.join(ECRII_DIR, "ecrii_results_*.csv")), key=os.path.getmtime, reverse=True)
if not ecrii_files:
    raise FileNotFoundError("No se encontraron CSV de ECRII en ecrii_outputs/")
ecrii_csv = ecrii_files[0]
print(f"📂 Usando ECRII más reciente: {os.path.basename(ecrii_csv)}")

df_ecrii = pd.read_csv(ecrii_csv)
ecrii_map = df_ecrii.set_index("author_id")["ECRII"].to_dict()
arch_map = df_ecrii.set_index("author_id")["archetype"].to_dict()

# === 2) Procesar cada CSV de cluster ===
cluster_files = glob.glob(os.path.join(CLUSTER_DIR, "scopus_cluster_*.csv"))
if not cluster_files:
    raise FileNotFoundError("No se encontraron CSV de clusters en cluster_exports/")

for cluster_file in cluster_files:
    df_cluster = pd.read_csv(cluster_file)

    ecrii_list = []
    arch_list = []

    for idx, row in df_cluster.iterrows():
        author_ids_raw = str(row.get("Author(s) ID", "")).strip()
        if not author_ids_raw:
            ecrii_list.append("")
            arch_list.append("")
            continue

        # Separar IDs y limpiar
        author_ids = [aid.strip() for aid in author_ids_raw.split(";") if aid.strip()]

        # Buscar ECRII y Archetype
        scores = [ecrii_map[aid] for aid in author_ids if aid in ecrii_map]
        archetypes = list({arch_map[aid] for aid in author_ids if aid in arch_map and pd.notna(arch_map[aid])})

        # Promedio de ECRII
        avg_score = round(sum(scores) / len(scores), 3) if scores else ""

        ecrii_list.append(avg_score)
        arch_list.append("; ".join(archetypes))

    # Añadir columnas
    df_cluster["ECRII"] = ecrii_list
    df_cluster["Archetype"] = arch_list

    # Guardar enriquecido
    out_path = os.path.join(OUTPUT_DIR, os.path.basename(cluster_file))
    df_cluster.to_csv(out_path, index=False)
    print(f"✅ Guardado enriquecido: {out_path}")

print("\n🎯 Proceso completado. Archivos enriquecidos en:", OUTPUT_DIR)
