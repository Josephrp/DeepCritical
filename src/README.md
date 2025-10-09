# Bibliometric Intelligence Suite

> Sistema modular para análisis bibliométrico y de patentes integrando múltiples APIs y tecnologías de grafos.

[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![Neo4j](https://img.shields.io/badge/Neo4j-5.x-green.svg)](https://neo4j.com/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## 📋 Tabla de Contenidos

- [Características](#características)
- [Estructura del Proyecto](#estructura-del-proyecto)
- [Requisitos Previos](#requisitos-previos)
- [Instalación](#instalación)
- [Configuración](#configuración)
- [Uso](#uso)
- [Módulos](#módulos)
- [Flujos de Trabajo](#flujos-de-trabajo)
- [Documentación](#documentación)
- [Contribuir](#contribuir)
- [Licencia](#licencia)

## ✨ Características

### 🔬 Fuentes de Datos
- **OpenAlex**: Importación directa y gratuita de publicaciones científicas
- **Scopus**: Pipeline completo con enriquecimiento bibliométrico
- **Crossref**: Referencias cruzadas y métricas de citas
- **EPO + Google Patents**: Web scraping de patentes

### 🧠 Análisis
- **Semantic Clustering**: SBERT, SPECTER2, ChemBERTa, TF-IDF
- **Algoritmos**: K-means, Agglomerative, Ensemble
- **Representantes**: MMR (Maximal Marginal Relevance), FPS (Farthest Point Sampling)
- **Validación**: Bootstrap ARI, Silhouette, Davies-Bouldin, Calinski-Harabasz

### 📊 Visualización
- **VOSviewer**: Exportación optimizada para mapas de cocitación
- **Neo4j**: Grafo de conocimiento interactivo
- **PCA 2D**: Visualización de clusters

### 🎯 Métricas de Diversidad
- Participation Ratio
- Spectral Entropy
- Coverage Curves (adaptive)
- Term Distance Statistics

## 📁 Estructura del Proyecto