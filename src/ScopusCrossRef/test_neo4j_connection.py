#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Test de conexión a Neo4j para Vector Store"""

import sys
from neo4j import GraphDatabase
from config_manager import get_config

config = get_config()

def test_connection():
    print("\n" + "="*70)
    print("TEST NEO4J - VECTOR STORE READINESS")
    print("="*70 + "\n")
    
    print(f"Configuration:")
    print(f"   URI: {config.neo4j_uri}")
    print(f"   User: {config.neo4j_user}")
    print(f"   Database: {config.neo4j_database}")
    print(f"   Vector Store: {'Enabled' if config.vector_store_enabled else 'Disabled'}")
    
    try:
        print("\nConnecting to Neo4j...")
        driver = GraphDatabase.driver(config.neo4j_uri, auth=(config.neo4j_user, config.neo4j_password))
        
        with driver.session(database=config.neo4j_database) as session:
            # Test connection
            session.run("RETURN 1").single()
            print("Connection successful!")
            
            # Check version
            version = session.run("CALL dbms.components() YIELD versions RETURN versions[0] as v").single()["v"]
            print(f"\nNeo4j Version: {version}")
            
            major, minor = map(int, version.split('.')[:2])
            if major >= 5 and minor >= 11:
                print("Vector indices SUPPORTED")
            else:
                print(f"WARNING: Vector indices NOT supported. Need Neo4j 5.11+ (you have {version})")
            
            # Check data
            print("\nExisting data:")
            labels = ["Publication", "Author", "Journal", "Concept", "Keyword"]
            total_pubs = 0
            
            for label in labels:
                try:
                    count = session.run(f"MATCH (n:{label}) RETURN count(n) as c").single()["c"]
                    if count > 0:
                        print(f"   {label}: {count:,}")
                        if label == "Publication":
                            total_pubs = count
                except:
                    pass
            
            # Check embeddings
            try:
                emb_count = session.run("""
                    MATCH (p:Publication)
                    WHERE p.abstract_embedding IS NOT NULL
                    RETURN count(p) as c
                """).single()["c"]
                print(f"\nEmbeddings status:")
                print(f"   Publications with embeddings: {emb_count:,}")
                print(f"   Publications without embeddings: {total_pubs - emb_count:,}")
                if emb_count == 0:
                    print("   ACTION: Run script6_embeddings.py to generate embeddings")
            except:
                print(f"\nEmbeddings status:")
                print(f"   Publications with embeddings: 0")
                print(f"   ACTION: Run script6_embeddings.py to generate embeddings")
            
            # Check indices
            print("\nExisting indices:")
            try:
                indices = session.run("SHOW INDEXES")
                vector_indices = []
                other_indices = []
                
                for idx in indices:
                    name = idx.get("name", "N/A")
                    idx_type = idx.get("type", "N/A")
                    state = idx.get("state", "N/A")
                    
                    if "vector" in idx_type.lower() or "vector" in name.lower():
                        vector_indices.append(f"   {name} ({state})")
                    else:
                        other_indices.append(f"   {name} ({idx_type})")
                
                if vector_indices:
                    print("   Vector indices:")
                    for idx in vector_indices:
                        print(idx)
                else:
                    print("   No vector indices found")
                    print("   ACTION: Run script5_vector_setup.py to create vector indices")
                
                if other_indices:
                    print(f"   Other indices: {len(other_indices)} found")
                    
            except Exception as e:
                print(f"   Could not list indices: {e}")
        
        driver.close()
        
        print("\n" + "="*70)
        print("TEST COMPLETED")
        print("="*70 + "\n")
        
        return True
        
    except Exception as e:
        print(f"\nERROR: {e}")
        print("\nPossible causes:")
        print("   1. Neo4j is not running")
        print("   2. Incorrect credentials in .env")
        print("   3. Wrong URI (check port 7687)")
        return False

if __name__ == "__main__":
    success = test_connection()
    sys.exit(0 if success else 1)