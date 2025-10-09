import os
from dotenv import load_dotenv
from neo4j import GraphDatabase

load_dotenv()  # lee .env del proyecto

uri = os.getenv("NEO4J_URI")
user = os.getenv("NEO4J_USER")
pwd  = os.getenv("NEO4J_PASSWORD")
db   = os.getenv("NEO4J_DATABASE","neo4j")

driver = GraphDatabase.driver(uri, auth=(user, pwd))
with driver.session(database=db) as s:
    print(s.run("RETURN 1 AS ok").single())
driver.close()
