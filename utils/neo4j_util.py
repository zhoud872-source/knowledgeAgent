import os
import logging
from neo4j import GraphDatabase

logger = logging.getLogger(__name__)
_neo4j_driver = None


def get_neo4j_driver()  -> GraphDatabase:

    global _neo4j_driver

    try:
        if _neo4j_driver is None:
            uri= os.getenv("NEO4J_URI")
            username= os.getenv("NEO4J_USERNAME")
            password= os.getenv("NEO4J_PASSWORD")

            logger.info("Connecting to Neo4j")

            _neo4j_driver = GraphDatabase.driver(uri=uri,
                                                 auth=(username,password))
            _neo4j_driver.verify_connectivity()

            logger.info("Connected to Neo4j")
            return _neo4j_driver
    except Exception as e:
        logger.error(f"Failed to connect to Neo4j: {e}")
        return None


