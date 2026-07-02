import asyncio
import json
import networkx as nx
import numpy as np
import logging
from itext2kg_atom.itext2kg.atom.models.knowledge_graph import KnowledgeGraph
from itext2kg_atom.itext2kg import Neo4jStorage
from itext2kg_atom.itext2kg.logging_config import get_logger, setup_logging
from itext2kg_atom.itext2kg.atom import Atom

from env_config import neo4j_uri, neo4j_username, neo4j_password
from models.models import get_default_model, get_default_embedding_model
from sanity_checks.test_config import validate_config

# Set up logger for this module
logger = get_logger(__name__)

def _convert_embeddings_to_arrays(obj):
    if isinstance(obj, dict):
        if "embeddings" in obj and isinstance(obj["embeddings"], list):
            obj["embeddings"] = np.array(obj["embeddings"])
        for v in obj.values():
            _convert_embeddings_to_arrays(v)
    elif isinstance(obj, list):
        for item in obj:
            _convert_embeddings_to_arrays(item)

# Custom encoder to transform numpy embedding arrays into standard lists for JSON
class GraphEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super().default(obj)

class GraphUtility():
    def __init__(self):
        self.export_file_path = "./data/knowledge_graph/knowledge_graph"
        self.import_file_path = self.export_file_path

        try:
            self.neo4j = Neo4jStorage(uri=neo4j_uri, username=neo4j_username, password=neo4j_password)
            self.atom = Atom(llm_model=get_default_model(), embeddings_model=get_default_embedding_model())
        except Exception as e:
            logger.error("Could not instanciate GraphUtility")
            raise RuntimeError(f"Detailed error log: {e}")
    
    def _fully_process_kg(self, kg: KnowledgeGraph) -> KnowledgeGraph:
        """
        In-place normalizes all names and labels within the KnowledgeGraph
        to prevent downstream hash mismatch bugs inside the ATOM framework.
        """
        for entity in kg.entities:
            entity.process()
        for rel in kg.relationships:
            rel.process()
            if rel.startEntity:
                rel.startEntity.process()
            if rel.endEntity:
                rel.endEntity.process()
        return kg

    async def export_graph(self):
        # Parse graph into your Pydantic model
        logger.info("📥 Extracting knowledge graph from Neo4j...")
        kg = KnowledgeGraph.from_neo4j(self.neo4j)
        
        if kg.is_empty():
            logger.warning("❌ Graph is empty. Nothing to export.")
            return

        # ---- FORMAT 1: STRUCTURED JSON ----
        # We use model_dump to extract data while preserving list structures
        graph_data = kg.model_dump()
        with open(f"{self.export_file_path}.json", "w", encoding="utf-8") as f:
            json.dump(graph_data, f, indent=2, cls=GraphEncoder)
        logger.info(f"✅ Exported to Structured JSON: {self.export_file_path}.json")

        # ---- FORMAT 2: GRAPHML (via NetworkX) ----
        logger.info("🔄 Converting to NetworkX for GraphML export...")
        G = nx.MultiDiGraph()
        
        for entity in kg.entities:
            # GraphML doesn't support raw lists/arrays; convert embeddings to string representation
            props = {}
            if entity.properties and entity.properties.embeddings is not None:
                props['embeddings'] = str(entity.properties.embeddings.tolist())
            G.add_node(entity.name, label=entity.label, **props)
            
        for rel in kg.relationships:
            # Flatten temporal and atomic arrays to string representations for GraphML compatibility
            props = {
                "t_obs": str(rel.properties.t_obs),
                "atomic_facts": str(rel.properties.atomic_facts),
                "t_start": str(rel.properties.t_start) if rel.properties.t_start else "",
                "t_end": str(rel.properties.t_end) if rel.properties.t_end else "",
            }
            if rel.properties.embeddings is not None:
                props['embeddings'] = str(rel.properties.embeddings.tolist())
                
            G.add_edge(rel.startEntity.name, rel.endEntity.name, key=rel.name, **props)
            
        nx.write_graphml(G, f"{self.export_file_path}.graphml")
        logger.info(f"✅ Exported to GraphML: {self.export_file_path}.graphml")

    def read_graph_from_file(self) -> KnowledgeGraph:
        logger.info("📤 Loading JSON file...")
        try:
            with open(f"{self.import_file_path}.json", "r", encoding="utf-8") as f:
                data = json.load(f)
        except Exception as e: 
            logger.error(f"Could not import {self.import_file_path}.json: {e}")
            raise RuntimeError(e)
        
        # Mutate list embeddings back into numpy arrays before passing to Pydantic
        _convert_embeddings_to_arrays(data)
        # Re-validate dictionary data back into your Pydantic classes
        kg = KnowledgeGraph.model_validate(data)
        return kg

    def store_graph(self, kg: KnowledgeGraph):
        logger.info("⚙️ Injecting graph data into Neo4j container...")
        self.neo4j.store_graph(knowledge_graph=kg)
        logger.info("✅ Import complete!")
    
    async def import_graph(self):
        kg = self.read_graph_from_file()
        await self.delete_graph()
        self.store_graph(kg)

    
    async def append_graph(self):
        existing_kg = self.neo4j.read_graph()
        imported_kg = self.read_graph_from_file()

        if existing_kg.is_empty() or imported_kg.is_empty():
            logger.warning("One of the two knowledge graphs is empty. Append operation aborted")
        else:
            # Pre-process both graphs to neutralize the framework's hash lookup bug
            existing_kg = self._fully_process_kg(existing_kg)
            imported_kg = self._fully_process_kg(imported_kg)

            kg = self.atom.parallel_atomic_merge(kgs=[existing_kg, imported_kg])
            await self.delete_graph()
            self.store_graph(kg)


    async def delete_graph(self):
        logger.info("🧹 Cleaning existing graph...")
        self.neo4j.delete_graph()



async def main():
    # Initialize logging configuration (only used if you are running this script as a standalone utility)
    setup_logging(
        log_file="graph_utility.log", 
        level="DEBUG",  # Your itext2kg logs at DEBUG level
        console_output=True,
        langchain_level="WARNING"  # Keep external library logs at WARNING for performance
    )

    config_ok = await validate_config()
    if not config_ok:
        logger.error("Configuration validation failed. Exiting.")
        return

    kg_utility = GraphUtility()
    choice = -1

    print("-"*90)
    print("Welcome to the Knowledge Graph Utility")
    print("This utility will specifically operates on these 2 files:")
    print(f"- export_file_path:   {kg_utility.export_file_path}.json")
    print(f"- import_file_path:   {kg_utility.import_file_path}.json")

    while(choice != 0):
        print("-"*90)
        print(f"1 - export the current Knowledge Graph from Neo4j (⚠️  This overwrites the {kg_utility.export_file_path}.json !)")
        print("2 - import an existing Knowledge Graph into Neo4j (⚠️  This overwrites the current Neo4j graph !)")
        print("3 - append an existing Knowledge Graph into Neo4j (ℹ️  Merges the two graphs using ATOM framework)")
        print("4 - delete the current Knowledge Graph from Neo4j (⚠️  Unrecoverable, no confirmation prompts)")
        print("0 - quit")
        print("-"*90)
        print("Type the number of an option: ")

        try:
            choice = int(input())
        except:
            print("Please enter an integer")
        
        if choice == 1:
            await kg_utility.export_graph()
        elif choice == 2:
            await kg_utility.import_graph()
        elif choice == 3:
            await kg_utility.append_graph()
        elif choice == 4:
            await kg_utility.delete_graph()
        elif choice != 0:
            print("Invalid choice")
    
    print("Logs saved into graph_utility.log")

if __name__ == "__main__":
    asyncio.run(main())