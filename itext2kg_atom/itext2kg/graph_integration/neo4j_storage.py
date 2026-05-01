from neo4j import GraphDatabase
import numpy as np
from typing import List, Optional
from itext2kg.atom.models import KnowledgeGraph
from itext2kg.graph_integration.storage_interface import GraphStorageInterface
from itext2kg.logging_config import get_logger

logger = get_logger(__name__)

class Neo4jStorage(GraphStorageInterface):
    """
    A class to integrate and manage graph data in a Neo4j database.
    """
    def __init__(self, uri: str, username: str, password: str, database: Optional[str] = None):
        """
        Initializes the Neo4jStorage with database connection parameters.
        
        Args:
            uri (str): URI for the Neo4j database.
            username (str): Username for database access.
            password (str): Password for database access.
            database (Optional[str]): The name of the database to connect to. Defaults to None, which uses the default database.
        """
        self.uri = uri
        self.username = username
        self.password = password
        self.database = database 
        self.driver = self.connect()
        
    def connect(self):
        """
        Establishes a connection to the Neo4j database.
        
        Returns:
            A Neo4j driver instance for executing queries.
        """
        driver = GraphDatabase.driver(self.uri, auth=(self.username, self.password))
        logger.debug("Created Neo4j driver: %s", driver)
        return driver

    def run_query(self, query: str):
        """
        Runs a Cypher query against the Neo4j database.
        
        Args:
            query (str): The Cypher query to run.
        """
        session = self.driver.session(database=self.database)
        try:
            session.run(query)
        finally:
            session.close()
            
    @staticmethod
    def transform_embeddings_to_str_list(embeddings: np.ndarray) -> str:
        """
        Transforms a NumPy array of embeddings into a comma-separated string.
        
        Args:
            embeddings (np.array): An array of embeddings.
        
        Returns:
            str: A comma-separated string of embeddings.
        """
        if embeddings is None:
            return ""
        return ",".join(list(embeddings.astype("str")))
    
    @staticmethod
    def transform_str_list_to_embeddings(embeddings: str) -> np.ndarray:
        """
        Transforms a comma-separated string of embeddings back into a NumPy array.
        
        Args:
            embeddings (str): A comma-separated string of embeddings.
        
        Returns:
            np.array: A NumPy array of embeddings.
        """
        if embeddings is None:
            return ""
        return np.array(embeddings.split(",")).astype(np.float64)
    
    @staticmethod
    def escape_str(s: str) -> str:
        """
        Escapes double quotes in a string for safe insertion into a Cypher query.
        """
        return s.replace('"', '\\"')
    
    @staticmethod
    def format_value(value) -> str:
        """
        Converts a value to a string and escapes it for safe Cypher insertion.
        """
        return Neo4jStorage.escape_str(str(value))
    
    @staticmethod
    def format_property_value(key: str, value) -> str:
        """
        Formats a property value for safe Cypher insertion, handling different data types.
        
        Args:
            key (str): The property key name
            value: The property value to format
            
        Returns:
            str: A formatted string for Cypher query
        """
        if key == "embeddings":
            return f'"{Neo4jStorage.transform_embeddings_to_str_list(value)}"'
        elif isinstance(value, list):
            # Handle list properties properly for Neo4j
            if not value:  # Empty list
                return "[]"
            # Convert list items to strings and create Neo4j list syntax
            formatted_items = []
            for item in value:
                if isinstance(item, str):
                    # Escape quotes in string items
                    escaped_item = Neo4jStorage.escape_str(item)
                    formatted_items.append(f'"{escaped_item}"')
                elif isinstance(item, (int, float)):
                    # For numbers, don't add quotes
                    formatted_items.append(str(item))
                else:
                    # For other types, convert to string and escape
                    escaped_item = Neo4jStorage.escape_str(str(item))
                    formatted_items.append(f'"{escaped_item}"')
            return f"[{', '.join(formatted_items)}]"
        elif isinstance(value, (int, float)):
            # For numeric values, don't add quotes
            return str(value)
        else:
            # Handle scalar values (strings, etc.)
            return f'"{Neo4jStorage.format_value(value)}"'

    def run_query_with_result(self, query: str):
        """
        Runs a Cypher query against the Neo4j database and returns results.
        
        Args:
            query (str): The Cypher query to run.
        
        Returns:
            List of records from the query result.
        """
        session = self.driver.session(database=self.database)
        try:
            result = session.run(query)
            return [record for record in result]
        finally:
            session.close()

    def create_nodes(self, knowledge_graph: KnowledgeGraph) -> List[str]:
        """
        Constructs Cypher queries for creating nodes in the graph database from a KnowledgeGraph object.
        
        Args:
            knowledge_graph (KnowledgeGraph): The KnowledgeGraph object containing entities.
        
        Returns:
            List[str]: A list of Cypher queries for node creation.
        """
        queries = []
        for node in knowledge_graph.entities:
            # Escape the node name and label if needed.
            node_name = Neo4jStorage.format_value(node.name)
            original_label = node.label
            node_label = Neo4jStorage.sanitize_label(node.label)
            
            # Log label sanitization for debugging
            if original_label != node_label:
                logger.info(f"Sanitized node label '{original_label}' to '{node_label}' for node '{node_name}'")
            
            properties = []
            for prop, value in node.properties.model_dump().items():
                if prop == "embeddings":
                    value_str = Neo4jStorage.transform_embeddings_to_str_list(value)
                    properties.append(f'SET n.{prop.replace(" ", "_")} = "{value_str}"')
                elif isinstance(value, (int, float)):
                    # For numeric values, don't add quotes
                    properties.append(f'SET n.{prop.replace(" ", "_")} = {value}')
                else:
                    value_str = Neo4jStorage.format_value(value)
                    # Build a SET clause for each property.
                    properties.append(f'SET n.{prop.replace(" ", "_")} = "{value_str}"')

            query = f'MERGE (n:{node_label} {{name: "{node_name}"}}) ' + ' '.join(properties)
            queries.append(query)
        return queries

    def create_relationships(self, knowledge_graph: KnowledgeGraph) -> List[str]:
        """
        Constructs Cypher queries for creating relationships in the graph database from a KnowledgeGraph object.
        
        Args:
            knowledge_graph (KnowledgeGraph): The KnowledgeGraph object containing relationships.
        
        Returns:
            List[str]: A list of Cypher queries for relationship creation.
        """
        rels = []
        for rel in knowledge_graph.relationships:
            # Escape start and end node names.
            original_start_label = rel.startEntity.label
            original_end_label = rel.endEntity.label
            original_rel_name = rel.name
            
            start_label = Neo4jStorage.sanitize_label(rel.startEntity.label)
            start_name = Neo4jStorage.format_value(rel.startEntity.name)
            end_label = Neo4jStorage.sanitize_label(rel.endEntity.label)
            end_name = Neo4jStorage.format_value(rel.endEntity.name)
            rel_name = Neo4jStorage.sanitize_relationship_type(rel.name)
            
            # Log sanitization for debugging
            if original_start_label != start_label:
                logger.info(f"Sanitized start entity label '{original_start_label}' to '{start_label}'")
            if original_end_label != end_label:
                logger.info(f"Sanitized end entity label '{original_end_label}' to '{end_label}'")
            if original_rel_name != rel_name:
                logger.info(f"Sanitized relationship type '{original_rel_name}' to '{rel_name}'")
            
            # Build property statements for setting all properties
            property_statements = []
            for key, value in rel.properties.model_dump().items():
                formatted_value = Neo4jStorage.format_property_value(key, value)
                property_key = key.replace(" ", "_")
                property_statements.append(f'r.{property_key} = {formatted_value}')
            
            # Build SET clause for properties
            set_clause = f'SET {", ".join(property_statements)}' if property_statements else ''
            
            # Use MERGE with only relationship name for uniqueness
            # ON MATCH SET will update existing relationships with new properties
            # This prefers incoming relationship properties over existing ones
            query = (
                f'MATCH (n:{start_label} {{name: "{start_name}"}}), '
                f'(m:{end_label} {{name: "{end_name}"}}) '
                f'MERGE (n)-[r:{rel_name}]->(m) '
                f'ON CREATE {set_clause} '
                f'ON MATCH {set_clause}'
            )
            rels.append(query)
            
        return rels


    def store_graph(self, knowledge_graph: KnowledgeGraph, parent_node_type: str = "Hadith") -> None:
        """
        Runs the necessary queries to store a graph structure in Neo4j from a KnowledgeGraph input.
        Also creates HAS_ENTITY relationships between existing nodes and knowledge graph entities.
        
        Args:
            knowledge_graph (KnowledgeGraph): The KnowledgeGraph object containing the graph structure.
            parent_node_type (str): The type of parent nodes to create HAS_ENTITY relationships with.
        """
        nodes = self.create_nodes(knowledge_graph=knowledge_graph)
        relationships = self.create_relationships(knowledge_graph=knowledge_graph)
        
        for node_query in nodes:
            self.run_query(node_query)

        for rel_query in relationships:
            self.run_query(rel_query)

    @staticmethod
    def sanitize_label(label: str) -> str:
        """
        Sanitizes a label to be Neo4j compliant.
        Neo4j labels cannot start with numbers and must follow specific naming conventions.
        
        Args:
            label (str): The original label to sanitize
            
        Returns:
            str: A sanitized label that is Neo4j compliant
        """
        if not label:
            return "Entity"
        
        # Remove any non-alphanumeric characters except underscores
        sanitized = ''.join(c for c in label if c.isalnum() or c == '_')
        
        # If the label starts with a number, prefix it with 'L'
        if sanitized and sanitized[0].isdigit():
            sanitized = 'L' + sanitized
        
        # If the label is empty after sanitization, use a default
        if not sanitized:
            sanitized = "Entity"
            
        return sanitized
    
    @staticmethod
    def sanitize_relationship_type(rel_type: str) -> str:
        """
        Sanitizes a relationship type to be Neo4j compliant.
        Neo4j relationship types cannot start with numbers and must follow specific naming conventions.
        
        Args:
            rel_type (str): The original relationship type to sanitize
            
        Returns:
            str: A sanitized relationship type that is Neo4j compliant
        """
        if not rel_type:
            return "RELATES_TO"
        
        # Remove any non-alphanumeric characters except underscores
        sanitized = ''.join(c for c in rel_type if c.isalnum() or c == '_')
        
        # If the relationship type starts with a number, prefix it with 'R'
        if sanitized and sanitized[0].isdigit():
            sanitized = 'R' + sanitized
        
        # If the relationship type is empty after sanitization, use a default
        if not sanitized:
            sanitized = "RELATES_TO"
            
        return sanitized
    
    def get_sanitization_mapping(self, knowledge_graph: KnowledgeGraph) -> dict:
        """
        Returns a mapping of original labels/relationship types to their sanitized versions.
        Useful for understanding how labels will be transformed before database insertion.
        
        Args:
            knowledge_graph (KnowledgeGraph): The KnowledgeGraph object to analyze
            
        Returns:
            dict: A dictionary with 'labels' and 'relationships' keys containing the mappings
        """
        label_mapping = {}
        relationship_mapping = {}
        
        # Map entity labels
        for entity in knowledge_graph.entities:
            original = entity.label
            sanitized = Neo4jStorage.sanitize_label(original)
            if original != sanitized:
                label_mapping[original] = sanitized
        
        # Map relationship types
        for rel in knowledge_graph.relationships:
            original = rel.name
            sanitized = Neo4jStorage.sanitize_relationship_type(original)
            if original != sanitized:
                relationship_mapping[original] = sanitized
        
        return {
            'labels': label_mapping,
            'relationships': relationship_mapping
        }

    def delete_graph(self) -> None:
        """
        Deletes all nodes and relationships from the Neo4j database.
        
        WARNING: This is destructive and cannot be undone. All data in the database will be removed.
        
        Returns:
            None
        """
        delete_query = "MATCH (n) DETACH DELETE n"
        try:
            self.run_query(delete_query)
            logger.info("Successfully deleted all nodes and relationships from Neo4j database")
        except Exception as e:
            logger.error(f"Error deleting graph from Neo4j: {e}")
            raise

    def read_graph(self) -> KnowledgeGraph:
        """
        Reads all nodes and relationships from Neo4j database and converts them into a KnowledgeGraph object.
        
        This method queries all existing nodes and relationships from the database and reconstructs
        the KnowledgeGraph structure, preserving all properties including embeddings and temporal information.
        
        Returns:
            KnowledgeGraph: A KnowledgeGraph object populated with all entities and relationships from Neo4j
            
        Raises:
            Exception: If there's an error querying the Neo4j database
        """
        from itext2kg.atom.models.entity import Entity, EntityProperties
        from itext2kg.atom.models.relationship import Relationship, RelationshipProperties
        
        entities = []
        relationships = []
        
        try:
            # Query to get all nodes with their properties
            nodes_query = "MATCH (n) RETURN n"
            node_records = self.run_query_with_result(nodes_query)
            
            # Build entities from nodes
            entity_dict = {}  # Map node element_id to Entity objects for relationship building
            for record in node_records:
                node = record["n"]
                
                # Extract node properties
                properties = dict(node.items())
                
                # Handle embeddings if present
                embeddings = None
                if "embeddings" in properties:
                    embeddings_str = properties.pop("embeddings")
                    if embeddings_str:
                        try:
                            embeddings = self.transform_str_list_to_embeddings(embeddings_str)
                        except Exception as e:
                            logger.warning(f"Could not parse embeddings: {e}")
                            embeddings = None
                
                # Create Entity with label from first node label
                entity = Entity(
                    name=properties.get("name", ""),
                    label=list(node.labels)[0] if node.labels else "Entity",
                    properties=EntityProperties(embeddings=embeddings)
                )
                
                entities.append(entity)
                entity_dict[node.element_id] = entity
            
            logger.info(f"Read {len(entities)} entities from Neo4j")
            
            # Query to get all relationships with their properties
            rels_query = "MATCH (n)-[r]->(m) RETURN n, r, m"
            rel_records = self.run_query_with_result(rels_query)
            
            # Build relationships
            for record in rel_records:
                start_node = record["n"]
                rel = record["r"]
                end_node = record["m"]
                
                # Get corresponding entities
                start_entity = entity_dict.get(start_node.element_id)
                end_entity = entity_dict.get(end_node.element_id)
                
                if start_entity and end_entity:
                    # Extract relationship properties
                    rel_properties = dict(rel.items())
                    
                    # Handle embeddings if present
                    embeddings = None
                    if "embeddings" in rel_properties:
                        embeddings_str = rel_properties.pop("embeddings")
                        if embeddings_str:
                            try:
                                embeddings = self.transform_str_list_to_embeddings(embeddings_str)
                            except Exception as e:
                                logger.warning(f"Could not parse relationship embeddings: {e}")
                                embeddings = None
                    
                    # Handle list properties (atomic_facts, t_obs, t_start, t_end)
                    atomic_facts = rel_properties.pop("atomic_facts", [])
                    t_obs = rel_properties.pop("t_obs", [])
                    t_start = rel_properties.pop("t_start", [])
                    t_end = rel_properties.pop("t_end", [])
                    
                    # Create RelationshipProperties
                    rel_props = RelationshipProperties(
                        embeddings=embeddings,
                        atomic_facts=atomic_facts if isinstance(atomic_facts, list) else [],
                        t_obs=t_obs if isinstance(t_obs, list) else [],
                        t_start=t_start if isinstance(t_start, list) else [],
                        t_end=t_end if isinstance(t_end, list) else []
                    )
                    
                    # Create Relationship
                    relationship = Relationship(
                        name=rel.type,
                        startEntity=start_entity,
                        endEntity=end_entity,
                        properties=rel_props
                    )
                    
                    relationships.append(relationship)
            
            logger.info(f"Read {len(relationships)} relationships from Neo4j")
            
            return KnowledgeGraph(entities=entities, relationships=relationships)
            
        except Exception as e:
            logger.error(f"Error reading graph from Neo4j: {e}")
            raise

    # def update_graph(self, 
    #                 knowledge_graph: KnowledgeGraph,
    #                 merge_function,
    #                 delete_existing_graph: bool = False,
    #                 ent_threshold: float = 0.8,
    #                 rel_threshold: float = 0.8,
    #                 max_workers: int = 8) -> KnowledgeGraph:
    #     """
    #     Updates the Neo4j database by merging existing graph with a new KnowledgeGraph.
        
    #     This method:
    #     1. Reads the existing KnowledgeGraph from Neo4j
    #     2. Merges it with the new KnowledgeGraph using the provided merge function
    #     3. Stores the merged result back to Neo4j
        
    #     Args:
    #         new_knowledge_graph (KnowledgeGraph): The new KnowledgeGraph to merge with existing data
    #         merge_function: The merge function to use (typically Atom.parallel_atomic_merge)
    #         ent_threshold (float): Entity matching threshold for merge (default: 0.8)
    #         rel_threshold (float): Relationship matching threshold for merge (default: 0.8)
    #         max_workers (int): Number of workers for parallel merge (default: 8)
            
    #     Returns:
    #         KnowledgeGraph: The merged KnowledgeGraph that was stored in Neo4j
            
    #     Raises:
    #         Exception: If there's an error reading, merging, or writing to Neo4j
    #     """
    #     try:
    #         existing_kg = None
            
    #         if delete_existing_graph:
    #             # Clear existing database and store merged graph
    #             logger.info("Deleting existing graph from Neo4j before storing merged graph...")
    #             self.delete_graph()
    #         else:
    #             # Read existing graph from Neo4j
    #             logger.info("Reading existing graph from Neo4j...")
    #             existing_kg = self.read_graph()
            
    #         if existing_kg is None or existing_kg.is_empty():
    #             logger.info("No existing graph in Neo4j. Storing new graph directly...")
    #             merged_kg = knowledge_graph
    #         else:
    #             # Merge existing and new graphs
    #             logger.info(f"Merging graphs: {len(existing_kg.entities)} existing entities + "
    #                       f"{len(knowledge_graph.entities)} new entities")
                
    #             # Create list of KGs to merge
    #             kgs_to_merge = [existing_kg, knowledge_graph]
                
    #             # Call the merge function (typically Atom.parallel_atomic_merge)
    #             merged_kg = merge_function(
    #                 kgs=kgs_to_merge,
    #                 ent_threshold=ent_threshold,
    #                 rel_threshold=rel_threshold,
    #                 max_workers=max_workers
    #             )
                
    #             logger.info(f"Merge complete: {len(merged_kg.entities)} merged entities, "
    #                       f"{len(merged_kg.relationships)} merged relationships")
            
    #         logger.info("Storing merged graph to Neo4j...")
    #         self.store_graph(merged_kg)
            
    #         logger.info("Graph update complete!")
    #         return merged_kg
            
    #     except Exception as e:
    #         logger.error(f"Error updating graph: {e}")
    #         raise