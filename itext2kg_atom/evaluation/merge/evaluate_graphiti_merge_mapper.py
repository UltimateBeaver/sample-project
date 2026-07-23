"""
This module's only target is to convert the knowledge graph obtained from ATOM or itext2kg framework to a suitable format 
for Graphiti framework, to be used in evaluate_graphiti_merge.py script
"""

class GraphData(dict):
    """
    A custom dictionary that supports both bracket notation graph_data["nodes"]
    and dot notation graph_data.nodes to prevent subscriptable/attribute errors.
    """
    def __init__(self, nodes, edges):
        super().__init__()
        self["nodes"] = nodes
        self["edges"] = edges
        self["relationships"] = edges
        self["relations"] = edges
        
    @property
    def nodes(self):
        return self["nodes"]
        
    @property
    def edges(self):
        return self["edges"]
        
    @property
    def relationships(self):
        return self["relationships"]

    @property
    def relations(self):
        return self["relations"]


class PropertiesWrapper(dict):
    """
    A hybrid dictionary/object wrapper that ensures properties can be accessed 
    via keys, attributes, or .get() calls as expected by the evaluation script.
    """
    def __init__(self, pydantic_props):
        props_dict = {}
        if pydantic_props:
            if hasattr(pydantic_props, "model_dump"):
                props_dict = pydantic_props.model_dump()
            elif hasattr(pydantic_props, "dict"):
                props_dict = pydantic_props.dict()
            elif isinstance(pydantic_props, dict):
                props_dict = pydantic_props
        
        super().__init__(props_dict)
        self._pydantic_props = pydantic_props

    def __getattr__(self, name):
        if name in self:
            return self[name]
        return getattr(self._pydantic_props, name)


class MappedNode:
    """
    Wraps an ATOM Entity object to expose standard graph attributes.
    """
    def __init__(self, entity):
        self._entity = entity
        self.name = entity.name
        self.label = entity.label
        self.labels = [entity.label, "Entity"] if entity.label else ["Entity"]
        
        # FIX: Explicitly define both public and protected properties 
        # using the hybrid dictionary/object wrapper
        wrapped_props = PropertiesWrapper(entity.properties)
        self.properties = wrapped_props
        self._properties = wrapped_props
        
        self.embedding = getattr(entity.properties, "embeddings", None)
        self.embeddings = getattr(entity.properties, "embeddings", None)

    def __getattr__(self, name):
        return getattr(self._entity, name)


class MappedEdge:
    """
    Wraps an ATOM Relationship object to expose standard edge attributes.
    """
    def __init__(self, relationship, start_node, end_node):
        self._relationship = relationship
        
        self.startEntity = start_node
        self.start_node = start_node
        self.source = start_node
        
        self.endEntity = end_node
        self.end_node = end_node
        self.target = end_node
        
        self.name = relationship.name
        self.type = relationship.name
        self.label = relationship.name
        
        # FIX: Explicitly define both public and protected properties here too
        wrapped_props = PropertiesWrapper(relationship.properties)
        self.properties = wrapped_props
        self._properties = wrapped_props
        
        self.embedding = getattr(relationship.properties, "embeddings", None)
        self.embeddings = getattr(relationship.properties, "embeddings", None)

    def __getattr__(self, name):
        return getattr(self._relationship, name)


def map_knowledge_graph_to_eval_format(kg):
    """
    Maps a KnowledgeGraph instance (from knowledge_graph.py schema)
    into the format expected by the evaluation script.
    """
    entity_to_mapped_node = {}
    mapped_nodes = []
    
    # 1. Map all entities to nodes
    for entity in kg.entities:
        mapped_node = MappedNode(entity)
        entity_to_mapped_node[entity] = mapped_node
        mapped_nodes.append(mapped_node)
        
    # 2. Map all relationships to edges
    mapped_edges = []
    for rel in kg.relationships:
        # Safeguard if start/end entities aren't pre-listed in kg.entities
        start_node = entity_to_mapped_node.get(rel.startEntity)
        if start_node is None:
            start_node = MappedNode(rel.startEntity)
            entity_to_mapped_node[rel.startEntity] = start_node
            mapped_nodes.append(start_node)
            
        end_node = entity_to_mapped_node.get(rel.endEntity)
        if end_node is None:
            end_node = MappedNode(rel.endEntity)
            entity_to_mapped_node[rel.endEntity] = end_node
            mapped_nodes.append(end_node)
            
        mapped_edge = MappedEdge(rel, start_node, end_node)
        mapped_edges.append(mapped_edge)
        
    return GraphData(nodes=mapped_nodes, edges=mapped_edges)