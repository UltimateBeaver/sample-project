# iText2KG - Detailed Method Reference

## ATOM Module Reference

### Class: `Atom` [atom/atom.py]

#### Constructor
```python
def __init__(self, llm_model, embeddings_model) -> None
    Attributes:
    - self.matcher: GraphMatcher()
    - self.llm_output_parser: LangchainOutputParser(llm_model, embeddings_model)
```
**Line**: L14-22

---

#### Method: `extract_quintuples()`
```python
async def extract_quintuples(
    self,
    atomic_facts: List[str], 
    observation_timestamp: str
) -> List[RelationshipsExtractor]
```
**Purpose**: Extract temporal relationships (quintuples) from atomic facts  
**Line**: L52-56  
**Returns**: List of RelationshipsExtractor objects with relationships containing t_start, t_end  
**Dependencies**: LangchainOutputParser.extract_information_as_json_for_context()

---

#### Method: `build_atomic_kg_from_quintuples()`
```python
async def build_atomic_kg_from_quintuples(
    self,
    relationships: list[RelationshipSchema],
    entity_name_weight: float = 0.8,
    entity_label_weight: float = 0.2,
    rel_threshold: float = 0.8,
    ent_threshold: float = 0.8,
    max_workers: int = 8
) -> KnowledgeGraph
```
**Purpose**: Converts quintuples (relationships with temporal info) into atomic KnowledgeGraphs  
**Line**: L88-127  
**Process**:
1. Create temporary KG with all entities
2. Embed entities (weighted by name and label)
3. Parse timestamps for t_start and t_end
4. Create relationships with temporal properties
5. Split into atomic KGs (one relationship per KG)
6. Merge atomically using GraphMatcher

**Returns**: Merged KnowledgeGraph with no duplicate relationships  
**Key Operations**:
- Timestamp parsing: `parser.parse(ts).timestamp()` with error handling
- Entity finding: Uses `temp_kg.get_entity()`
- Embedding: Uses `llm_output_parser.calculate_embeddings()`

---

#### Method: `merge_two_kgs()`
```python
def merge_two_kgs(
    self,
    kg1,
    kg2,
    rel_threshold: float = 0.8,
    ent_threshold: float = 0.8
) -> KnowledgeGraph
```
**Purpose**: Merges two KGs using matcher logic  
**Line**: L60-66  
**Returns**: Single merged KnowledgeGraph  
**Uses**: `self.matcher.match_entities_and_update_relationships()`

---

#### Method: `parallel_atomic_merge()`
```python
def parallel_atomic_merge(
    self,
    kgs: List[KnowledgeGraph],
    existing_kg: Optional[KnowledgeGraph] = None,
    rel_threshold: float = 0.8,
    ent_threshold: float = 0.8,
    max_workers: int = 4
) -> KnowledgeGraph
```
**Purpose**: Merges multiple KGs in parallel using tree reduction  
**Line**: L68-86  
**Algorithm**: 
- Pairs KGs and merges them in parallel
- Repeats until single KG remains
- Optionally merges with existing_kg

**Uses**: `concurrent.futures.ThreadPoolExecutor`  
**Returns**: Final merged KnowledgeGraph

---

#### Method: `build_graph()` - Main Entry Point
```python
async def build_graph(
    self,
    atomic_facts: List[str],
    obs_timestamp: str,
    existing_knowledge_graph: KnowledgeGraph = None,
    ent_threshold: float = 0.8,
    rel_threshold: float = 0.7,
    entity_name_weight: float = 0.8,
    entity_label_weight: float = 0.2,
    max_workers: int = 8
) -> KnowledgeGraph
```
**Purpose**: Main method - builds temporal KG from atomic facts at one timestamp  
**Line**: L131-161  
**Process**:
1. Extract quintuples from all atomic facts
2. Build atomic KGs from quintuples
3. Add atomic facts to relationships
4. Merge atomic KGs in parallel
5. Add observation timestamp to relationships
6. Optionally merge with existing KG

**Returns**: Temporal KnowledgeGraph  
**Logging**: Extensive info logging at each step

---

#### Method: `build_graph_from_different_obs_times()` - Multi-Timestamp Entry Point
```python
async def build_graph_from_different_obs_times(
    self,
    atomic_facts_with_obs_timestamps: dict,
    existing_knowledge_graph: KnowledgeGraph = None,
    ent_threshold: float = 0.8,
    rel_threshold: float = 0.7,
    entity_name_weight: float = 0.8,
    entity_label_weight: float = 0.2,
    max_workers: int = 8
) -> KnowledgeGraph
```
**Purpose**: Builds temporal KG across multiple observation timestamps  
**Line**: L163-+  
**Input Format**: `{timestamp: [atomic_facts], ...}`  
**Process**:
1. For each timestamp, call `build_graph()`
2. Incrementally merge all KGs
3. Return final merged KG

---

### Class: `GraphMatcher` [atom/graph_matching/matcher.py]

#### Constructor
```python
def __init__(self) -> None
```
**Line**: L18

---

#### Method: `_batch_match_entities()`
```python
def _batch_match_entities(
    self,
    entities1: List[Entity],
    entities2: List[Entity],
    threshold: float = 0.8
) -> Tuple[List[Entity], List[Entity]]
```
**Purpose**: Batch-matches entities from entities1 against entities2  
**Line**: L25-89  
**Algorithm**:
1. **Exact matching**: Check `e1 in entities2` (uses __eq__)
2. **Embedding matching**: 
   - Build cosine similarity matrix (sklearn)
   - Find best match for each unmatched e1
   - Apply threshold
3. **Union**: Combine matched entities with entities2, remove duplicates

**Returns**: 
- `matched_entities1`: Matched or original entities in same order
- `global_entities`: Union of matched_entities1 + entities2 (deduplicated)

**Complexity**: O(n*m) where n=len(entities1), m=len(entities2)

---

#### Method: `_batch_match_relationships()`
```python
def _batch_match_relationships(
    self,
    rels1: List[Relationship],
    rels2: List[Relationship],
    threshold: float = 0.8
) -> List[Relationship]
```
**Purpose**: Matches relationships by name embeddings  
**Line**: L94-150+  
**Algorithm**:
1. Extract name embeddings from both lists
2. Compute cosine similarity matrix
3. For each rel1, find best match in rels2
4. If score >= threshold, rename rel1 to rel2's name
5. Track fully matched relationships

**Returns**: Updated rels1 with matched names  
**Key Property**: Preserves startEntity and endEntity references

---

#### Method: `match_entities_and_update_relationships()`
```python
def match_entities_and_update_relationships(
    self,
    entities_1: List[Entity],
    entities_2: List[Entity],
    relationships_1: List[Relationship],
    relationships_2: List[Relationship],
    ent_threshold: float = 0.8,
    rel_threshold: float = 0.8,
    entity_name_weight: float = 0.8,
    entity_label_weight: float = 0.2
) -> Tuple[List[Entity], List[Relationship]]
```
**Purpose**: Comprehensive match of entities and relationships across two KGs  
**Line**: (interface definition)  
**Process**:
1. Match entities from set 1 against set 2
2. Match relationships from set 1 against set 2
3. Create entity name mapping (old → matched)
4. Update relationship endpoints using mapping
5. Extend relationships_2 with updated relationships_1

**Returns**: 
- Global entities (merged + deduplicated)
- Global relationships (updated + merged)

---

## iText2KG Module Reference

### Class: `iText2KG` [itext2kg_star/itext2kg.py]

#### Constructor
```python
def __init__(
    self,
    llm_model,
    embeddings_model,
    sleep_time: int = 5
) -> None
```
**Purpose**: Initialize with LLM and embeddings models  
**Line**: L12-24  
**Attributes**:
- `self.ientities_extractor`: iEntitiesExtractor(...)
- `self.irelations_extractor`: iRelationsExtractor(...)
- `self.matcher`: Matcher()
- `self.langchain_output_parser`: LangchainOutputParser(...)

---

#### Method: `build_graph()` - Main Entry Point
```python
async def build_graph(
    self,
    sections: List[str],
    existing_knowledge_graph: KnowledgeGraph = None,
    ent_threshold: float = 0.7,
    rel_threshold: float = 0.7,
    max_tries: int = 5,
    max_tries_isolated_entities: int = 3,
    entity_name_weight: float = 0.6,
    entity_label_weight: float = 0.4,
    observation_date: str = ""
) -> KnowledgeGraph
```
**Purpose**: Builds KG from multiple document sections  
**Line**: L33-107  
**Process**:
1. Extract entities from section 0
2. Extract and verify relationships for section 0
3. For sections 1..n:
   - Extract entities
   - Match with global entities
   - Extract relationships
   - Match with global relationships
4. Merge with existing KG if provided
5. Remove duplicates

**Returns**: Merged KnowledgeGraph  
**Entity Weights**: 
- `entity_name_weight=0.6` (default): Weight for name embedding
- `entity_label_weight=0.4` (default): Weight for label embedding

---

### Class: `iEntitiesExtractor` [itext2kg_star/ientities_extraction/ientities_extractor.py]

#### Constructor
```python
def __init__(
    self,
    llm_model,
    embeddings_model,
    sleep_time: int = 5
) -> None
```
**Purpose**: Initialize entity extractor  
**Line**: L10-17  
**Attributes**:
- `self.langchain_output_parser`: LangchainOutputParser(...)

---

#### Method: `extract_entities()`
```python
async def extract_entities(
    self,
    context: str,
    max_tries: int = 5,
    entity_name_weight: float = 0.6,
    entity_label_weight: float = 0.4
) -> List[Entity]
```
**Purpose**: Extracts entities from text with embeddings  
**Line**: L24-83  
**Process**:
1. Call LLM with context to extract entities as JSON
2. Parse result to EntitiesExtractor.entities
3. Create Entity objects (name, label only)
4. Create temporary KnowledgeGraph
5. Embed all entities with weighted name+label embeddings
6. Return entities with embeddings

**Returns**: List of Entity objects with embeddings  
**Retries**: Up to `max_tries` on extraction failure  
**Embedding Weights**: Defaults to name-weighted (0.6) + label-weighted (0.4)

---

### Class: `iRelationsExtractor` [itext2kg_star/irelations_extraction/irelations_extractor.py]

#### Constructor
```python
def __init__(
    self,
    llm_model,
    embeddings_model,
    sleep_time: int = 5
) -> None
```
**Purpose**: Initialize relationship extractor  
**Line**: L10-19  
**Attributes**:
- `self.langchain_output_parser`: LangchainOutputParser(...)
- `self.matcher`: Matcher()

---

#### Method: `extract_relations()`
```python
async def extract_relations(
    self,
    context: str,
    entities: List[Entity],
    isolated_entities_without_relations: List[Entity] = None,
    max_tries: int = 5,
    entity_name_weight: float = 0.6,
    entity_label_weight: float = 0.4
) -> List[Relationship]
```
**Purpose**: Extracts relationships from context, handles invented entities  
**Line**: L30-107  
**Process**:
1. Format context with simplified entity list
2. Call LLM to extract relationships
3. Verify each relationship's entities exist in input list
4. Handle invented entities:
   - Embed invented entities
   - Match to closest input entities (threshold=0.5)
   - Update relationship endpoints
5. Embed all relationships
6. Return curated relationships

**Returns**: List of Relationship objects with embeddings  
**Invented Entity Handling**: Core feature for robustness against LLM hallucinations

---

#### Method: `extract_verify_and_correct_relations()`
```python
async def extract_verify_and_correct_relations(
    self,
    context: str,
    entities: List[Entity],
    rel_threshold: float = 0.7,
    max_tries: int = 5,
    max_tries_isolated_entities: int = 3,
    entity_name_weight: float = 0.6,
    entity_label_weight: float = 0.4,
    observation_date: str = ""
) -> List[Relationship]
```
**Purpose**: Main method - extracts, verifies, and corrects relationships  
**Line**: L109-166  
**Process**:
1. Call `extract_relations()`
2. Find isolated entities (entities without relationships)
3. While isolated entities exist and tries < max:
   - Re-extract relationships for isolated entities
   - Link isolated entities to other entities
   - Match corrected relationships with existing ones
   - Update isolated entity list
4. Add observation dates to relationships
5. Return curated relationships

**Returns**: List of verified Relationship objects  
**Iteration**: Up to `max_tries_isolated_entities` attempts to resolve isolated entities

---

### Class: `Matcher` [itext2kg_star/graph_matching/matcher.py]

#### Constructor
```python
def __init__(self) -> None
```
**Line**: L13

---

#### Method: `find_match()`
```python
def find_match(
    self,
    obj1: Union[Entity, Relationship],
    list_objects: List[Union[Entity, Relationship]],
    threshold: float = 0.8
) -> Union[Entity, Relationship]
```
**Purpose**: Finds best match for an entity or relationship  
**Line**: L25-62  
**Algorithm**:
1. Extract name and embeddings from obj1
2. For each obj2 in list_objects:
   - Check exact name+label match (for entities)
   - Compute cosine similarity
   - Track best match
3. If found best_match with score > threshold:
   - For Entity: return the best_match object
   - For Relationship: update obj1 name and embeddings
4. Else: return obj1 unchanged

**Returns**: Matched or original object  
**Logging**: Logs entity/relationship matches

---

#### Method: `process_lists()`
```python
def process_lists(
    self,
    list1: List[Union[Entity, Relationship]],
    list2: List[Union[Entity, Relationship]],
    threshold: float = 0.8
) -> Tuple[List[Union[Entity, Relationship]], List[Union[Entity, Relationship]]]
```
**Purpose**: Processes two lists to generate matched items and union  
**Line**: L92-110  
**Process**:
1. For each item in list1, find best match in list2 → list3
2. Create union of list3 + list2 → list4
3. Remove duplicates from union

**Returns**: 
- `matched_local_items`: list3 (matched version of list1)
- `new_global_items`: list4 (deduplicated union)

---

#### Method: `create_union_list()`
```python
def create_union_list(
    self,
    list1: List[Union[Entity, Relationship]],
    list2: List[Union[Entity, Relationship]]
) -> List[Union[Entity, Relationship]]
```
**Purpose**: Creates union avoiding duplicates  
**Line**: L65-91  
**Duplicate Detection**:
- **Entities**: By (name, label) tuple
- **Relationships**: By name only

**Returns**: Union list with duplicates removed

---

#### Method: `match_entities_and_update_relationships()`
```python
def match_entities_and_update_relationships(
    self,
    entities1: List[Entity],
    entities2: List[Entity],
    relationships1: List[Relationship],
    relationships2: List[Relationship],
    rel_threshold: float = 0.8,
    ent_threshold: float = 0.8
) -> Tuple[List[Entity], List[Relationship]]
```
**Purpose**: Matches entities and updates relationships  
**Line**: L112-157+  
**Process**:
1. Match entities from set 1 against set 2
2. Match relationships from set 1 against set 2
3. Create entity name mapping (old → new)
4. Update relationship startEntity/endEntity names
5. Combine relationships

**Returns**: (global_entities, global_relationships)

---

## Documents Distiller Module

### Class: `DocumentsDistiller` [documents_distiller/documents_distiller.py]

#### Constructor
```python
def __init__(self, llm_model) -> None
```
**Purpose**: Initialize with LLM model  
**Line**: L16-21  
**Attributes**:
- `self.langchain_output_parser`: LangchainOutputParser(...)

---

#### Method: `distill()`
```python
async def distill(
    self,
    documents: List[str],
    output_data_structure,
    IE_query: str
) -> Union[dict, BaseModel]
```
**Purpose**: Distills and combines information from multiple documents  
**Line**: L139-161  
**Process**:
1. Batch extract structured info from all documents
2. Call `extract_information_as_json_for_context()` for all documents
3. Combine results using `__combine_objects()`
4. Return unified object

**Returns**: Combined dict or Pydantic model (same type as input structure)  
**Flexibility**: Works with any Pydantic model or dict output structure

---

#### Static Method: `__combine_objects()`
```python
@staticmethod
def __combine_objects(
    object_list: List[Union[dict, BaseModel]]
) -> Union[dict, BaseModel]
```
**Purpose**: Combines list of objects (dicts or Pydantic models)  
**Line**: L27-48  
**Logic**: 
- If all are same Pydantic type → use `__combine_pydantic_objects()`
- Otherwise → use `__combine_via_dicts()`

---

#### Static Method: `__combine_pydantic_objects()`
```python
@staticmethod
def __combine_pydantic_objects(
    pydantic_objects: List[BaseModel],
    dict_objects: List[dict] = None
) -> BaseModel
```
**Purpose**: Combines Pydantic objects of same type  
**Line**: L50-95  
**Process**:
1. Extract all field names from objects
2. For each field:
   - Collect values from all objects
   - Merge values using `__merge_field_values()`
3. Create new object with merged fields

---

#### Static Method: `__merge_field_values()`
```python
@staticmethod
def __merge_field_values(values: List[Any]) -> Any
```
**Purpose**: Merges multiple field values based on type  
**Line**: L97-130  
**Type-Specific Merging**:
- **Lists**: Extend all lists together
- **Strings**: Concatenate with spaces
- **Dicts**: Update/merge all dicts
- **Others**: Return last non-None value

---

## Graph Integration Module

### Class: `Neo4jStorage` [graph_integration/neo4j_storage.py]

#### Constructor
```python
def __init__(
    self,
    uri: str,
    username: str,
    password: str,
    database: Optional[str] = None
)
```
**Purpose**: Initialize Neo4j storage connection  
**Line**: L14-27  
**Attributes**:
- `self.uri`: Database URI
- `self.username`: Database username
- `self.password`: Database password
- `self.database`: Optional database name
- `self.driver`: Neo4j driver instance

---

#### Method: `connect()`
```python
def connect(self)
```
**Purpose**: Establishes Neo4j connection  
**Line**: L31-36  
**Returns**: Neo4j driver instance

---

#### Method: `run_query()`
```python
def run_query(self, query: str)
```
**Purpose**: Executes Cypher query without results  
**Line**: L38-47  
**Usage**: For CREATE, UPDATE, DELETE operations

---

#### Method: `run_query_with_result()`
```python
def run_query_with_result(self, query: str)
```
**Purpose**: Executes Cypher query and returns results  
**Line**: L152-161  
**Returns**: List of records

---

#### Method: `create_nodes()`
```python
def create_nodes(
    self,
    knowledge_graph: KnowledgeGraph
) -> List[str]
```
**Purpose**: Generates Cypher queries for node creation  
**Line**: L166-191  
**Process**:
1. For each entity in KG:
   - Create node label (sanitized)
   - Format properties
   - Generate MERGE query

**Returns**: List of Cypher queries  
**Example Query**:
```cypher
MERGE (n:EntityLabel {name: "entity_name"}) 
SET n.embeddings = "..."
```

---

#### Method: `create_relationships()`
```python
def create_relationships(
    self,
    knowledge_graph: KnowledgeGraph
) -> List[str]
```
**Purpose**: Generates Cypher queries for relationship creation  
**Line**: L194-+  
**Process**:
1. For each relationship in KG:
   - Sanitize start/end entity labels
   - Format relationship name
   - Generate MATCH + CREATE query

**Returns**: List of Cypher queries  
**Example Query**:
```cypher
MATCH (start:StartLabel {name: "..."})
MATCH (end:EndLabel {name: "..."})
MERGE (start)-[r:RELATIONSHIP_NAME]->(end)
SET r.properties = ...
```

---

#### Static Methods - Utility Functions

##### `transform_embeddings_to_str_list()`
```python
@staticmethod
def transform_embeddings_to_str_list(embeddings: np.ndarray) -> str
```
**Line**: L49-56  
**Returns**: Comma-separated string of embeddings

##### `transform_str_list_to_embeddings()`
```python
@staticmethod
def transform_str_list_to_embeddings(embeddings: str) -> np.ndarray
```
**Line**: L58-67  
**Returns**: Reconstructed numpy array

##### `escape_str()`
```python
@staticmethod
def escape_str(s: str) -> str
```
**Line**: L69-72  
**Purpose**: Escapes double quotes for Cypher

##### `format_value()`
```python
@staticmethod
def format_value(value) -> str
```
**Line**: L74-77  
**Purpose**: Converts value to escaped string

##### `format_property_value()`
```python
@staticmethod
def format_property_value(key: str, value) -> str
```
**Line**: L79-104  
**Purpose**: Formats property values for Cypher  
**Handles**:
- Embeddings (arrays → strings)
- Lists (with proper escaping)
- Numbers (without quotes)
- Strings (with escaping)

##### `sanitize_label()` & `sanitize_relationship_type()`
**Line**: (methods for cleaning Neo4j identifiers)

---

## LLM Output Parsing Module

### Class: `LangchainOutputParser` [llm_output_parsing/langchain_output_parser.py]

#### Constructor
```python
def __init__(
    self,
    llm_model,
    embeddings_model,
    sleep_time: int = 5,
    provider_type: Optional[ProviderType] = None
) -> None
```
**Purpose**: Initialize provider-agnostic parser  
**Line**: L61-80  
**Auto-Detection**: Detects provider if not specified  
**Attributes**:
- `self.model`: LLM instance
- `self.embeddings_model`: Embeddings instance
- `self.provider_type`: Detected provider (OpenAI, Mistral, Claude, etc.)
- `self.config`: Provider-specific rate limiting config

---

#### Method: `extract_information_as_json_for_context()`
```python
async def extract_information_as_json_for_context(
    self,
    contexts: List[str],
    output_data_structure,
    system_query: str
) -> List[BaseModel]
```
**Purpose**: Batch extracts structured information from contexts  
**Key Features**:
- Batch processing with rate limiting
- Provider-aware token limits
- Automatic retries with exponential backoff
- Returns Pydantic models

**Returns**: List of output_data_structure instances (one per context)

---

#### Method: `calculate_embeddings()`
```python
async def calculate_embeddings(
    self,
    texts: Union[str, List[str]]
) -> np.ndarray
```
**Purpose**: Generates embeddings for texts  
**Returns**: 
- Single embedding (if input is string)
- Array of embeddings (if input is list)

---

#### Method: `_detect_provider()`
```python
def _detect_provider(self) -> ProviderType
```
**Purpose**: Auto-detects LLM provider from model instance  
**Returns**: ProviderType enum value

---

## Data Models Reference

### Entity (ATOM version)

```python
class Entity(BaseModelWithConfig):
    label: str = ""              # Semantic category (Person, Organization, etc.)
    name: str = ""               # Unique identifier
    properties: EntityProperties = Field(default_factory=EntityProperties)
    
    def process(self) -> "Entity":
        """Normalizes label and name"""
        # Label: [^a-zA-Z0-9]+ → _, & → and, lowercase
        # Name: _, ", - → space, lowercase, strip
        
    def __eq__(self, other) -> bool:
        """Equality by (name, label) tuple"""
        
    def __hash__(self) -> int:
        """Hash by (name, label) tuple"""
```

### Relationship (ATOM version)

```python
class Relationship(BaseModelWithConfig):
    startEntity: Entity = Field(default_factory=Entity)
    endEntity: Entity = Field(default_factory=Entity)
    name: str = ""               # Predicate (e.g., "is_CEO")
    properties: RelationshipProperties = Field(default_factory=RelationshipProperties)
    
    class RelationshipProperties:
        embeddings: Optional[np.ndarray] = None
        atomic_facts: List[str] = []
        t_obs: List[float] = []     # Observation timestamps
        t_start: List[float] = []   # Start timestamps
        t_end: List[float] = []     # End timestamps
    
    def process(self) -> "Relationship":
        """Normalizes name"""
        
    def combine_timestamps(timestamps, temporal_aspect):
        """Adds timestamps to specified temporal aspect"""
        
    def combine_atomic_facts(atomic_facts):
        """Adds atomic facts to relationship"""
```

### KnowledgeGraph (ATOM version)

```python
class KnowledgeGraph(BaseModelWithConfig):
    entities: List[Entity] = Field(default_factory=list)
    relationships: List[Relationship] = Field(default_factory=list)
    
    def is_empty(self) -> bool
    def remove_duplicates_entities(self) -> None
    def remove_duplicates_relationships(self) -> None
    
    async def embed_entities(
        embeddings_function,
        entity_name_weight: float = 0.8,
        entity_label_weight: float = 0.2
    ) -> None
    
    async def embed_relationships(
        embeddings_function
    ) -> None
    
    def get_entity(other_entity: Entity) -> Optional[Entity]
    def get_relationship(other_relationship: Relationship) -> Optional[Relationship]
    
    def add_t_obs_to_relationships(t_obs: Union[List[float], List[str]]) -> None
    def add_atomic_facts_to_relationships(atomic_facts: List[str]) -> None
    
    def find_isolated_entities() -> List[Entity]
    def split_into_atomic_kgs() -> List['KnowledgeGraph']
    
    @staticmethod
    def from_neo4j(graph_storage) -> 'KnowledgeGraph'
```

---

## Schema Definitions for LLM Output

### Temporal Relationships (ATOM) [atom/models/schemas.py]

```python
class Entity(BaseModel):
    label: str  # Semantic category
    name: str   # Entity name/identifier

class Relationship(BaseModel):
    startNode: Entity
    endNode: Entity
    name: str                      # Present-tense predicate
    t_start: Optional[list[str]] = []  # Start times (resolved)
    t_end: Optional[list[str]] = []    # End times (resolved)

class RelationshipsExtractor(BaseModel):
    relationships: List[Relationship]
```

### Generic Relationships (iText2KG)

```python
class EntitiesExtractor(BaseModel):
    entities: List[Entity]

class RelationshipsExtractor(BaseModel):
    relationships: List[Relationship]
```

---

