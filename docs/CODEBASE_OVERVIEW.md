# iText2KG Codebase Overview

## Table of Contents
1. [Module Structure](#module-structure)
2. [Main Components](#main-components)
3. [Entry Points](#entry-points)
4. [Core Data Models](#core-data-models)
5. [Class Relationships](#class-relationships)
6. [Key Methods](#key-methods)

---

## Module Structure

### Top-Level Modules
```
itext2kg/
├── atom/                      # Temporal knowledge graph extraction from atomic facts
├── itext2kg_star/             # Multi-section knowledge graph extraction from documents
├── documents_distiller/       # Document information consolidation
├── graph_integration/         # Storage and visualization (Neo4j)
├── llm_output_parsing/        # LLM interaction and embeddings
└── logging_config.py          # Logging configuration
```

### Exported Classes (from `itext2kg/__init__.py`)
```python
from itext2kg.atom import Atom
from itext2kg.itext2kg_star import iText2KG, iText2KG_Star
from itext2kg.graph_integration import Neo4jStorage
from itext2kg.documents_distiller import DocumentsDistiller
from itext2kg.llm_output_parsing import LangchainOutputParser
```

---

## Main Components

### 1. **ATOM Module** (`itext2kg/atom/`)
Processes atomic facts with temporal information to extract knowledge graphs.

#### Key Classes:

##### `Atom` [atom.py, L1-220]
**Purpose**: Main orchestrator for temporal knowledge graph extraction from atomic facts.

**Constructor**:
```python
def __init__(self, llm_model, embeddings_model) -> None
```
- `llm_model`: Language model for extracting entities and relationships
- `embeddings_model`: For generating vector representations

**Key Methods**:

| Method | Signature | Purpose | Line |
|--------|-----------|---------|------|
| `extract_quintuples()` | `async def extract_quintuples(atomic_facts: List[str], observation_timestamp: str) -> List[RelationshipsExtractor]` | Extracts temporal relationships from atomic facts | L52 |
| `build_graph()` | `async def build_graph(atomic_facts: List[str], obs_timestamp: str, existing_knowledge_graph: KnowledgeGraph=None, ent_threshold: float=0.8, rel_threshold: float=0.7, entity_name_weight: float=0.8, entity_label_weight: float=0.2, max_workers: int=8) -> KnowledgeGraph` | Main method: builds temporal KG from atomic facts | L131 |
| `build_graph_from_different_obs_times()` | `async def build_graph_from_different_obs_times(atomic_facts_with_obs_timestamps: dict, existing_knowledge_graph: KnowledgeGraph=None, ...) -> KnowledgeGraph` | Builds KG across multiple observation timestamps | L163 |
| `build_atomic_kg_from_quintuples()` | `async def build_atomic_kg_from_quintuples(relationships: list[RelationshipSchema], ...) -> KnowledgeGraph` | Converts quintuples to atomic knowledge graphs | L88 |
| `parallel_atomic_merge()` | `def parallel_atomic_merge(kgs: List[KnowledgeGraph], existing_kg: Optional[KnowledgeGraph]=None, rel_threshold: float=0.8, ent_threshold: float=0.8, max_workers: int=4) -> KnowledgeGraph` | Merges multiple KGs in parallel | L68 |
| `merge_two_kgs()` | `def merge_two_kgs(kg1, kg2, rel_threshold: float=0.8, ent_threshold: float=0.8)` | Merges two knowledge graphs | L60 |

**Dependencies**: 
- `GraphMatcher` - for entity/relationship matching
- `LangchainOutputParser` - for LLM interaction
- `KnowledgeGraph` - data model

---

##### `GraphMatcher` [atom/graph_matching/matcher.py]
**Purpose**: Matches and merges entities and relationships using exact matching and embedding similarity.

**Key Methods**:

| Method | Signature | Purpose | Line |
|--------|-----------|---------|------|
| `_batch_match_entities()` | `def _batch_match_entities(entities1: List[Entity], entities2: List[Entity], threshold: float=0.8) -> Tuple[List[Entity], List[Entity]]` | Batch-matches entities using embeddings | L25 |
| `_batch_match_relationships()` | `def _batch_match_relationships(rels1: List[Relationship], rels2: List[Relationship], threshold: float=0.8) -> List[Relationship]` | Matches relationships by name similarity | L94 |
| `match_entities_and_update_relationships()` | `def match_entities_and_update_relationships(entities_1: List[Entity], entities_2: List[Entity], relationships_1: List[Relationship], relationships_2: List[Relationship], ent_threshold: float=0.8, rel_threshold: float=0.8, entity_name_weight: float=0.8, entity_label_weight: float=0.2) -> Tuple[List[Entity], List[Relationship]]` | Matches entities and updates relationships accordingly | (interface) |

**Algorithm**:
1. Exact match by name+label
2. Embedding-based matching (cosine similarity) for remaining entities
3. Union creation with duplicate removal

---

### 2. **iText2KG_Star Module** (`itext2kg/itext2kg_star/`)
Extracts knowledge graphs from document sections using entity and relationship extraction.

#### Key Classes:

##### `iText2KG` [itext2kg_star/itext2kg.py, L1-121]
**Purpose**: Extracts knowledge from text across multiple sections and merges them.

**Constructor**:
```python
def __init__(self, llm_model, embeddings_model, sleep_time: int=5) -> None
```

**Key Methods**:

| Method | Signature | Purpose | Line |
|--------|-----------|---------|------|
| `build_graph()` | `async def build_graph(sections: List[str], existing_knowledge_graph: KnowledgeGraph=None, ent_threshold: float=0.7, rel_threshold: float=0.7, max_tries: int=5, max_tries_isolated_entities: int=3, entity_name_weight: float=0.6, entity_label_weight: float=0.4, observation_date: str="") -> KnowledgeGraph` | Extracts KG from multiple document sections | L33 |

**Workflow**:
1. Extract entities from first section
2. Extract and verify relationships
3. For each additional section:
   - Extract entities and match with global entities
   - Extract relationships and verify isolated entities
   - Merge with global graph
4. Merge with existing KG if provided

**Components**:
- `iEntitiesExtractor` - entity extraction
- `iRelationsExtractor` - relationship extraction
- `Matcher` - entity/relationship matching

---

##### `iEntitiesExtractor` [itext2kg_star/ientities_extraction/ientities_extractor.py, L1-85]
**Purpose**: Extracts entities from text using LLM.

**Constructor**:
```python
def __init__(self, llm_model, embeddings_model, sleep_time: int=5) -> None
```

**Key Methods**:

| Method | Signature | Purpose | Line |
|--------|-----------|---------|------|
| `extract_entities()` | `async def extract_entities(context: str, max_tries: int=5, entity_name_weight: float=0.6, entity_label_weight: float=0.4) -> List[Entity]` | Extracts entities from text with embeddings | L38 |

**Process**:
1. Calls LLM to extract entities as JSON
2. Retries up to `max_tries` times on failure
3. Generates embeddings for each entity
4. Returns list of `Entity` objects with embeddings

---

##### `iRelationsExtractor` [itext2kg_star/irelations_extraction/irelations_extractor.py, L1-216]
**Purpose**: Extracts, verifies, and corrects relationships between entities.

**Constructor**:
```python
def __init__(self, llm_model, embeddings_model, sleep_time: int=5) -> None
```

**Key Methods**:

| Method | Signature | Purpose | Line |
|--------|-----------|---------|------|
| `extract_relations()` | `async def extract_relations(context: str, entities: List[Entity], isolated_entities_without_relations: List[Entity]=None, max_tries: int=5, entity_name_weight: float=0.6, entity_label_weight: float=0.4) -> List[Relationship]` | Extracts relationships with invented entity handling | L30 |
| `extract_verify_and_correct_relations()` | `async def extract_verify_and_correct_relations(context: str, entities: List[Entity], rel_threshold: float=0.7, max_tries: int=5, max_tries_isolated_entities: int=3, entity_name_weight: float=0.6, entity_label_weight: float=0.4, observation_date: str="") -> List[Relationship]` | Main method: extracts and corrects relationships | L109 |

**Process**:
1. Extract relationships from context
2. Verify entities are in input list (handle invented entities)
3. Match invented entities to closest input entities using embeddings
4. Find isolated entities and re-prompt LLM
5. Add observation dates to relationships

---

##### `Matcher` [itext2kg_star/graph_matching/matcher.py, L1-155]
**Purpose**: Matches entities and relationships using similarity metrics.

**Key Methods**:

| Method | Signature | Purpose | Line |
|--------|-----------|---------|------|
| `find_match()` | `def find_match(obj1: Union[Entity, Relationship], list_objects: List[Union[Entity, Relationship]], threshold: float=0.8) -> Union[Entity, Relationship]` | Finds best match for an entity/relationship | L25 |
| `process_lists()` | `def process_lists(list1: List[Union[Entity, Relationship]], list2: List[Union[Entity, Relationship]], threshold: float=0.8) -> Tuple[List[Union[Entity, Relationship]], List[Union[Entity, Relationship]]]` | Processes two lists and returns matched + union | L92 |
| `match_entities_and_update_relationships()` | `def match_entities_and_update_relationships(entities1: List[Entity], entities2: List[Entity], relationships1: List[Relationship], relationships2: List[Relationship], rel_threshold: float=0.8, ent_threshold: float=0.8) -> Tuple[List[Entity], List[Relationship]]` | Matches entities and updates relationship endpoints | L112 |

**Matching Strategy**:
1. Exact name+label match first
2. Cosine similarity on embeddings (> threshold)
3. Merge relationships with updated entity references

---

### 3. **Documents Distiller Module** (`itext2kg/documents_distiller/`)

##### `DocumentsDistiller` [documents_distiller.py, L1-199]
**Purpose**: Consolidates information from multiple documents into a single combined structure.

**Constructor**:
```python
def __init__(self, llm_model) -> None
```

**Key Methods**:

| Method | Signature | Purpose | Line |
|--------|-----------|---------|------|
| `distill()` | `async def distill(documents: List[str], output_data_structure, IE_query: str) -> Union[dict, BaseModel]` | Extracts and combines information from documents | L139 |
| `__combine_objects()` | `@staticmethod def __combine_objects(object_list: List[Union[dict, BaseModel]]) -> Union[dict, BaseModel]` | Combines dictionaries or Pydantic objects | L27 |
| `__combine_pydantic_objects()` | `@staticmethod def __combine_pydantic_objects(pydantic_objects: List[BaseModel], dict_objects: List[dict]=None) -> BaseModel` | Combines Pydantic objects of same type | L50 |
| `__merge_field_values()` | `@staticmethod def __merge_field_values(values: List[Any]) -> Any` | Merges field values based on type (lists, strings, dicts) | L97 |

---

### 4. **Graph Integration Module** (`itext2kg/graph_integration/`)

##### `Neo4jStorage` [neo4j_storage.py, L1-220]
**Purpose**: Manages graph data storage and visualization in Neo4j.

**Constructor**:
```python
def __init__(self, uri: str, username: str, password: str, database: Optional[str]=None)
```

**Key Methods**:

| Method | Signature | Purpose | Line |
|--------|-----------|---------|------|
| `connect()` | `def connect()` | Establishes Neo4j connection | L31 |
| `run_query()` | `def run_query(query: str)` | Executes Cypher query without results | L38 |
| `run_query_with_result()` | `def run_query_with_result(query: str)` | Executes Cypher query and returns results | L152 |
| `create_nodes()` | `def create_nodes(knowledge_graph: KnowledgeGraph) -> List[str]` | Generates Cypher queries for node creation | L166 |
| `create_relationships()` | `def create_relationships(knowledge_graph: KnowledgeGraph) -> List[str]` | Generates Cypher queries for relationship creation | L194 |
| `visualize_graph()` | `def visualize_graph(knowledge_graph: KnowledgeGraph) -> None` | (Interface method) Stores KG in Neo4j | (interface) |

**Utility Methods**:
- `transform_embeddings_to_str_list()` - Convert numpy arrays to strings
- `transform_str_list_to_embeddings()` - Convert strings back to arrays
- `escape_str()` - Escape quotes for Cypher
- `format_value()` - Format values for Cypher
- `format_property_value()` - Format properties for Cypher
- `sanitize_label()` - Clean labels for Neo4j

---

##### `GraphStorageInterface` [storage_interface.py, L1-17]
**Purpose**: Protocol/interface for storage implementations.

**Interface Methods**:
```python
def visualize_graph(self, knowledge_graph: KnowledgeGraph) -> None:
    """Visualizes the knowledge graph."""
```

---

### 5. **LLM Output Parsing Module** (`itext2kg/llm_output_parsing/`)

##### `LangchainOutputParser` [langchain_output_parser.py, L1-100+]
**Purpose**: Provider-agnostic parser for LLM interaction with rate limiting.

**Constructor**:
```python
def __init__(self, llm_model, embeddings_model, sleep_time: int=5, provider_type: Optional[ProviderType]=None) -> None
```

**Key Methods** (partial list):

| Method | Signature | Purpose |
|--------|-----------|---------|
| `extract_information_as_json_for_context()` | `async def extract_information_as_json_for_context(contexts: List[str], output_data_structure, system_query: str) -> List[BaseModel]` | Batch extracts structured info from contexts |
| `calculate_embeddings()` | `async def calculate_embeddings(texts: Union[str, List[str]]) -> np.ndarray` | Generates embeddings for texts |
| `_detect_provider()` | `def _detect_provider() -> ProviderType` | Auto-detects LLM provider |

**Supported Providers**: OpenAI, Mistral, Claude, Unknown (with rate limiting configs)

---

## Core Data Models

### 1. **Entity (ATOM version)** [atom/models/entity.py]
```python
class Entity(BaseModelWithConfig):
    label: str = ""
    name: str = ""
    properties: EntityProperties = Field(default_factory=EntityProperties)
    
    def process(self) -> "Entity": # Normalizes label and name
    def __eq__(self, other) -> bool: # Equality by name+label
    def __hash__(self) -> int: # Hash for set operations
```

**Properties**: `embeddings: Optional[np.ndarray]`

**Normalization**: 
- Label: `[^a-zA-Z0-9]+` → `_`, `&` → `and`, lowercase
- Name: remove `_`, `"`, `-`, strip whitespace, lowercase

---

### 2. **Relationship (ATOM version)** [atom/models/relationship.py]
```python
class Relationship(BaseModelWithConfig):
    startEntity: Entity = Field(default_factory=Entity)
    endEntity: Entity = Field(default_factory=Entity)
    name: str = ""
    properties: RelationshipProperties = Field(default_factory=RelationshipProperties)
    
    def process(self) -> "Relationship": # Normalizes name
    def combine_timestamps(timestamps: Union[List[float], List[str]], temporal_aspect: str)
    def combine_atomic_facts(atomic_facts: List[str])
```

**Properties**:
```python
class RelationshipProperties:
    embeddings: Optional[np.ndarray] = None
    atomic_facts: List[str] = []
    t_obs: List[float] = []      # Observation timestamps
    t_start: List[float] = []    # Start timestamps
    t_end: List[float] = []      # End timestamps
```

**Normalization**: `[^a-zA-Z0-9]+` → `_`, `&` → `and`, lowercase

---

### 3. **KnowledgeGraph (ATOM version)** [atom/models/knowledge_graph.py, L1-160]
```python
class KnowledgeGraph(BaseModelWithConfig):
    entities: List[Entity] = Field(default_factory=list)
    relationships: List[Relationship] = Field(default_factory=list)
    
    def is_empty(self) -> bool
    def remove_duplicates_entities(self) -> None
    async def embed_entities(embeddings_function, entity_name_weight=0.8, entity_label_weight=0.2) -> None
    async def embed_relationships(embeddings_function) -> None
    def get_entity(other_entity: Entity) -> Optional[Entity]
    def get_relationship(other_relationship: Relationship) -> Optional[Relationship]
    def add_t_obs_to_relationships(t_obs: Union[List[float], List[str]]) -> None
    def add_atomic_facts_to_relationships(atomic_facts: List[str]) -> None
    def find_isolated_entities() -> List[Entity]
    def split_into_atomic_kgs() -> List['KnowledgeGraph']  # One relationship per KG
    @staticmethod
    def from_neo4j(graph_storage) -> 'KnowledgeGraph'
```

---

### 4. **Pydantic Schemas for LLM Output** [atom/models/schemas.py & itext2kg_star/models/schemas.py]

#### Temporal Relationship (ATOM) [atom/models/schemas.py]
```python
class Entity(BaseModel):
    label: str  # Semantic category
    name: str   # Unique identifier

class Relationship(BaseModel):
    startNode: Entity
    endNode: Entity
    name: str  # Present-tense predicate (e.g., "is_CEO", "works_at")
    t_start: Optional[list[str]] = []  # Start timestamps (resolved from relative refs)
    t_end: Optional[list[str]] = []    # End timestamps

class RelationshipsExtractor(BaseModel):
    relationships: List[Relationship]
```

#### Generic Relationship (iText2KG_Star) [itext2kg_star/models/schemas.py]
```python
class Entity(BaseModel):
    label: str
    name: str

class Relationship(BaseModel):
    startNode: Entity
    endNode: Entity
    name: str

class RelationshipsExtractor(BaseModel):
    relationships: List[Relationship]

class EntitiesExtractor(BaseModel):
    entities: List[Entity]
```

#### Document-Level Schemas [itext2kg_star/models/schemas.py]
```python
class InformationRetriever  # Company website info
class Article              # Scientific articles
class CV                   # Curriculum vitae
class Facts                # Extracted facts
```

---

## Entry Points

### 1. **Main Script** [main.py, L1-64]
```python
async def main():
    # 1. Load data
    news_covid = pd.read_pickle("./small_pickle.pkl")
    news_covid_dict = to_dictionary(news_covid)
    
    # 2. Initialize ATOM
    atom = Atom(llm_model=openai_llm_model, embeddings_model=openai_embeddings_model)
    
    # 3. Build temporal KG
    kg = await atom.build_graph_from_different_obs_times(
        atomic_facts_with_obs_timestamps=news_covid_dict,
    )
    
    # 4. Visualize in Neo4j
    Neo4jStorage(uri=URI, username=USERNAME, password=PASSWORD).visualize_graph(kg)
```

**Key Functions**:
- `to_dictionary(df, max_elements)` - Converts DataFrame to timestamp-factoids dict

---

## Class Relationships

### Data Flow Diagram

```
┌─────────────────────────────────────────────────────────────────┐
│                        Entry Point: main.py                     │
└────────────────────────────┬────────────────────────────────────┘
                             │
                    ┌────────▼─────────┐
                    │      Atom        │ ◄─── Initialize
                    └────────┬─────────┘
                             │
         ┌───────────────────┼───────────────────┐
         │                   │                   │
    ┌────▼──────┐   ┌────────▼────────┐  ┌──────▼──────┐
    │ Extract   │   │ Build Atomic KG │  │ Parallel    │
    │ Quintuples│   │ from Quintuples │  │ Merge       │
    │ (LLM)     │   │                 │  │             │
    └────┬──────┘   └────────┬────────┘  └──────┬──────┘
         │                   │                  │
         └───────────────────┼──────────────────┘
                             │
                      ┌──────▼──────┐
                      │ KnowledgeGraph
                      │ (temporal KG)
                      └──────┬──────┘
                             │
                      ┌──────▼─────────┐
                      │ Neo4jStorage   │
                      │ visualize_graph|
                      └────────────────┘
```

### Component Interactions

**ATOM Processing Pipeline**:
```
Atomic Facts
    ↓
[LangchainOutputParser: extract_quintuples()]
    ↓
Relationships (with t_start, t_end, t_obs)
    ↓
[build_atomic_kg_from_quintuples()]
    ↓
Atomic KGs (one rel per KG)
    ↓
[parallel_atomic_merge() + GraphMatcher]
    ↓
Merged KnowledgeGraph
    ↓
[Optional: match with existing KG]
    ↓
Final Temporal KnowledgeGraph
```

**iText2KG Processing Pipeline**:
```
Document Sections
    ↓
[iEntitiesExtractor: extract_entities()]
    ↓
Entities with Embeddings
    ↓
[iRelationsExtractor: extract_verify_and_correct_relations()]
    ↓
Relationships with Embeddings
    ↓
[Matcher: process_lists(), match_entities_and_update_relationships()]
    ↓
Merged KnowledgeGraph
    ↓
[Optional: match with existing KG]
    ↓
Final KnowledgeGraph
```

---

## Key Methods

### Configuration and Initialization

**Model Initialization** [models/models.py]:
```python
def get_default_model() -> ChatOpenAI
def get_default_embedding_model() -> OpenAIEmbeddings
```

**LangChain Provider Configs** [llm_output_parsing/langchain_output_parser.py]:
```python
PROVIDER_CONFIGS = {
    ProviderType.OPENAI: ProviderConfig(
        max_elements_per_batch=40,
        max_tokens_per_batch=8000,
        max_context_window=128000,
        sleep_between_batches=2.0
    ),
    ProviderType.MISTRAL: ProviderConfig(
        max_elements_per_batch=1,
        max_tokens_per_batch=10000,
        max_context_window=128000,
        sleep_between_batches=0.2
    ),
    # ... Claude, Unknown configs
}
```

---

## Summary of Workflows

### Workflow 1: Temporal Knowledge Graph from Atomic Facts (ATOM)

**Input**: Dictionary of {timestamp: [atomic_facts]}
**Output**: Temporal KnowledgeGraph with t_obs, t_start, t_end

**Steps**:
1. For each timestamp → extract quintuples (entities + relations)
2. Build atomic KGs (one relationship per KG)
3. Merge atomic KGs in parallel using GraphMatcher
4. Add temporal information
5. Merge with existing KG if provided
6. Store in Neo4j

---

### Workflow 2: Knowledge Graph from Document Sections (iText2KG)

**Input**: List of document sections
**Output**: KnowledgeGraph with entities and relationships

**Steps**:
1. Extract entities from first section
2. Extract and verify relationships
3. For each additional section:
   - Extract entities → match with global
   - Extract relationships → verify isolated entities → re-prompt if needed
   - Merge with global graph
4. Merge with existing KG if provided
5. Remove duplicates

---

### Workflow 3: Document Information Consolidation

**Input**: List of documents, output schema
**Output**: Single consolidated object (dict or Pydantic model)

**Steps**:
1. Batch process all documents through LLM
2. Combine results by merging field values
3. Return unified object

---

## Architecture Patterns

### 1. **Async/Await Pattern**
- All LLM interactions are async
- Entity/relationship extraction is async
- Embedding calculations are async
- Parallel merging with ThreadPoolExecutor

### 2. **Matching Strategy**
- **Exact match first** (name+label equality)
- **Embedding-based matching** (cosine similarity > threshold)
- **Union creation** with duplicate removal using sets and hash()

### 3. **Provider-Agnostic LLM Interface**
- Auto-detects LLM provider (OpenAI, Mistral, Claude)
- Provider-specific rate limiting configs
- Batch processing with token/request limits

### 4. **Temporal Tracking**
- `t_start`: When relationship begins
- `t_end`: When relationship ends
- `t_obs`: When relationship was observed
- Resolved from relative references (today → exact date)

---

## Configuration Files

### `env_config.py`
- Neo4j connection parameters
- LLM/embeddings model configurations

### `models_config.py`
- LangChain model instances
- Default model getters

### `pyproject.toml`
- Package metadata
- Dependencies

### `docker-compose.yml`
- Neo4j container setup
- Database initialization

---

## Testing & Evaluation

### Test Files
- `itext2kg-1.0.0/tests/atom/test_atom_matching.py` - ATOM matching tests
- `itext2kg-1.0.0/tests/itext2kg/test_itext2kg_matching.py` - iText2KG matching tests

### Evaluation Scripts
- `evaluation/exhaustivity/` - Factoid extraction coverage
- `evaluation/latency/` - Performance comparison
- `evaluation/quintuples_quality/` - Triple quality metrics
- `evaluation/stability/` - Consistency testing
- `evaluation/merge/` - Graph merge quality

---

## Key Design Decisions

1. **Separate Modules for Different Tasks**:
   - ATOM for temporal facts (atomic_facts → quintuples → temporal KG)
   - iText2KG for general documents (sections → entities → relationships)
   - DocumentsDistiller for consolidation

2. **Embedding-Based Similarity**:
   - Both entity and relationship matching use cosine similarity on embeddings
   - Weighted entity embeddings (name + label)

3. **Temporal Information in Relationships**:
   - t_start, t_end, t_obs stored in RelationshipProperties
   - Supports temporal queries and graph evolution tracking

4. **Neo4j Integration**:
   - Graph storage for visualization and querying
   - Cypher query generation for node/relationship creation

5. **Pydantic Models for LLM Output**:
   - Type-safe extraction of structured information
   - Field descriptions serve as LLM prompts

---

## Notes

- All embeddings use numpy arrays stored in properties
- Names are lowercase; labels are lowercase with underscores
- Duplicates are removed using set() with custom __hash__ and __eq__
- Parallel processing uses ThreadPoolExecutor for KG merging
- Rate limiting is provider-aware with batch processing
- Neo4j labels are sanitized to valid Neo4j identifiers

