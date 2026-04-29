# iText2KG - Complete Index & Line Number Reference

## File Structure Index

```
itext2kg-1.0.0/
├── itext2kg/
│   ├── __init__.py                           [Exports: Atom, iText2KG, iText2KG_Star, Neo4jStorage, DocumentsDistiller, LangchainOutputParser]
│   ├── logging_config.py                     [get_logger() function]
│   │
│   ├── atom/                                 [Temporal KG from atomic facts]
│   │   ├── __init__.py                       [Exports: Atom, GraphMatcher]
│   │   ├── atom.py                           [L1-220: Main ATOM class]
│   │   ├── graph_matching/
│   │   │   ├── matcher.py                    [L1-150: GraphMatcher class]
│   │   │   └── matcher_interface.py          [Interface definition]
│   │   └── models/
│   │       ├── entity.py                     [L1-80: Entity class (ATOM)]
│   │       ├── relationship.py               [L1-80: Relationship class (ATOM)]
│   │       ├── knowledge_graph.py            [L1-160: KnowledgeGraph class (ATOM)]
│   │       └── schemas.py                    [L1-260: Pydantic schemas for temporal extraction]
│   │
│   ├── itext2kg_star/                        [Generic KG from document sections]
│   │   ├── __init__.py                       [Exports: iText2KG, iText2KG_Star]
│   │   ├── itext2kg.py                       [L1-121: iText2KG class]
│   │   ├── itext2kg_star.py                  [iText2KG_Star class (if exists)]
│   │   ├── ientities_extraction/
│   │   │   ├── ientities_extractor.py       [L1-85: iEntitiesExtractor class]
│   │   │   └── __init__.py
│   │   ├── irelations_extraction/
│   │   │   ├── irelations_extractor.py      [L1-216: iRelationsExtractor class]
│   │   │   └── __init__.py
│   │   ├── graph_matching/
│   │   │   ├── matcher.py                    [L1-155: Matcher class]
│   │   │   └── matcher_interface.py
│   │   └── models/
│   │       ├── knowledge_graph.py            [L1-160: KnowledgeGraph class (iText2KG)]
│   │       ├── entity.py                     [Entity class definitions]
│   │       ├── relationship.py               [Relationship class definitions]
│   │       └── schemas.py                    [L1-197: Pydantic schemas for generic extraction]
│   │
│   ├── documents_distiller/                  [Document consolidation]
│   │   ├── __init__.py                       [Exports: DocumentsDistiller]
│   │   └── documents_distiller.py            [L1-199: DocumentsDistiller class]
│   │
│   ├── graph_integration/                    [Neo4j storage]
│   │   ├── __init__.py                       [Exports: Neo4jStorage]
│   │   ├── neo4j_storage.py                  [L1-220: Neo4jStorage class]
│   │   └── storage_interface.py              [L1-17: GraphStorageInterface protocol]
│   │
│   └── llm_output_parsing/                   [LLM interaction & embeddings]
│       ├── __init__.py                       [Exports: LangchainOutputParser]
│       ├── langchain_output_parser.py        [L1-100+: LangchainOutputParser class]
│       └── llm_output_parser_interface.py    [Interface definition]
│
├── models/
│   ├── models.py                             [L1-19: get_default_model(), get_default_embedding_model()]
│   └── models_config.py                      [LLM & embeddings configurations]
│
├── main.py                                   [L1-64: Entry point script]
├── env_config.py                             [Environment configuration]
├── pyproject.toml                            [Project metadata]
├── requirements.txt                          [Dependencies]
└── docker-compose.yml                        [Neo4j container setup]
```

---

## Class Location Index

### ATOM Module Classes

| Class | File | Line | Purpose |
|-------|------|------|---------|
| `Atom` | atom/atom.py | L14-220 | Main temporal KG orchestrator |
| `GraphMatcher` | atom/graph_matching/matcher.py | L18+ | Entity/relationship matching |
| `Entity` (ATOM) | atom/models/entity.py | L1-80 | ATOM entity data model |
| `Relationship` (ATOM) | atom/models/relationship.py | L1-80 | ATOM relationship with temporal props |
| `KnowledgeGraph` (ATOM) | atom/models/knowledge_graph.py | L1-160 | ATOM KG with temporal support |
| `Factoid` | atom/models/schemas.py | L1+ | Pydantic schema for factoids |
| `AtomicFact` | atom/models/schemas.py | L1+ | Pydantic schema for atomic facts |
| `Entity` (Schema) | atom/models/schemas.py | L1+ | Pydantic entity for extraction |
| `Relationship` (Schema) | atom/models/schemas.py | L1+ | Pydantic relationship with t_start/t_end |
| `RelationshipsExtractor` | atom/models/schemas.py | L1+ | Pydantic container for relationships |

### iText2KG Module Classes

| Class | File | Line | Purpose |
|-------|------|------|---------|
| `iText2KG` | itext2kg_star/itext2kg.py | L1-121 | Multi-section KG extractor |
| `iEntitiesExtractor` | itext2kg_star/ientities_extraction/ientities_extractor.py | L1-85 | Entity extraction from text |
| `iRelationsExtractor` | itext2kg_star/irelations_extraction/irelations_extractor.py | L1-216 | Relationship extraction & verification |
| `Matcher` | itext2kg_star/graph_matching/matcher.py | L1-155 | Entity/relationship matching |
| `Entity` (iText2KG) | itext2kg_star/models/entity.py | (varies) | iText2KG entity data model |
| `Relationship` (iText2KG) | itext2kg_star/models/relationship.py | (varies) | iText2KG relationship data model |
| `KnowledgeGraph` (iText2KG) | itext2kg_star/models/knowledge_graph.py | L1-160 | iText2KG generic KG |
| `Entity` (Schema) | itext2kg_star/models/schemas.py | L1-197 | Pydantic entity for iText2KG |
| `Relationship` (Schema) | itext2kg_star/models/schemas.py | L1-197 | Pydantic relationship for iText2KG |
| `EntitiesExtractor` | itext2kg_star/models/schemas.py | L1-197 | Pydantic container |
| `RelationshipsExtractor` | itext2kg_star/models/schemas.py | L1-197 | Pydantic container |

### Supporting Classes

| Class | File | Line | Purpose |
|-------|------|------|---------|
| `DocumentsDistiller` | documents_distiller/documents_distiller.py | L1-199 | Document consolidation |
| `Neo4jStorage` | graph_integration/neo4j_storage.py | L1-220 | Neo4j graph storage |
| `GraphStorageInterface` | graph_integration/storage_interface.py | L1-17 | Storage protocol/interface |
| `LangchainOutputParser` | llm_output_parsing/langchain_output_parser.py | L1-100+ | LLM interaction & embeddings |
| `LLMOutputParserInterface` | llm_output_parsing/llm_output_parser_interface.py | (interface) | Parser protocol |
| `ProviderType` | llm_output_parsing/langchain_output_parser.py | L13+ | LLM provider enum |
| `ProviderConfig` | llm_output_parsing/langchain_output_parser.py | L16+ | Rate limiting config |

---

## Method Location Index

### Atom Class Methods

| Method | Line | Signature |
|--------|------|-----------|
| `__init__()` | L14-22 | `def __init__(self, llm_model, embeddings_model)` |
| `extract_quintuples()` | L52-56 | `async def extract_quintuples(self, atomic_facts, observation_timestamp)` |
| `merge_two_kgs()` | L60-66 | `def merge_two_kgs(self, kg1, kg2, rel_threshold, ent_threshold)` |
| `parallel_atomic_merge()` | L68-86 | `def parallel_atomic_merge(self, kgs, existing_kg, ...)` |
| `build_atomic_kg_from_quintuples()` | L88-127 | `async def build_atomic_kg_from_quintuples(self, relationships, ...)` |
| `build_graph()` | L131-161 | `async def build_graph(self, atomic_facts, obs_timestamp, ...)` |
| `build_graph_from_different_obs_times()` | L163+ | `async def build_graph_from_different_obs_times(self, ...)` |

### GraphMatcher Class Methods

| Method | Line | Signature |
|--------|------|-----------|
| `__init__()` | L18 | `def __init__(self)` |
| `_batch_match_entities()` | L25-89 | `def _batch_match_entities(self, entities1, entities2, threshold)` |
| `_batch_match_relationships()` | L94-150 | `def _batch_match_relationships(self, rels1, rels2, threshold)` |
| `match_entities_and_update_relationships()` | (interface) | `def match_entities_and_update_relationships(...)` |

### iText2KG Class Methods

| Method | Line | Signature |
|--------|------|-----------|
| `__init__()` | L12-24 | `def __init__(self, llm_model, embeddings_model, sleep_time)` |
| `build_graph()` | L33-107 | `async def build_graph(self, sections, existing_knowledge_graph, ...)` |

### iEntitiesExtractor Class Methods

| Method | Line | Signature |
|--------|------|-----------|
| `__init__()` | L10-17 | `def __init__(self, llm_model, embeddings_model, sleep_time)` |
| `extract_entities()` | L24-83 | `async def extract_entities(self, context, max_tries, ...)` |

### iRelationsExtractor Class Methods

| Method | Line | Signature |
|--------|------|-----------|
| `__init__()` | L10-19 | `def __init__(self, llm_model, embeddings_model, sleep_time)` |
| `extract_relations()` | L30-107 | `async def extract_relations(self, context, entities, ...)` |
| `extract_verify_and_correct_relations()` | L109-166 | `async def extract_verify_and_correct_relations(self, context, ...)` |

### Matcher Class Methods

| Method | Line | Signature |
|--------|------|-----------|
| `__init__()` | L13 | `def __init__(self)` |
| `find_match()` | L25-62 | `def find_match(self, obj1, list_objects, threshold)` |
| `create_union_list()` | L65-91 | `def create_union_list(self, list1, list2)` |
| `process_lists()` | L92-110 | `def process_lists(self, list1, list2, threshold)` |
| `match_entities_and_update_relationships()` | L112-157+ | `def match_entities_and_update_relationships(...)` |

### DocumentsDistiller Class Methods

| Method | Line | Signature |
|--------|------|-----------|
| `__init__()` | L16-21 | `def __init__(self, llm_model)` |
| `distill()` | L139-161 | `async def distill(self, documents, output_data_structure, IE_query)` |
| `__combine_objects()` | L27-48 | `@staticmethod def __combine_objects(object_list)` |
| `__combine_pydantic_objects()` | L50-95 | `@staticmethod def __combine_pydantic_objects(...)` |
| `__combine_via_dicts()` | L163+ | `@staticmethod def __combine_via_dicts(object_list)` |
| `__merge_field_values()` | L97-130 | `@staticmethod def __merge_field_values(values)` |

### Neo4jStorage Class Methods

| Method | Line | Signature |
|--------|------|-----------|
| `__init__()` | L14-27 | `def __init__(self, uri, username, password, database)` |
| `connect()` | L31-36 | `def connect(self)` |
| `run_query()` | L38-47 | `def run_query(self, query)` |
| `run_query_with_result()` | L152-161 | `def run_query_with_result(self, query)` |
| `create_nodes()` | L166-191 | `def create_nodes(self, knowledge_graph)` |
| `create_relationships()` | L194+ | `def create_relationships(self, knowledge_graph)` |
| `transform_embeddings_to_str_list()` | L49-56 | `@staticmethod def transform_embeddings_to_str_list(embeddings)` |
| `transform_str_list_to_embeddings()` | L58-67 | `@staticmethod def transform_str_list_to_embeddings(embeddings)` |
| `escape_str()` | L69-72 | `@staticmethod def escape_str(s)` |
| `format_value()` | L74-77 | `@staticmethod def format_value(value)` |
| `format_property_value()` | L79-104 | `@staticmethod def format_property_value(key, value)` |
| `sanitize_label()` | (varies) | `@staticmethod def sanitize_label(label)` |
| `sanitize_relationship_type()` | (varies) | `@staticmethod def sanitize_relationship_type(rel_type)` |

### LangchainOutputParser Class Methods

| Method | Line | Signature |
|--------|------|-----------|
| `__init__()` | L61-80 | `def __init__(self, llm_model, embeddings_model, sleep_time, provider_type)` |
| `_detect_provider()` | (varies) | `def _detect_provider(self)` |
| `extract_information_as_json_for_context()` | (varies) | `async def extract_information_as_json_for_context(...)` |
| `calculate_embeddings()` | (varies) | `async def calculate_embeddings(self, texts)` |

### KnowledgeGraph Class Methods

| Method | File | Line | Purpose |
|--------|------|------|---------|
| `__init__()` | both | varies | Constructor |
| `is_empty()` | both | varies | Check if empty |
| `remove_duplicates_entities()` | both | varies | Deduplicate entities |
| `remove_duplicates_relationships()` | both | varies | Deduplicate relationships |
| `embed_entities()` | both | varies | Generate entity embeddings |
| `embed_relationships()` | both | varies | Generate relationship embeddings |
| `get_entity()` | both | varies | Find entity by name+label |
| `get_relationship()` | both | varies | Find relationship by components |
| `add_t_obs_to_relationships()` | ATOM | varies | Add observation timestamps |
| `add_atomic_facts_to_relationships()` | ATOM | varies | Add atomic facts |
| `find_isolated_entities()` | both | varies | Find unconnected entities |
| `split_into_atomic_kgs()` | ATOM | varies | Split into single-relationship KGs |
| `from_neo4j()` | ATOM | varies | Load KG from Neo4j |

### Entity Class Methods

| Method | File | Line | Purpose |
|--------|------|------|---------|
| `process()` | both | varies | Normalize name/label |
| `__eq__()` | both | varies | Equality check |
| `__hash__()` | both | varies | Hash for set operations |
| `__repr__()` | both | varies | String representation |

### Relationship Class Methods

| Method | File | Line | Purpose |
|--------|------|------|---------|
| `process()` | both | varies | Normalize name |
| `combine_timestamps()` | ATOM | varies | Add temporal info |
| `combine_atomic_facts()` | ATOM | varies | Add atomic facts |
| `__eq__()` | both | varies | Equality check |
| `__hash__()` | both | varies | Hash for set operations |
| `__repr__()` | both | varies | String representation |

---

## Schema Classes Index

### ATOM Schemas [atom/models/schemas.py]

| Class | Fields | Purpose |
|-------|--------|---------|
| `Factoid` | `phrase: list[str]` | Temporal factoid guidelines |
| `AtomicFact` | `atomic_fact: list[str]` | Atomic fact extraction schema |
| `Entity` | `label: str`, `name: str` | Entity for quintuples extraction |
| `Relationship` | `startNode: Entity`, `endNode: Entity`, `name: str`, `t_start: list[str]`, `t_end: list[str]` | Temporal relationship schema |
| `RelationshipsExtractor` | `relationships: List[Relationship]` | Container for relationships |
| `EntitiesExtractor` | `entities: List[Entity]` | Container for entities |

### iText2KG Schemas [itext2kg_star/models/schemas.py]

| Class | Fields | Purpose |
|-------|--------|---------|
| `Entity` | `label: str`, `name: str` | Basic entity |
| `Relationship` | `startNode: Entity`, `endNode: Entity`, `name: str` | Basic relationship |
| `RelationshipsExtractor` | `relationships: List[Relationship]` | Container for relationships |
| `EntitiesExtractor` | `entities: List[Entity]` | Container for entities |
| `InformationRetriever` | (company website fields) | Company info extraction |
| `Author`, `ArticleDescription`, `Article` | (article fields) | Scientific article extraction |
| `WorkExperience`, `Education`, `CV` | (CV fields) | Curriculum vitae extraction |
| `Facts` | `facts: list[str]` | Extracted facts container |

---

## Configuration Index

### Models Configuration [models/models_config.py]

| Variable | Purpose |
|----------|---------|
| `model_ollama_llama3` | Local Ollama LLM (Llama 3) |
| `model_openai` | OpenAI GPT model |
| `embeddings_ollama_nomic` | Local Ollama embeddings (Nomic) |
| `embeddings_openai` | OpenAI embeddings |

### Default Model Selection [models/models.py]

```python
_DEFAULT_LLM = model_ollama_llama3
_DEFAULT_EMBEDDINGS = embeddings_ollama_nomic

def get_default_model() -> ChatOpenAI
def get_default_embedding_model() -> OpenAIEmbeddings
```

### Rate Limiting Configuration [llm_output_parsing/langchain_output_parser.py]

| Provider | Max Elements | Max Tokens | Context Window | Sleep |
|----------|--------------|-----------|-----------------|-------|
| OpenAI | 40 | 8000 | 128000 | 2.0s |
| Mistral | 1 | 10000 | 128000 | 0.2s |
| Claude | 50 | 8000 | 200000 | 1.2s |
| Unknown | 5 | 4000 | 32000 | 10.0s |

---

## Entry Point Index

### Main Script [main.py, L1-64]

```python
async def main():
    # Load data
    news_covid = pd.read_pickle("./small_pickle.pkl")
    news_covid_dict = to_dictionary(news_covid)
    
    # Initialize ATOM
    atom = Atom(llm_model=openai_llm_model, 
                embeddings_model=openai_embeddings_model)
    
    # Build KG
    kg = await atom.build_graph_from_different_obs_times(
        atomic_facts_with_obs_timestamps=news_covid_dict
    )
    
    # Visualize
    Neo4jStorage(...).visualize_graph(knowledge_graph=kg)

if __name__ == "__main__":
    asyncio.run(main())
```

**Key Helper Function**:
- `to_dictionary(df, max_elements)` [L29] - Convert DataFrame to timestamp-factoids dict

---

## Test Files Index

| File | Purpose | Line Coverage |
|------|---------|---|
| `tests/atom/test_atom_matching.py` | Test ATOM entity/relationship matching | (varies) |
| `tests/itext2kg/test_itext2kg_matching.py` | Test iText2KG section consolidation | (varies) |

---

## Evaluation Scripts Index

| Script | File | Purpose |
|--------|------|---------|
| Exhaustivity | `evaluation/exhaustivity/` | Measure factoid extraction coverage |
| Latency | `evaluation/latency/` | Performance comparison |
| Quintuples Quality | `evaluation/quintuples_quality/` | Assess triple quality |
| Stability | `evaluation/stability/` | Test consistency |
| Merge Quality | `evaluation/merge/` | Evaluate graph merge effectiveness |

---

## Import Paths Reference

### Public Exports
```python
from itext2kg import (
    Atom,
    iText2KG,
    iText2KG_Star,
    Neo4jStorage,
    DocumentsDistiller,
    LangchainOutputParser
)
```

### Component Imports
```python
from itext2kg.atom import Atom, GraphMatcher
from itext2kg.itext2kg_star import iText2KG
from itext2kg.itext2kg_star.ientities_extraction import iEntitiesExtractor
from itext2kg.itext2kg_star.irelations_extraction import iRelationsExtractor
from itext2kg.itext2kg_star.graph_matching import Matcher
from itext2kg.documents_distiller import DocumentsDistiller
from itext2kg.graph_integration import Neo4jStorage, GraphStorageInterface
from itext2kg.llm_output_parsing import LangchainOutputParser
```

### Model Imports
```python
from itext2kg.atom.models import Entity, Relationship, KnowledgeGraph
from itext2kg.atom.models.schemas import Relationship as RelationshipSchema
from itext2kg.itext2kg_star.models import Entity, Relationship, KnowledgeGraph
```

---

## Quick Search Guide

### How to find...

**The entry point** → [main.py, L1-64](main.py#L1-L64)

**ATOM temporal processing** → [atom/atom.py, L131-161](atom/atom.py#L131-L161)

**Entity/relationship matching** → [atom/graph_matching/matcher.py, L25-157](atom/graph_matching/matcher.py#L25-L157)

**Document section processing** → [itext2kg_star/itext2kg.py, L33-107](itext2kg_star/itext2kg.py#L33-L107)

**Entity extraction** → [itext2kg_star/ientities_extraction/ientities_extractor.py, L24-83](itext2kg_star/ientities_extraction/ientities_extractor.py#L24-L83)

**Relationship extraction** → [itext2kg_star/irelations_extraction/irelations_extractor.py, L109-166](itext2kg_star/irelations_extraction/irelations_extractor.py#L109-L166)

**Document consolidation** → [documents_distiller/documents_distiller.py, L139-161](documents_distiller/documents_distiller.py#L139-L161)

**Neo4j storage** → [graph_integration/neo4j_storage.py, L14-220](graph_integration/neo4j_storage.py#L14-L220)

**LLM interaction** → [llm_output_parsing/langchain_output_parser.py, L61-80](llm_output_parsing/langchain_output_parser.py#L61-L80)

**Entity data model (temporal)** → [atom/models/entity.py, L1-80](atom/models/entity.py#L1-L80)

**Relationship data model (temporal)** → [atom/models/relationship.py, L1-80](atom/models/relationship.py#L1-L80)

**Knowledge graph data model (temporal)** → [atom/models/knowledge_graph.py, L1-160](atom/models/knowledge_graph.py#L1-L160)

**Pydantic schemas** → [atom/models/schemas.py, L1-260](atom/models/schemas.py#L1-L260) (ATOM) or [itext2kg_star/models/schemas.py, L1-197](itext2kg_star/models/schemas.py#L1-L197) (iText2KG)

