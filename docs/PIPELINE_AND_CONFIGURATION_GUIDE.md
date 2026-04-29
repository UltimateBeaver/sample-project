# iText2KG & ATOM Pipeline Documentation

## 📋 Table of Contents
1. [Configuration Mismatches](#configuration-mismatches)
2. [Theoretical Pipeline Overview](#theoretical-pipeline-overview)
3. [Key Classes & Objects](#key-classes--objects)
4. [Method Reference by Pipeline Stage](#method-reference-by-pipeline-stage)
5. [Execution Flow Diagrams](#execution-flow-diagrams)
6. [Quick Integration Guide](#quick-integration-guide)

---

## 🔴 Configuration Mismatches

### **CRITICAL: Type Annotation Mismatch in `models.py`**

**Location:** `models/models.py` lines 16-19

**Issue:**
```python
def get_default_embedding_model() -> OpenAIEmbeddings:
    """Return the default embeddings model instance."""
    return _DEFAULT_EMBEDDINGS
```

**Mismatch:** 
- **Declared Return Type:** `OpenAIEmbeddings` (from `langchain_openai`)
- **Actual Return Type:** `OllamaEmbeddings` (from `langchain_ollama`)
- **Assigned Value:** `embeddings_ollama_nomic` (line 15)

**Current Configuration State:**
```python
_DEFAULT_LLM        = model_ollama_llama3        # Local Ollama (Qwen 3.6 27B)
_DEFAULT_EMBEDDINGS = embeddings_ollama_nomic    # Local Ollama (nomic-embed-text)
```

**Recommendation:**
Fix the type annotation to reflect actual runtime type:
```python
from langchain_ollama import OllamaEmbeddings

def get_default_embedding_model() -> OllamaEmbeddings:
    """Return the default embeddings model instance."""
    return _DEFAULT_EMBEDDINGS
```

Or use a Union type for flexibility:
```python
from typing import Union
from langchain_openai import OpenAIEmbeddings
from langchain_ollama import OllamaEmbeddings

def get_default_embedding_model() -> Union[OpenAIEmbeddings, OllamaEmbeddings]:
    """Return the default embeddings model instance."""
    return _DEFAULT_EMBEDDINGS
```

### **Configuration Dependencies**

| Config Variable | Location | Type | Current Value | Purpose |
|---|---|---|---|---|
| `model_ollama_llama3` | `models_config.py:69` | `ChatOllama` | `qwen3.6:27b-q4_K_M` | LLM for entity/relation extraction |
| `embeddings_ollama_nomic` | `models_config.py:109` | `OllamaEmbeddings` | `nomic-embed-text` | Entity/relation embeddings |
| `ollama_base_url` | `env_config.py:9` | str | `http://localhost:11434/v1` | Ollama service endpoint |

---

## 🏗️ Theoretical Pipeline Overview

Based on the paper (ATOM architecture), the codebase implements **two distinct pipelines**:

### **Pipeline 1: ATOM (Temporal Atomic Facts to Knowledge Graphs)**

**Input:** Atomic facts with observation timestamps (dictionary format)
**Output:** Temporal Knowledge Graph with entities, relationships, and temporal bounds

**Flow:**
```
Atomic Facts Dict {t₁: [facts], t₂: [facts], ...}
    ↓
[1] EXTRACT QUINTUPLES (extract_quintuples)
    ↓ (Per factoid: RelationshipsExtractor)
    ├─ Entity pair extraction (start_node, end_node)
    └─ Relationship extraction (predicate, t_start, t_end)
    ↓
[2] BUILD ATOMIC KG (build_atomic_kg_from_quintuples)
    ↓
    ├─ Embed entities (entity_name × 0.8 + entity_label × 0.2)
    ├─ Embed relationships (predicate embeddings)
    ├─ Create atomic KGs (1 relationship per KG)
    └─ Remove duplicates
    ↓
[3] MERGE PARALLEL (parallel_atomic_merge)
    ↓ (Binary tree merge, max_workers=8)
    ├─ Match entities (exact + cosine similarity)
    ├─ Update relationships (merge properties)
    └─ Recursively merge until 1 KG
    ↓
[4] STORE (Neo4jStorage.visualize_graph)
    ↓
Final Temporal KG with t_start, t_end, t_obs
```

**Key Characteristics:**
- Tracks observation timestamps (`t_obs`)
- Temporal relationship bounds (`t_start`, `t_end`)
- Parallel merge strategy using thread pools
- Atomic fact embeddings for retrieval

---

### **Pipeline 2: iText2KG (Document Sections to Knowledge Graphs)**

**Input:** List of document sections (strings)
**Output:** Non-temporal Knowledge Graph with global entities and relationships

**Flow:**
```
Document Sections [section₁, section₂, ..., sectionₙ]
    ↓
[1] EXTRACT ENTITIES FROM FIRST SECTION (iEntitiesExtractor.extract_entities)
    ↓ (returns: Entity[] with embeddings)
    ├─ LLM extraction
    └─ Embed: entity_name × 0.6 + entity_label × 0.4
    ↓
[2] EXTRACT RELATIONS FROM FIRST SECTION (iRelationsExtractor.extract_verify_and_correct_relations)
    ↓ (returns: Relationship[] with embeddings)
    ├─ Extract candidate relations
    ├─ Verify against entity list (retry if invalid)
    └─ Correct isolated entities (max_tries=5)
    ↓
FOR EACH REMAINING SECTION:
    ├─ Extract entities
    ├─ Match with global entities (threshold=0.7)
    ├─ Consolidate duplicates
    ├─ Extract relations
    ├─ Verify & correct relations
    └─ Merge with global relations
    ↓
[3] MERGE WITH EXISTING KG (if provided)
    ↓ (Matcher.match_entities_and_update_relationships)
    ├─ Match entities across KGs
    └─ Update relationship endpoints
    ↓
[4] REMOVE DUPLICATES
    ↓
[5] STORE (Neo4jStorage.create_nodes, create_relationships)
    ↓
Final Non-temporal KG
```

**Key Characteristics:**
- Section-by-section incremental merging
- Entity deduplication across sections
- Relationship verification loop (ensures all relations reference known entities)
- Configurable matching thresholds

---

## 📦 Key Classes & Objects

### **Core Data Models**

#### **Entity** (itext2kg/atom/models/entity.py)
```python
class Entity(BaseModel):
    label: str              # Semantic category (Person, Location, Event, etc.)
    name: str               # Unique identifier
    properties: EntityProperties
        embeddings: np.ndarray  # Label + Name weighted embeddings
        
# Usage: Entity(label="Person", name="Steve Jobs")
```

#### **Relationship** (itext2kg/atom/models/relationship.py)
```python
class Relationship(BaseModel):
    name: str                   # Predicate (e.g., "is_CEO", "works_at")
    startEntity: Entity         # Subject
    endEntity: Entity           # Object
    properties: RelationshipProperties
        embeddings: np.ndarray     # Predicate embeddings
        t_start: List[float]       # Start timestamps (Unix format)
        t_end: List[float]         # End timestamps (Unix format)
        t_obs: List[float]         # Observation timestamps (ATOM only)
        atomic_facts: List[str]    # Original source facts (ATOM only)

# Usage in ATOM: Relationship(name="is_CEO", startEntity=..., endEntity=...)
```

#### **KnowledgeGraph** (itext2kg/atom/models/knowledge_graph.py)
```python
class KnowledgeGraph(BaseModel):
    entities: List[Entity]
    relationships: List[Relationship]
    
# Key Methods:
# - embed_entities(embeddings_function, weights)
# - embed_relationships(embeddings_function)
# - split_into_atomic_kgs() -> List[KnowledgeGraph]
# - add_t_obs_to_relationships(t_obs)
```

#### **Schema Classes** (itext2kg/atom/models/schemas.py)
```python
class Relationship(BaseModel):
    startNode: Entity           # Schema node (for LLM extraction)
    endNode: Entity
    name: str                   # Predicate (PRESENT TENSE)
    t_start: List[str]          # String dates (e.g., "01-01-2023")
    t_end: List[str]            # String dates

class RelationshipsExtractor(BaseModel):
    relationships: List[Relationship]  # LLM output wrapper
    
# Usage: LLM returns RelationshipsExtractor instance
```

---

### **Pipeline Orchestrators**

#### **ATOM** (itext2kg/atom/atom.py)
```python
class Atom:
    def __init__(self, llm_model, embeddings_model):
        self.matcher = GraphMatcher()
        self.llm_output_parser = LangchainOutputParser(...)
    
    # Main Entry Point
    async def build_graph_from_different_obs_times(
        atomic_facts_with_obs_timestamps: Dict[str, List[str]]
    ) -> KnowledgeGraph:
        """
        Builds temporal KG from atomic facts with observation timestamps.
        
        Args:
            atomic_facts_with_obs_timestamps: {
                "2020-01-09": ["Fact 1", "Fact 2", ...],
                "2020-01-23": ["Fact 3", "Fact 4", ...],
                ...
            }
        
        Returns:
            KnowledgeGraph with t_start, t_end, t_obs properties
        """
    
    # Stage 1: Extract quintuples
    async def extract_quintuples(
        atomic_facts: List[str],
        observation_timestamp: str
    ) -> List[RelationshipsExtractor]
    
    # Stage 2: Build atomic KG
    async def build_atomic_kg_from_quintuples(
        relationships: List[RelationshipSchema],
        entity_name_weight: float = 0.8,
        entity_label_weight: float = 0.2,
        rel_threshold: float = 0.8,
        ent_threshold: float = 0.8,
        max_workers: int = 8
    ) -> KnowledgeGraph
    
    # Stage 3: Merge pairwise
    def merge_two_kgs(
        kg1: KnowledgeGraph,
        kg2: KnowledgeGraph,
        rel_threshold: float = 0.8,
        ent_threshold: float = 0.8
    ) -> KnowledgeGraph
    
    # Stage 3 (Parallel): Merge tree
    def parallel_atomic_merge(
        kgs: List[KnowledgeGraph],
        existing_kg: Optional[KnowledgeGraph] = None,
        rel_threshold: float = 0.8,
        ent_threshold: float = 0.8,
        max_workers: int = 4
    ) -> KnowledgeGraph
```

#### **iText2KG** (itext2kg/itext2kg_star/itext2kg.py)
```python
class iText2KG:
    def __init__(self, llm_model, embeddings_model, sleep_time: int = 5):
        self.ientities_extractor = iEntitiesExtractor(...)
        self.irelations_extractor = iRelationsExtractor(...)
        self.matcher = Matcher()
    
    # Main Entry Point
    async def build_graph(
        sections: List[str],
        existing_knowledge_graph: KnowledgeGraph = None,
        ent_threshold: float = 0.7,
        rel_threshold: float = 0.7,
        max_tries: int = 5,
        max_tries_isolated_entities: int = 3,
        entity_name_weight: float = 0.6,
        entity_label_weight: float = 0.4,
        observation_date: str = ""
    ) -> KnowledgeGraph:
        """
        Builds non-temporal KG from document sections.
        
        Process:
        1. Extract entities from first section
        2. Extract & verify relations
        3. For each subsequent section:
           - Extract entities → match & consolidate
           - Extract & verify relations → merge
        4. Match with existing KG (if provided)
        5. Remove duplicates
        
        Returns:
            KnowledgeGraph with global consolidated entities/relations
        """
```

---

### **Supporting Components**

#### **GraphMatcher** (itext2kg/atom/graph_matching/matcher.py)
```python
class GraphMatcher:
    def _batch_match_entities(
        entities1: List[Entity],
        entities2: List[Entity],
        threshold: float = 0.8
    ) -> Tuple[List[Entity], List[Entity]]:
        """
        Matches entities using:
        1. Exact match (name + label equality)
        2. Cosine similarity (embeddings, if exact miss)
        
        Returns:
            (matched_entities1, union_of_both_lists)
        """
    
    def match_entities_and_update_relationships(
        entities_1: List[Entity],
        relationships_1: List[Relationship],
        entities_2: List[Entity],
        relationships_2: List[Relationship],
        rel_threshold: float = 0.8,
        ent_threshold: float = 0.8
    ) -> Tuple[List[Entity], List[Relationship]]:
        """
        Matches entities across two KGs and updates relationship endpoints.
        """
```

#### **Matcher** (itext2kg/itext2kg_star/graph_matching/matcher.py)
```python
class Matcher:
    def process_lists(
        list1: List[Union[Entity, Relationship]],
        list2: List[Union[Entity, Relationship]],
        threshold: float = 0.7
    ) -> Tuple[List, List]:
        """
        Incremental matcher for iText2KG pipeline.
        Returns: (processed_list1, updated_list2)
        """
    
    def match_entities_and_update_relationships(...) -> Tuple[List, List]:
        """Same signature as GraphMatcher."""
```

#### **LangchainOutputParser** (itext2kg/llm_output_parsing/langchain_output_parser.py)
```python
class LangchainOutputParser:
    async def extract_information_as_json_for_context(
        output_data_structure: Type[BaseModel],  # RelationshipsExtractor, etc.
        contexts: List[str],
        system_query: str,
        max_retries: int = 2
    ) -> List[output_data_structure]:
        """
        Calls LLM to extract structured info (with retry logic).
        
        - Handles rate limits with backoff
        - Validates JSON parsing
        - Converts timestamps (relative→absolute)
        """
    
    async def calculate_embeddings(
        texts: List[str]
    ) -> np.ndarray:
        """
        Computes embeddings asynchronously.
        Shape: (len(texts), embedding_dim)
        """
```

#### **Neo4jStorage** (itext2kg/graph_integration/neo4j_storage.py)
```python
class Neo4jStorage(GraphStorageInterface):
    def __init__(self, uri: str, username: str, password: str, database: Optional[str] = None):
        self.driver = GraphDatabase.driver(uri, auth=(username, password))
    
    def visualize_graph(self, knowledge_graph: KnowledgeGraph):
        """
        Converts KnowledgeGraph to Neo4j nodes and relationships.
        - Creates Entity nodes with properties (embeddings, label, name)
        - Creates Relationship edges with t_start, t_end, t_obs
        """
    
    def create_nodes(self, entities: List[Entity]):
        """Creates Neo4j nodes from entities."""
    
    def create_relationships(self, relationships: List[Relationship]):
        """Creates Neo4j relationships with properties."""
```

---

## 🔄 Method Reference by Pipeline Stage

### **Stage 1: Entity & Relationship Extraction**

| Method | Location | Input | Output | Async | Purpose |
|---|---|---|---|---|---|
| `extract_entities()` | itext2kg_star/ientities_extraction/ | str (context) | Entity[] | ✓ | Extracts entities from text |
| `extract_quintuples()` | atom/atom.py:32 | List[str] (facts), str (timestamp) | RelationshipsExtractor[] | ✓ | Extracts temporal relationships |
| `extract_verify_and_correct_relations()` | itext2kg_star/irelations_extraction/ | str (context), Entity[] | Relationship[] | ✓ | Validates relationships against entity list |

### **Stage 2: Embedding & Normalization**

| Method | Location | Input | Output | Async | Notes |
|---|---|---|---|---|---|
| `embed_entities()` | atom/models/knowledge_graph.py | KG, embeddings_func, weights | void (modifies KG) | ✓ | Applies weighted combination: name×0.8 + label×0.2 |
| `embed_relationships()` | atom/models/knowledge_graph.py | KG, embeddings_func | void (modifies KG) | ✓ | Embeds relationship predicates |
| `calculate_embeddings()` | llm_output_parsing/ | List[str] (texts) | np.ndarray | ✓ | Batch embedding computation |

### **Stage 3: Matching & Merging**

| Method | Location | Input | Output | Match Strategy |
|---|---|---|---|---|
| `_batch_match_entities()` | atom/graph_matching/matcher.py:25 | Entity[], Entity[], threshold | (Entity[], Entity[]) | Exact + Cosine |
| `match_entities_and_update_relationships()` | atom/graph_matching/matcher.py:100 | Ent[], Rel[], Ent[], Rel[], thresholds | (Ent[], Rel[]) | Cascading match |
| `merge_two_kgs()` | atom/atom.py:50 | KG, KG, thresholds | KG | Entity match + Rel update |
| `parallel_atomic_merge()` | atom/atom.py:65 | KG[], threshold, max_workers | KG | Binary tree merge (threads) |

### **Stage 4: Storage & Visualization**

| Method | Location | Input | Output |
|---|---|---|---|
| `visualize_graph()` | graph_integration/neo4j_storage.py | KnowledgeGraph | Cypher queries → Neo4j |
| `create_nodes()` | graph_integration/neo4j_storage.py | Entity[] | Cypher CREATE node statements |
| `create_relationships()` | graph_integration/neo4j_storage.py | Relationship[] | Cypher CREATE relationship statements |

---

## 🎯 Execution Flow Diagrams

### **ATOM Pipeline Execution (main.py)**

```
ENTRY: main.py:main()
  ↓
  1. Load pickle: atomic_facts_with_obs_timestamps
     Dict[str, List[str]]  ← news_covid_dict
     Keys: observation dates, Values: atomic facts
  ↓
  2. Initialize ATOM(llm_model, embeddings_model)
     - GraphMatcher()
     - LangchainOutputParser()
  ↓
  3. AWAIT atom.build_graph_from_different_obs_times(news_covid_dict)
     │
     ├─ FOR EACH (timestamp, facts) in news_covid_dict:
     │  ├─ AWAIT extract_quintuples(facts, timestamp)
     │  │  └─ LLM extraction → RelationshipsExtractor
     │  │
     │  └─ AWAIT build_atomic_kg_from_quintuples(relationships)
     │     ├─ Embed entities (weighted: name + label)
     │     ├─ Embed relationships (predicates)
     │     ├─ split_into_atomic_kgs() → KG per relationship
     │     └─ parallel_atomic_merge(atomic_kgs)
     │        ├─ Binary tree merge (ThreadPoolExecutor)
     │        ├─ Match entities (exact + cosine)
     │        └─ Merge relationships
     │
     └─ FINAL: temporal_kg (merged across all timestamps)
  ↓
  4. Neo4jStorage(uri, username, password)
     └─ visualize_graph(temporal_kg)
        ├─ create_nodes(entities)
        ├─ create_relationships(relationships)
        └─ → Neo4j database

EXIT
```

### **iText2KG Pipeline Execution (custom integration)**

```
ENTRY: iText2KG(llm_model, embeddings_model)
  ↓
  1. Initialize components:
     - iEntitiesExtractor
     - iRelationsExtractor
     - Matcher
  ↓
  2. AWAIT build_graph(sections=[section1, section2, ...])
     │
     ├─ SECTION 0:
     │  ├─ AWAIT extract_entities(sections[0])
     │  │  └─ Entity[] (with embeddings: name×0.6 + label×0.4)
     │  │
     │  └─ AWAIT extract_verify_and_correct_relations(sections[0])
     │     └─ Relationship[] (verified against entities)
     │
     ├─ FOR i IN 1..n:
     │  ├─ AWAIT extract_entities(sections[i])
     │  │  └─ Entity[] (local)
     │  │
     │  ├─ process_lists(local_entities, global_entities, threshold=0.7)
     │  │  └─ Consolidate duplicates
     │  │
     │  ├─ AWAIT extract_verify_and_correct_relations(sections[i])
     │  │  └─ Relationship[] (local)
     │  │
     │  └─ process_lists(local_rels, global_rels, threshold=0.7)
     │     └─ Merge relationships
     │
     ├─ IF existing_knowledge_graph provided:
     │  └─ match_entities_and_update_relationships(
     │     global_entities, global_relationships,
     │     existing_entities, existing_relationships)
     │
     ├─ remove_duplicates_entities()
     ├─ remove_duplicates_relationships()
     │
     └─ RETURN constructed_kg

EXIT: KnowledgeGraph (non-temporal)
```

---

## 📝 Quick Integration Guide

### **ATOM Pipeline (Temporal KG from Atomic Facts)**

```python
import asyncio
from itext2kg.atom import Atom
from itext2kg import Neo4jStorage
from models.models import get_default_model, get_default_embedding_model

async def build_temporal_kg():
    # 1. Initialize
    llm = get_default_model()
    embeddings = get_default_embedding_model()
    atom = Atom(llm_model=llm, embeddings_model=embeddings)
    
    # 2. Prepare data (Dict[str, List[str]])
    atomic_facts_dict = {
        "2024-01-09": [
            "Virus identified in Wuhan on December 2019",
            "Virus spread to 10 other countries"
        ],
        "2024-01-23": [
            "Wuhan coronavirus caused 26 deaths in China",
            "Death toll increased by January 27"
        ]
    }
    
    # 3. Build temporal KG
    temporal_kg = await atom.build_graph_from_different_obs_times(
        atomic_facts_with_obs_timestamps=atomic_facts_dict
    )
    
    # 4. Store in Neo4j
    storage = Neo4jStorage(
        uri="bolt://localhost:7687",
        username="neo4j",
        password="your_password"
    )
    storage.visualize_graph(temporal_kg)
    
    return temporal_kg

# Run
asyncio.run(build_temporal_kg())
```

### **iText2KG Pipeline (Non-Temporal KG from Document Sections)**

```python
import asyncio
from itext2kg.itext2kg_star import iText2KG
from models.models import get_default_model, get_default_embedding_model

async def build_document_kg():
    # 1. Initialize
    llm = get_default_model()
    embeddings = get_default_embedding_model()
    itext = iText2KG(llm_model=llm, embeddings_model=embeddings, sleep_time=5)
    
    # 2. Prepare document sections
    sections = [
        "Apple Inc. was founded by Steve Jobs...",
        "In 2007, Apple launched the iPhone 2G...",
        "Steve Jobs designs online services..."
    ]
    
    # 3. Build KG from sections
    kg = await itext.build_graph(
        sections=sections,
        ent_threshold=0.7,
        rel_threshold=0.7,
        entity_name_weight=0.6,
        entity_label_weight=0.4
    )
    
    # 4. (Optional) Merge with existing KG
    kg_with_history = await itext.build_graph(
        sections=new_sections,
        existing_knowledge_graph=kg
    )
    
    return kg_with_history

# Run
asyncio.run(build_document_kg())
```

### **Configuration Switching**

```python
# In models/models.py, switch these two lines:

# Option 1: Use OpenAI GPT-4o
_DEFAULT_LLM        = model_gpt4o_mini            # GPT-4o Mini
_DEFAULT_EMBEDDINGS = embeddings_text_embedding_3_small  # text-embedding-3-small

# Option 2: Use Together AI (Free)
_DEFAULT_LLM        = model_llama3_3_70b          # Llama 3.3 70B
_DEFAULT_EMBEDDINGS = embeddings_bge_large        # BAAI/bge-large

# Option 3: Use Local Ollama (Current - requires local setup)
_DEFAULT_LLM        = model_ollama_llama3         # Qwen 3.6 27B
_DEFAULT_EMBEDDINGS = embeddings_ollama_nomic     # nomic-embed-text
```

---

## 🔍 Key Parameters & Their Effects

### **Entity Matching Parameters**

| Parameter | Default | Effect | Stage |
|---|---|---|---|
| `entity_name_weight` | 0.8 (ATOM), 0.6 (iText2KG) | Higher → name more important | Embedding |
| `entity_label_weight` | 0.2 (ATOM), 0.4 (iText2KG) | Higher → label more important | Embedding |
| `ent_threshold` | 0.8 (ATOM), 0.7 (iText2KG) | Cosine similarity cutoff for match | Matching |

### **Relationship Matching Parameters**

| Parameter | Default | Effect | Stage |
|---|---|---|---|
| `rel_threshold` | 0.8 (ATOM), 0.7 (iText2KG) | Cosine similarity cutoff | Matching |

### **Extraction Parameters**

| Parameter | Default | Effect | Stage |
|---|---|---|---|
| `max_tries` | 5 | Retry attempts for relation verification | iText2KG only |
| `max_tries_isolated_entities` | 3 | Retry attempts for entities without relations | iText2KG only |
| `max_workers` | 8 | Thread pool size for parallel merge | ATOM merge stage |

---

## 📚 Class Hierarchy & Inheritance

```
BaseModel (Pydantic)
├── Entity
├── Relationship
├── KnowledgeGraph
├── Entity (schemas.py - LLM output)
├── Relationship (schemas.py - LLM output)
├── RelationshipsExtractor
└── ...other schemas

GraphStorageInterface (ABC)
└── Neo4jStorage

GraphMatcherInterface (ABC)
├── GraphMatcher (ATOM)
└── Matcher (iText2KG)

LangchainOutputParser
├── extract_information_as_json_for_context()
└── calculate_embeddings()
```

---

## ⚠️ Important Notes

1. **Temporal Semantics (ATOM only):**
   - All relationship names must be in **PRESENT TENSE** (e.g., "is_CEO", not "was_CEO")
   - Temporal bounds stored separately: `t_start`, `t_end`
   - Observation timestamp `t_obs` tracks when the fact was observed

2. **Embedding Weights:**
   - ATOM uses name-heavy embeddings (0.8 name, 0.2 label)
   - iText2KG uses balanced embeddings (0.6 name, 0.4 label)
   - This affects entity consolidation accuracy

3. **Entity Deduplication:**
   - First pass: exact match (name + label equality)
   - Second pass: cosine similarity on embeddings
   - Threshold-based acceptance

4. **Asynchronous Execution:**
   - Both pipelines use `async/await` extensively
   - Always call with `asyncio.run()` or `await` in async context
   - Rate limiting built into `LangchainOutputParser`

5. **Neo4j Schema:**
   - Entities become `:Entity` nodes with properties:
     - `name`, `label`, `embeddings_str`
   - Relationships become edges with properties:
     - `predicate`, `t_start`, `t_end`, `t_obs`, `atomic_facts`, `embeddings_str`

