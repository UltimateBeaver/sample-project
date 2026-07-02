# iText2KG & ATOM - Pipeline & Configuration Guide

## 📋 Table of Contents

1. [Pipeline Overview](#pipeline-overview)
2. [LLM & Model Setup](#llm--model-setup)
3. [Translation Service](#translation-service)
4. [Configuration Validation](#configuration-validation)
5. [Logging & Debugging](#logging--debugging)
6. [Schema classes](#schema-classes-itext2kgatommodelsschemaspy)
7. [Methods reference by pipeline stage](#-method-reference-by-pipeline-stage)
8. [Execution flow diagrams](#-execution-flow-diagrams)
9. [Quick integration guide](#-quick-integration-guide)
10. [Key parameters and their effects](#-key-parameters--their-effects)
11. [Important notes](#️-important-notes)

---

## Pipeline Overview

### ATOM: Temporal KG from Atomic Facts

**Purpose**: Extracts temporal knowledge graphs from atomic facts grouped by observation timestamp

**Input Format**:

```python
{
    "2020-01-09": ["Virus identified in Wuhan", "Initial cases confirmed"],
    "2020-01-23": ["Cases spread to 10 countries", "First death reported"],
    ...
}
```

**Processing Steps**:

1. **Extract Quintuples** — LLM extracts (subject, predicate, object, t_start, t_end) from each fact
2. **Build Atomic KGs** — Create individual KGs with embeddings (1 relationship per KG)
3. **Parallel Merge** — Binary tree merge using thread pool for efficiency
4. **Store in Neo4j** — Persist with temporal properties

**Output**: Single temporal KnowledgeGraph with:

- Global entity set
- Merged relationships with `t_start`, `t_end`, `t_obs` timestamps

**Entry Point**: `Atom.build_graph_from_different_obs_times(atomic_facts_dict)`

---

### iText2KG: Non-Temporal KG from Document Sections

**Purpose**: Extracts consolidated knowledge graph from multiple document sections

**Input Format**:

```python
[
    "Section text with entities and relationships...",
    "Another section with more information...",
    ...
]
```

**Processing Steps**:

1. Extract entities from first section
2. Extract and verify relationships for first section
3. For each remaining section:
    - Extract entities → match with global entities
    - Extract relationships → verify and correct → merge
4. Optional: merge with existing KG
5. Remove duplicates

**Output**: Single non-temporal KnowledgeGraph with global consolidated entities/relationships

**Entry Point**: `iText2KG.build_graph(sections_list)`

---

## LLM & Model Setup

### Current Configuration

**LLM**: Local Llama.cpp (Gemma 4 model)  
**Embeddings**: Local Llama.cpp (Nomic Embed Text, 1024-dim)  
**Location**: `models/models.py`

```python
_DEFAULT_LLM = model_llamacpp_gemma4
_DEFAULT_EMBEDDINGS = embeddings_llamacpp_nomic

def get_default_model():
    return _DEFAULT_LLM

def get_default_embedding_model():
    return _DEFAULT_EMBEDDINGS
```

### Switching Models

Depending on which models you want to use, there are several python LangChain APIs. In the original ATOM framework code, only langchain_openai was used. I've added the support for ollama through langchain_ollama.

**Example: how to use another LLM backend through ChatOpenAI python API**:

1. Look for the following variables in `.env` file:

    ```
    OPENAI_API_KEY=your-key-here
    OPENAI_API_BASE=https://api.openai.com/v1  # Optional for custom base
    ### Langchain Output Parser: Provider-specific configurations
    ### NOTE: the following configs act as a safeguard to prevent exceeding the maximum context window of the LLM provider. If you encounter errors related to context window limits, consider adjusting these values.
    # ProviderType.OPENAI
    PROVIDER_<PROVIDER_NAME>_MAX_ELEMENTS_PER_BATCH=8
    PROVIDER_<PROVIDER_NAME>_MAX_TOKENS_PER_BATCH=8192
    PROVIDER_<PROVIDER_NAME>_MAX_CONTEXT_WINDOW=16384
    ```

2. Define a configuration (or use an existing one) for the llm model and embedding model you want to use in `models_config.py`

    ```python
    # Example for llama.cpp, using langchain_openai API
    # LLM model (local llama.cpp Gemma4)
    model_llamacpp_gemma4 = ChatOpenAI(
        api_key=openai_api_key,
        base_url=openai_api_base,
        model="gemma4",  # The server ignores this string, but ChatOpenAI requires it
        temperature=0,
        streaming=False,
    )
    # Embedding model (local llama.cpp nomic-embed-text)
    embeddings_llamacpp_nomic = OpenAIEmbeddings(
        api_key="llama_cpp", # Dummy key required by LangChain
        base_url=llamacpp_embed_base,
        model="nomic-embed-text",
    )
    ```

3. Update `models.py`:

    ```python
    _DEFAULT_LLM = model_gpt4o_mini  # or other OpenAI model
    _DEFAULT_EMBEDDINGS = embeddings_text_embedding_3_small
    ```

4. Make sure to properly configure your backend infrastructure (llama.cpp, Ollama, ...). For more info see 'Installation' section on [README.md](../README.md)

**Available Models** (see `models_config.py`):
- **Llama.cpp** (DEFAULT): `model_llamacpp_gemma4`. (local server on port 8080)
- **Ollama** (Alternative): `model_ollama_gemma4` (if Ollama server running on port 11434)

---

## Translation Service

### Purpose

Automatically translates non-English documents (e.g., Italian) to English before processing. Improves LLM extraction accuracy by providing prompts and data in consistent language.

### Setup

**1. Enable in `.env`:**

```
ENABLE_TRANSLATION=true
```

**2. Optional: Use few-shot examples:**

```
ENABLE_TRANSLATOR_FEW_SHOT=true  # Provides 5-sample examples for better translation
TRANSLATOR_BATCH_SIZE=8           # Parallel translations (adjust if OOM)
```

### Features

- **Device Auto-Detection**: GPU (CUDA/ROCm) → DirectML (Windows AMD) → CPU
- **Batch Processing**: Translates multiple paragraphs in parallel
- **Sentiment Analysis** (optional): Optional per-paragraph sentiment scores (1-5 scale)

### How It Works

1. Detects language of input paragraph
2. If non-English → calls LLM translation
3. Returns (English translation, optional sentiment)
4. Cached for repeated texts
5. Processing continues with English text + LLM prompts

---

## Configuration Validation

### Pre-Run Checks

Before executing the pipeline, validate your configuration:

**Location**: `sanity_checks/test_config.py`

**Validates**:

- ✅ LLM endpoint responds

**Run validation**:

```bash
python sanity_checks/test_config.py
```

---

## Logging & Debugging

### Logging Configuration

**Location**: `itext2kg_atom/itext2kg/logging_config.py`

**Default Setup**:

```python
from itext2kg_atom.itext2kg.logging_config import setup_logging

# INFO level for app, WARNING for LangChain (better performance)
setup_logging(level="INFO", langchain_level="WARNING")

# DEBUG with full LangChain output (verbose)
setup_logging(level="DEBUG", langchain_level="DEBUG")
```

### Why Suppress LangChain Debug Output?

LangChain debug logging causes ~2x slowdown:

- Logs every LLM request/response
- Logs embedding calculations
- Logs prompt templates and token counts

**Recommendation**: Keep `langchain_level="WARNING"` for production. Use `"DEBUG"` only when troubleshooting specific issues.

### Common Issues & Solutions

**Issue**: `t_start`/`t_end` fields return `None` instead of `[]`

**Cause**: Language mismatch (English prompts + non-English text)  
**Solution**: Set `ENABLE_TRANSLATION=true` in `.env`

**Issue**: Extraction failures with "Unexpected error in batch" messages

**Cause**: LLM rate limits or context window exceeded  
**Solution**:

- Reduce `DOC_PARSER_BATCH_SIZE`
- Reduce `PROVIDER_*_MAX_ELEMENTS_PER_BATCH`
- Use a larger model or OpenAI (higher limits)

**Issue**: Neo4j connection refused

**Cause**: Database not running or wrong credentials  
**Solution**:

```bash
docker-compose up -d  # Start Neo4j
# Or verify NEO4J_URI, NEO4J_USERNAME, NEO4J_PASSWORD in .env
```

---

## **Schema Classes** (itext2kg/atom/models/schemas.py)

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

| Method                                   | Location                             | Input                              | Output                   | Async | Purpose                                     |
| ---------------------------------------- | ------------------------------------ | ---------------------------------- | ------------------------ | ----- | ------------------------------------------- |
| `extract_entities()`                     | itext2kg_star/ientities_extraction/  | str (context)                      | Entity[]                 | ✓     | Extracts entities from text                 |
| `extract_quintuples()`                   | atom/atom.py:32                      | List[str] (facts), str (timestamp) | RelationshipsExtractor[] | ✓     | Extracts temporal relationships             |
| `extract_verify_and_correct_relations()` | itext2kg_star/irelations_extraction/ | str (context), Entity[]            | Relationship[]           | ✓     | Validates relationships against entity list |

### **Stage 2: Embedding & Normalization**

| Method                   | Location                       | Input                        | Output             | Async | Notes                                              |
| ------------------------ | ------------------------------ | ---------------------------- | ------------------ | ----- | -------------------------------------------------- |
| `embed_entities()`       | atom/models/knowledge_graph.py | KG, embeddings_func, weights | void (modifies KG) | ✓     | Applies weighted combination: name×0.8 + label×0.2 |
| `embed_relationships()`  | atom/models/knowledge_graph.py | KG, embeddings_func          | void (modifies KG) | ✓     | Embeds relationship predicates                     |
| `calculate_embeddings()` | llm_output_parsing/            | List[str] (texts)            | np.ndarray         | ✓     | Batch embedding computation                        |

### **Stage 3: Matching & Merging**

| Method                                      | Location                           | Input                                  | Output               | Match Strategy              |
| ------------------------------------------- | ---------------------------------- | -------------------------------------- | -------------------- | --------------------------- |
| `_batch_match_entities()`                   | atom/graph_matching/matcher.py:25  | Entity[], Entity[], threshold          | (Entity[], Entity[]) | Exact + Cosine              |
| `match_entities_and_update_relationships()` | atom/graph_matching/matcher.py:100 | Ent[], Rel[], Ent[], Rel[], thresholds | (Ent[], Rel[])       | Cascading match             |
| `merge_two_kgs()`                           | atom/atom.py:50                    | KG, KG, thresholds                     | KG                   | Entity match + Rel update   |
| `parallel_atomic_merge()`                   | atom/atom.py:65                    | KG[], threshold, max_workers           | KG                   | Binary tree merge (threads) |

### **Stage 4: Storage & Visualization**

| Method                   | Location                           | Input          | Output                                |
| ------------------------ | ---------------------------------- | -------------- | ------------------------------------- |
| `visualize_graph()`      | graph_integration/neo4j_storage.py | KnowledgeGraph | Cypher queries → Neo4j                |
| `create_nodes()`         | graph_integration/neo4j_storage.py | Entity[]       | Cypher CREATE node statements         |
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

---

## 🔍 Key Parameters & Their Effects

### **Entity Matching Parameters**

| Parameter             | Default                    | Effect                             | Stage     |
| --------------------- | -------------------------- | ---------------------------------- | --------- |
| `entity_name_weight`  | 0.8 (ATOM), 0.6 (iText2KG) | Higher → name more important       | Embedding |
| `entity_label_weight` | 0.2 (ATOM), 0.4 (iText2KG) | Higher → label more important      | Embedding |
| `ent_threshold`       | 0.8 (ATOM), 0.7 (iText2KG) | Cosine similarity cutoff for match | Matching  |

### **Relationship Matching Parameters**

| Parameter       | Default                    | Effect                   | Stage    |
| --------------- | -------------------------- | ------------------------ | -------- |
| `rel_threshold` | 0.8 (ATOM), 0.7 (iText2KG) | Cosine similarity cutoff | Matching |

### **Extraction Parameters**

| Parameter                     | Default | Effect                                        | Stage            |
| ----------------------------- | ------- | --------------------------------------------- | ---------------- |
| `max_tries`                   | 5       | Retry attempts for relation verification      | iText2KG only    |
| `max_tries_isolated_entities` | 3       | Retry attempts for entities without relations | iText2KG only    |
| `max_workers`                 | 8       | Thread pool size for parallel merge           | ATOM merge stage |

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
