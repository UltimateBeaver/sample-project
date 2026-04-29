# iText2KG - Visual Architecture & Quick Reference

## Architecture Diagram

```
┌───────────────────────────────────────────────────────────────────────┐
│                          itext2kg Package                              │
├───────────────────────────────────────────────────────────────────────┤
│                                                                         │
│  ┌─────────────────────────────────────────────────────────────────┐  │
│  │                    ATOM Module (Temporal)                        │  │
│  ├─────────────────────────────────────────────────────────────────┤  │
│  │                                                                   │  │
│  │  atomic_facts          extract_quintuples()     quintuples       │  │
│  │  {timestamp:           ────────────────────►    with t_start,    │  │
│  │   [facts]}                                       t_end, t_obs    │  │
│  │      │                                                 │          │  │
│  │      │              build_atomic_kg_from_quintuples()│          │  │
│  │      │              ────────────────────────────────►│          │  │
│  │      │                                                 │          │  │
│  │      │         atomic KGs (one rel per KG)           │          │  │
│  │      │         with temporal properties              │          │  │
│  │      └──────────────┬────────────────────────────────┘          │  │
│  │                     │ parallel_atomic_merge()                    │  │
│  │                     ├───────────► GraphMatcher                   │  │
│  │                     │             (embeddings + cosine sim)      │  │
│  │                     ▼                                             │  │
│  │            Merged KnowledgeGraph (temporal)                      │  │
│  │            (entities, relationships with t_*)                   │  │
│  │                                                                   │  │
│  └─────────────────────────────────────────────────────────────────┘  │
│                                   ▲                                     │
│                                   │ Optional: merge existing KG         │
│                                   │                                     │
│  ┌─────────────────────────────────────────────────────────────────┐  │
│  │              iText2KG_Star Module (Sections)                     │  │
│  ├─────────────────────────────────────────────────────────────────┤  │
│  │                                                                   │  │
│  │  document_sections    extract_entities()    entities             │  │
│  │  [section0, ...]  ─────────────────────►    with embeddings     │  │
│  │      │                                            │              │  │
│  │      │            extract_verify_and_correct_rel_()              │  │
│  │      │            ──────────────────────────────►│              │  │
│  │      │                                            │              │  │
│  │      └───────────────┬──────────────────────────┘              │  │
│  │                      │ Matcher.match_entities_and...()           │  │
│  │                      ├───────────► Matcher                       │  │
│  │                      │             (exact + embedding match)     │  │
│  │                      ▼                                            │  │
│  │             Merged KnowledgeGraph (generic)                      │  │
│  │             (entities, relationships)                            │  │
│  │                                                                   │  │
│  └─────────────────────────────────────────────────────────────────┘  │
│                                   ▲                                     │
│                                   │                                     │
│  ┌─────────────────────────────────────────────────────────────────┐  │
│  │         Documents Distiller Module (Consolidation)               │  │
│  ├─────────────────────────────────────────────────────────────────┤  │
│  │                                                                   │  │
│  │  documents           distill()             consolidated          │  │
│  │  [doc1, doc2, ...]  ────────────────►      object                │  │
│  │                     (LLM batch extract +    (dict or              │  │
│  │                      combine by type)       Pydantic model)       │  │
│  │                                                                   │  │
│  └─────────────────────────────────────────────────────────────────┘  │
│                                   │                                     │
│                                   ▼                                     │
│  ┌─────────────────────────────────────────────────────────────────┐  │
│  │     Graph Integration Module (Neo4j Storage & Visualization)     │  │
│  ├─────────────────────────────────────────────────────────────────┤  │
│  │                                                                   │  │
│  │  KnowledgeGraph      create_nodes()        Cypher queries        │  │
│  │       │              create_relationships()      │              │  │
│  │       └──────────────────────────────────────────┤              │  │
│  │                                                   │              │  │
│  │       Neo4j Database                             │              │  │
│  │       ┌─────────────────┐  ◄─────────────────────┘              │  │
│  │       │ Nodes (Entities)│ MERGE queries                          │  │
│  │       │ Edges (Rels)    │                                        │  │
│  │       │ Properties      │                                        │  │
│  │       └─────────────────┘                                        │  │
│  │                                                                   │  │
│  └─────────────────────────────────────────────────────────────────┘  │
│                                   │                                     │
│                                   ▼                                     │
│                       Neo4j Visualization UI                            │
│                                                                         │
│  ┌─────────────────────────────────────────────────────────────────┐  │
│  │          LLM Output Parsing (Shared Infrastructure)              │  │
│  ├─────────────────────────────────────────────────────────────────┤  │
│  │                                                                   │  │
│  │  LangchainOutputParser                                           │  │
│  │  ├── Auto-detect LLM provider (OpenAI, Mistral, Claude, etc.)   │  │
│  │  ├── Provider-specific rate limiting                            │  │
│  │  ├── Batch processing with token limits                         │  │
│  │  ├── JSON extraction with Pydantic models                       │  │
│  │  └── Embedding generation                                       │  │
│  │                                                                   │  │
│  └─────────────────────────────────────────────────────────────────┘  │
│                                                                         │
└───────────────────────────────────────────────────────────────────────┘
```

---

## Data Flow - ATOM Pipeline (Example)

```
Input: Atomic Facts at different timestamps
  {
    "2024-06-18": [
      "Real Madrid won the Champions League final",
      "The match ended 2-1",
      "Fans celebrated in the city"
    ],
    "2024-06-19": [
      "Real Madrid received the trophy",
      "Coach praised the team"
    ]
  }
    │
    ├─► For each timestamp:
    │   1. Extract quintuples (entities + relationships + temporal bounds)
    │      ├─ "Real Madrid" (Organization) –is_winner→ "Champions League" (Event)
    │      │  t_start: [2024-06-18], t_end: []
    │      ├─ "Match" (Event) –has_score→ "2-1" (Result)
    │      │  t_start: [2024-06-18], t_end: []
    │      └─ ...
    │
    │   2. Build atomic KGs (separate graph per relationship)
    │      kg1: {entities: [Real Madrid, Champions League], rels: [is_winner]}
    │      kg2: {entities: [Match, Result], rels: [has_score]}
    │      ...
    │
    │   3. Parallel merge atomic KGs
    │      merged_timestamp_kg = merge_all_atomic_kgs()
    │      └─ Deduplicates entities and relationships
    │
    │   4. Add observation timestamp
    │      merged_timestamp_kg.relationships[*].t_obs = [2024-06-18, 2024-06-19]
    │
    └─► Merge across timestamps
        final_kg = merge(timestamp_kg_1, timestamp_kg_2, ...)
        └─ Final graph tracks entity/relationship evolution over time

Output: Temporal KnowledgeGraph
  {
    entities: [Real Madrid, Champions League, Match, ...],
    relationships: [
      {
        startEntity: Real Madrid,
        endEntity: Champions League,
        name: "is_winner",
        properties: {
          t_start: [1718659200.0],    # 2024-06-18
          t_end: [],
          t_obs: [1718659200.0, 1718745600.0],
          atomic_facts: ["Real Madrid won the Champions League final"]
        }
      },
      ...
    ]
  }
    │
    ▼
  Store in Neo4j via Neo4jStorage.visualize_graph()
```

---

## Data Flow - iText2KG Pipeline (Example)

```
Input: Multiple document sections
  [
    "Section 1: Real Madrid is a Spanish football club based in Madrid...",
    "Section 2: The club won the Champions League in 2024...",
    "Section 3: Current CEO is Florentino Pérez..."
  ]
    │
    ├─► Section 1:
    │   1. Extract entities
    │      ├─ "Real Madrid" (Organization)
    │      ├─ "Madrid" (Location)
    │      └─ "Spain" (Country)
    │
    │   2. Extract relationships
    │      ├─ Real Madrid –is_based_in→ Madrid
    │      └─ Real Madrid –is_located_in→ Spain
    │
    │   3. Create global_entities = extracted_entities
    │       global_relationships = extracted_relationships
    │
    ├─► Section 2:
    │   1. Extract entities
    │      ├─ "Real Madrid" (Organization)
    │      ├─ "Champions League" (Event)
    │      └─ 2024 (Time)
    │
    │   2. Match with global_entities
    │      ├─ "Real Madrid" matched (exact match)
    │      ├─ "Champions League" is NEW
    │      └─ 2024 is NEW
    │
    │   3. Extract relationships for new + existing entities
    │      └─ Real Madrid –won_trophy→ Champions League
    │
    │   4. Merge with global
    │       global_entities += ["Champions League", 2024]
    │       global_relationships += [won_trophy rel]
    │
    ├─► Section 3:
    │   1. Extract entities
    │      ├─ "Florentino Pérez" (Person)
    │      └─ "CEO" (Position)
    │
    │   2. Extract relationships
    │      └─ Florentino Pérez –is_CEO_of→ Real Madrid
    │
    │   3. Merge with global
    │       global_entities += ["Florentino Pérez", "CEO"]
    │       global_relationships += [is_CEO_of rel]
    │
    └─► Final merge
        kg = KnowledgeGraph(
          entities = global_entities,
          relationships = global_relationships
        )
        kg.remove_duplicates_entities()
        kg.remove_duplicates_relationships()

Output: KnowledgeGraph (no temporal info)
  {
    entities: [
      Real Madrid (Organization),
      Madrid (Location),
      Spain (Country),
      Champions League (Event),
      Florentino Pérez (Person),
      CEO (Position)
    ],
    relationships: [
      Real Madrid –is_based_in→ Madrid,
      Real Madrid –is_located_in→ Spain,
      Real Madrid –won_trophy→ Champions League,
      Florentino Pérez –is_CEO_of→ Real Madrid
    ]
  }
    │
    ▼
  Store in Neo4j
```

---

## Class Hierarchy & Dependencies

```
LangchainOutputParser (Shared Infrastructure)
    ├── Uses: OpenAI/Mistral/Claude LLM
    ├── Uses: Embeddings model
    └── Provides: extract_information_as_json_for_context(), calculate_embeddings()

ATOM
    ├── Uses: LangchainOutputParser
    ├── Uses: GraphMatcher (ATOM version)
    ├── Provides: build_graph(), build_graph_from_different_obs_times()
    └── Returns: Temporal KnowledgeGraph

iText2KG
    ├── Uses: iEntitiesExtractor
    ├── Uses: iRelationsExtractor
    ├── Uses: Matcher (iText2KG version)
    ├── Uses: LangchainOutputParser
    ├── Provides: build_graph()
    └── Returns: Generic KnowledgeGraph

iEntitiesExtractor
    ├── Uses: LangchainOutputParser
    └── Provides: extract_entities()

iRelationsExtractor
    ├── Uses: LangchainOutputParser
    ├── Uses: Matcher
    └── Provides: extract_relations(), extract_verify_and_correct_relations()

GraphMatcher (ATOM)
    ├── Uses: scikit-learn (cosine_similarity)
    ├── Uses: KnowledgeGraph
    └── Provides: _batch_match_entities(), _batch_match_relationships(),
                   match_entities_and_update_relationships()

Matcher (iText2KG)
    ├── Uses: scikit-learn (cosine_similarity)
    ├── Uses: KnowledgeGraph
    └── Provides: find_match(), process_lists(), 
                   match_entities_and_update_relationships()

DocumentsDistiller
    ├── Uses: LangchainOutputParser
    └── Provides: distill()

Neo4jStorage
    ├── Uses: Neo4j driver
    ├── Provides: create_nodes(), create_relationships(), visualize_graph()
    └── Returns: Stored in Neo4j database

KnowledgeGraph (Data Model)
    ├── Contains: List[Entity], List[Relationship]
    ├── Provides: embed_entities(), embed_relationships(), 
                   remove_duplicates_*(), find_*(), add_*_to_relationships()
    └── Subtypes: Temporal (ATOM), Generic (iText2KG)

Entity (Data Model)
    ├── Fields: label, name, properties (embeddings)
    ├── Methods: process(), __eq__(), __hash__()
    └── Normalization: sanitize name/label

Relationship (Data Model)
    ├── Fields: startEntity, endEntity, name, properties
    ├── Properties: embeddings, atomic_facts, t_obs, t_start, t_end
    ├── Methods: process(), combine_timestamps(), combine_atomic_facts()
    └── Temporal extensions: Only in ATOM version
```

---

## Quick Reference - Key Methods by Task

### If you want to...

#### Extract temporal knowledge graph from atomic facts
```python
atom = Atom(llm_model, embeddings_model)
kg = await atom.build_graph_from_different_obs_times({
    "2024-06-18": ["fact1", "fact2"],
    "2024-06-19": ["fact3"]
})
```
**Main methods**: `build_graph_from_different_obs_times()` → `build_graph()` → `extract_quintuples()`

#### Extract knowledge graph from document sections
```python
itext2kg = iText2KG(llm_model, embeddings_model)
kg = await itext2kg.build_graph([section1, section2, section3])
```
**Main methods**: `build_graph()` → `iEntitiesExtractor.extract_entities()` → `iRelationsExtractor.extract_verify_and_correct_relations()`

#### Consolidate information from multiple documents
```python
distiller = DocumentsDistiller(llm_model)
consolidated = await distiller.distill(documents, OutputSchema, query)
```
**Main methods**: `distill()` → `__combine_objects()` → `__merge_field_values()`

#### Store knowledge graph in Neo4j
```python
storage = Neo4jStorage(uri, username, password)
storage.visualize_graph(knowledge_graph)
```
**Main methods**: `create_nodes()` + `create_relationships()` → `run_query()`

#### Match and merge two knowledge graphs
```python
# ATOM version
matcher = GraphMatcher()
entities, rels = matcher.match_entities_and_update_relationships(
    kg1.entities, kg2.entities,
    kg1.relationships, kg2.relationships
)

# iText2KG version
matcher = Matcher()
matched_entities, new_entities = matcher._batch_match_entities(
    entities1, entities2, threshold
)
```
**Main methods**: `_batch_match_entities()` + `_batch_match_relationships()`

#### Generate embeddings for entities/relationships
```python
parser = LangchainOutputParser(llm_model, embeddings_model)
entity_embeddings = await parser.calculate_embeddings(["entity1", "entity2"])
```
**Main methods**: `calculate_embeddings()`

#### Extract structured JSON from text using LLM
```python
parser = LangchainOutputParser(llm_model, embeddings_model)
results = await parser.extract_information_as_json_for_context(
    contexts=["text1", "text2"],
    output_data_structure=MyPydanticModel,
    system_query="instructions"
)
```
**Main methods**: `extract_information_as_json_for_context()`

---

## Configuration Constants

### Entity Weighting (for embeddings)
```python
entity_name_weight: float = 0.8   # (ATOM) or 0.6 (iText2KG)
entity_label_weight: float = 0.2  # (ATOM) or 0.4 (iText2KG)
# Combined: entity_embedding = name_weight * name_emb + label_weight * label_emb
```

### Matching Thresholds
```python
ent_threshold: float = 0.8   # Entity matching cosine similarity threshold
rel_threshold: float = 0.7   # Relationship matching cosine similarity threshold
# Used in: cosine_similarity(emb1, emb2) >= threshold
```

### Extraction Retries
```python
max_tries: int = 5                      # LLM extraction retries
max_tries_isolated_entities: int = 3    # Isolated entity fix attempts
sleep_time: int = 5                     # Seconds to sleep on error
```

### Rate Limiting (by provider)
```python
# OpenAI
max_elements_per_batch=40
max_tokens_per_batch=8000
max_context_window=128000
sleep_between_batches=2.0

# Mistral
max_elements_per_batch=1
max_tokens_per_batch=10000
max_context_window=128000
sleep_between_batches=0.2

# Claude
max_elements_per_batch=50
max_tokens_per_batch=8000
max_context_window=200000
sleep_between_batches=1.2
```

---

## Error Handling Patterns

### Entity Extraction Failures
```python
# Retries up to max_tries times
# Returns empty list on failure
# Log level: WARNING for each retry
```

### Relationship Extraction with Invented Entities
```python
# If entity not in input list:
#   1. Embed invented entity
#   2. Match to closest input entity (threshold=0.5)
#   3. Use matched entity instead
# Log level: INFO for each invented entity
```

### Isolated Entities
```python
# After initial relationship extraction:
# 1. Find entities with no relationships
# 2. Re-extract relationships for isolated entities
# 3. Link to existing entities
# 4. Repeat up to max_tries_isolated_entities times
# Log level: INFO for each iteration
```

### Timestamp Parsing
```python
# Try: parser.parse(timestamp_str) → .timestamp()
# Except: Log warning, skip timestamp, continue
# Result: List may have gaps
```

---

## Testing Entry Points

### ATOM Tests [tests/atom/test_atom_matching.py]
- Tests entity/relationship matching
- Validates embedding-based similarity

### iText2KG Tests [tests/itext2kg/test_itext2kg_matching.py]
- Tests multi-section KG building
- Validates entity consolidation

### Evaluation Scripts [evaluation/]
- `exhaustivity/`: Factoid coverage metrics
- `latency/`: Performance comparison
- `merge/`: Graph merge quality
- `quintuples_quality/`: Triple quality
- `stability/`: Consistency checks

---

## Important Notes

1. **Embedding Weights**: ATOM uses different defaults (0.8 name / 0.2 label) vs iText2KG (0.6 / 0.4)

2. **Temporal Properties**: Only Relationship objects support t_start, t_end, t_obs in ATOM

3. **Matching Threshold**: Default 0.8 is strict; 0.5-0.7 more permissive

4. **Neo4j Labels**: Sanitized to valid Neo4j identifiers (alphanumeric + underscores)

5. **Deduplication**: Uses Python sets with __hash__ and __eq__ methods

6. **Parallel Merge**: TreeReduce pattern with ThreadPoolExecutor (not async!)

7. **Provider Detection**: Auto-detects based on model instance type (infer from class name)

8. **Batch Processing**: Token-aware; splits batches if exceeding provider limits

9. **Neo4j Queries**: Escaped for injection; embeddings stored as comma-separated strings

10. **Pydantic Models**: Field descriptions used as LLM prompt engineering

