# ATOM & iText2KG - Visual Code Execution Flow

## Complete Walkthrough: From Paper to Code

This document maps the paper's architecture directly to code execution paths.

---

## Part I: ATOM Pipeline Execution Flow

### Paper Architecture (Figure 1 in EACL paper)

The paper shows:
- **Module-1:** Document Distiller (atomic facts extraction)
- **Module-2:** Parallel Atomic TKG Extraction (entities + relationships)
- **Module-3:** Parallel Atomic Merge (consolidation)
- **Module-4:** Neo4j Integration (storage/visualization)

### Code Execution Path: main.py

```
START: main.py:main()
│
├─ Line 19: openai_llm_model = get_default_model()
│   │ Returns: ChatOllama(model="qwen3.6:27b-q4_K_M")
│   └─ Uses: models/models.py:14
│
├─ Line 20: openai_embeddings_model = get_default_embedding_model()
│   │ Returns: OllamaEmbeddings(model="nomic-embed-text")
│   └─ Uses: models/models.py:18 (⚠️ Type annotation mismatch here)
│
├─ Line 35: news_covid = pd.read_pickle("./small_pickle.pkl")
│   └─ Contains: DataFrame with columns ['factoids_g_truth', 'date']
│
├─ Line 38: news_covid_dict = to_dictionary(news_covid)
│   │ Converts: DataFrame → Dict[str, List[str]]
│   │ Example: {"2020-01-09": ["fact1", "fact2"], "2020-01-23": ["fact3"]}
│   └─ Uses: main.py:14-26
│
├─ Line 41: atom = Atom(llm_model=openai_llm_model, embeddings_model=openai_embeddings_model)
│   │ Initializes ATOM components:
│   │ ├─ self.matcher = GraphMatcher()
│   │ └─ self.llm_output_parser = LangchainOutputParser(llm, embeddings)
│   └─ Uses: itext2kg/atom/atom.py:1-28
│
├─ Line 44: kg = await atom.build_graph_from_different_obs_times(
│   │              atomic_facts_with_obs_timestamps=news_covid_dict)
│   │
│   ├─ [LOOP: FOR EACH timestamp, facts IN news_covid_dict]
│   │
│   │ ✏️ FIRST ITERATION: timestamp="2020-01-09", facts=["fact1", "fact2", ...]
│   │ │
│   │ ├─ Step 1: Extract Quintuples
│   │ │  └─ Line 32: quintuples = await atom.extract_quintuples(facts, "2020-01-09")
│   │ │     │
│   │ │     └─ LLM Call:
│   │ │        ├─ System Prompt: temporal_system_query("2020-01-09") + EXAMPLES
│   │ │        ├─ LLM Input: ["fact1", "fact2", ...]
│   │ │        └─ LLM Output: RelationshipsExtractor object
│   │ │           └─ .relationships: [
│   │ │                 Relationship(
│   │ │                   startNode=Entity(label="Location", name="Wuhan"),
│   │ │                   endNode=Entity(label="Event", name="Virus Outbreak"),
│   │ │                   name="is_location_of",
│   │ │                   t_start=["09-01-2020"],
│   │ │                   t_end=[]
│   │ │                 ),
│   │ │                 ...
│   │ │              ]
│   │ │
│   │ ├─ Step 2: Build Atomic KG from Quintuples
│   │ │  └─ Line 100+: kg_t = await atom.build_atomic_kg_from_quintuples(
│   │ │                          relationships=quintuples.relationships)
│   │ │     │
│   │ │     ├─ 2a. Create Temporary KG with Entities
│   │ │     │  └─ temp_kg = KnowledgeGraph(entities=[...all unique entities...])
│   │ │     │
│   │ │     ├─ 2b. Embed Entities
│   │ │     │  └─ await temp_kg.embed_entities(
│   │ │     │          embeddings_function=llm_parser.calculate_embeddings,
│   │ │     │          entity_name_weight=0.8,
│   │ │     │          entity_label_weight=0.2)
│   │ │     │     │
│   │ │     │     └─ For each entity:
│   │ │     │        embeddings[i] = 0.2 * label_emb[i] + 0.8 * name_emb[i]
│   │ │     │
│   │ │     ├─ 2c. Parse Timestamps & Create Relationship Objects
│   │ │     │  └─ FOR each relationship in quintuples:
│   │ │     │        ├─ Parse t_start="09-01-2020" → Unix timestamp
│   │ │     │        ├─ Parse t_end=[] → []
│   │ │     │        └─ Create Relationship(
│   │ │     │             name="is_location_of",
│   │ │     │             startEntity=entity_object,
│   │ │     │             endEntity=entity_object,
│   │ │     │             properties=RelationshipProperties(
│   │ │     │               t_start=[1578596800.0],
│   │ │     │               t_end=[],
│   │ │     │               ...
│   │ │     │             )
│   │ │     │           )
│   │ │     │
│   │ │     ├─ 2d. Embed Relationships
│   │ │     │  └─ await kg.embed_relationships(embeddings_function)
│   │ │     │     └─ embeddings[i] = embedding of predicate name
│   │ │     │
│   │ │     ├─ 2e. Split into Atomic KGs
│   │ │     │  └─ atomic_kgs = kg.split_into_atomic_kgs()
│   │ │     │     └─ Returns: [KG(rel1 + entities), KG(rel2 + entities), ...]
│   │ │     │
│   │ │     └─ 2f. Parallel Merge Atomic KGs
│   │ │        └─ return atom.parallel_atomic_merge(
│   │ │                  kgs=atomic_kgs,
│   │ │                  rel_threshold=0.8,
│   │ │                  ent_threshold=0.8,
│   │ │                  max_workers=8)
│   │ │           │
│   │ │           └─ ThreadPoolExecutor Strategy:
│   │ │              Round 1: merge(kg0,kg1) || merge(kg2,kg3) || merge(kg4,kg5)
│   │ │              Round 2: merge(result_01,result_23) || merge(result_45)
│   │ │              Round 3: merge(result_0123, result_45)
│   │ │              → Single merged KG for timestamp 2020-01-09
│   │ │
│   │ └─ RESULT: kg_t (temporal KG for 2020-01-09)
│   │
│   │ ✏️ SECOND ITERATION: timestamp="2020-01-23", facts=[...]
│   │ │
│   │ ├─ [Repeat steps 1-2 for new timestamp]
│   │ └─ RESULT: kg_t2 (temporal KG for 2020-01-23)
│   │
│   │ ✏️ FINAL MERGE: Merge kg_t + kg_t2
│   │ │
│   │ ├─ Call: atom.merge_two_kgs(kg_t, kg_t2, rel_threshold=0.8, ent_threshold=0.8)
│   │ │ └─ matcher.match_entities_and_update_relationships(...)
│   │ │    └─ Returns: (merged_entities, merged_relationships)
│   │ │
│   │ └─ RESULT: final_kg (merged across all timestamps)
│   │
│   └─ RETURNS: kg (Temporal KnowledgeGraph)
│       └─ kg.entities: All unique entities across timestamps
│       └─ kg.relationships: All unique relationships with t_start, t_end, t_obs
│
├─ Line 51: Neo4jStorage(uri=URI, username=USERNAME, password=PASSWORD)
│   └─ Initializes Neo4j connection
│
└─ Line 52: Neo4jStorage(...).visualize_graph(knowledge_graph=kg)
   └─ Converts KG to Neo4j nodes and relationships
       ├─ FOR each entity in kg.entities:
       │  └─ CREATE (n:Entity { name: ..., label: ..., embeddings: ... })
       └─ FOR each relationship in kg.relationships:
          └─ CREATE (start_entity)-[r:PREDICATE_NAME { 
                       t_start: ..., 
                       t_end: ..., 
                       t_obs: ..., 
                       atomic_facts: ...
                     }]->(end_entity)

END
```

---

## Part II: Key Method Calls & Returns

### extract_quintuples() - Detailed

```python
# CALL:
quintuple_list = await atom.extract_quintuples(
    atomic_facts=["Virus identified in Wuhan in December 2019"],
    observation_timestamp="2020-01-09"
)

# INTERNALLY:
# 1. Builds system prompt:
system_prompt = (
    "Extract quintuples (subject, predicate, object, t_start, t_end) "
    "from atomic facts observed on 2020-01-09.\n"
    "Examples: ...\n"
    "Present tense predicates only. Resolve relative dates using 2020-01-09 as reference."
)

# 2. Calls LLM:
response = await llm.invoke({
    "system": system_prompt,
    "user": "Virus identified in Wuhan in December 2019"
})

# 3. Parses JSON response:
json_response = json.loads(response.content)
# {
#   "relationships": [
#     {
#       "startNode": {"label": "Location", "name": "Wuhan"},
#       "endNode": {"label": "Event", "name": "Virus Outbreak"},
#       "name": "is_location_of",
#       "t_start": ["01-12-2019"],
#       "t_end": []
#     }
#   ]
# }

# 4. Creates Pydantic model:
extractor = RelationshipsExtractor(**json_response)

# RETURNS:
# [
#   RelationshipsExtractor(
#     relationships=[
#       Relationship(
#         startNode=Entity(label="Location", name="Wuhan"),
#         endNode=Entity(label="Event", name="Virus Outbreak"),
#         name="is_location_of",
#         t_start=["01-12-2019"],
#         t_end=[]
#       )
#     ]
#   )
# ]
```

---

### embed_entities() - Detailed

```python
# CALL:
await kg.embed_entities(
    embeddings_function=llm_parser.calculate_embeddings,
    entity_name_weight=0.8,
    entity_label_weight=0.2
)

# INTERNALLY:
# 1. Extract labels and names:
labels = ["Location", "Event", "Person"]
names = ["Wuhan", "Virus Outbreak", "Steve Jobs"]

# 2. Compute embeddings via Ollama:
label_embs = await embeddings_model.aembed_documents(labels)
# Returns: np.array(shape=(3, 1024)) - Nomic Embed dimension

name_embs = await embeddings_model.aembed_documents(names)
# Returns: np.array(shape=(3, 1024))

# 3. Compute weighted sum:
for i, entity in enumerate(kg.entities):
    entity.properties.embeddings = (
        0.2 * label_embs[i] +
        0.8 * name_embs[i]
    )
    # entity.properties.embeddings now has shape (1024,)

# 4. No explicit return (modifies kg in-place)
```

---

### parallel_atomic_merge() - Detailed

```python
# CALL:
merged_kg = atom.parallel_atomic_merge(
    kgs=[kg1, kg2, kg3, kg4],  # 4 atomic KGs (1 relationship each)
    rel_threshold=0.8,
    ent_threshold=0.8,
    max_workers=8
)

# INTERNALLY:
# Round 1: Process pairs
# ├─ Thread 1: merge(kg1, kg2, 0.8, 0.8) → merged_12
# ├─ Thread 2: merge(kg3, kg4, 0.8, 0.8) → merged_34
# └─ Wait for completion

current = [merged_12, merged_34]

# Round 2: Merge results
# └─ Thread 1: merge(merged_12, merged_34, 0.8, 0.8) → final_kg

current = [final_kg]

# RETURNS:
# KnowledgeGraph(
#   entities=[...all 8 unique entities (deduplicated)...],
#   relationships=[...all 4 relationships (possibly consolidated)...]
# )
```

---

## Part III: iText2KG Execution Flow

### Code Path: Custom Implementation

```python
# Pseudo-code for iText2KG usage (not in provided files, but pattern):

async def build_document_kg_example():
    # SETUP
    llm = get_default_model()
    embeddings = get_default_embedding_model()
    itext = iText2KG(llm_model=llm, embeddings_model=embeddings, sleep_time=5)
    
    sections = [
        "Apple Inc. was founded by Steve Jobs on April 1, 1976.",
        "In 2007, Apple launched the iPhone, designed by Steve Jobs.",
        "Steve Jobs served as CEO until 2011 when Tim Cook took over."
    ]
    
    # BUILD GRAPH
    kg = await itext.build_graph(
        sections=sections,
        ent_threshold=0.7,
        rel_threshold=0.7,
        entity_name_weight=0.6,
        entity_label_weight=0.4
    )
    
    # INTERNALLY:
    # │
    # ├─ Section 0: "Apple Inc. was founded by Steve Jobs..."
    # │ ├─ extract_entities(sections[0]) → [Apple Inc., Steve Jobs, April 1 1976]
    # │ │  └─ Embed with weights: 0.4*label + 0.6*name
    # │ │     (Different from ATOM: 0.2*label + 0.8*name)
    # │ │
    # │ ├─ extract_verify_and_correct_relations(sections[0], entities)
    # │ │  │ LLM extracts: "Steve Jobs FOUNDED Apple Inc."
    # │ │  │ Verifies: startNode in entities? ✓ endNode in entities? ✓
    # │ │  └─ Returns: [Relationship(Steve Jobs → FOUNDED → Apple Inc.)]
    # │ │
    # │ └─ global_entities = [Apple Inc., Steve Jobs, ...]
    #     global_relationships = [...]
    #
    # ├─ Section 1: "In 2007, Apple launched the iPhone..."
    # │ ├─ local_entities = [Apple, iPhone, 2007]
    # │ ├─ matcher.process_lists(local_entities, global_entities, threshold=0.7)
    # │ │  └─ Matches "Apple" to "Apple Inc." (cosine sim > 0.7)
    # │ │  └─ Returns: ([iPhone, 2007], updated_global_entities)
    # │ │
    # │ ├─ extract_verify_and_correct_relations(...)
    # │ │  └─ "Apple LAUNCHED iPhone"
    # │ │
    # │ └─ global_entities updated + global_relationships updated
    #
    # ├─ Section 2: "Steve Jobs served as CEO..."
    # │ ├─ extract_entities(...) → [Steve Jobs, CEO, Tim Cook, ...]
    # │ ├─ match entities
    # │ ├─ extract_verify_and_correct_relations(...)
    # │ │  └─ "Steve Jobs IS_CEO_OF Apple Inc. (until 2011)"
    # │ │     "Tim Cook IS_CEO_OF Apple Inc. (from 2011)"
    # │ │
    # │ └─ global_entities + global_relationships updated
    #
    # └─ DEDUP + RETURN: consolidated_kg
    
    # RESULT:
    # kg.entities: [Apple Inc., Steve Jobs, iPhone, Tim Cook, ...]
    # kg.relationships: [
    #   Relationship(Steve Jobs, FOUNDED, Apple Inc.),
    #   Relationship(Apple Inc., LAUNCHED, iPhone),
    #   Relationship(Steve Jobs, WAS_CEO_OF, Apple Inc., t_end=2011),
    #   Relationship(Tim Cook, IS_CEO_OF, Apple Inc., t_start=2011),
    #   ...
    # ]
    
    return kg
```

---

## Part IV: Data Transformation Examples

### ATOM: Timestamp to Unix

```python
# Input from LLM:
t_start = ["09-01-2020"]
t_end = []

# Processing:
from dateutil import parser

t_start_timestamps = []
for ts_str in t_start:
    parsed_dt = parser.parse(ts_str)  # dateutil infers: 2020-01-09
    unix_ts = parsed_dt.timestamp()   # 1578596800.0
    t_start_timestamps.append(unix_ts)

t_end_timestamps = []  # Empty, so stays []

# Result in Relationship:
relationship.properties.t_start = [1578596800.0]
relationship.properties.t_end = []
```

### Entity Deduplication: Exact Match

```python
# Scenario: Same entity extracted from two sections

entity_1 = Entity(label="Organization", name="Apple Inc")
entity_2 = Entity(label="Organization", name="apple inc")

# After normalization:
entity_1.process() = Entity(label="organization", name="apple inc")
entity_2.process() = Entity(label="organization", name="apple inc")

# Hash comparison:
hash(entity_1.process()) == hash(entity_2.process())  # ✓ True

# Result: Deduplicated to single entity in KG
```

### Entity Embedding-based Matching

```python
# Scenario: Similar entities from different sections

entities_section1 = [
    Entity(label="Person", name="Steven Jobs", properties=EntityProperties(
        embeddings=np.array([...1024 dims...])  # nomic-embed-text
    )),
]

entities_section2 = [
    Entity(label="Person", name="Steve Jobs", properties=EntityProperties(
        embeddings=np.array([...1024 dims...])
    ))
]

# Embedding similarity:
from sklearn.metrics.pairwise import cosine_similarity

sim = cosine_similarity(
    entities_section1[0].properties.embeddings.reshape(1, -1),
    entities_section2[0].properties.embeddings.reshape(1, -1)
)
# Result: 0.95 (high similarity)

# Matching:
if sim[0, 0] >= threshold (0.7):
    # Match found! Use one entity reference
    unified_entity = entities_section2[0]  # Or consolidate both
```

---

## Part V: Neo4j Output Schema

### Entities → Neo4j Nodes

```python
# Python KnowledgeGraph entity:
entity = Entity(
    label="Person",
    name="Steve Jobs",
    properties=EntityProperties(
        embeddings=np.array([...1024 floats...])
    )
)

# Converted to Neo4j Cypher:
CREATE (n:Entity {
    name: "Steve Jobs",
    label: "Person",
    embeddings_str: "0.123,0.456,...,0.789",  # CSV of embeddings
    embeddings_dim: 1024
})

# OR with relationship type based on label:
CREATE (n:Person {
    name: "Steve Jobs",
    embeddings_str: "...",
})
```

### Relationships → Neo4j Edges

```python
# Python Relationship:
rel = Relationship(
    name="FOUNDED",
    startEntity=steve_jobs,
    endEntity=apple_inc,
    properties=RelationshipProperties(
        embeddings=np.array([...1024 floats...]),
        t_start=[1145952000.0],  # 2006-04-16 (when Apple founded)
        t_end=[],
        t_obs=[1578596800.0],    # 2020-01-09 (observation date)
        atomic_facts=["Steve Jobs founded Apple Inc on April 16 2006"]
    )
)

# Converted to Neo4j Cypher:
MATCH (s:Entity {name: "Steve Jobs"}), (e:Entity {name: "Apple Inc"})
CREATE (s)-[r:FOUNDED {
    embeddings_str: "...",
    t_start: [1145952000.0],
    t_end: [],
    t_obs: [1578596800.0],
    atomic_facts: ["Steve Jobs founded Apple Inc on April 16 2006"]
}]->(e)
```

---

## Part VI: Error Handling Flows

### Configuration Mismatch Error

```python
# Current code:
from langchain_openai import OpenAIEmbeddings
from .models_config import *  # Contains embeddings_ollama_nomic

def get_default_embedding_model() -> OpenAIEmbeddings:  # ❌ WRONG
    return _DEFAULT_EMBEDDINGS  # Returns OllamaEmbeddings instance

# IDE/Type Checker Error:
# Expected: OpenAIEmbeddings
# Got: OllamaEmbeddings
# Error: Type mismatch

# Runtime Error (if strict checking):
# AttributeError: 'OllamaEmbeddings' has no attribute 'api_key'
# (because IDE assumes OpenAIEmbeddings interface)
```

### Entity Not Found Error (iText2KG)

```python
# Scenario: LLM extracts relationship with non-existent entity

entities = [
    Entity(label="Person", name="Steve Jobs"),
    Entity(label="Company", name="Apple Inc")
]

extracted_rel = Relationship(
    startNode=Entity(label="Person", name="Steve"),  # ❌ Doesn't match "Steve Jobs"
    endNode=Entity(label="Company", name="Apple"),   # ❌ Doesn't match "Apple Inc"
    name="FOUNDED"
)

# Verification fails:
if extracted_rel.startNode not in entities:
    # RETRY: Extract isolated entities
    isolated = [extracted_rel.startNode, extracted_rel.endNode]
    new_entities = await extractor.extract_entities(context)
    entities.extend(new_entities)  # Add "Steve" and "Apple" entities
    
    # RE-EXTRACT: Get relations again
    extracted_rel = await extractor.extract_verify_and_correct_relations(
        context,
        updated_entities=entities
    )
    # This time, hopefully matches correctly
```

---

## Summary: Key Execution Checkpoints

| Stage | Location | Input | Process | Output |
|---|---|---|---|---|
| 1. Init | main.py:19-20 | Config | Load LLM + Embeddings | ChatOllama, OllamaEmbeddings |
| 2. Load | main.py:35 | File | Load pickle | DataFrame |
| 3. Format | main.py:38 | DataFrame | to_dictionary() | Dict[timestamp → facts] |
| 4. Init Pipeline | main.py:41 | Models | Atom() | GraphMatcher + Parser |
| 5. Extract | atom.py:32 | Facts + Timestamp | LLM call | RelationshipsExtractor |
| 6. Build | atom.py:100 | Relationships | Embed + Split + Merge | KG (per timestamp) |
| 7. Merge | atom.py:65 | KG[] | Parallel binary merge | Final KG |
| 8. Store | storage.py | KG | Cypher generation | Neo4j nodes/edges |

