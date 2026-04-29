# ATOM & iText2KG - Detailed Method Reference & Pipeline Mapping

## 📖 Complete Method Signatures with Paper References

Based on the paper: "ATOM: Temporal atomic facts from web news with knowledge graphs" (EACL 2026 Findings)

---

## Part I: ATOM Pipeline (Temporal KG)

### Core Orchestrator: `Atom` Class

**File:** `itext2kg/atom/atom.py`

#### Constructor
```python
def __init__(self, llm_model, embeddings_model) -> None:
    """
    Initialize ATOM pipeline components.
    
    Args:
        llm_model: LangChain LLM instance (ChatOpenAI, ChatOllama, etc.)
        embeddings_model: LangChain embeddings instance (OllamaEmbeddings, etc.)
    
    Attributes Created:
        self.matcher: GraphMatcher()
        self.llm_output_parser: LangchainOutputParser(llm_model, embeddings_model)
    """
```

#### Main Entry Point
```python
async def build_graph_from_different_obs_times(
    self,
    atomic_facts_with_obs_timestamps: Dict[str, List[str]]
) -> KnowledgeGraph:
    """
    [PAPER: Module-3 Update DTKG]
    
    Builds temporal knowledge graph from atomic facts observed at different timestamps.
    
    Args:
        atomic_facts_with_obs_timestamps: Dictionary mapping observation dates to lists of atomic facts
            Example: {
                "2024-01-09": [
                    "Virus identified in Wuhan",
                    "Spread to 10 countries"
                ],
                "2024-01-23": [
                    "26 deaths in China",
                    "80+ deaths reported"
                ]
            }
    
    Returns:
        KnowledgeGraph: Temporal KG with:
            - entities: Global entity set across all timestamps
            - relationships: Merged relationships with t_start, t_end, t_obs
    
    Process Flow:
        FOR EACH (timestamp, atomic_facts) in atomic_facts_with_obs_timestamps:
            1. extract_quintuples(atomic_facts, timestamp)
            2. build_atomic_kg_from_quintuples(relationships)
            3. parallel_atomic_merge(atomic_kgs)
        
        FINAL: Merge all timestamp KGs into one global temporal KG
    """
    # Implementation pseudo-code:
    # result_kg = KnowledgeGraph()
    # for timestamp, facts in atomic_facts_with_obs_timestamps.items():
    #     quintuples = await extract_quintuples(facts, timestamp)
    #     kg_t = await build_atomic_kg_from_quintuples(quintuples)
    #     kg_t.add_t_obs_to_relationships([parse_timestamp(timestamp)])
    #     result_kg = merge_two_kgs(result_kg, kg_t)
    # return result_kg
```

---

#### Stage 1: Quintuple Extraction
```python
async def extract_quintuples(
    self,
    atomic_facts: List[str],
    observation_timestamp: str
) -> List[RelationshipsExtractor]:
    """
    [PAPER: Module-2 - Parallel 5-tuples Extraction]
    
    Extracts (subject, predicate, object, t_start, t_end) tuples from atomic facts.
    
    Args:
        atomic_facts: List of atomic factoid strings
            Example: ["Wuhan coronavirus caused 26 deaths in China"]
        observation_timestamp: Date when facts were observed
            Format: "DD-MM-YYYY" or ISO format
    
    Returns:
        List[RelationshipsExtractor]: LLM output wrapped objects, each containing:
            - relationships: List[Relationship] (schema-based, not runtime objects)
                └─ startNode: Entity (subject)
                └─ endNode: Entity (object)
                └─ name: str (predicate, PRESENT TENSE)
                └─ t_start: List[str] (relationship start dates)
                └─ t_end: List[str] (relationship end dates)
    
    Details:
        - Calls LLM with temporal prompts (see Prompt.temporal_system_query)
        - LLM extracts facts in context of observation_timestamp
        - Converts relative dates ("today", "yesterday") to absolute dates
        - Includes EXAMPLES in prompt for few-shot learning
    
    Line Numbers: atom.py:32-38
    """
    return await self.llm_output_parser.extract_information_as_json_for_context(
        output_data_structure=RelationshipsExtractor,
        contexts=atomic_facts,
        system_query=Prompt.temporal_system_query(observation_timestamp) + Prompt.EXAMPLES.value
    )
```

---

#### Stage 2: Atomic KG Building
```python
async def build_atomic_kg_from_quintuples(
    self,
    relationships: list[RelationshipSchema],
    entity_name_weight: float = 0.8,
    entity_label_weight: float = 0.2,
    rel_threshold: float = 0.8,
    ent_threshold: float = 0.8,
    max_workers: int = 8
) -> KnowledgeGraph:
    """
    [PAPER: Module-2 - Atomic TKG Construction + Module-3 Parallel Merge]
    
    Builds atomic KGs (1 relationship per KG) from quintuples, then merges them.
    
    Args:
        relationships: Output from extract_quintuples (schema-based Relationship objects)
        entity_name_weight: Weight of entity name in embedding (0.0-1.0)
        entity_label_weight: Weight of entity label in embedding (0.0-1.0)
        rel_threshold: Cosine similarity threshold for relationship matching
        ent_threshold: Cosine similarity threshold for entity matching
        max_workers: Thread pool size for parallel merge
    
    Returns:
        KnowledgeGraph: Merged atomic KG (typically 1 relationship, unless many entities)
    
    Sub-Process:
        1. Create temporary KG with all entities from relationships
        2. Embed entities:
            embeddings = entity_label_weight * label_emb + entity_name_weight * name_emb
        3. Parse timestamps (string → Unix floats)
        4. Create Relationship objects with embedded entities
        5. Embed relationships (predicate names)
        6. Split into atomic KGs (1 relationship each)
        7. Parallel merge all atomic KGs (see parallel_atomic_merge)
    
    Line Numbers: atom.py:100-160
    """
    # Key steps:
    # - temp_kg = KnowledgeGraph(entities=[...from relationships...])
    # - await temp_kg.embed_entities(embeddings_func, weights)
    # - Create Relationship objects with t_start, t_end parsed
    # - await kg.embed_relationships(embeddings_func)
    # - atomic_kgs = kg.split_into_atomic_kgs()
    # - return parallel_atomic_merge(atomic_kgs, ...)
```

---

#### Stage 3a: Binary KG Merge
```python
def merge_two_kgs(
    self,
    kg1: KnowledgeGraph,
    kg2: KnowledgeGraph,
    rel_threshold: float = 0.8,
    ent_threshold: float = 0.8
) -> KnowledgeGraph:
    """
    [PAPER: Module-3 - Parallel Merge (binary operation)]
    
    Merges two KGs using entity and relationship matching.
    
    Args:
        kg1: First knowledge graph
        kg2: Second knowledge graph
        rel_threshold: Relationship matching cosine similarity threshold
        ent_threshold: Entity matching cosine similarity threshold
    
    Returns:
        KnowledgeGraph: Union of kg1 and kg2 with deduplicated entities/relationships
    
    Implementation:
        updated_entities, updated_relationships = matcher.match_entities_and_update_relationships(
            entities_2=kg1.entities,
            relationships_2=kg1.relationships,
            entities_1=kg2.entities,
            relationships_1=kg2.relationships,
            rel_threshold=rel_threshold,
            ent_threshold=ent_threshold
        )
        return KnowledgeGraph(entities=updated_entities, relationships=updated_relationships)
    
    Line Numbers: atom.py:50-60
    """
```

---

#### Stage 3b: Parallel Binary Tree Merge
```python
def parallel_atomic_merge(
    self,
    kgs: List[KnowledgeGraph],
    existing_kg: Optional[KnowledgeGraph] = None,
    rel_threshold: float = 0.8,
    ent_threshold: float = 0.8,
    max_workers: int = 4
) -> KnowledgeGraph:
    """
    [PAPER: Module-3 - Parallel Atomic Merge]
    
    Merges multiple KGs using binary tree strategy with ThreadPoolExecutor.
    
    Args:
        kgs: List of KGs to merge (typically atomic KGs with 1 relationship each)
        existing_kg: Optional existing global KG to merge with final result
        rel_threshold: Relationship matching threshold
        ent_threshold: Entity matching threshold
        max_workers: Number of threads in executor
    
    Returns:
        KnowledgeGraph: Single merged KG from all inputs
    
    Algorithm:
        1. While len(kgs) > 1:
            a. Pair up KGs: [(kg0, kg1), (kg2, kg3), ...]
            b. Submit all pairs to ThreadPoolExecutor.submit(merge_two_kgs, ...)
            c. Collect merged results
            d. Handle leftover KG (if odd count)
            e. kgs = merged_results + [leftover]
        
        2. Final merge with existing_kg if provided
        3. Return result
    
    Example:
        Input: [kg1, kg2, kg3, kg4]
        Round 1: merge(kg1, kg2) || merge(kg3, kg4) → [merged_12, merged_34]
        Round 2: merge(merged_12, merged_34) → [final_kg]
        Return: final_kg
    
    Line Numbers: atom.py:65-95
    """
```

---

## Part II: iText2KG Pipeline (Non-Temporal KG)

### Core Orchestrator: `iText2KG` Class

**File:** `itext2kg/itext2kg_star/itext2kg.py`

#### Main Entry Point
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
) -> KnowledgeGraph:
    """
    [PAPER: iText2KG Multi-Document KG Extraction]
    
    Builds non-temporal KG from document sections with incremental consolidation.
    
    Main Process:
        1. Extract entities from sections[0]
        2. Extract & verify relations from sections[0]
        3. FOR i IN 1..len(sections)-1:
            a. Extract entities from sections[i]
            b. Match with global entities (matcher.process_lists)
            c. Extract & verify relations from sections[i]
            d. Match with global relations
            e. Consolidate duplicates
        4. IF existing_knowledge_graph:
            Match and merge with existing KG
        5. Remove duplicates (entities & relationships)
        6. Return final KG
    """
```

---

## Key Data Flow

### ATOM Data Flow:
```
Dict[timestamp → facts]
    ↓
FOR EACH timestamp:
    ├─ extract_quintuples() → RelationshipsExtractor[]
    ├─ build_atomic_kg_from_quintuples() → KnowledgeGraph
    │  ├─ Embed entities (name_weight × emb_name + label_weight × emb_label)
    │  ├─ Parse timestamps (string→Unix)
    │  ├─ split_into_atomic_kgs() → KG[] (1 rel per KG)
    │  └─ parallel_atomic_merge() → KG (merged atomic)
    └─ merge_two_kgs(result, kg_t) → result KG updated
    ↓
Final Temporal KG (all entities + relationships merged across timestamps)
```

### iText2KG Data Flow:
```
[section₁, section₂, ..., sectionₙ]
    ↓
Section 0:
    ├─ extract_entities() → Entity[]
    └─ extract_verify_and_correct_relations() → Relationship[]
    ↓ (set as global)
FOR Section 1..n:
    ├─ extract_entities() → local_entities
    ├─ matcher.process_lists(local, global) → (new_entities, updated_global)
    ├─ extract_verify_and_correct_relations() → local_relationships
    ├─ matcher.process_lists(local, global) → (new_rels, updated_global)
    └─ Extend global with new items
    ↓
Remove duplicates → Final Non-Temporal KG
```

