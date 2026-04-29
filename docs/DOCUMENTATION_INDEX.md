# Complete Documentation Index & Quick Reference

## 📚 Documentation Files Generated

This documentation package contains 5 comprehensive guides covering your iText2KG & ATOM pipelines.

### Generated Documents

1. **[PIPELINE_AND_CONFIGURATION_GUIDE.md](PIPELINE_AND_CONFIGURATION_GUIDE.md)** ⭐ **START HERE**
   - High-level overview of both pipelines
   - Configuration details and variable table
   - Key parameters and their effects
   - Quick integration examples
   - **Best for:** Understanding the big picture

2. **[METHOD_REFERENCE_SIGNATURES.md](METHOD_REFERENCE_SIGNATURES.md)** 📖 **API REFERENCE**
   - Complete method signatures with docstrings
   - Stage-by-stage breakdown (extraction → embedding → matching → storage)
   - Parameter descriptions
   - Data flow diagrams
   - **Best for:** Understanding method contracts

3. **[VISUAL_CODE_EXECUTION_FLOW.md](VISUAL_CODE_EXECUTION_FLOW.md)** 🎯 **EXECUTION WALKTHROUGH**
   - Line-by-line execution trace of main.py
   - Concrete examples with actual data
   - Error scenarios and solutions
   - Neo4j output schema
   - **Best for:** Debugging and tracing execution

4. **[PIPELINE_AND_CONFIGURATION_GUIDE.md](PIPELINE_AND_CONFIGURATION_GUIDE.md)** (From Subagent)
   - Codebase overview
   - Class hierarchy
   - Integration guide
   - **Best for:** High-level understanding

---

## 🎯 Quick Start by Use Case

### "I want to understand the pipeline"
→ Read: [PIPELINE_AND_CONFIGURATION_GUIDE.md](PIPELINE_AND_CONFIGURATION_GUIDE.md) §1-2

### "I need to call a specific method"
→ Read: [METHOD_REFERENCE_SIGNATURES.md](METHOD_REFERENCE_SIGNATURES.md) + search for method name

### "I want to trace what main.py does"
→ Read: [VISUAL_CODE_EXECUTION_FLOW.md](VISUAL_CODE_EXECUTION_FLOW.md) §Part I

---

## 🔴 CRITICAL FINDINGS

### Configuration Mismatch: Type Annotation

**Location:** `models/models.py` lines 16-19

**Problem:**
```python
def get_default_embedding_model() -> OpenAIEmbeddings:  # ❌ WRONG TYPE
    return _DEFAULT_EMBEDDINGS  # Actually returns OllamaEmbeddings
```

**Impact:** Type checkers fail; IDE autocomplete incorrect

**Fix:**
```python
from typing import Union
from langchain_openai import OpenAIEmbeddings
from langchain_ollama import OllamaEmbeddings

def get_default_embedding_model() -> Union[OpenAIEmbeddings, OllamaEmbeddings]:
    return _DEFAULT_EMBEDDINGS
```


---

## 📊 Pipeline Architecture Summary

### ATOM (Temporal KG from Atomic Facts)

**Input:** `Dict[timestamp → List[facts]]`

**Pipeline:**
```
Facts → [Extract Quintuples] → [Build Atomic KGs] → [Parallel Merge] → [Store Neo4j]
         ↓                        ↓                   ↓
      RelationshipSchema[]    KnowledgeGraph[]   KnowledgeGraph
```

**Entry Point:** `Atom.build_graph_from_different_obs_times()`

**Key Methods:**
- `extract_quintuples()` - LLM extraction with temporal context
- `build_atomic_kg_from_quintuples()` - Create 1-relationship KGs
- `parallel_atomic_merge()` - Binary tree merge using threads
- `embed_entities()` - Weighted embedding: 0.8×name + 0.2×label

**Output:** `KnowledgeGraph` with t_start, t_end, t_obs properties

---

### iText2KG (Non-Temporal KG from Document Sections)

**Input:** `List[section_str]`

**Pipeline:**
```
Sections → [Extract Entities] → [Extract Relations] → [Consolidate] → [Store Neo4j]
           ↓                    ↓                     ↓
           Entity[]             Relationship[]       KnowledgeGraph
```

**Entry Point:** `iText2KG.build_graph()`

**Key Methods:**
- `extract_entities()` - Entity extraction with embeddings
- `extract_verify_and_correct_relations()` - Relations with verification loop
- `process_lists()` - Incremental matching & consolidation
- `embed_entities()` - Balanced embedding: 0.6×name + 0.4×label

**Output:** `KnowledgeGraph` with global entities/relationships

---

## 🔑 Key Classes

| Class | File | Purpose | Key Method |
|-------|------|---------|-----------|
| `Atom` | itext2kg/atom/atom.py | Temporal KG orchestrator | `build_graph_from_different_obs_times()` |
| `iText2KG` | itext2kg/itext2kg_star/itext2kg.py | Document KG orchestrator | `build_graph()` |
| `GraphMatcher` | itext2kg/atom/graph_matching/matcher.py | ATOM entity/rel matching | `_batch_match_entities()` |
| `Matcher` | itext2kg/itext2kg_star/graph_matching/matcher.py | iText2KG incremental match | `process_lists()` |
| `KnowledgeGraph` | itext2kg/atom/models/knowledge_graph.py | KG container | `embed_entities()`, `split_into_atomic_kgs()` |
| `Entity` | itext2kg/atom/models/entity.py | Node container | (properties: label, name, embeddings) |
| `Relationship` | itext2kg/atom/models/relationship.py | Edge container | (properties: name, t_start, t_end, t_obs) |
| `LangchainOutputParser` | itext2kg/llm_output_parsing/ | LLM interface | `extract_information_as_json_for_context()` |
| `Neo4jStorage` | itext2kg/graph_integration/neo4j_storage.py | Graph database | `visualize_graph()` |

---

## 📝 Configuration Files

### env_config.py
Loads environment variables from `.env` file:
```python
ollama_base_url = "http://localhost:11434/v1"  # Ollama service
together_api_key = "..."  # Together AI (optional)
openai_api_key = "..."  # OpenAI (optional)
neo4j_uri = "bolt://localhost:7687"  # Neo4j database
```

### models_config.py
Defines available model instances:
```python
# Ollama (Local)
model_ollama_llama3 = ChatOllama(model="qwen3.6:27b-q4_K_M", ...)
embeddings_ollama_nomic = OllamaEmbeddings(model="nomic-embed-text", ...)

# OpenAI
model_gpt4o_mini = ChatOpenAI(model="gpt-4o-mini", ...)
embeddings_text_embedding_3_small = OpenAIEmbeddings(model="text-embedding-3-small", ...)

# Together AI
model_llama3_3_70b = ChatOpenAI(model="meta-llama/Llama-3.3-70B-Instruct-Turbo-Free", ...)
embeddings_bge_large = OpenAIEmbeddings(model="BAAI/bge-large-en-v1.5", ...)
```

### models.py
Selects active models:
```python
_DEFAULT_LLM = model_ollama_llama3
_DEFAULT_EMBEDDINGS = embeddings_ollama_nomic  # ⚠️ Type mismatch!

def get_default_model() -> ChatOllama:
    return _DEFAULT_LLM

def get_default_embedding_model() -> OpenAIEmbeddings:  # ❌ WRONG!
    return _DEFAULT_EMBEDDINGS
```

---

## 🔄 Data Model Hierarchy

### KnowledgeGraph
```
KnowledgeGraph
├── entities: List[Entity]
│   └── Entity
│       ├── label: str (e.g., "Person", "Location")
│       ├── name: str (unique identifier)
│       └── properties: EntityProperties
│           └── embeddings: np.ndarray (1024-dim Nomic)
│
└── relationships: List[Relationship]
    └── Relationship
        ├── name: str (predicate, PRESENT TENSE)
        ├── startEntity: Entity
        ├── endEntity: Entity
        └── properties: RelationshipProperties
            ├── embeddings: np.ndarray
            ├── t_start: List[float] (Unix timestamps)
            ├── t_end: List[float]
            ├── t_obs: List[float] (ATOM only)
            └── atomic_facts: List[str] (ATOM only)
```

---

## 🚀 Execution Sequence: ATOM Pipeline

```
main.py
  ↓
1. Load config: get_default_model(), get_default_embedding_model()
2. Load data: pd.read_pickle() → DataFrame
3. Format: to_dictionary() → Dict[timestamp → facts]
4. Initialize: Atom(llm, embeddings)
5. Build: FOR EACH timestamp:
           ├─ extract_quintuples() → LLM output
           ├─ build_atomic_kg_from_quintuples()
           │  ├─ embed_entities()
           │  ├─ embed_relationships()
           │  ├─ split_into_atomic_kgs()
           │  └─ parallel_atomic_merge()
           └─ merge_two_kgs() with accumulated result
6. Store: Neo4jStorage.visualize_graph()
```

**Total Async Operations:** 3+ per timestamp (extract + embed entities + embed relationships)

**Parallelization:** ThreadPoolExecutor in merge (max_workers=8)

**Typical Duration:** Minutes to hours (depends on data size + LLM API)

---

## 🛠️ Common Tasks

### Switch to OpenAI GPT-4o Mini

1. Update `.env`: Add `OPENAI_API_KEY`
2. Update `models/models.py`:
   ```python
   _DEFAULT_LLM = model_gpt4o_mini
   _DEFAULT_EMBEDDINGS = embeddings_text_embedding_3_small
   ```
3. Uncomment definitions in `models_config.py` (currently commented)

### Switch to Together AI (Free Tier)

1. Update `.env`: Add `TOGETHER_API_KEY`
2. Update `models/models.py`:
   ```python
   _DEFAULT_LLM = model_llama3_3_70b
   _DEFAULT_EMBEDDINGS = embeddings_bge_large
   ```

### Increase Merge Parallelism

```python
# In main.py, pass max_workers parameter:
kg = await atom.build_graph_from_different_obs_times(
    atomic_facts_with_obs_timestamps=news_covid_dict,
    max_workers=16  # Increase from default 8
)
```

### Change Entity Matching Threshold

```python
kg = await atom.build_graph_from_different_obs_times(
    atomic_facts_with_obs_timestamps=news_covid_dict,
    ent_threshold=0.6  # More lenient matching
)
```

---

## 🐛 Debugging Guide

### Check Configuration
```python
# test_config.py
from models.models import get_default_model, get_default_embedding_model

llm = get_default_model()
emb = get_default_embedding_model()

print(f"LLM: {type(llm).__name__}")
print(f"Embeddings: {type(emb).__name__}")
```

### Trace Extraction
```python
# In your code, after extract_quintuples():
print(f"Extracted {len(quintuples)} relationships")
for rel in quintuples[0].relationships:
    print(f"  {rel.startNode.name} --{rel.name}--> {rel.endNode.name}")
    print(f"    t_start: {rel.t_start}, t_end: {rel.t_end}")
```

### Check Neo4j Connection
```python
# test_neo4j.py
from itext2kg import Neo4jStorage

storage = Neo4jStorage(
    uri="bolt://localhost:7687",
    username="neo4j",
    password="your_password"
)

# Try a simple query
storage.run_query("MATCH (n) RETURN count(n) as count")
```

---

## 📋 Checklist: Project Setup

- [ ] Python 3.9+ installed
- [ ] Virtual environment activated
- [ ] Dependencies installed: `pip install -r itext2kg-1.0.0/requirements.txt`
- [ ] `.env` file created with required keys
- [ ] For Ollama: Service running (`ollama serve`)
- [ ] For Ollama: Models pulled (`ollama pull qwen3.6:27b-q4_K_M`, `ollama pull nomic-embed-text`)
- [ ] For Neo4j: Database running (Docker or local)
- [ ] Type annotation mismatch fixed in `models/models.py`
- [ ] Configuration test passes (`python test_config.py`)
- [ ] Able to run `main.py` without errors

---

## 📚 Reference Links

### Within Repository
- **ATOM class:** [itext2kg/atom/atom.py](itext2kg-1.0.0/itext2kg/atom/atom.py)
- **iText2KG class:** [itext2kg/itext2kg_star/itext2kg.py](itext2kg-1.0.0/itext2kg/itext2kg_star/itext2kg.py)
- **KnowledgeGraph model:** [itext2kg/atom/models/knowledge_graph.py](itext2kg-1.0.0/itext2kg/atom/models/knowledge_graph.py)
- **Schemas:** [itext2kg/atom/models/schemas.py](itext2kg-1.0.0/itext2kg/atom/models/schemas.py)
- **Storage:** [itext2kg/graph_integration/neo4j_storage.py](itext2kg-1.0.0/itext2kg/graph_integration/neo4j_storage.py)

### External Documentation
- **LangChain:** https://python.langchain.com/
- **Ollama:** https://ollama.ai/
- **Neo4j:** https://neo4j.com/developer/
- **Paper (EACL 2026):** https://aclanthology.org/2026.findings-eacl.49.pdf

---

## 💡 Pro Tips

1. **Always use `async`/`await`** - Both pipelines are fully asynchronous
2. **Start with thresholds=0.7** - Too high causes missed matches
3. **Monitor embeddings quality** - Use cosine_similarity to check clustering
4. **Batch process large datasets** - Split by timestamp or section
5. **Test configuration first** - Before running full pipeline
6. **Use Neo4j for visualization** - Much easier than raw KG inspection

---

## 📞 Support

For issues, check:
1. Is Ollama running? `curl http://localhost:11434/v1/models`
2. Are models available? `ollama list`
3. Is Neo4j running? `bolt://localhost:7687`
4. Check type annotations in `models/models.py`
5. Review [CONFIGURATION_MISMATCH_ANALYSIS.md](CONFIGURATION_MISMATCH_ANALYSIS.md)

