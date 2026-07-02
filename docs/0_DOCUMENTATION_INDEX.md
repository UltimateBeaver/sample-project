# Documentation Index & Navigation Guide

## 📚 Documentation Structure (5 Core Guides)

This documentation covers your iText2KG & ATOM knowledge graph extraction pipelines.

### Core Documentation Files

1. **[CONFIGURATION_GUIDE.md](./1_CONFIGURATION_GUIDE.md)** ⭐ **START HERE**
   - High-level overview of both ATOM and iText2KG pipelines
   - Complete configuration reference (environment variables, models, batch settings)
   - Translation service setup
   - Configuration validation
   - Logging tuning
   - **Best for:** Getting started and understanding the big picture

2. **[VISUAL_ARCHITECTURE_AND_REFERENCE.md](./2_VISUAL_ARCHITECTURE_AND_REFERENCE.md)** 🎨 **VISUAL GUIDE**
   - Architecture diagrams showing module interactions
   - Data flow visualizations for both pipelines
   - Class hierarchy and relationships
   - Quick Reference - Key Methods by Task
   - **Best for:** Visual learners and high-level understanding

3. **[DETAILED_METHOD_REFERENCE.md](./3_DETAILED_METHOD_REFERENCE.md)** 📋 **METHOD DEEP-DIVE**
   - Comprehensive method documentation with detailed process flows
   - ATOM pipeline methods (extract_quintuples, build_atomic_kg_from_quintuples, parallel_atomic_merge)
   - iText2KG pipeline methods (build_graph, extract_entities, extract_verify_and_correct_relations)
   - Matching algorithms (GraphMatcher, Matcher, entity/relationship deduplication)
   - Supporting components (DocumentsDistiller, Neo4jStorage, LangchainOutputParser)
   - **Best for:** Deep-diving into specific methods and understanding implementation details

4. **[DOCUMENTATION_INDEX.md](./0_DOCUMENTATION_INDEX.md)** (This File)
   - Navigation guide
   - Quick start by use case
   - Configuration overview
   - **Best for:** Navigating the documentation

---

## 🎯 Quick Start by Use Case

### "I want to understand the pipeline architecture"
→ Read: [CONFIGURATION_GUIDE.md](./1_CONFIGURATION_GUIDE.md) §Pipeline Overview

### "I need to understand a specific method"
→ Read: [DETAILED_METHOD_REFERENCE.md](./3_DETAILED_METHOD_REFERENCE.md) (search for method name)

### "I want to see how modules interact"
→ Read: [VISUAL_ARCHITECTURE_AND_REFERENCE.md](./2_VISUAL_ARCHITECTURE_AND_REFERENCE.md)

### "I need to configure the system"
→ Read: [CONFIGURATION_GUIDE.md](./1_CONFIGURATION_GUIDE.md) §Configuration sections

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
- [ ] For Ollama: Models pulled (`ollama pull gemma4:e4b`, `ollama pull nomic-embed-text`)
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

## 📞 Support & Troubleshooting

For issues, check:
1. Is Ollama running? `curl http://localhost:11434/v1/models`
2. Are models available? `ollama list`
3. Is Neo4j running? `bolt://localhost:7687`
4. Check type annotations in `models/models.py` (should match actual return types)
5. Verify default models in `models/models.py` match your setup

