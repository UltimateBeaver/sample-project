# sample-project
Sample repo for master's degree thesis

# Requirements
* Python (3.9 or greater) at https://www.python.org/downloads/
* Docker at https://www.docker.com/
* Ollama at https://ollama.com/download (used as a temporarily free model)
* At least 16 GB of GPU VRAM memory (to execute gemma4 model)

# Installation
1. Open up a terminal (Windows users must use Powershell), move to your desired directory and copy-paste the following commands:
```shell
git clone https://github.com/UltimateBeaver/sample-project.git
cd sample-project
python -m venv venv
venv/Scripts/activate
pip install -r requirements.txt
```
2. Create a `.env` file on the root directory of the project. Insert your own API keys
```bash
# OpenaiAPI
OPENAI_API_BASE=vvv
OPENAI_API_KEY=your-actual-key-here
# TogetherAPI
TOGETHER_API_BASE=https://api.together.xyz/v1
TOGETHER_API_KEY=your-actual-key-here
# Ollama
OLLAMA_BASE_URL=http://localhost:11434
# Neo4j
NEO4J_URI=bolt://localhost:7687
NEO4J_USERNAME=neo4j
NEO4J_PASSWORD=password
```
3. Run/download all the required containers (Neo4j) through the following commands:
```bash
# Start the containers (install image if it does not exists)
docker compose up -d
# Stop the containers
docker compose stop
# Stopping and removing containers
docker compose down
# Remove everything (including volumes)
docker compose down -v
```
4. If you use Ollama, as default settings in models/models.py, download the required models:
```bash
# Default llm model
ollama pull gemma4:e4b
# Default embedding model
ollama pull nomic-embed-text:latest
```
Note: you can download other ollama models from here: https://ollama.com/search

# Run the application
```bash
# Move to the python virtual environment (if not already there)
venv/Scripts/activate
# Make sure Docker and Ollama are running!
# Finally execute the app
python main.py
```

---
# Developer tips
- When adding a new package, add the definition also in pyproject.toml
- When changing models, please refer to [langchain_output_parser.py](./itext2kg_atom/itext2kg/llm_output_parsing/langchain_output_parser.py), updating PROVIDER_CONFIGS object. This is the default config for Ollama:
```python
    ProviderType.OLLAMA: ProviderConfig(
        name="Ollama",
        max_elements_per_batch=32,    # Conservative for local GPU: smaller batches prevent OOM on 16GB VRAM
        max_tokens_per_batch=12000,   # Increased to 12K per request (local inference, not API limits)
        max_context_window=32768,   # Ollama context window
        max_pending_requests=None,   # Ollama doesn't have explicit pending request limits
        sleep_between_batches=0.1,   # 100ms between batches to prevent GPU thrashing
    )
```

# Bugfix Checklist
[] Self loop relationships without any sense
[] Entities with empty names
[] Redundant relationships