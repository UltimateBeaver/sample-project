# sample-project
Sample repo for master's degree thesis

# Requirements
* Python (3.9 or greater) at https://www.python.org/downloads/
* Docker
* Ollama at https://ollama.com/download (used as a temporarily free model)

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
OLLAMA_BASE_URL=http://localhost:11434/v1
# Neo4j
NEO4J_URI=bolt://localhost:7687
NEO4J_USERNAME=neo4j
NEO4J_PASSWORD=password
```
3. Run/download all the required containers (Neo4j) through the following commands:
```bash
cd docker
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
ollama pull gemma3:1b
# Default embedding model
ollama pull nomic-embed-text
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

# Developer tips
- When adding a new package, add the definition also in pyproject.toml