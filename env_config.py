import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()
# API keys
openai_api_key = os.getenv("OPENAI_API_KEY")
openai_api_base = os.getenv("OPENAI_API_BASE")          # e.g. https://openrouter.ai/api/v1
together_api_base = os.getenv("TOGETHER_API_BASE")
together_api_key = os.getenv("TOGETHER_API_KEY")
ollama_base_url = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434/v1")
# Neo4j
neo4j_uri = os.getenv("NEO4J_URI")
neo4j_username = os.getenv("NEO4J_USERNAME")
neo4j_password = os.getenv("NEO4J_PASSWORD")