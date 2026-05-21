import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()
# API keys
openai_api_key = os.getenv("OPENAI_API_KEY")
llamacpp_embed_base = os.getenv("LLAMACPP_EMBED_BASE")
openai_api_base = os.getenv("OPENAI_API_BASE")          # e.g. https://openrouter.ai/api/v1
together_api_base = os.getenv("TOGETHER_API_BASE")
together_api_key = os.getenv("TOGETHER_API_KEY")
ollama_base_url = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
# Neo4j
neo4j_uri = os.getenv("NEO4J_URI")
neo4j_username = os.getenv("NEO4J_USERNAME")
neo4j_password = os.getenv("NEO4J_PASSWORD")

# Static constants
num_rows_to_process = abs(int(os.getenv("NUM_ROWS_TO_PROCESS", 0)))  # 0 means process all rows
doc_parser_input_excel_path = os.getenv("DOC_PARSER_INPUT_EXCEL_PATH")
doc_parser_output_excel_path = os.getenv("DOC_PARSER_OUTPUT_EXCEL_PATH")
column_name_date = os.getenv("COLUMN_NAME_DATE", "date")
column_name_paragraph = os.getenv("COLUMN_NAME_PARAGRAPH", "lead_paragraph")