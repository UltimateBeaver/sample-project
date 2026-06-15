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
neo4j_uri = os.getenv("NEO4J_URI", "bolt://localhost:7687")
neo4j_username = os.getenv("NEO4J_USERNAME")
neo4j_password = os.getenv("NEO4J_PASSWORD")

# Langchain output parser configuration
provider_openai_max_elements_per_batch =    int(os.getenv("PROVIDER_OPENAI_MAX_ELEMENTS_PER_BATCH"))
provider_openai_max_tokens_per_batch =      int(os.getenv("PROVIDER_OPENAI_MAX_TOKENS_PER_BATCH"))
provider_openai_max_context_window =        int(os.getenv("PROVIDER_OPENAI_MAX_CONTEXT_WINDOW"))
provider_ollama_max_elements_per_batch =    int(os.getenv("PROVIDER_OLLAMA_MAX_ELEMENTS_PER_BATCH"))
provider_ollama_max_tokens_per_batch =      int(os.getenv("PROVIDER_OLLAMA_MAX_TOKENS_PER_BATCH"))
provider_ollama_max_context_window =        int(os.getenv("PROVIDER_OLLAMA_MAX_CONTEXT_WINDOW"))

# Static constants
num_rows_to_process = abs(int(os.getenv("NUM_ROWS_TO_PROCESS", 0)))  # 0 means process all rows
doc_parser_input_excel_path = os.getenv("DOC_PARSER_INPUT_EXCEL_PATH", "./data/dataset.xlsx")
doc_parser_output_excel_path = os.getenv("DOC_PARSER_OUTPUT_EXCEL_PATH", "./data/dataset_with_factoids.xlsx")
doc_parser_enable_parallel_processing = bool(os.getenv("DOC_PARSER_ENABLE_PARALLEL_PROCESSING", "false").lower() == "true")
doc_parser_batch_size = int(os.getenv("DOC_PARSER_BATCH_SIZE", 2))
column_name_date = os.getenv("COLUMN_NAME_DATE", "date")
column_name_paragraph = os.getenv("COLUMN_NAME_PARAGRAPH", "lead_paragraph")

# Language configuration
input_language = os.getenv("INPUT_LANGUAGE", "en")  # Source language ("en", "it", etc.)
translation_model_name = os.getenv("TRANSLATION_MODEL_NAME", "it-en")  # Language pair for translation
enable_translation = os.getenv("ENABLE_TRANSLATION", "false").lower() == "true"  # Auto-translate if not English
translator_sentence_batch_size = int(os.getenv("TRANSLATOR_SENTENCE_BATCH_SIZE", 32))   # Batch size for sentence processing
translator_sentence_max_length = int(os.getenv("TRANSLATOR_SENTENCE_MAX_LENGTH", 256))  # Max length for sentence translation