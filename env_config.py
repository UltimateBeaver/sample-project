import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()
# API keys
openai_api_key =        os.getenv("OPENAI_API_KEY")
llamacpp_embed_base =   os.getenv("LLAMACPP_EMBED_BASE")
openai_api_base =       os.getenv("OPENAI_API_BASE")          # e.g. https://openrouter.ai/api/v1
together_api_base =     os.getenv("TOGETHER_API_BASE")
together_api_key =      os.getenv("TOGETHER_API_KEY")
ollama_base_url =       os.getenv("OLLAMA_BASE_URL",    "http://localhost:11434")
# Neo4j
neo4j_uri =             os.getenv("NEO4J_URI",          "bolt://localhost:7687")
neo4j_username =        os.getenv("NEO4J_USERNAME")
neo4j_password =        os.getenv("NEO4J_PASSWORD")

# Langchain output parser configuration
provider_openai_max_elements_per_batch =    int(os.getenv("PROVIDER_OPENAI_MAX_ELEMENTS_PER_BATCH"))
provider_openai_max_tokens_per_batch =      int(os.getenv("PROVIDER_OPENAI_MAX_TOKENS_PER_BATCH"))
provider_openai_max_context_window =        int(os.getenv("PROVIDER_OPENAI_MAX_CONTEXT_WINDOW"))
provider_ollama_max_elements_per_batch =    int(os.getenv("PROVIDER_OLLAMA_MAX_ELEMENTS_PER_BATCH"))
provider_ollama_max_tokens_per_batch =      int(os.getenv("PROVIDER_OLLAMA_MAX_TOKENS_PER_BATCH"))
provider_ollama_max_context_window =        int(os.getenv("PROVIDER_OLLAMA_MAX_CONTEXT_WINDOW"))

# Document parser configuration
num_rows_to_process =                       abs(int(os.getenv("NUM_ROWS_TO_PROCESS", 0)))  # 0 means process all rows
doc_parser_input_excel_path =               os.getenv("DOC_PARSER_INPUT_EXCEL_PATH",                "./data/dataset.xlsx")
doc_parser_output_excel_path =              os.getenv("DOC_PARSER_OUTPUT_EXCEL_PATH",               "./data/dataset_with_factoids.xlsx")
doc_parser_enable_parallel_processing =     bool(os.getenv("DOC_PARSER_ENABLE_PARALLEL_PROCESSING", "false").lower() == "true")
doc_parser_batch_size =                     int(os.getenv("DOC_PARSER_BATCH_SIZE",                  2))
column_name_date =                          os.getenv("COLUMN_NAME_DATE",                           "date")
column_name_paragraph =                     os.getenv("COLUMN_NAME_PARAGRAPH",                      "lead_paragraph")
column_name_sentiment =                     os.getenv("COLUMN_NAME_SENTIMENT",                      "sentiment_score")
column_name_translated_paragraph =          os.getenv("COLUMN_NAME_TRANSLATED_PARAGRAPH",           "translated_paragraph")
column_name_translated_sentiment =          os.getenv("COLUMN_NAME_TRANSLATED_SENTIMENT",           "translated_sentiment")
column_name_date_translated_paragraph =     os.getenv("COLUMN_NAME_DATE_TRANSLATED_PARAGRAPH",      "lead_paragraph_observation_date")
column_name_factoids_extracted =            os.getenv("COLUMN_NAME_FACTOIDS_EXTRACTED",             "factoids_extracted")
column_name_factoids_ground_truth =         os.getenv("COLUMN_NAME_FACTOIDS_GROUND_TRUTH",          "factoids_g_truth")

# Language configuration
enable_translation =                        os.getenv("ENABLE_TRANSLATION",         "false").lower() == "true"  # Auto-translate if not English
enable_translator_few_shot =                os.getenv("ENABLE_TRANSLATOR_FEW_SHOT", "false").lower() == "true"  # Whether to use few shot examples for sentiment evaluation or not
translator_batch_size =                     int(os.getenv("TRANSLATOR_BATCH_SIZE",  2))                  # Batch size for translation
translator_few_shot_seed =                  os.getenv("TRANSLATOR_FEW_SHOT_SEED",   42)                   # Seed for extracting random 5 samples for few shot sentiment context