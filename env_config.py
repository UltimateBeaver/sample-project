import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()
# LLama.cpp config
llamacpp_num_parallel_slots = int(os.getenv("LLAMACPP_MODEL_NUM_PARALLEL_SLOTS", 1))
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

# ATOM
# used in main.py pipeline
similarity_threshold_entity = float(os.getenv("SIMILARITY_THRESHOLD_ENTITY", 0.8))
similarity_threshold_relationship = float(os.getenv("SIMILARITY_THRESHOLD_RELATIONSHIP", 0.7))
# used in evaluation pipeline
similarity_threshold_eval_factoid = float(os.getenv("SIMILARITY_THRESHOLD_EVAL_FACTOID", 0.7))
similarity_threshold_eval_quintuple = float(os.getenv("SIMILARITY_THRESHOLD_EVAL_QUINTUPLE", 0.7))
similarity_threshold_eval_merge = float(os.getenv("SIMILARITY_THRESHOLD_EVAL_MERGE", 0.8))

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
enable_parallel_quintuples_extraction =     bool(os.getenv("ENABLE_PARALLEL_QUINTUPLES_EXTRACTION", "false").lower() == "true")
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
column_name_quintuples_ground_truth =       os.getenv("COLUMN_NAME_QUINTUPLES_GROUND_TRUTH",        "quintuples_g_truth")
column_name_quintuples_extracted =          os.getenv("COLUMN_NAME_QUINTUPLES_EXTRACTED",           "quintuples_extracted")
column_name_quintuples_extracted_from_raw_text = os.getenv("COLUMN_NAME_QUINTUPLES_EXTRACTED_FROM_RAW_TEXT",           "quintuples_extracted_from_raw_text")
column_name_factoids_prompt_tokenc =        os.getenv("COLUMN_NAME_FACTOIDS_EXTRACTION_PROMPT_TOKEN_COUNT",           "factoids_prompt_tokenc")
column_name_quintuples_prompt_tokenc =      os.getenv("COLUMN_NAME_QUINTUPLES_EXTRACTION_PROMPT_TOKEN_COUNT",         "quintuples_prompt_tokenc")
column_name_quintuples_raw_prompt_tokenc =  os.getenv("COLUMN_NAME_QUINTUPLES_RAW_EXTRACTION_PROMPT_TOKEN_COUNT",     "quintuples_raw_prompt_tokenc")

# Language configuration
enable_translation =                        os.getenv("ENABLE_TRANSLATION",         "false").lower() == "true"  # Auto-translate if not English
enable_translator_few_shot =                os.getenv("ENABLE_TRANSLATOR_FEW_SHOT", "false").lower() == "true"  # Whether to use few shot examples for sentiment evaluation or not
translator_batch_size =                     int(os.getenv("TRANSLATOR_BATCH_SIZE",  2))                         # Batch size for translation
translator_few_shot_seed =                  int(os.getenv("TRANSLATOR_FEW_SHOT_SEED",   42))                    # Seed for extracting random 5 samples for few shot sentiment context

# Evaluation configuration
eval_input_dataset_path =                   os.getenv("EVAL_INPUT_DATASET_PATH",            "./datasets/atom/my_test_datasets/dataset.pkl")
eval_input_knowledge_graph_path =           os.getenv("EVAL_INPUT_KNOWLEDGE_GRAPH_PATH",    "./datasets/atom/my_test_datasets/eval_kg.pkl")
eval_output_dataset_path =                  os.getenv("EVAL_OUTPUT_DATASET_PATH",           "./datasets/atom/my_test_datasets/dataset_with_factoids.pkl")
eval_output_results_path =                  os.getenv("EVAL_OUTPUT_RESULTS_PATH",           "./datasets/atom/my_test_datasets/evaluation_results")
eval_cache_path =                           os.getenv("EVAL_CACHE_PATH",                    "./datasets/atom/my_test_datasets/cache")
eval_checkpoint_factoids_path =             os.getenv("EVAL_CHECKPOINT_FACTOIDS_PATH",      "./datasets/atom/my_test_datasets/factoids_checkpoint.json")
eval_checkpoint_quintuples_path =           os.getenv("EVAL_CHECKPOINT_QUINTUPLES_PATH",    "./datasets/atom/my_test_datasets/quintuples_checkpoint.json")
eval_model_postfixes_list =                 (os.getenv("EVAL_MODEL_POSTFIXES_LIST",          "_llamacpp_gemma4 _ollama_gemma4")).split(" ")
eval_model_postfixes_to_plot_list =         (os.getenv("EVAL_MODEL_POSTFIXES_TO_PLOT_LIST",  "_llamacpp_gemma4 _ollama_gemma4")).split(" ")