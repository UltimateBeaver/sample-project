"""
Factoids Extraction from NYT COVID-19 News Articles

This script extracts atomic facts (factoids) from news article paragraphs using Large Language Models.
It processes the NYT COVID-19 dataset and decomposes each paragraph into individual factoids. 
The script supports batch processing with
checkpointing and can be used with different LLM models (OpenAI GPT, Claude, Mistral).

Usage:
    python factoids_extraction_nyt.py

Output:
    - A pickle file containing the original dataset with an additional column for extracted factoids
    - Checkpoint files for resuming interrupted processing
"""

import sys
import asyncio
import logging
import time
import json
from pathlib import Path
import argparse

import pandas as pd
import numpy as np


# from langchain_mistralai import ChatMistralAI
# from langchain_mistralai import MistralAIEmbeddings
from langchain_openai import ChatOpenAI, OpenAIEmbeddings

from itext2kg.llm_output_parsing.langchain_output_parser import LangchainOutputParser
from itext2kg.atom.models import AtomicFact

from document_parser.parser_prompt import ParserPrompt
from models.models import get_default_model, get_default_embedding_model
from env_config import (
    column_name_date, column_name_date_translated_paragraph, column_name_factoids_extracted, column_name_factoids_prompt_tokenc, doc_parser_batch_size, 
    eval_input_dataset_path, eval_output_dataset_path, eval_checkpoint_factoids_path, eval_model_postfixes_list, num_rows_to_process
)
# from langchain_anthropic import ChatAnthropic


# Add the project root to Python path
current_file = Path(__file__).resolve()
project_root = current_file.parent.parent.parent
sys.path.append(str(project_root))


# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),  # Console output
    ]
)
logger = logging.getLogger(__name__)

print("🚀 Starting factoids extraction script...")
logger.info("Setting up API connections...")

# ==========================
# Global configuration vars
# ==========================
# Paths
# INPUT_DATASET_PATH: Path = project_root / "datasets" / "atom" / "nyt_news" / "2020_nyt_COVID_last_version_ready.pkl"
# OUTPUT_DATASET_PATH: Path = project_root / "datasets" / "atom" / "nyt_news" / "2020_nyt_COVID_last_version_ready_factoids_llamacpp.pkl"
INPUT_DATASET_PATH: Path =  project_root / eval_input_dataset_path
OUTPUT_DATASET_PATH: Path = project_root / eval_output_dataset_path
NUM_ROWS_TO_PROCESS = num_rows_to_process

# Column names
# It could be used on the cumulative lead_paragraph_observation_date. You can change "lead_paragraph_observation_date" 
# to "cumul_lead_paragraph_observation_date" if you want to use the cumulative lead_paragraph_observation_date.
DATA_COL = column_name_date
PARAGRAPHS_COL_NAME: str = column_name_date_translated_paragraph
FACTOIDS_COL_NAME: str = column_name_factoids_extracted

# Sampling: number of uniformly spaced indices to process. Set to None or 0 to process all
SAMPLER_K: int | None = None

# Batch processing configuration
BATCH_SIZE: int = doc_parser_batch_size
CHECKPOINT_FILE: Path = project_root / eval_checkpoint_factoids_path


# --- Local / Native Llama.cpp Server --------------------------------------
model_llamacpp_gemma4 = get_default_model()
# --- Local / Native Llama.cpp Embeddings ----------------------------------
embeddings_llamacpp_nomic = get_default_embedding_model()

lg_kg_construction = LangchainOutputParser(
   llm_model=model_llamacpp_gemma4,
   embeddings_model=embeddings_llamacpp_nomic
)

logger.info("✅ LangchainOutputParser initialized successfully")

print("📊 Loading dataset...")
df_nyt = pd.read_pickle(INPUT_DATASET_PATH)
if num_rows_to_process > 0:
    df_nyt = df_nyt.head(num_rows_to_process)
logger.info(f"📋 Loaded dataset with {len(df_nyt)} rows")

def load_checkpoint() -> dict:
    """Load checkpoint data if it exists, otherwise return empty checkpoint."""
    if CHECKPOINT_FILE.exists():
        with open(CHECKPOINT_FILE, 'r') as f:
            checkpoint = json.load(f)
        logger.info(f"📂 Loaded checkpoint: {len(checkpoint.get('completed_batches', []))} batches completed")
        return checkpoint
    else:
        logger.info("📂 No checkpoint found, starting fresh")
        return {"completed_batches": [], "results": {}}

def save_checkpoint(checkpoint: dict):
    """Save current progress to checkpoint file."""
    with open(CHECKPOINT_FILE, 'w') as f:
        json.dump(checkpoint, f, indent=2)
    logger.info(f"💾 Checkpoint saved: {len(checkpoint['completed_batches'])} batches completed")

def _uniform_indices(num_rows: int, k: int | None) -> list[int]:
    """Return k uniformly spaced indices in [0, num_rows-1]. If k is None or <=0 or >= num_rows, return all indices."""
    if num_rows <= 0:
        return []
    if k is None or k <= 0 or k >= num_rows:
        return list(range(num_rows))
    # Use linspace to include first and last; round to nearest int and ensure uniqueness and sorted order
    raw = np.linspace(0, num_rows - 1, num=k)
    idx = sorted({int(round(v)) for v in raw})
    # If rounding caused duplicates and we have fewer than k, pad by sampling remaining uniformly
    while len(idx) < k:
        # Increase resolution and try to add more points
        cand = int(round((num_rows - 1) * (len(idx) / (k - 1) if k > 1 else 0)))
        if cand not in idx:
            idx.append(cand)
        else:
            # fallback linear sweep
            for j in range(num_rows):
                if j not in idx:
                    idx.append(j)
                    break
        idx = sorted(idx)
    return idx[:k]


async def extract_factoid_batch(contexts: list[str]) -> list[list[str]]:
    """Extract factoids for a batch of contexts. Returns list of factoid lists, one per context."""
    logger.info(f"🔍 Starting factoid extraction for batch of {len(contexts)} contexts...")
    
    atomic_facts = await lg_kg_construction.extract_information_as_json_for_context(
        AtomicFact, contexts, ParserPrompt._create_temporal_system_query("2000-01-01")
    )
    
    logger.info(f"✅ Extracted {len(atomic_facts)} atomic fact objects")
    
    # Each atomic_facts[i] corresponds to contexts[i] and contains an AtomicFact object
    # Return the atomic_fact lists, ensuring we have one result per input context
    results = []
    for i, atomic_fact_obj in enumerate(atomic_facts):
        if atomic_fact_obj and hasattr(atomic_fact_obj, 'atomic_fact'):
            results.append(atomic_fact_obj.atomic_fact)
        else:
            results.append([])  # Empty list for failed extractions
    
    # Ensure we have exactly one result per input context
    while len(results) < len(contexts):
        results.append([])
    
    return results[:len(contexts)]

def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Extract factoids from raw news paragraphs - Factoids Analysis')
    parser.add_argument('--model-postfix', '-p', type=str, required=True,
                       help='The postfix representing the backend and model you are executing the test. You can define all supported postfixes inside your .env file, through $EVAL_MODEL_POSTFIXES_LIST variable')
    return parser.parse_args()

async def main():
    start_time = time.time()

    # Parse command line arguments
    args = parse_arguments()

    if not args.model_postfix:
        logger.error('--model-postfix arg not provided')
        return
    if args.model_postfix not in eval_model_postfixes_list:
        logger.error(f'Unsupported --model-postfix arg. Supported ones are: {eval_model_postfixes_list}')
        return
    
    factoids_col_with_postfix = f"{FACTOIDS_COL_NAME}_{args.model_postfix}"
    
    try:
        print("🎯 Starting main extraction process...")
        logger.info("Beginning factoid extraction from NYT COVID data")
        
        # Load checkpoint
        checkpoint = load_checkpoint()
        
        # Determine indices to process
        num_rows = len(df_nyt)
        selected_indices = _uniform_indices(num_rows=num_rows, k=SAMPLER_K)
        logger.info(f"📝 Processing {len(selected_indices)} rows out of {num_rows} total")

        # Create batches from selected indices
        batches = []
        for i in range(0, len(selected_indices), BATCH_SIZE):
            batch_indices = selected_indices[i:i + BATCH_SIZE]
            batches.append(batch_indices)
        
        logger.info(f"📦 Created {len(batches)} batches of size {BATCH_SIZE}")

        # Initialize the results column if not exists
        if factoids_col_with_postfix not in df_nyt.columns:
            df_nyt[factoids_col_with_postfix] = None

        # Load existing results from checkpoint
        for idx_str, result in checkpoint.get("results", {}).items():
            idx = int(idx_str)
            if idx < len(df_nyt):
                df_nyt.at[df_nyt.index[idx], factoids_col_with_postfix] = result

        # Process batches
        for batch_idx, batch_indices in enumerate(batches):
            if batch_idx in checkpoint["completed_batches"]:
                logger.info(f"⏩ Skipping batch {batch_idx + 1}/{len(batches)} (already completed)")
                continue
                
            logger.info(f"🔄 Processing batch {batch_idx + 1}/{len(batches)} ({len(batch_indices)} items)")
            
            # Prepare contexts for this batch
            batch_contexts = [df_nyt.iloc[i][PARAGRAPHS_COL_NAME] for i in batch_indices]
            
            # Extract factoids for this batch
            batch_results = await extract_factoid_batch(batch_contexts)
            
            # Store results in dataframe and checkpoint
            for idx, result in zip(batch_indices, batch_results):
                df_nyt.at[df_nyt.index[idx], factoids_col_with_postfix] = result
                checkpoint["results"][str(idx)] = result
            
            # Mark batch as completed and save checkpoint
            checkpoint["completed_batches"].append(batch_idx)
            save_checkpoint(checkpoint)
            
            logger.info(f"✅ Batch {batch_idx + 1}/{len(batches)} completed and saved")

        # Compute token count for each row
        df_nyt[column_name_factoids_prompt_tokenc] = [
            lg_kg_construction.count_tokens(f"# Context: {txt}\n\n# Question: {ParserPrompt._create_temporal_system_query(date.strftime('%Y-%m-%d'))}\n\nAnswer: ")
            for txt, date in zip(df_nyt[PARAGRAPHS_COL_NAME], df_nyt[DATA_COL])
        ]

        # Save final results (do not overwrite the whole output dataset, just merge new columns)
        print(f"💾 Saving final results to: {OUTPUT_DATASET_PATH}")
        if Path.exists(OUTPUT_DATASET_PATH):
            df_out_existing = pd.read_pickle(OUTPUT_DATASET_PATH)
            df_out_existing[factoids_col_with_postfix] = df_nyt[factoids_col_with_postfix]
            df_out_existing[column_name_factoids_prompt_tokenc] = df_nyt[column_name_factoids_prompt_tokenc]
            df_out_existing.to_pickle(OUTPUT_DATASET_PATH)
        else:
            df_nyt.to_pickle(OUTPUT_DATASET_PATH)
        
        # Clean up checkpoint file
        if CHECKPOINT_FILE.exists():
            CHECKPOINT_FILE.unlink()
            logger.info("🧹 Checkpoint file cleaned up")
        
        elapsed_time = time.time() - start_time
        logger.info(f"🎉 Processing completed successfully in {elapsed_time:.2f} seconds!")
        print(f"🎉 Factoid extraction completed successfully in {elapsed_time:.2f} seconds!")
        
    except Exception as e:
        elapsed_time = time.time() - start_time
        logger.error(f"❌ Error occurred after {elapsed_time:.2f} seconds: {str(e)}")
        print(f"❌ Error occurred: {str(e)}")
        print("💡 Progress has been saved. Re-run the script to resume from where it left off.")
        raise

if __name__ == "__main__":
    print("=" * 50)
    print("  FACTOIDS EXTRACTION FROM NYT COVID DATA")
    print("=" * 50)
    asyncio.run(main())