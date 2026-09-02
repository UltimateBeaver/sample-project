"""
Quintuples Extraction from NYT COVID-19 News Articles

This script extracts knowledge graph quintuples (head_entity, relationship, tail_entity, t_start, t_end)
directly from news article paragraphs using Large Language Models. It processes the NYT COVID-19
dataset and converts each paragraph into structured knowledge quintuples that can be used for
knowledge graph construction and evaluation. Supports batch processing with checkpointing.

Usage:
    python quintuples_extraction_nyt.py

Output:
    - A pickle file containing the original dataset with an additional column for extracted quintuples
    - Checkpoint files for resuming interrupted processing
"""

import gc
import sys
import asyncio
import logging
import time
import json
from pathlib import Path
import argparse

import pandas as pd
import numpy as np

# Add the project root to Python path
current_file = Path(__file__).resolve()
project_root = current_file.parent.parent.parent
sys.path.append(str(project_root))

# from langchain_mistralai import ChatMistralAI
# from langchain_mistralai import MistralAIEmbeddings
# from langchain_openai import ChatOpenAI, OpenAIEmbeddings
# from langchain_anthropic import ChatAnthropic

from itext2kg.llm_output_parsing.langchain_output_parser import LangchainOutputParser
from itext2kg.atom.models import RelationshipsExtractor, Prompt
from models.models import get_default_model_no_reasoning, get_default_embedding_model
from env_config import (
    column_name_quintuples_extracted_from_raw_text, column_name_date, column_name_date_translated_paragraph, column_name_quintuples_raw_prompt_tokenc,
    eval_input_dataset_path, eval_output_dataset_path, num_rows_to_process, doc_parser_batch_size, eval_checkpoint_quintuples_path, eval_model_postfixes_list
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),  # Console output
    ]
)
logger = logging.getLogger(__name__)

print("🚀 Starting quintuples extraction script...")
logger.info("Setting up API connections...")

# ==========================
# Global configuration vars
# ==========================
# Paths
INPUT_DATASET_PATH: Path =  project_root / eval_input_dataset_path
OUTPUT_DATASET_PATH: Path = project_root / eval_output_dataset_path

# Column names
# It could be used on the cumulative lead_paragraph_observation_date. You can change "lead_paragraph_observation_date" 
# to "cumul_lead_paragraph_observation_date" if you want to use the cumulative lead_paragraph_observation_date.
PARAGRAPHS_COL_NAME: str = column_name_date_translated_paragraph
DATE_COL_NAME: str = column_name_date
QUINTUPLES_COL_NAME: str = column_name_quintuples_extracted_from_raw_text

# Sampling: number of uniformly spaced indices to process. Set to None or 0 to process all
SAMPLER_K: int | None = None

# Batch processing configuration
BATCH_SIZE: int = doc_parser_batch_size 
CHECKPOINT_FILE: Path = project_root / eval_checkpoint_quintuples_path

lg_kg_construction = LangchainOutputParser(
   llm_model=get_default_model_no_reasoning(),
   embeddings_model=get_default_embedding_model()
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

async def extract_quintuples_batch(contexts: list[str], timestamps: list[str]) -> list[list[tuple]]:
    """Extract quintuples for a batch of contexts. Returns list of quintuple lists, one per context."""
    logger.info(f"🔍 Starting quintuple extraction for batch of {len(contexts)} contexts...")
    
    # Process each context individually with its timestamp
    batch_results = []
    
    for i, (context, obs_timestamp) in enumerate(zip(contexts, timestamps)):
        logger.info(f"🔍 Processing context {i+1}/{len(contexts)} with timestamp {obs_timestamp}")
        
        try:
            quintuples_all_data = await lg_kg_construction.extract_information_as_json_for_context(
                output_data_structure=RelationshipsExtractor,
                contexts=[context],  # Process single context
                system_query=Prompt.temporal_system_query(obs_timestamp=obs_timestamp) + Prompt.EXAMPLES.value,
            )
            
            # Handle cases where no relationships are extracted
            if not quintuples_all_data:
                logger.info(f"✅ Context {i+1}: Extracted 0 quintuples (empty result)")
                batch_results.append([])
                continue
            
            safe_results: list[tuple] = []
            try:
                for relationships_container in quintuples_all_data:
                    if not relationships_container:
                        continue
                    relationships_list = getattr(relationships_container, "relationships", None)
                    if not relationships_list:
                        continue
                    for relationship in relationships_list:
                        if not relationship:
                            continue
                        try:
                            safe_results.append(
                                (
                                    getattr(getattr(relationship, "startNode", None), "name", None),
                                    getattr(relationship, "name", None),
                                    getattr(getattr(relationship, "endNode", None), "name", None),
                                    getattr(relationship, "t_start", None),
                                    getattr(relationship, "t_end", None),
                                )
                            )
                        except Exception:
                            # Skip malformed relationship entries
                            continue
            except Exception:
                # In case the structure is not as expected, return what we safely collected so far
                pass
            
            logger.info(f"✅ Context {i+1}: Extracted {len(safe_results)} quintuples")
            batch_results.append(safe_results)
            
        except Exception as e:
            logger.error(f"❌ Error processing context {i+1}: {str(e)}")
            batch_results.append([])  # Empty result for failed context
    
    total_quintuples = sum(len(result) for result in batch_results)
    logger.info(f"✅ Batch extraction completed: {total_quintuples} quintuples across {len(contexts)} contexts")
    return batch_results

async def extract_raw_quintuples_wrapper(df: pd.DataFrame, quintuples_col_with_postfix: str):
    start_time = time.time()
    try:
        print("🎯 Starting main extraction process...")
        logger.info("Beginning quintuple extraction from NYT COVID data")
        
        # Load checkpoint
        checkpoint = load_checkpoint()
        
        # Determine indices to process
        num_rows = len(df)
        selected_indices = _uniform_indices(num_rows=num_rows, k=SAMPLER_K)
        logger.info(f"📝 Processing {len(selected_indices)} rows out of {num_rows} total")

        # Create batches from selected indices
        batches = []
        for i in range(0, len(selected_indices), BATCH_SIZE):
            batch_indices = selected_indices[i:i + BATCH_SIZE]
            batches.append(batch_indices)
        
        logger.info(f"📦 Created {len(batches)} batches of size {BATCH_SIZE}")

        # Initialize the results column if not exists
        if quintuples_col_with_postfix not in df.columns:
            df[quintuples_col_with_postfix] = None

        # Load existing results from checkpoint
        for idx_str, result in checkpoint.get("results", {}).items():
            idx = int(idx_str)
            if idx < len(df):
                df.at[df.index[idx], quintuples_col_with_postfix] = result

        # Process batches
        for batch_idx, batch_indices in enumerate(batches):
            if batch_idx in checkpoint["completed_batches"]:
                logger.info(f"⏩ Skipping batch {batch_idx + 1}/{len(batches)} (already completed)")
                continue
                
            logger.info(f"🔄 Processing batch {batch_idx + 1}/{len(batches)} ({len(batch_indices)} items)")
            
            # Prepare contexts and timestamps for this batch
            batch_contexts = [df.iloc[i][PARAGRAPHS_COL_NAME] for i in batch_indices]
            batch_timestamps = [df.iloc[i][DATE_COL_NAME] for i in batch_indices]
            
            # Extract quintuples for this batch
            batch_results = await extract_quintuples_batch(batch_contexts, batch_timestamps)
            
            # Store results in dataframe and checkpoint
            for idx, result in zip(batch_indices, batch_results):
                df.at[df.index[idx], quintuples_col_with_postfix] = result
                checkpoint["results"][str(idx)] = result
            
            # Mark batch as completed and save checkpoint
            checkpoint["completed_batches"].append(batch_idx)
            save_checkpoint(checkpoint)
            
            logger.info(f"✅ Batch {batch_idx + 1}/{len(batches)} completed and saved")

        # Compute token count for each row
        df[column_name_quintuples_raw_prompt_tokenc] = [
            lg_kg_construction.count_tokens(f"# Context: {txt}\n\n# Question: {Prompt.temporal_system_query(date.strftime('%Y-%m-%d') + Prompt.EXAMPLES.value)}\n\nAnswer: ")
            for txt, date in zip(df[PARAGRAPHS_COL_NAME], df[DATE_COL_NAME])
        ]

        # Save final results (do not overwrite the whole output dataset, just merge new columns)
        print(f"💾 Saving final results to: {OUTPUT_DATASET_PATH}")
        if Path.exists(OUTPUT_DATASET_PATH):
            df_out_existing = pd.read_pickle(OUTPUT_DATASET_PATH)
            df_out_existing[quintuples_col_with_postfix] = df[quintuples_col_with_postfix]
            df_out_existing[column_name_quintuples_raw_prompt_tokenc] = df[column_name_quintuples_raw_prompt_tokenc]
            df_out_existing.to_pickle(OUTPUT_DATASET_PATH)
        else:
            df.to_pickle(OUTPUT_DATASET_PATH)
        
        # Clean up checkpoint file
        if CHECKPOINT_FILE.exists():
            CHECKPOINT_FILE.unlink()
            logger.info("🧹 Checkpoint file cleaned up")
        
        elapsed_time = time.time() - start_time
        logger.info(f"🎉 Processing completed successfully in {elapsed_time:.2f} seconds!")
        print(f"🎉 Quintuple extraction completed successfully in {elapsed_time:.2f} seconds!")
        
    except Exception as e:
        elapsed_time = time.time() - start_time
        logger.error(f"❌ Error occurred after {elapsed_time:.2f} seconds: {str(e)}")
        print(f"❌ Error occurred: {str(e)}")
        print("💡 Progress has been saved. Re-run the script to resume from where it left off.")
        raise
    gc.collect()


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Extract factoids from raw news paragraphs - Factoids Analysis')
    parser.add_argument('--model-postfix', '-p', type=str, required=True,
                       help='The postfix representing the backend and model you are executing the test. You can define all supported postfixes inside your .env file, through $EVAL_MODEL_POSTFIXES_LIST variable')
    return parser.parse_args()

async def main():

    # Parse command line arguments
    args = parse_arguments()

    if not args.model_postfix:
        logger.error('--model-postfix arg not provided')
        return
    if args.model_postfix not in eval_model_postfixes_list:
        logger.error(f'Unsupported --model-postfix arg. Supported ones are: {eval_model_postfixes_list}')
        return
    
    quintuples_col_with_postfix = f"{QUINTUPLES_COL_NAME}_{args.model_postfix}"
    await extract_raw_quintuples_wrapper(df_nyt, quintuples_col_with_postfix)
    

if __name__ == "__main__":
    print("=" * 50)
    print("  QUINTUPLES EXTRACTION FROM NYT COVID DATA")
    print("=" * 50)
    asyncio.run(main())
