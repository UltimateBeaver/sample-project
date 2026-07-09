"""
Quintuples Extraction from Previously Extracted Factoids

This script converts previously extracted factoids into knowledge graph quintuples
(head_entity, relationship, tail_entity, t_start, t_end) using Large Language Models.
Unlike direct quintuple extraction, this two-step approach first decomposes text into
factoids then structures them into quintuples, potentially improving precision.

Usage:
    python quintuples_extraction_nyt_from_factoids.py

Output:
    - A pickle file containing the dataset with quintuples extracted from factoids
    - Checkpoint files for resuming interrupted processing
"""

import sys
import asyncio
import logging
import time
import ast
from pathlib import Path
import argparse

import pandas as pd
import numpy as np

# Add the project root to Python path
current_file = Path(__file__).resolve()
project_root = current_file.parent.parent.parent
sys.path.append(str(project_root))

from itext2kg.llm_output_parsing.langchain_output_parser import LangchainOutputParser
from itext2kg.atom.models import RelationshipsExtractor, Prompt
from models.models import get_default_model, get_default_embedding_model
from env_config import (
    column_name_quintuples_extracted, column_name_date, column_name_factoids_ground_truth, column_name_factoids_extracted, column_name_quintuples_prompt_tokenc, 
    eval_input_dataset_path, eval_output_dataset_path, eval_model_postfixes_list, num_rows_to_process, num_rows_to_process
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
NUM_ROWS_TO_PROCESS = num_rows_to_process

# Column names
FACTOIDS_COL_NAME: str = column_name_factoids_ground_truth
FACTOIDS_EXTRACTED_COL_NAME: str = column_name_factoids_extracted
DATE_COL_NAME: str = column_name_date
QUINTUPLES_COL_NAME: str = column_name_quintuples_extracted

# Sampling: number of uniformly spaced indices to process. Set to None or 0 to process all
SAMPLER_K: int | None = None

lg_kg_construction = LangchainOutputParser(
   llm_model=get_default_model(),
   embeddings_model=get_default_embedding_model()
)

logger.info("✅ LangchainOutputParser initialized successfully")

print("📊 Loading dataset (only first row, for testing)...")
df_nyt = pd.read_pickle(INPUT_DATASET_PATH)
if num_rows_to_process > 0:
    df_nyt = df_nyt.head(num_rows_to_process)
logger.info(f"📋 Loaded dataset with {len(df_nyt)} rows")

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


async def extract_quintuples(contexts: list[list[str]], timestamps: list[str]) -> list[list[tuple]]:
    logger.info(f"🔍 Starting quintuple extraction for {len(contexts)} contexts...")
    print(f"🔍 Processing {len(contexts)} factoid contexts...")
    
    # Process each context with its corresponding timestamp
    all_results: list[list[tuple]] = []
    
    for i, (context, obs_timestamp) in enumerate(zip(contexts, timestamps)):
        logger.info(f"🔍 Processing context {i+1}/{len(contexts)} with timestamp {obs_timestamp}")
        
        quintuples_all_data = await lg_kg_construction.extract_information_as_json_for_context(
            output_data_structure=RelationshipsExtractor,
            contexts=context,
            system_query=Prompt.temporal_system_query(obs_timestamp=obs_timestamp) + Prompt.EXAMPLES.value,
        )
        
        # Handle cases where no relationships are extracted
        if not quintuples_all_data:
            logger.info(f"✅ Context {i+1}: Extracted 0 quintuples (empty result)")
            all_results.append([])
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
        all_results.append(safe_results)
    
    total_quintuples = sum(len(result) for result in all_results)
    logger.info(f"✅ Total extracted {total_quintuples} quintuples across {len(contexts)} contexts")
    return all_results


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
    
    quintuples_col_with_postfix = f"{QUINTUPLES_COL_NAME}_{args.model_postfix}"
    #factoids_extracted_col_with_postfix = f"{FACTOIDS_EXTRACTED_COL_NAME}_{args.model_postfix}"
    
    try:
        print("🎯 Starting main extraction process...")
        logger.info("Beginning quintuple extraction from NYT COVID data")
        
        # Determine indices to process
        num_rows = len(df_nyt)
        selected_indices = _uniform_indices(num_rows=num_rows, k=SAMPLER_K)
        logger.info(f"📝 Processing {len(selected_indices)} rows out of {num_rows} total")

        # Prepare contexts and timestamps for selected rows only
        context_data = [ast.literal_eval(df_nyt.iloc[i][FACTOIDS_COL_NAME]) for i in selected_indices]
        timestamp_data = [df_nyt.iloc[i][DATE_COL_NAME] for i in selected_indices]

        # Extract quintuples for selected contexts
        extracted = await extract_quintuples(context_data, timestamp_data)

        # Initialize column with empty values, then fill only selected indices
        empty_value = None
        df_nyt[quintuples_col_with_postfix] = empty_value
        for idx, value in zip(selected_indices, extracted):
            df_nyt.at[df_nyt.index[idx], quintuples_col_with_postfix] = value

        # Compute token count for each row
        df_nyt[column_name_quintuples_prompt_tokenc] = [
            lg_kg_construction.count_tokens(f"# Context: {txt}\n\n# Question: {Prompt.temporal_system_query(date.strftime('%Y-%m-%d') + Prompt.EXAMPLES.value)}\n\nAnswer: ")
            for txt, date in zip(df_nyt[FACTOIDS_COL_NAME], df_nyt[DATE_COL_NAME])
        ]
        
        # Save final results (do not overwrite the whole output dataset, just merge new columns)
        print(f"💾 Saving final results to: {OUTPUT_DATASET_PATH}")
        if Path.exists(OUTPUT_DATASET_PATH):
            df_out_existing = pd.read_pickle(OUTPUT_DATASET_PATH)
            df_out_existing[quintuples_col_with_postfix] = df_nyt[quintuples_col_with_postfix]
            df_out_existing[column_name_quintuples_prompt_tokenc] = df_nyt[column_name_quintuples_prompt_tokenc]
            df_out_existing.to_pickle(OUTPUT_DATASET_PATH)
        else:
            df_nyt.to_pickle(OUTPUT_DATASET_PATH)
        
        elapsed_time = time.time() - start_time
        logger.info(f"🎉 Processing completed successfully in {elapsed_time:.2f} seconds!")
        print(f"🎉 Quintuples extraction completed successfully in {elapsed_time:.2f} seconds!")
        
    except Exception as e:
        elapsed_time = time.time() - start_time
        logger.error(f"❌ Error occurred after {elapsed_time:.2f} seconds: {str(e)}")
        print(f"❌ Error occurred: {str(e)}")
        raise

if __name__ == "__main__":
    print("=" * 50)
    print("  QUINTUPLES EXTRACTION FROM NYT COVID DATA")
    print("=" * 50)
    asyncio.run(main())