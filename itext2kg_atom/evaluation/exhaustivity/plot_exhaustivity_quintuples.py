"""
Quintuple Exhaustivity Plot Generation

This script generates plots showing the exhaustivity (recall) of quintuple extraction,
analyzing how well different LLM models maintain extraction quality as context size increases.

Usage:
    python plot_exhaustivity_quintuples.py

Output:
    - PNG and PDF plots showing quintuple extraction exhaustivity
    - Analysis of extraction quality across different models
    - JSON file with the results
"""

import ast
import asyncio
import json
import logging
import time
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import dateparser
from datetime import datetime
from sklearn.metrics.pairwise import cosine_similarity
import sys
from pathlib import Path

from models.models import get_default_model, get_default_embedding_model
from env_config import (
    column_name_quintuples_ground_truth, column_name_quintuples_extracted, column_name_quintuples_prompt_tokenc,
    eval_output_dataset_path, eval_output_results_path, eval_model_postfixes_list, eval_model_postfixes_to_plot_list,
    similarity_threshold_eval_quintuple
)

# Add the project root to Python path (same pattern as exhaustivity_evaluation_nyt.py)
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

print("🚀 Starting exhaustivity plot generation script...")
logger.info("Setting up configuration and API connections...")

# ============================================================================
# GLOBAL CONFIGURATION VARIABLES
# ============================================================================

# Models to evaluate (all available models - will be filtered for publication quality in plotting)
MODEL_NAMES = eval_model_postfixes_list

# Data configuration
DATA_PATH = project_root / eval_output_dataset_path
PREDICTED_COL_TEMPLATE = f"{column_name_quintuples_extracted}_{{}}"
GOLD_COL = column_name_quintuples_ground_truth
TOKEN_COL = column_name_quintuples_prompt_tokenc

# Analysis parameters
SIMILARITY_THRESHOLD = similarity_threshold_eval_quintuple
MAX_SAMPLES = None  # Set to None for all samples, or integer for limit

# Output configuration
Path(project_root / eval_output_results_path).mkdir(parents=True, exist_ok=True)
OUTPUT_JSON = project_root / eval_output_results_path / "exhaustivity_quintuples_results.json"
OUTPUT_PLOT_PNG = project_root / eval_output_results_path / "exhaustivity_quintuples_plot_publication.png"
OUTPUT_PLOT_PDF = project_root / eval_output_results_path / "exhaustivity_quintuples_plot_publication.pdf"

# Publication-quality plot settings
FIGURE_WIDTH = 4.8  # inches (wider to accommodate right-side legend)
FIGURE_HEIGHT = 2.8  # inches (maintain good aspect ratio)
DPI = 300

# Models for publication plot (all available models)
PUBLICATION_MODELS = eval_model_postfixes_to_plot_list

# Publication color palette (colorblind-friendly)
COLORS = {
    'llamacpp_gemma4': '#1f77b4',    # Blue
    'ollama_gemma4': '#ff7f0e',     # Orange   
    # 'mistral': '#2ca02c',   # Green
    # 'o3mini': '#d62728',    # Red
    # 'gpt41': '#9467bd'      # Purple
}

# Precise model names for legend display
MODEL_DISPLAY_NAMES = {
    'llamacpp_gemma4': 'llama.cpp-gemma4-e4b',
    'ollama_gemma4': 'ollama-gemma4-e4b',
}

# Font sizes for publication (increased as requested)
FONT_SIZES = {
    'axis_labels': 13,      # Increased from 11 to 13
    'tick_labels': 11,      # Increased from 10 to 11  
    'legend': 8,            # Compact legend for right-side placement
    'title': 14
}


# ============================================================================
# CORE FUNCTIONS
# ============================================================================

def normalize_quintuples_list(value):
    """
    Normalizes a pandas cell value into a list of quintuples (5-tuples).
    Handles raw lists/tuples, stringified representations, and missing data.
    """
    #parsed = value

    # Convert string representation to Python objects
    if isinstance(value, str):
        value = value.strip()
        if not value or value in ('[]', '()'):
            return []
        try:
            value = ast.literal_eval(value)
        except Exception:
            return []
    
    # If it's a NumPy array, convert it to a standard Python list
    if type(value).__name__ == 'ndarray':
        value = value.tolist()

    if isinstance(value, (list, tuple)):
        if len(value) == 0:
            return []

        # Edge Case: Single quintuple passed directly
        if len(value) == 5 and isinstance(value[0], str):
            return [tuple(value)]

        # Standard Case: Extract valid 5-tuples
        normalized = []
        for item in value:
            if isinstance(item, (list, tuple)) and len(item) == 5:
                normalized.append(tuple(item))
        return normalized

    if pd.isna(value):
        return []
    # Catch-all fallback for unexpected types
    return []

async def find_matches_quintuples_optimized(quintuples, gold_quintuples, lg_kg_construction, threshold=0.7):
    """
    Optimized function to find matches between quintuples using embeddings and temporal analysis.
    
    Args:
        quintuples: List of predicted quintuples (head, relation, tail, t_start, t_end)
        gold_quintuples: List of gold standard quintuples
        lg_kg_construction: Language model construction object for embeddings
        threshold: Similarity threshold for matching
        
    Returns:
        Dict with recall and recall_t metrics
    """
    logger.debug(f"Finding matches for {len(quintuples)} predicted vs {len(gold_quintuples)} gold quintuples")
    
    if not quintuples or not gold_quintuples:
        logger.warning("Empty quintuples or gold_quintuples provided")
        return {'recall': 0.0, 'recall_t': 0.0}
    
    # Format quintuples as text for embedding
    quintuple_texts = [f"{q[0]} {q[1]} {q[2]}" for q in quintuples]
    gold_quintuple_texts = [f"{gq[0]} {gq[1]} {gq[2]}" for gq in gold_quintuples]
    
    # Calculate embeddings in batches
    quintuple_embeddings = await lg_kg_construction.calculate_embeddings(text=quintuple_texts)
    gold_quintuple_embeddings = await lg_kg_construction.calculate_embeddings(text=gold_quintuple_texts)
    
    # Convert to numpy arrays and ensure proper shape
    quintuple_embeddings = np.array(quintuple_embeddings)
    gold_quintuple_embeddings = np.array(gold_quintuple_embeddings)
    
    if quintuple_embeddings.ndim == 1:
        quintuple_embeddings = quintuple_embeddings.reshape(1, -1)
    if gold_quintuple_embeddings.ndim == 1:
        gold_quintuple_embeddings = gold_quintuple_embeddings.reshape(1, -1)
    
    # Compute similarity matrix
    similarity_matrix = cosine_similarity(quintuple_embeddings, gold_quintuple_embeddings)
    
    # Find matches and analyze temporal information
    # Use sets to track unique gold quintuples that are matched
    matched_gold_indices = set()
    temporal_matched_gold_indices = set()
    
    def is_empty_temporal(value):
        """Check if temporal value is empty"""
        return value is None or value == '' or str(value).lower() == 'none'
    
    def temporal_similar(pred_val, gold_val):
        """Check if temporal values are similar using dateparser"""
        if is_empty_temporal(pred_val) and is_empty_temporal(gold_val):
            return True
        if is_empty_temporal(pred_val) or is_empty_temporal(gold_val):
            return False
        
        try:
            pred_date = dateparser.parse(str(pred_val).strip())
            gold_date = dateparser.parse(str(gold_val).strip())
            
            if pred_date is not None and gold_date is not None:
                return pred_date.date() == gold_date.date()
            
            return str(pred_val).strip().lower() == str(gold_val).strip().lower()
        except (ValueError, TypeError, AttributeError):
            return str(pred_val).strip().lower() == str(gold_val).strip().lower()
    
    # Process each predicted quintuple
    for i, quintuple in enumerate(quintuples):
        similarities = similarity_matrix[i]
        max_similarity_idx = np.argmax(similarities)
        max_similarity = similarities[max_similarity_idx]
        
        if max_similarity > threshold:
            # Track which gold quintuple was matched (avoid double-counting)
            matched_gold_indices.add(max_similarity_idx)
            matched_gold = gold_quintuples[max_similarity_idx]
            
            # Check temporal similarity
            pred_t_start = quintuple[3] if len(quintuple) > 3 else None
            pred_t_end = quintuple[4] if len(quintuple) > 4 else None
            gold_t_start = matched_gold[3] if len(matched_gold) > 3 else None
            gold_t_end = matched_gold[4] if len(matched_gold) > 4 else None
            
            if (temporal_similar(pred_t_start, gold_t_start) and 
                temporal_similar(pred_t_end, gold_t_end)):
                temporal_matched_gold_indices.add(max_similarity_idx)
    
    # Calculate recall metrics based on unique gold quintuples matched
    total_gold = len(gold_quintuples)
    unique_gold_matches = len(matched_gold_indices)
    unique_temporal_gold_matches = len(temporal_matched_gold_indices)
    
    recall = unique_gold_matches / total_gold if total_gold > 0 else 0.0
    recall_t = unique_temporal_gold_matches / total_gold if total_gold > 0 else 0.0
    
    return {'recall': recall, 'recall_t': recall_t}


async def evaluate_models_by_token_count(df, model_names, lg_kg_construction, threshold=0.7, max_samples=None):
    """
    Evaluate multiple models and return results by token count.
    
    Args:
        df: DataFrame containing the data
        model_names: List of model names to evaluate
        lg_kg_construction: Language model construction object
        threshold: Similarity threshold
        max_samples: Maximum number of samples to process per model
        
    Returns:
        Dictionary with results for each model
    """
    print(f"🚀 Evaluating {len(model_names)} models: {model_names}")
    logger.info(f"Starting evaluation for {len(model_names)} models with threshold {threshold}")
    
    results = {}
    
    for model_name in model_names:
        print(f"📊 Processing model: {model_name.upper()}")
        logger.info(f"Processing model: {model_name}")
        
        predicted_col = PREDICTED_COL_TEMPLATE.format(model_name)
        
        # Check if columns exist
        if predicted_col not in df.columns:
            print(f"⚠️  Column {predicted_col} not found. Skipping {model_name}")
            logger.warning(f"Column {predicted_col} not found in dataframe. Skipping {model_name}")
            continue
            
        # Filter valid rows
        valid_indices = (df[predicted_col].notna() & 
                        df[GOLD_COL].notna() & 
                        df[TOKEN_COL].notna())
        valid_df = df[valid_indices].copy()
        
        if max_samples:
            valid_df = valid_df.head(max_samples)
        
        if len(valid_df) == 0:
            print(f"⚠️  No valid data for {model_name}")
            logger.warning(f"No valid data found for model {model_name}")
            continue
        
        logger.info(f"Processing {len(valid_df)} valid samples for {model_name}")
        model_results = []
        
        # Process each row
        for row_idx, idx in enumerate(valid_df.index):
            if row_idx % 10 == 0:  # Log progress every 10 rows
                logger.debug(f"Processing row {row_idx + 1}/{len(valid_df)} for {model_name}")
            quintuples = normalize_quintuples_list(valid_df[predicted_col].loc[idx])
            gold_quintuples = normalize_quintuples_list(valid_df[GOLD_COL].loc[idx])
            token_count = valid_df[TOKEN_COL].loc[idx]
            
            if not quintuples or not gold_quintuples:
                continue
                
            # Calculate recall metrics
            result = await find_matches_quintuples_optimized(
                quintuples=quintuples,
                gold_quintuples=gold_quintuples,
                lg_kg_construction=lg_kg_construction,
                threshold=threshold
            )
            
            model_results.append({
                'token_count': int(token_count),
                'recall': float(result['recall']),
                'recall_t': float(result['recall_t']),
                'row_idx': int(idx)
            })
        
        results[model_name] = model_results
        print(f"   ✅ Processed {len(model_results)} samples")
        logger.info(f"Completed processing {model_name}: {len(model_results)} samples")
    
    logger.info(f"Evaluation completed for all models. Total results: {sum(len(v) for v in results.values())} samples")
    return results

## For bucket visualization: commented as it left blank bars on the plot
# def create_publication_exhaustivity_plot(results, model_names=None):
#     """
#     Create a publication-quality bar plot showing semantic and temporal recall by token count.
    
#     Args:
#         results: Dictionary with results for each model
#         model_names: List of model names (defaults to PUBLICATION_MODELS for cleaner plot)
        
#     Returns:
#         matplotlib figure and axes objects
#     """
#     # Use publication models for cleaner plot if not specified
#     if model_names is None:
#         model_names = PUBLICATION_MODELS
    
#     logger.info(f"Creating publication-quality exhaustivity plot for models: {model_names}")
    
#     # Set matplotlib parameters for publication quality
#     plt.rcParams.update({
#         'font.size': FONT_SIZES['tick_labels'],
#         'axes.labelsize': FONT_SIZES['axis_labels'],
#         'axes.titlesize': FONT_SIZES['title'],
#         'legend.fontsize': FONT_SIZES['legend'],
#         'xtick.labelsize': FONT_SIZES['tick_labels'],
#         'ytick.labelsize': FONT_SIZES['tick_labels'],
#         'font.family': 'serif',
#         'font.serif': ['Times New Roman'],
#         'text.usetex': False,  # Set to True if LaTeX is available
#         'figure.dpi': DPI,
#         'savefig.dpi': DPI,
#         'axes.linewidth': 0.8,
#         'grid.linewidth': 0.5,
#         'lines.linewidth': 1.0
#     })
    
#     # Prepare data
#     plot_data = []
#     for model_name, model_results in results.items():
#         for result in model_results:
#             plot_data.append({
#                 'model': model_name.lower(),
#                 'token_count': result['token_count'],
#                 'recall': result['recall'],
#                 'recall_t': result['recall_t']
#             })
    
#     if not plot_data:
#         logger.error("No data to plot!")
#         return None, None
    
#     df_plot = pd.DataFrame(plot_data)
#     logger.info(f"Prepared plot data with {len(df_plot)} data points")
    
#     min_tc = df_plot['token_count'].min()
#     max_tc = df_plot['token_count'].max()
#     num_unique = df_plot['token_count'].nunique()

#     # Helper function for pretty human-readable numbers (e.g., 1500 -> 1.5k)
#     def format_val(val):
#         if val >= 1000:
#             return f"{val/1000:.1f}k"
#         return f"{int(round(val))}"
    
#     if num_unique <= 1 or min_tc == max_tc:
#         # Fallback if there's only 1 data point or all token counts are completely identical
#         bin_label = format_val(min_tc) if not pd.isna(min_tc) else "0"
#         df_plot['token_bins'] = bin_label
#         unique_bins = [bin_label]
#     else:
#         # Determine appropriate number of bins (max 10 bins, or fewer if there are very few unique records)
#         num_bins = min(10, num_unique)
        
#         # Calculate exactly spaced edges to wrap our specific data range cleanly
#         edges = np.linspace(min_tc, max_tc, num_bins + 1)
#         edges[0] -= 1   # Extend lower boundary slightly to guarantee inclusion of min_tc
#         edges[-1] += 1  # Extend upper boundary slightly to guarantee inclusion of max_tc
        
#         # Build clean interval labels dynamically matching the exact spans
#         labels = []
#         step = (max_tc - min_tc) / num_bins
#         use_exact = step < 100  # Fallback to exact integers if bin range is narrow

#         for i in range(num_bins):
#             start = min_tc + i * step
#             end = min_tc + (i + 1) * step
#             s_str = str(int(round(start))) if use_exact else format_val(start)
#             e_str = str(int(round(end))) if use_exact else format_val(end)
#             lbl = f"{s_str}-{e_str}" if s_str != e_str else s_str
#             labels.append(lbl)

#         df_plot['token_bins'] = pd.cut(df_plot['token_count'], bins=edges, labels=labels, ordered=False)
#         unique_bins = list(dict.fromkeys(labels))

#     # Group by the adaptive bins instead of individual integers
#     grouped = df_plot.groupby(['token_bins', 'model'], observed=False).agg({
#         'recall': 'mean',
#         'recall_t': 'mean'
#     }).reset_index()

#     n_models = len([m for m in model_names if m in results])
    
#     # Create figure with publication dimensions
#     fig, ax = plt.subplots(figsize=(FIGURE_WIDTH, FIGURE_HEIGHT), dpi=DPI)
    
#     # Set up bar positions
#     x = np.arange(len(unique_bins))
#     width = 0.12  # Slightly reduced bar width to prevent touching
    
#     # Plot bars for each model
#     legend_elements = []
#     for i, model_name in enumerate(model_names):
#         if model_name not in results:
#             continue
            
#         model_data = grouped[grouped['model'] == model_name]
        
#         recalls = []
#         recalls_t = []
        
#         for bin_name in unique_bins:
#             token_data = model_data[model_data['token_bins'] == bin_name]
#             if len(token_data) > 0:
#                 recalls.append(token_data['recall'].iloc[0])
#                 recalls_t.append(token_data['recall_t'].iloc[0])
#             else:
#                 recalls.append(0)
#                 recalls_t.append(0)
        
#         # Calculate bar positions
#         x_pos = x + (i - (n_models-1)/2) * width
        
#         # Get color for this model
#         color = COLORS.get(model_name, '#666666')
        
#         # Semantic recall bars (full bars with subtle background)
#         bars_semantic = ax.bar(x_pos, recalls, width, 
#                               color=color, alpha=0.85, 
#                               edgecolor='black', linewidth=0.6,
#                               zorder=2)
        
#         # Temporal recall bars (overlaid with pattern and slight color variation)
#         temporal_color = color  # Same base color
#         bars_temporal = ax.bar(x_pos, recalls_t, width,
#                               color=temporal_color, alpha=0.65,
#                               edgecolor='black', linewidth=0.6,
#                               hatch='///', zorder=3)
        
#         # Add to legend with precise model names
#         display_name = MODEL_DISPLAY_NAMES.get(model_name, model_name.upper())
#         legend_elements.append((bars_semantic[0], f'{display_name} - Semantic'))
#         legend_elements.append((bars_temporal[0], f'{display_name} - Temporal'))
    
#     # Customize plot for publication
#     ax.set_xlabel('Token count as context', fontsize=FONT_SIZES['axis_labels'], fontweight='bold')
#     ax.set_ylabel('Exhaustivity', fontsize=FONT_SIZES['axis_labels'], fontweight='bold')
#     ax.set_title('Exhaustivity of quintuples', fontsize=FONT_SIZES['title'], fontweight='bold', pad=20)
    
#     # Set y-axis range [0, 0.6] as requested
#     ax.set_ylim(0, 0.6)
    
#     # Add horizontal gridlines at specified intervals (improved visibility)
#     gridlines = [0.1, 0.2, 0.3, 0.4, 0.5]
#     for gridline in gridlines:
#         ax.axhline(y=gridline, color='gray', linestyle='-', alpha=0.4, linewidth=0.6, zorder=1)
    
#     # Add minor tick marks on y-axis
#     ax.set_yticks([0.05, 0.15, 0.25, 0.35, 0.45, 0.55], minor=True)
#     ax.tick_params(axis='y', which='minor', length=3, width=0.5)
    
#     # Set x-axis with improved readability
#     ax.set_xticks(x)
    
#     ax.set_xticklabels(unique_bins, 
#                        rotation=45, ha='right', fontsize=FONT_SIZES['tick_labels'])
    
#     # Extend x-axis limits to use full plot width
#     ax.set_xlim(-0.5, len(unique_bins) - 0.5)
    
#     # Create custom legend with single-column layout for right-side placement
#     handles = []
#     labels = []
#     for handle, label in legend_elements:
#         handles.append(handle)
#         labels.append(label)
    
#     # Place legend outside plot area on the right side with single-column layout
#     ax.legend(handles, labels, bbox_to_anchor=(1.02, 1), loc='upper left', 
#               fontsize=FONT_SIZES['legend'], frameon=True, fancybox=False, shadow=False,
#               ncol=1, handletextpad=0.5, handlelength=1.2,
#               framealpha=0.9, edgecolor='black', facecolor='white')
    
#     # Increase axis border line width for better print quality
#     for spine in ax.spines.values():
#         spine.set_linewidth(1.2)
    
#     # Remove top and right spines for cleaner look
#     ax.spines['top'].set_visible(False)
#     ax.spines['right'].set_visible(False)
    
#     # Set grid below data
#     ax.set_axisbelow(True)
    
#     # Improve tick parameters for better print quality
#     ax.tick_params(axis='both', which='major', width=1.0, length=4)
#     ax.tick_params(axis='x', which='major', pad=8)  # Add padding for rotated labels
    
#     # Adjust layout to accommodate right-side legend and rotated x-axis labels
#     plt.tight_layout()
#     plt.subplots_adjust(bottom=0.15, right=0.75)  # Extra space for rotated labels and right legend
    
#     # Save both PNG and PDF formats
#     logger.info("Saving plot in multiple formats")
#     plt.savefig(str(OUTPUT_PLOT_PNG), dpi=DPI, bbox_inches='tight', 
#                 facecolor='white', edgecolor='none', pad_inches=0.1)
#     plt.savefig(str(OUTPUT_PLOT_PDF), dpi=DPI, bbox_inches='tight', 
#                 facecolor='white', edgecolor='none', format='pdf', pad_inches=0.1)
    
#     print("📊 Publication plot saved to:")
#     print(f"   PNG: {OUTPUT_PLOT_PNG}")
#     print(f"   PDF: {OUTPUT_PLOT_PDF}")
#     logger.info(f"Plot saved to {OUTPUT_PLOT_PNG} and {OUTPUT_PLOT_PDF}")
    
#     return fig, ax



def create_publication_exhaustivity_plot(results, model_names=None):
    """
    Create a publication-quality bar plot showing semantic and temporal recall by token count.
    
    Args:
        results: Dictionary with results for each model
        model_names: List of model names (defaults to PUBLICATION_MODELS for cleaner plot)
        
    Returns:
        matplotlib figure and axes objects
    """
    # Use publication models for cleaner plot if not specified
    if model_names is None:
        model_names = PUBLICATION_MODELS
    
    logger.info(f"Creating publication-quality exhaustivity plot for models: {model_names}")
    
    # Set matplotlib parameters for publication quality
    plt.rcParams.update({
        'font.size': FONT_SIZES['tick_labels'],
        'axes.labelsize': FONT_SIZES['axis_labels'],
        'axes.titlesize': FONT_SIZES['title'],
        'legend.fontsize': FONT_SIZES['legend'],
        'xtick.labelsize': FONT_SIZES['tick_labels'],
        'ytick.labelsize': FONT_SIZES['tick_labels'],
        'font.family': 'serif',
        'font.serif': ['Times New Roman'],
        'text.usetex': False,  # Set to True if LaTeX is available
        'figure.dpi': DPI,
        'savefig.dpi': DPI,
        'axes.linewidth': 0.8,
        'grid.linewidth': 0.5,
        'lines.linewidth': 1.0
    })
    
    # Prepare data
    plot_data = []
    for model_name, model_results in results.items():
        for result in model_results:
            plot_data.append({
                'model': model_name.lower(),
                'token_count': result['token_count'],
                'recall': result['recall'],
                'recall_t': result['recall_t']
            })
    
    if not plot_data:
        logger.error("No data to plot!")
        return None, None
    
    df_plot = pd.DataFrame(plot_data)
    logger.info(f"Prepared plot data with {len(df_plot)} data points")
    
    min_tc = df_plot['token_count'].min()
    max_tc = df_plot['token_count'].max()
    num_unique = df_plot['token_count'].nunique()

    # Helper function for pretty human-readable numbers (e.g., 1500 -> 1.5k)
    def format_val(val):
        if val >= 1000:
            return f"{val/1000:.1f}k"
        return f"{int(round(val))}"
    
    if num_unique <= 1 or min_tc == max_tc:
        bin_label = format_val(min_tc) if not pd.isna(min_tc) else "0"
        df_plot['token_bins'] = bin_label
        unique_bins = [bin_label]
    else:
        num_bins = min(10, num_unique)
        edges = np.linspace(min_tc, max_tc, num_bins + 1)
        edges[0] -= 1   
        edges[-1] += 1  
        
        labels = []
        for i in range(num_bins):
            start = min_tc + i * (max_tc - min_tc) / num_bins
            end = min_tc + (i + 1) * (max_tc - min_tc) / num_bins
            s_str, e_str = format_val(start), format_val(end)
            labels.append(f"{s_str}-{e_str}" if s_str != e_str else s_str)
            
        df_plot['token_bins'] = pd.cut(df_plot['token_count'], bins=edges, labels=labels, ordered=False)
        unique_bins = list(dict.fromkeys(labels))

    # Group by the adaptive bins instead of individual integers
    grouped = df_plot.groupby(['token_bins', 'model'], observed=True).agg({
        'recall': 'mean',
        'recall_t': 'mean'
    }).reset_index()

    # 2. Add immediately after, to prune unique_bins to match actual data
    bins_in_data = set(grouped['token_bins'].astype(str))
    unique_bins = [b for b in unique_bins if b in bins_in_data]
    
    n_models = len([m for m in model_names if m in results])
    
    # Create figure with publication dimensions
    fig, ax = plt.subplots(figsize=(FIGURE_WIDTH, FIGURE_HEIGHT), dpi=DPI)
    
    # Set up bar positions
    x = np.arange(len(unique_bins))
    width = 0.12
    
    # Plot bars for each model
    legend_elements = []
    for i, model_name in enumerate(model_names):
        if model_name not in results:
            continue
            
        model_data = grouped[grouped['model'] == model_name.lower()]
        
        recalls = []
        recalls_t = []
        
        for bin_name in unique_bins:
            token_data = model_data[model_data['token_bins'] == bin_name]
            if len(token_data) > 0:
                recalls.append(token_data['recall'].iloc[0])
                recalls_t.append(token_data['recall_t'].iloc[0])
            else:
                recalls.append(0)
                recalls_t.append(0)
        
        # Calculate bar positions
        x_pos = x + (i - (n_models-1)/2) * width
        
        # Get color for this model
        color = COLORS.get(model_name, '#666666')
        
        # Semantic recall bars (full bars with subtle background)
        bars_semantic = ax.bar(x_pos, recalls, width, 
                              color=color, alpha=0.85, 
                              edgecolor='black', linewidth=0.6,
                              zorder=2)
        
        # Temporal recall bars (overlaid with pattern and slight color variation)
        temporal_color = color  # Same base color
        bars_temporal = ax.bar(x_pos, recalls_t, width,
                              color=temporal_color, alpha=0.65,
                              edgecolor='black', linewidth=0.6,
                              hatch='///', zorder=3)
        
        # Add to legend with precise model names
        display_name = MODEL_DISPLAY_NAMES.get(model_name, model_name.upper())
        legend_elements.append((bars_semantic[0], f'{display_name} - Semantic'))
        legend_elements.append((bars_temporal[0], f'{display_name} - Temporal'))
    
    # Customize plot for publication
    ax.set_xlabel('Token count as context', fontsize=FONT_SIZES['axis_labels'], fontweight='bold')
    ax.set_ylabel('Exhaustivity', fontsize=FONT_SIZES['axis_labels'], fontweight='bold')
    ax.set_title('Exhaustivity of quintuples', fontsize=FONT_SIZES['title'], fontweight='bold', pad=20)
    
    # Set y-axis range [0, 1.0]
    ax.set_ylim(0, 1.0)
    
    # Add horizontal gridlines at specified intervals (improved visibility)
    gridlines = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    for gridline in gridlines:
        ax.axhline(y=gridline, color='gray', linestyle='-', alpha=0.4, linewidth=0.6, zorder=1)
    
    # Add minor tick marks on y-axis
    ax.set_yticks([0.05, 0.15, 0.25, 0.35, 0.45, 0.55, 0.65, 0.75, 0.85, 0.95], minor=True)
    ax.tick_params(axis='y', which='minor', length=3, width=0.5)
    
    # Set x-axis with improved readability
    ax.set_xticks(x)
    
    # Format x-axis labels with scientific notation for readability
    # def format_token_count(tc):
    #     if tc >= 10000:
    #         return f'{tc/1000:.1f}k'
    #     elif tc >= 1000:
    #         return f'{tc/1000:.1f}k'
    #     else:
    #         return f'{int(tc)}'
    
    ax.set_xticklabels(unique_bins, 
                       rotation=45, ha='right', fontsize=FONT_SIZES['tick_labels'])
    
    # Extend x-axis limits to use full plot width 
    ax.set_xlim(-0.5, len(unique_bins) - 0.5)
    
    # Create custom legend with single-column layout for right-side placement
    handles = []
    labels = []
    for handle, label in legend_elements:
        handles.append(handle)
        labels.append(label)
    
    # Place legend outside plot area on the right side with single-column layout
    ax.legend(handles, labels, bbox_to_anchor=(1.02, 1), loc='upper left', 
              fontsize=FONT_SIZES['legend'], frameon=True, fancybox=False, shadow=False,
              ncol=1, handletextpad=0.5, handlelength=1.2,
              framealpha=0.9, edgecolor='black', facecolor='white')
    
    # Increase axis border line width for better print quality
    for spine in ax.spines.values():
        spine.set_linewidth(1.2)
    
    # Remove top and right spines for cleaner look
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    # Set grid below data
    ax.set_axisbelow(True)
    
    # Improve tick parameters for better print quality
    ax.tick_params(axis='both', which='major', width=1.0, length=4)
    ax.tick_params(axis='x', which='major', pad=8)  # Add padding for rotated labels
    
    # Adjust layout to accommodate right-side legend and rotated x-axis labels
    plt.tight_layout()
    plt.subplots_adjust(bottom=0.15, right=0.75)  # Extra space for rotated labels and right legend
    
    # Save both PNG and PDF formats
    logger.info("Saving plot in multiple formats")
    plt.savefig(str(OUTPUT_PLOT_PNG), dpi=DPI, bbox_inches='tight', 
                facecolor='white', edgecolor='none', pad_inches=0.1)
    plt.savefig(str(OUTPUT_PLOT_PDF), dpi=DPI, bbox_inches='tight', 
                facecolor='white', edgecolor='none', format='pdf', pad_inches=0.1)
    
    print("📊 Publication plot saved to:")
    print(f"   PNG: {OUTPUT_PLOT_PNG}")
    print(f"   PDF: {OUTPUT_PLOT_PDF}")
    logger.info(f"Plot saved to {OUTPUT_PLOT_PNG} and {OUTPUT_PLOT_PDF}")
    
    # Log the improvements made
    logger.info("Quintuples plot improvements applied:")
    logger.info("  ✅ X-axis labels rotated 45° with scientific notation (k format)")
    logger.info("  ✅ X-axis limits extended to full plot width")
    #logger.info("  ✅ All 5 models included in legend (claude, gpt4o, mistral, o3mini, gpt41)")
    logger.info("  ✅ Horizontal gridlines added at 0.1-0.5 intervals")
    logger.info("  ✅ Font sizes optimized (axis: 13pt, ticks: 11pt, legend: 8pt)")
    logger.info("  ✅ Single-column legend positioned outside plot area (right side)")
    logger.info("  ✅ Figure width increased to 4.8\" to accommodate right legend")
    logger.info("  ✅ Minor tick marks and improved axis borders")
    logger.info("  ✅ Reduced bar width and improved spacing")
    
    return fig, ax


def load_existing_results(json_path):
    """
    Load existing results from JSON file if it exists.
    
    Args:
        json_path: Path to JSON results file
        
    Returns:
        Dictionary with results or None if file doesn't exist/invalid
    """
    try:
        if not Path(json_path).exists():
            logger.info(f"No existing results file found at {json_path}")
            return None
            
        with open(str(json_path), 'r') as f:
            data = json.load(f)
        
        # Validate the structure
        if 'results' not in data or 'metadata' not in data:
            logger.warning(f"Invalid JSON structure in {json_path}")
            return None
            
        # Check if metadata matches current configuration
        metadata = data['metadata']
        current_models = set(MODEL_NAMES)
        existing_models = set(metadata.get('model_names', []))
        
        if metadata.get('similarity_threshold') != SIMILARITY_THRESHOLD:
            logger.warning(f"Threshold mismatch: existing={metadata.get('similarity_threshold')}, current={SIMILARITY_THRESHOLD}")
            return None
            
        if not current_models.issubset(existing_models):
            missing_models = current_models - existing_models
            logger.warning(f"Missing models in existing results: {missing_models}")
            return None
            
        # Filter results to only include current models
        filtered_results = {model: data['results'][model] for model in MODEL_NAMES if model in data['results']}
        
        logger.info(f"✅ Loaded existing results from {json_path}")
        logger.info(f"   Models: {list(filtered_results.keys())}")
        logger.info(f"   Total samples: {sum(len(v) for v in filtered_results.values())}")
        logger.info(f"   Timestamp: {metadata.get('timestamp', 'unknown')}")
        
        return filtered_results
        
    except Exception as e:
        logger.error(f"Error loading existing results from {json_path}: {e}")
        return None


def save_results_to_json(results, output_path):
    """
    Save results to JSON file for later analysis.
    
    Args:
        results: Dictionary with results
        output_path: Path to save JSON file
    """
    logger.info(f"Saving results to JSON file: {output_path}")
    
    # Add metadata
    output_data = {
        'metadata': {
            'timestamp': datetime.now().isoformat(),
            'model_names': MODEL_NAMES,
            'similarity_threshold': SIMILARITY_THRESHOLD,
            'data_path': str(DATA_PATH),
            'total_samples': sum(len(v) for v in results.values())
        },
        'results': results
    }
    
    try:
        with open(str(output_path), 'w') as f:
            json.dump(output_data, f, indent=2)
        
        print(f"💾 Results saved to: {output_path}")
        logger.info(f"Successfully saved {len(results)} model results to {output_path}")
    except Exception as e:
        logger.error(f"Failed to save results to {output_path}: {e}")
        raise


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Generate exhaustivity plots for ATOM models')
    parser.add_argument('--force-recalculate', '-f', action='store_true',
                       help='Force recalculation even if existing results are found')
    parser.add_argument('--max-samples', '-m', type=int, default=None,
                       help='Maximum number of samples to process per model (for testing)')
    return parser.parse_args()


async def main():
    """
    Main function to run the exhaustivity analysis.
    """
    start_time = time.time()
    
    # Parse command line arguments
    args = parse_arguments()
    
    print("🎯 Starting Exhaustivity Analysis")
    print("=" * 50)
    logger.info("Beginning exhaustivity plot generation analysis")
    
    if args.force_recalculate:
        print("🔄 Force recalculation mode enabled")
        logger.info("Force recalculation mode enabled - will skip existing results")
    
    if args.max_samples:
        global MAX_SAMPLES
        MAX_SAMPLES = args.max_samples
        print(f"🎯 Limited to {MAX_SAMPLES} samples per model (testing mode)")
        logger.info(f"Testing mode: limited to {MAX_SAMPLES} samples per model")
    
    try:
        # First, try to load existing results (unless force recalculation is enabled)
        results = None
        if not args.force_recalculate:
            print("🔍 Checking for existing results...")
            logger.info(f"Looking for existing results in {OUTPUT_JSON}")
            results = load_existing_results(OUTPUT_JSON)
        
        if results is not None:
            print("   ✅ Found existing results! Using cached data.")
            print(f"   📊 Loaded {len(results)} models with {sum(len(v) for v in results.values())} total samples")
            logger.info("Using existing results, skipping evaluation")
        else:
            print("   ⚠️  No existing results found or force recalculation enabled. Running full evaluation...")
            logger.info("No existing results found or force recalculation enabled, proceeding with full evaluation")
            
            # Import ATOM modules (these need to be imported after sys.path modification)
            try:
                from itext2kg.llm_output_parsing.langchain_output_parser import LangchainOutputParser
                from langchain_openai import ChatOpenAI, OpenAIEmbeddings
                print("   ✅ ATOM modules imported successfully")
                logger.info("ATOM modules imported successfully")
            except ImportError as e:
                print(f"❌ Error importing ATOM modules: {e}")
                print(f"Current working directory: {Path.cwd()}")
                print(f"Project root: {project_root}")
                logger.error(f"Failed to import ATOM modules: {e}")
                print("Make sure you're running this from the correct directory")
                return
            
            # Load data
            print(f"📁 Loading data from: {DATA_PATH}")
            logger.info(f"Loading dataset from {DATA_PATH}")
            try:
                df = pd.read_pickle(DATA_PATH)
                print(f"   ✅ Loaded {len(df)} samples")
                logger.info(f"Successfully loaded dataset with {len(df)} samples")
                
                # Log dataset info
                logger.info(f"Dataset columns: {list(df.columns)}")
                logger.info(f"Available models in dataset: {[col.replace('cumul_quintuples_', '') for col in df.columns if col.startswith('cumul_quintuples_') and col != GOLD_COL]}")
            except Exception as e:
                print(f"❌ Error loading data: {e}")
                logger.error(f"Failed to load dataset: {e}")
                return
            
            # Initialize language model components (same pattern as exhaustivity_evaluation_nyt.py)
            print("🤖 Initializing language model components...")
            logger.info("Initializing language model components")
            try:
                lg_kg_construction = LangchainOutputParser(
                    llm_model=get_default_model(),
                    embeddings_model=get_default_embedding_model()
                )
                print("   ✅ Language model components initialized")
                logger.info("Language model components initialized successfully")
            except Exception as e:
                print(f"❌ Error initializing language model: {e}")
                logger.error(f"Failed to initialize language model: {e}")
                return
            
            # Run evaluation
            print("🔍 Running evaluation...")
            logger.info(f"Starting evaluation with threshold {SIMILARITY_THRESHOLD}")
            if MAX_SAMPLES:
                logger.info(f"Limited to {MAX_SAMPLES} samples per model")
            
            results = await evaluate_models_by_token_count(
                df=df,
                model_names=MODEL_NAMES,
                lg_kg_construction=lg_kg_construction,
                threshold=SIMILARITY_THRESHOLD,
                max_samples=MAX_SAMPLES
            )
            
            # Save results to JSON
            save_results_to_json(results, OUTPUT_JSON)
        
        # Create and save publication-quality plot (using top models for cleaner visualization)
        print("📊 Creating publication-quality visualization...")
        logger.info("Creating publication-quality plot")
        fig, ax = create_publication_exhaustivity_plot(results)  # Uses PUBLICATION_MODELS by default
        
        if fig is not None:
            print("   ✅ Plot created successfully")
            logger.info("Plot created and saved successfully")
        else:
            print("   ⚠️  No plot generated")
            logger.warning("No plot was generated")
        
        elapsed_time = time.time() - start_time
        print("\n✨ Analysis complete!")
        print(f"📊 Results saved to: {OUTPUT_JSON}")
        print("🖼️  Publication plots saved to:")
        print(f"   PNG: {OUTPUT_PLOT_PNG}")
        print(f"   PDF: {OUTPUT_PLOT_PDF}")
        print(f"📈 Plot includes models: {PUBLICATION_MODELS}")
        print(f"⏱️  Total time: {elapsed_time:.2f} seconds")
        logger.info(f"Analysis completed successfully in {elapsed_time:.2f} seconds")
        
    except Exception as e:
        elapsed_time = time.time() - start_time
        logger.error(f"Error occurred after {elapsed_time:.2f} seconds: {str(e)}")
        print(f"❌ Error occurred: {str(e)}")
        print("💡 Check the logs for more details.")
        raise


if __name__ == "__main__":
    print("=" * 50)
    print("  EXHAUSTIVITY PLOT GENERATION FOR NYT COVID DATA")
    print("=" * 50)
    asyncio.run(main())
