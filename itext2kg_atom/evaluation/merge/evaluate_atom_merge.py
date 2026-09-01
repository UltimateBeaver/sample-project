"""
Evaluate ATOM Entity and Relation Resolution (Merge) Precision and Recall
This script evaluates the precision and recall of entity resolution (ER) and relation resolution (RR)
by finding similar entities and relations using embeddings.

Precision: Measures how many entities ATOM should merge but didn't (false negatives)
Recall: Measures how well ATOM is merging entities compared to ground truth
"""

import argparse
import ast
import os
from pathlib import Path
import pickle
import sys
import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from typing import List, Dict, Any, Tuple
import asyncio
from models.models import get_default_model, get_default_embedding_model
from env_config import (
    eval_output_dataset_path, eval_input_knowledge_graph_path, eval_cache_path, num_rows_to_process,
    column_name_quintuples_extracted, eval_model_postfixes_list
)

# Add the project root to Python path (same pattern as other scripts)
current_file = Path(__file__).resolve()
project_root = current_file.parent.parent.parent
sys.path.append(str(project_root))

# ============================================================================
# GLOBAL CONFIGURATION
# ============================================================================

# Path to the ATOM knowledge graph pickle file
ATOM_KG_PATH = project_root / eval_input_knowledge_graph_path

# Path to the df_nyt pickle file (contains ground truth data)
DF_NYT_PATH = project_root / eval_output_dataset_path
COL_NAME_QUINTUPLES = "will be overwritten in main"

# Similarity threshold for determining duplicates
THRESHOLD = 0.8

# Cache file for ground truth entity embeddings
ENTITY_EMBEDDINGS_CACHE = f"{project_root}/{eval_cache_path}/cache_atom/entity_embeddings_ground_truth_atom.pkl"
Path(ENTITY_EMBEDDINGS_CACHE).parent.mkdir(parents=True, exist_ok=True)


# ============================================================================
# UTILITY FUNCTIONS
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

def load_atom_kg(path: str) -> Any:
    """Load ATOM knowledge graph from pickle file."""
    print(f"📂 Loading ATOM knowledge graph from: {path}")
    with open(path, 'rb') as f:
        kg = pickle.load(f)
    print(f"   ✅ Loaded KG with {len(kg.entities)} entities and {len(kg.relationships)} relationships")
    return kg


def load_df_nyt(path: str) -> pd.DataFrame:
    """Load NYT dataframe from pickle file."""
    print(f"📂 Loading NYT dataframe from: {path}")
    df = pd.read_pickle(path)
    if num_rows_to_process > 0:
        df = df.head(num_rows_to_process)
    
    # Normalize quintuples list of eah row
    df[COL_NAME_QUINTUPLES] = df[COL_NAME_QUINTUPLES].apply(normalize_quintuples_list)
    print(f"   ✅ Loaded dataframe with {len(df)} rows")
    return df


def load_cached_embeddings(cache_path: str) -> Dict[str, Any]:
    """
    Load cached embeddings if they exist.
    
    Args:
        cache_path: Path to the cache file
    
    Returns:
        Dictionary with cached embeddings or None if cache doesn't exist
    """
    if os.path.exists(cache_path):
        try:
            with open(cache_path, 'rb') as f:
                cache = pickle.load(f)
            print(f"   ✅ Loaded cached embeddings for {len(cache['entity_names'])} entities")
            return cache
        except Exception as e:
            print(f"   ⚠️  Error loading cache: {e}")
            return None
    return None


def save_embeddings_cache(cache_path: str, entity_names: List[str], embeddings: np.ndarray):
    """
    Save embeddings to cache.
    
    Args:
        cache_path: Path to save the cache file
        entity_names: List of entity names
        embeddings: Numpy array of embeddings
    """
    try:
        cache = {
            'entity_names': entity_names,
            'embeddings': embeddings,
            'model': 'text-embedding-3-large'
        }
        with open(cache_path, 'wb') as f:
            pickle.dump(cache, f)
        print(f"   💾 Saved embeddings cache to {cache_path}")
    except Exception as e:
        print(f"   ⚠️  Error saving cache: {e}")


# ============================================================================
# ENTITY RESOLUTION FUNCTIONS
# ============================================================================

def find_similar_nodes_atom(atom_kg: Any, similarity_threshold: float = 0.9) -> List[Dict[str, Any]]:
    """
    Find nodes with cosine similarity greater than the specified threshold for ATOM graph structure.
    
    Args:
        atom_kg: ATOM knowledge graph object with .entities attribute
        similarity_threshold: Minimum cosine similarity score
    
    Returns:
        List of dictionaries containing similar entity pairs with their similarity scores
    """
    print(f"\n🔍 Finding similar entities with threshold > {similarity_threshold}")
    
    # Extract entities
    entities = atom_kg.entities
    print(f"   Found {len(entities)} total entities")
    
    # Filter entities that have embeddings
    entities_with_embeddings = []
    embeddings = []
    
    for entity in entities:
        if hasattr(entity.properties, 'embeddings') and entity.properties.embeddings is not None:
            entities_with_embeddings.append(entity)
            embeddings.append(entity.properties.embeddings)
    
    print(f"   Found {len(entities_with_embeddings)} entities with embeddings")
    
    if len(embeddings) < 2:
        print("   ⚠️  Not enough entities with embeddings found.")
        return []
    
    # Convert to numpy array
    embeddings_matrix = np.array(embeddings)
    print(f"   Embeddings matrix shape: {embeddings_matrix.shape}")
    
    # Calculate cosine similarity matrix
    similarity_matrix = cosine_similarity(embeddings_matrix)
    
    # Find similar pairs
    similar_pairs = []
    n_entities = len(entities_with_embeddings)
    
    for i in range(n_entities):
        for j in range(i + 1, n_entities):  # Only upper triangle to avoid duplicates
            similarity_score = similarity_matrix[i][j]
            
            if similarity_score > similarity_threshold:
                similar_pairs.append({
                    'entity_1_name': entities_with_embeddings[i].name,
                    'entity_1_label': entities_with_embeddings[i].label,
                    'entity_2_name': entities_with_embeddings[j].name,
                    'entity_2_label': entities_with_embeddings[j].label,
                    'similarity_score': float(similarity_score)
                })
    
    # Sort by similarity score (descending)
    similar_pairs.sort(key=lambda x: x['similarity_score'], reverse=True)
    
    print(f"   ✅ Found {len(similar_pairs)} pairs with similarity > {similarity_threshold}")
    
    return similar_pairs

def calculate_number_of_entities(df_nyt):
    all_entities = [relation[0] for relation in df_nyt[COL_NAME_QUINTUPLES].cumsum().iloc[-1]] + [relation[2] for relation in df_nyt[COL_NAME_QUINTUPLES].cumsum().iloc[-1]]
    #non_duplicated_entities = list(set(all_entities))
    return len(all_entities)

def calculate_number_of_entities_(df_nyt: pd.DataFrame) -> int:
    """Calculate the number of unique entities in the ground truth."""
    all_entities = []
    
    # Collect all entities from quintuples_g_truth
    for quintuples in df_nyt[COL_NAME_QUINTUPLES]:
        if isinstance(quintuples, list):
            for relation in quintuples:
                if len(relation) >= 3:
                    all_entities.append(relation[0])  # head entity
                    all_entities.append(relation[2])  # tail entity
    
    # Remove duplicates (case-insensitive)
    non_duplicated_entities = list(set([entity.lower() for entity in all_entities if entity]))
    
    return len(non_duplicated_entities)


def calculate_number_of_entities_atom(atom_kg: Any) -> int:
    """Calculate the number of entities in ATOM graph."""
    return len(atom_kg.entities)


async def number_ground_truth_merged_entities(
    df_nyt: pd.DataFrame, 
    embeddings_model,
    threshold: float = 0.9,
    cache_path: str = None
) -> int:
    """
    Calculate the number of entities that should be merged in ground truth based on embeddings similarity.
    Uses caching to avoid re-embedding on subsequent runs.
    
    This function:
    1. Gets all entities from ground truth (with duplicates)
    2. Creates unique set of entities (case-insensitive)
    3. Embeds the unique entities (with caching)
    4. Finds pairs with cosine similarity above threshold
    5. Returns: total_entities - unique_entities - similar_pairs
    
    Args:
        df_nyt: Ground truth dataframe
        embeddings_model: OpenAI embeddings model
        threshold: Similarity threshold for determining duplicates
        cache_path: Optional path to cache embeddings
    
    Returns:
        Number of entities that should be merged according to ground truth
    """
    print(f"\n🔍 Calculating ground truth merged entities with threshold > {threshold}")
    
    # Step 1: Get all entities (with duplicates)
    all_entities = []
    for quintuples in df_nyt[COL_NAME_QUINTUPLES]:
        if isinstance(quintuples, list):
            for relation in quintuples:
                if len(relation) >= 3:
                    all_entities.append(relation[0])  # head entity
                    all_entities.append(relation[2])  # tail entity
    
    total_entities = len(all_entities)
    print(f"   Total entities (with duplicates): {total_entities}")
    
    # Step 2: Get unique entities (case-insensitive, remove None/empty)
    unique_entities_list = list(set([entity.lower() for entity in all_entities if entity]))
    num_unique = len(unique_entities_list)
    print(f"   Unique entities: {num_unique}")
    
    if num_unique < 2:
        print("   ⚠️  Not enough unique entities found.")
        return 0
    
    # Step 3: Try to load from cache
    embeddings_array = None
    if cache_path:
        cached = load_cached_embeddings(cache_path)
        if cached is not None:
            # Check if the cached entity names match
            if set(cached['entity_names']) == set(unique_entities_list):
                print("   🎯 Cache matches current entities, using cached embeddings")
                embeddings_array = cached['embeddings']
                # Reorder if necessary
                cached_names = cached['entity_names']
                if cached_names != unique_entities_list:
                    name_to_embedding = {name: emb for name, emb in zip(cached_names, embeddings_array)}
                    embeddings_array = np.array([name_to_embedding[name] for name in unique_entities_list])
            else:
                print("   ⚠️  Cache doesn't match current entities, re-embedding...")
    
    # Step 4: Embed unique entities if not cached
    if embeddings_array is None:
        print(f"   🔮 Embedding {num_unique} unique entities...")
        embeddings = await embeddings_model.aembed_documents(unique_entities_list)
        embeddings_array = np.array(embeddings)
        print(f"   ✅ Generated embeddings with shape: {embeddings_array.shape}")
        
        # Save to cache
        if cache_path:
            save_embeddings_cache(cache_path, unique_entities_list, embeddings_array)
    
    # Step 5: Calculate cosine similarity and find pairs above threshold
    similarity_matrix = cosine_similarity(embeddings_array)
    
    similar_pairs_count = 0
    n_entities = len(unique_entities_list)
    similar_nodes_set = set()
    
    for i in range(n_entities):
        for j in range(i + 1, n_entities):
            if similarity_matrix[i][j] > threshold:
                similar_nodes_set.add(i)
                similar_nodes_set.add(j)
    
    print(f"   Nodes involved in similar pairs: {len(similar_nodes_set)}")
    
    # Step 6: Calculate ground truth merged entities
    # Formula: total - unique - similar_pairs
    ground_truth_merged = total_entities - num_unique + (len(similar_nodes_set) // 2)
    
    print(f"   ✅ Ground truth merged entities: {ground_truth_merged}")
    
    return ground_truth_merged


def calculate_ER_precision(atom_kg: Any, df_nyt: pd.DataFrame, threshold: float = 0.9) -> Tuple[float, List[Dict[str, Any]]]:
    """
    Calculate Entity Resolution (ER) precision.
    
    Args:
        atom_kg: ATOM knowledge graph object
        df_nyt: Ground truth dataframe
        threshold: Similarity threshold for finding duplicates
    
    Returns:
        Tuple of (ER precision score, list of similar entity pairs)
    """
    print("\n📊 Calculating Entity Resolution (ER) Precision")
    
    similar_nodes = find_similar_nodes_atom(atom_kg, similarity_threshold=threshold)
    n_entities_atom = calculate_number_of_entities_atom(atom_kg)
    n_entities_ground_truth = calculate_number_of_entities(df_nyt)
    
    print(f"   Ground truth entities: {n_entities_ground_truth}")
    print(f"   ATOM entities: {n_entities_atom}")
    print(f"   Similar entity pairs (potential duplicates): {len(similar_nodes)}")
    
    # Precision formula: 1 - (duplicates / expected_duplicates)
    # where expected_duplicates = ground_truth_count - atom_count
    expected_duplicates = n_entities_ground_truth - n_entities_atom
    
    if expected_duplicates <= 0:
        print("   ⚠️  Warning: ATOM has more entities than ground truth!")
        return 0.0, similar_nodes
    
    er_precision = 1.0 - (len(similar_nodes) / expected_duplicates)
    er_precision = max(0.0, min(1.0, er_precision))  # Clamp between 0 and 1
    
    print(f"   ✅ ER Precision: {er_precision:.4f}")
    
    return er_precision, similar_nodes


async def calculate_ER_recall(
    atom_kg: Any, 
    df_nyt: pd.DataFrame, 
    embeddings_model,
    threshold: float = 0.9,
    cache_path: str = None
) -> Tuple[float, int]:
    """
    Calculate Entity Resolution (ER) recall.
    
    Recall measures how well ATOM is merging entities compared to ground truth.
    Formula: recall = 1 - (len(similar_nodes) / ground_truth_merged)
    
    Args:
        atom_kg: ATOM knowledge graph object
        df_nyt: Ground truth dataframe
        embeddings_model: OpenAI embeddings model
        threshold: Similarity threshold for finding duplicates
        cache_path: Optional path to cache embeddings
    
    Returns:
        Tuple of (ER recall score, ground truth merged count)
    """
    print("\n📊 Calculating Entity Resolution (ER) Recall")
    
    similar_nodes = find_similar_nodes_atom(atom_kg, similarity_threshold=threshold)
    ground_truth_merged = await number_ground_truth_merged_entities(
        df_nyt, 
        embeddings_model, 
        threshold, 
        cache_path
    )
    
    print(f"   Ground truth merged entities: {ground_truth_merged}")
    print(f"   ATOM similar pairs (unresolved): {len(similar_nodes)}")
    
    if ground_truth_merged <= 0:
        print("   ⚠️  Warning: No entities should be merged in ground truth!")
        return 1.0, ground_truth_merged
    
    er_recall = 1.0 - (len(similar_nodes) / ground_truth_merged)
    er_recall = max(0.0, min(1.0, er_recall))  # Clamp between 0 and 1
    
    print(f"   ✅ ER Recall: {er_recall:.4f}")
    
    return er_recall, ground_truth_merged


# ============================================================================
# RELATION RESOLUTION FUNCTIONS
# ============================================================================

def extract_unique_relations_with_embeddings(atom_kg: Any) -> Tuple[List[str], np.ndarray]:
    """
    Extract unique relation names and their corresponding embeddings from ATOM graph.
    IMPORTANT: Deduplicates based on relation name only, keeping first occurrence's embedding.
    
    Args:
        atom_kg: ATOM knowledge graph object
    
    Returns:
        Tuple of (unique_relation_names list, embeddings array)
    """
    print("\n📋 Extracting unique relations with embeddings...")
    
    # Dictionary to store first occurrence of each relation name with its embedding
    relation_dict = {}
    
    for relationship in atom_kg.relationships:
        rel_name = relationship.name
        
        # Only add if we haven't seen this relation name before
        if rel_name not in relation_dict:
            # Check if relationship has embeddings
            if hasattr(relationship.properties, 'embeddings') and relationship.properties.embeddings is not None:
                relation_dict[rel_name] = relationship.properties.embeddings
    
    # Extract unique names and their embeddings
    unique_relation_names = list(relation_dict.keys())
    embeddings_list = [relation_dict[name] for name in unique_relation_names]
    
    print(f"   Found {len(atom_kg.relationships)} total relationships")
    print(f"   Deduplicated to {len(unique_relation_names)} unique relation names")
    print(f"   {len(embeddings_list)} relations have embeddings")
    
    if embeddings_list:
        embeddings_array = np.array(embeddings_list)
        print(f"   Embeddings shape: {embeddings_array.shape}")
        return unique_relation_names, embeddings_array
    else:
        print("   ⚠️  No relations with embeddings found")
        return [], np.array([])


def find_similar_relations_atom(
    atom_kg: Any,
    threshold: float = 0.9
) -> List[Dict[str, Any]]:
    """
    Find similar relations based on cosine similarity of embeddings.
    
    Args:
        atom_kg: ATOM knowledge graph object
        threshold: Minimum cosine similarity score
    
    Returns:
        List of dictionaries containing similar relation pairs with their similarity scores
    """
    print(f"\n🔍 Finding similar relations with threshold > {threshold}")
    
    # Extract unique relation names and embeddings
    relation_names, embeddings = extract_unique_relations_with_embeddings(atom_kg)
    
    if len(relation_names) < 2 or len(embeddings) < 2:
        print("   ⚠️  Not enough relations with embeddings found.")
        return []
    
    # Calculate cosine similarity matrix
    similarity_matrix = cosine_similarity(embeddings)
    
    # Find similar pairs
    similar_pairs = []
    n_relations = len(relation_names)
    
    for i in range(n_relations):
        for j in range(i + 1, n_relations):  # Only upper triangle to avoid duplicates
            similarity_score = similarity_matrix[i][j]
            
            if similarity_score > threshold:
                similar_pairs.append({
                    'relation_1': relation_names[i],
                    'relation_2': relation_names[j],
                    'similarity_score': float(similarity_score)
                })
    
    # Sort by similarity score (descending)
    similar_pairs.sort(key=lambda x: x['similarity_score'], reverse=True)
    
    print(f"   ✅ Found {len(similar_pairs)} pairs with similarity > {threshold}")
    
    return similar_pairs


def calculate_number_of_relations_atom(atom_kg: Any) -> int:
    """Calculate the number of unique relations in ATOM graph."""
    # unique_relation_names = list(set([relationship.name for relationship in atom_kg.relationships]))
    # return len(unique_relation_names)

    # We need to return the complete relationship instances instead of the unique types, otherwise this would produce wrong Precision and Recall metrics!
    return len(atom_kg.relationships)

def calculate_number_of_relations(df_nyt):
    all_relations = [relation[1] for relation in df_nyt[COL_NAME_QUINTUPLES].cumsum().iloc[-1]]
    return len(all_relations)


def calculate_number_of_relations_(df_nyt: pd.DataFrame) -> int:
    """Calculate the number of unique relations in the ground truth."""
    all_relations = []
    
    # Collect all relations from quintuples_g_truth
    for quintuples in df_nyt[COL_NAME_QUINTUPLES]:
        if isinstance(quintuples, list):
            for relation in quintuples:
                if len(relation) >= 3:
                    all_relations.append(relation[1].lower())  # relation type
    
    # Remove duplicates
    non_duplicated_relations = list(set(all_relations))
    
    return len(non_duplicated_relations)


def calculate_RR_precision(
    atom_kg: Any, 
    df_nyt: pd.DataFrame,
    threshold: float = 0.9
) -> Tuple[float, List[Dict[str, Any]]]:
    """
    Calculate Relation Resolution (RR) precision.
    
    Args:
        atom_kg: ATOM knowledge graph object
        df_nyt: Ground truth dataframe
        threshold: Similarity threshold for finding duplicates
    
    Returns:
        Tuple of (RR precision score, list of similar relation pairs)
    """
    print("\n📊 Calculating Relation Resolution (RR) Precision")
    
    similar_relations = find_similar_relations_atom(atom_kg, threshold)
    n_relations_atom = calculate_number_of_relations_atom(atom_kg)
    n_relations_ground_truth = calculate_number_of_relations(df_nyt)
    
    print(f"   Ground truth relations: {n_relations_ground_truth}")
    print(f"   ATOM relations: {n_relations_atom}")
    print(f"   Similar relation pairs (potential duplicates): {len(similar_relations)}")
    
    # Precision formula: 1 - (duplicates / expected_duplicates)
    expected_duplicates = n_relations_ground_truth - n_relations_atom
    
    if expected_duplicates <= 0:
        print("   ⚠️  Warning: ATOM has more or equal relations than ground truth!")
        precision = 1.0 if n_relations_atom == n_relations_ground_truth else 0.0
        return precision, similar_relations
    
    rr_precision = 1.0 - (len(similar_relations) / expected_duplicates)
    rr_precision = max(0.0, min(1.0, rr_precision))  # Clamp between 0 and 1
    
    print(f"   ✅ RR Precision: {rr_precision:.4f}")
    
    return rr_precision, similar_relations


async def number_ground_truth_merged_relations(
    df_nyt: pd.DataFrame, 
    embeddings_model,
    threshold: float = 0.9,
    cache_path: str = None
) -> int:
    """
    Calculate the number of relations that should be merged in ground truth based on embeddings similarity.
    Uses caching to avoid re-embedding on subsequent runs.
    
    This function:
    1. Gets all relations from ground truth (with duplicates)
    2. Creates unique set of relations (case-insensitive)
    3. Embeds the unique relations (with caching)
    4. Finds pairs with cosine similarity above threshold
    5. Returns: total_relations - unique_relations - similar_pairs
    
    Args:
        df_nyt: Ground truth dataframe
        embeddings_model: OpenAI embeddings model
        threshold: Similarity threshold for determining duplicates
        cache_path: Optional path to cache embeddings
    
    Returns:
        Number of relations that should be merged according to ground truth
    """
    print(f"\n🔍 Calculating ground truth merged relations with threshold > {threshold}")
    
    # Step 1: Get all relations (with duplicates)
    all_relations = []
    for quintuples in df_nyt[COL_NAME_QUINTUPLES]:
        if isinstance(quintuples, list):
            for relation in quintuples:
                if len(relation) >= 3:
                    all_relations.append(relation[1])  # relation type
    
    total_relations = len(all_relations)
    print(f"   Total relations (with duplicates): {total_relations}")
    
    # Step 2: Get unique relations (case-insensitive, remove None/empty)
    unique_relations_list = list(set([relation.lower() for relation in all_relations if relation]))
    num_unique = len(unique_relations_list)
    print(f"   Unique relations: {num_unique}")
    
    if num_unique < 2:
        print("   ⚠️  Not enough unique relations found.")
        return 0
    
    # Step 3: Try to load from cache
    cache_path_relations = cache_path.replace('entity_', 'relation_') if cache_path else None
    embeddings_array = None
    if cache_path_relations:
        cached = load_cached_embeddings(cache_path_relations)
        if cached is not None and 'entity_names' in cached:
            # Check if the cached relation names match (using entity_names key for consistency)
            if set(cached['entity_names']) == set(unique_relations_list):
                print("   🎯 Cache matches current relations, using cached embeddings")
                embeddings_array = cached['embeddings']
                # Reorder if necessary
                cached_names = cached['entity_names']
                if cached_names != unique_relations_list:
                    name_to_embedding = {name: emb for name, emb in zip(cached_names, embeddings_array)}
                    embeddings_array = np.array([name_to_embedding[name] for name in unique_relations_list])
            else:
                print("   ⚠️  Cache doesn't match current relations, re-embedding...")
    
    # Step 4: Embed unique relations if not cached
    if embeddings_array is None:
        print(f"   🔮 Embedding {num_unique} unique relations...")
        embeddings = await embeddings_model.aembed_documents(unique_relations_list)
        embeddings_array = np.array(embeddings)
        print(f"   ✅ Generated embeddings with shape: {embeddings_array.shape}")
        
        # Save to cache
        if cache_path_relations:
            save_embeddings_cache(cache_path_relations, unique_relations_list, embeddings_array)
    
    # Step 5: Calculate cosine similarity and find pairs above threshold
    similarity_matrix = cosine_similarity(embeddings_array)
    
    similar_pairs_count = 0
    n_relations = len(unique_relations_list)
    similar_nodes_set = set()
    
    for i in range(n_relations):
        for j in range(i + 1, n_relations):
            if similarity_matrix[i][j] > threshold:
                similar_nodes_set.add(i)
                similar_nodes_set.add(j)
    
    print(f"   Nodes involved in similar pairs: {len(similar_nodes_set)}")
    
    # Step 6: Calculate ground truth merged relations
    # Formula: total - unique - similar_pairs
    ground_truth_merged = total_relations - num_unique + (len(similar_nodes_set) // 2)
    
    print(f"   ✅ Ground truth merged relations: {ground_truth_merged}")
    
    return ground_truth_merged


async def calculate_RR_recall(
    atom_kg: Any, 
    df_nyt: pd.DataFrame, 
    embeddings_model,
    threshold: float = 0.9,
    cache_path: str = None
) -> Tuple[float, int]:
    """
    Calculate Relation Resolution (RR) recall.
    
    Recall measures how well ATOM is merging relations compared to ground truth.
    Formula: recall = 1 - (len(similar_relations) / ground_truth_merged)
    
    Args:
        atom_kg: ATOM knowledge graph object
        df_nyt: Ground truth dataframe
        embeddings_model: OpenAI embeddings model
        threshold: Similarity threshold for finding duplicates
        cache_path: Optional path to cache embeddings
    
    Returns:
        Tuple of (RR recall score, ground truth merged count)
    """
    print("\n📊 Calculating Relation Resolution (RR) Recall")
    
    similar_relations = find_similar_relations_atom(atom_kg, threshold)
    ground_truth_merged = await number_ground_truth_merged_relations(
        df_nyt, 
        embeddings_model, 
        threshold, 
        cache_path
    )
    
    print(f"   Ground truth merged relations: {ground_truth_merged}")
    print(f"   ATOM similar pairs (unresolved): {len(similar_relations)}")
    
    if ground_truth_merged <= 0:
        print("   ⚠️  Warning: No relations should be merged in ground truth!")
        return 1.0, ground_truth_merged
    
    rr_recall = 1.0 - (len(similar_relations) / ground_truth_merged)
    rr_recall = max(0.0, min(1.0, rr_recall))  # Clamp between 0 and 1
    
    print(f"   ✅ RR Recall: {rr_recall:.4f}")
    
    return rr_recall, ground_truth_merged


# ============================================================================
# DISPLAY FUNCTIONS
# ============================================================================

def display_similar_entities_examples(similar_entities: List[Dict[str, Any]], n_examples: int = 20):
    """
    Display examples of similar entity pairs.
    
    Args:
        similar_entities: List of similar entity pairs
        n_examples: Number of examples to display
    """
    if not similar_entities:
        print("   No similar entities found.")
        return
    
    print(f"\n{'='*80}")
    print(f"🔍 UNRESOLVED ENTITIES - Top {min(n_examples, len(similar_entities))} Examples")
    print(f"{'='*80}")
    
    for i, pair in enumerate(similar_entities[:n_examples], 1):
        print(f"\n{i}. Similarity: {pair['similarity_score']:.4f}")
        print(f"   Entity 1: {pair['entity_1_name']} ({pair['entity_1_label']})")
        print(f"   Entity 2: {pair['entity_2_name']} ({pair['entity_2_label']})")
        if i < min(n_examples, len(similar_entities)):
            print("   " + "-" * 70)


def display_similar_relations_examples(similar_relations: List[Dict[str, Any]], n_examples: int = 20):
    """
    Display examples of similar relation pairs.
    
    Args:
        similar_relations: List of similar relation pairs
        n_examples: Number of examples to display
    """
    if not similar_relations:
        print("   No similar relations found.")
        return
    
    print(f"\n{'='*80}")
    print(f"🔍 UNRESOLVED RELATIONS - Top {min(n_examples, len(similar_relations))} Examples")
    print(f"{'='*80}")
    
    for i, pair in enumerate(similar_relations[:n_examples], 1):
        print(f"\n{i}. Similarity: {pair['similarity_score']:.4f}")
        print(f"   Relation 1: {pair['relation_1']}")
        print(f"   Relation 2: {pair['relation_2']}")
        if i < min(n_examples, len(similar_relations)):
            print("   " + "-" * 70)

def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Extract factoids from raw news paragraphs - Factoids Analysis')
    parser.add_argument('--model-postfix', '-p', type=str, required=True,
                       help='The postfix representing the backend and model you are executing the test. You can define all supported postfixes inside your .env file, through $EVAL_MODEL_POSTFIXES_LIST variable')
    return parser.parse_args()

# ============================================================================
# MAIN FUNCTION
# ============================================================================

async def main():
    """Main function to evaluate entity and relation resolution precision and recall."""
    print("=" * 80)
    print("🚀 ATOM Entity & Relation Resolution Evaluation")
    print("=" * 80)

    # Parse command line arguments
    args = parse_arguments()

    if not args.model_postfix:
        print('--model-postfix arg not provided')
        return
    if args.model_postfix not in eval_model_postfixes_list:
        print(f'Unsupported --model-postfix arg. Supported ones are: {eval_model_postfixes_list}')
        return

    globals()['COL_NAME_QUINTUPLES'] = f"{column_name_quintuples_extracted}_{args.model_postfix}"
    
    # Load data
    atom_kg = load_atom_kg(ATOM_KG_PATH)
    df_nyt = load_df_nyt(DF_NYT_PATH)
    embeddings_model = get_default_embedding_model()
    
    print(f"💾 Entity embeddings cache: {ENTITY_EMBEDDINGS_CACHE}")
    if os.path.exists(ENTITY_EMBEDDINGS_CACHE):
        print("   ✅ Cache file exists - will use cached embeddings if they match")
    else:
        print("   📝 Cache file doesn't exist - will create after first run")
    
    # ==========================================
    # ENTITY RESOLUTION (ER) METRICS
    # ==========================================
    print("\n" + "=" * 80)
    print("📊 Calculating Entity Resolution (ER) Metrics")
    
    similar_entities = find_similar_nodes_atom(atom_kg, similarity_threshold=THRESHOLD)
    n_entities_atom = calculate_number_of_entities_atom(atom_kg)
    n_entities_ground_truth = calculate_number_of_entities(df_nyt)
    ground_truth_merged = await number_ground_truth_merged_entities(df_nyt, embeddings_model, THRESHOLD, ENTITY_EMBEDDINGS_CACHE)
    
    # Math Fix: Calculate TP, FP, FN based on merge counts
    actual_merges_er = n_entities_ground_truth - n_entities_atom
    
    tp_er = max(0, min(actual_merges_er, ground_truth_merged) - len(similar_entities))
    fp_er = max(0, actual_merges_er - ground_truth_merged) # Penalty for over-merging
    fn_er = max(0, ground_truth_merged - actual_merges_er) + len(similar_entities) # Penalty for under-merging
    
    er_precision = tp_er / (tp_er + fp_er) if (tp_er + fp_er) > 0 else 0.0
    er_recall = tp_er / (tp_er + fn_er) if (tp_er + fn_er) > 0 else 0.0
    
    # ==========================================
    # RELATION RESOLUTION (RR) METRICS
    # ==========================================
    print("\n" + "=" * 80)
    print("📊 Calculating Relation Resolution (RR) Metrics")
    
    similar_relations = find_similar_relations_atom(atom_kg, THRESHOLD)
    n_relations_atom = calculate_number_of_relations_atom(atom_kg)
    n_relations_ground_truth = calculate_number_of_relations(df_nyt)
    ground_truth_merged_relations = await number_ground_truth_merged_relations(df_nyt, embeddings_model, THRESHOLD, ENTITY_EMBEDDINGS_CACHE)
    
    # Math Fix: Calculate TP, FP, FN based on merge counts
    actual_merges_rr = n_relations_ground_truth - n_relations_atom
    
    tp_rr = max(0, min(actual_merges_rr, ground_truth_merged_relations) - len(similar_relations))
    fp_rr = max(0, actual_merges_rr - ground_truth_merged_relations) # Penalty for over-merging
    fn_rr = max(0, ground_truth_merged_relations - actual_merges_rr) + len(similar_relations) # Penalty for under-merging
    
    rr_precision = tp_rr / (tp_rr + fp_rr) if (tp_rr + fp_rr) > 0 else 0.0
    rr_recall = tp_rr / (tp_rr + fn_rr) if (tp_rr + fn_rr) > 0 else 0.0

    # ==========================================
    # DISPLAY & OUTPUT
    # ==========================================
    display_similar_entities_examples(similar_entities, n_examples=20)
    display_similar_relations_examples(similar_relations, n_examples=20)
    
    print("\n" + "=" * 80)
    print("📋 FINAL RESULTS")
    print("=" * 80)
    print(f"Similarity Threshold: {THRESHOLD}")
    print("\n--- Entity Resolution (ER) ---")
    print(f"ER Precision: {er_precision:.4f} ({er_precision*100:.2f}%)")
    print(f"ER Recall:    {er_recall:.4f} ({er_recall*100:.2f}%)")
    if (er_precision + er_recall) > 0:
        er_f1 = 2 * (er_precision * er_recall) / (er_precision + er_recall)
        print(f"ER F1-Score:  {er_f1:.4f} ({er_f1*100:.2f}%)")
        
    print("\n--- Relation Resolution (RR) ---")
    print(f"RR Precision: {rr_precision:.4f} ({rr_precision*100:.2f}%)")
    print(f"RR Recall:    {rr_recall:.4f} ({rr_recall*100:.2f}%)")
    if (rr_precision + rr_recall) > 0:
        rr_f1 = 2 * (rr_precision * rr_recall) / (rr_precision + rr_recall)
        print(f"RR F1-Score:  {rr_f1:.4f} ({rr_f1*100:.2f}%)")
        
    print("\n--- Diagnostic Details ---")
    print(f"ATOM Entity Merges Performed:   {actual_merges_er} (Expected: {ground_truth_merged})")
    print(f"ATOM Relation Merges Performed: {actual_merges_rr} (Expected: {ground_truth_merged_relations})")
    print(f"False Positives (Over-merging):  Entities = {fp_er}, Relations = {fp_rr}")
    print(f"False Negatives (Under-merging): Entities = {fn_er}, Relations = {fn_rr}")
    print("=" * 80)
    
    return {
        'er_precision': er_precision,
        'er_recall': er_recall,
        'rr_precision': rr_precision,
        'rr_recall': rr_recall,
        'threshold': THRESHOLD
    }


if __name__ == "__main__":
    # Run the async main function
    results = asyncio.run(main())

