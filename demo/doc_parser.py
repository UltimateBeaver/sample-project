"""
Demo script for converting raw news paragraphs into atomic facts (factoids)

This script:
1. Reads an Excel file with columns: date (yyyy-mm-dd), lead_paragraph
2. Extracts atomic facts from each paragraph following:
   - Atomicity: Each fact contains exactly one piece of information
   - Decontextualization: Pronouns replaced with full entity names
   - Temporal normalization: Relative time references converted to absolute dates
   - End actions: Termination of roles/actions captured with timestamps
3. Saves the results to a new column: factoids_g_truth

Usage:
    parser = DocumentParser(llm_model)
    df_with_factoids = await parser.parse_excel(input_excel_path, output_excel_path)

## Architecture Note: System Query vs AtomicFact Schema

The `_create_temporal_system_query()` method and the `AtomicFact.atomic_fact` field description in 
schemas.py serve complementary purposes and intentionally have overlapping content:

1. **schemas.py (AtomicFact.atomic_fact description)**: 
   - Purpose: Defines the Pydantic model structure for JSON schema validation
   - Audience: Language model responding to `extract_information_as_json_for_context()`
   - Scope: Stable reference guidelines for atomic fact extraction rules
   - Usage: Part of the output_data_structure parameter

2. **_create_temporal_system_query() in doc_parser.py**:
   - Purpose: Runtime system prompt tailored to the specific document and observation date
   - Audience: Language model receiving the actual extraction request
   - Scope: Expanded with exhaustiveness emphasis, specific date examples, and detailed guidance
   - Usage: Passed as system_query parameter to override/supplement schema description
   
The system_query takes precedence at runtime and is more comprehensive because:
- It includes dynamic date conversion examples based on the observation_date parameter
- It emphasizes exhaustive extraction to improve coverage (addresses Issue #1)
- It provides better deduplication guidance (addresses Issue #4)
- It includes explicit examples of what NOT to extract (irrelevant minutiae)

This design allows the schema to remain stable documentation while the system query 
remains flexible and optimized for extraction quality at runtime.
"""

import pandas as pd
import asyncio
import logging
import re
from pathlib import Path
from typing import List, Optional, Tuple
from datetime import datetime, timedelta
from difflib import SequenceMatcher

from itext2kg_atom.itext2kg.llm_output_parsing.langchain_output_parser import LangchainOutputParser
from itext2kg_atom.itext2kg.atom.models import AtomicFact


# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


# ============================================================================
# HELPER FUNCTIONS FOR DATE NORMALIZATION AND DUPLICATE REMOVAL
# ============================================================================

def convert_day_name_to_date(day_name: str, observation_date: str) -> Optional[str]:
    """
    Convert day names (Monday, Tuesday, etc., yesterday, today) to absolute dates.
    
    Args:
        day_name: Day name or relative reference (e.g., "Monday", "yesterday", "thursday")
        observation_date: Reference date in YYYY-MM-DD format
        
    Returns:
        Absolute date in YYYY-MM-DD format, or None if not convertible
    """
    day_name_lower = day_name.lower().strip()
    obs_date = datetime.strptime(observation_date, '%Y-%m-%d')
    
    # Map day names to weekday numbers (0=Monday, 6=Sunday)
    day_map = {
        'monday': 0, 'tuesday': 1, 'wednesday': 2, 'thursday': 3,
        'friday': 4, 'saturday': 5, 'sunday': 6
    }
    
    # Handle special relative dates
    if day_name_lower == 'today':
        return observation_date
    elif day_name_lower == 'yesterday':
        return (obs_date - timedelta(days=1)).strftime('%Y-%m-%d')
    elif day_name_lower == 'tomorrow':
        return (obs_date + timedelta(days=1)).strftime('%Y-%m-%d')
    
    # Handle day of week
    if day_name_lower in day_map:
        target_day = day_map[day_name_lower]
        current_day = obs_date.weekday()
        
        # Calculate days to target day (looking backward to most recent occurrence)
        days_back = (current_day - target_day) % 7
        if days_back == 0 and obs_date.weekday() != target_day:
            days_back = 7
        
        target_date = obs_date - timedelta(days=days_back)
        return target_date.strftime('%Y-%m-%d')
    
    return None


def normalize_relative_dates_in_fact(fact: str, observation_date: str) -> str:
    """
    Replace relative date references in a fact with absolute dates.
    
    Args:
        fact: The atomic fact string
        observation_date: Reference date in YYYY-MM-DD format
        
    Returns:
        Fact with normalized dates
    """
    obs_date = datetime.strptime(observation_date, '%Y-%m-%d')
    normalized_fact = fact
    
    # Pattern 1: "on Monday", "on Thursday", etc.
    day_pattern = r'\b(on\s+)?(?:Monday|Tuesday|Wednesday|Thursday|Friday|Saturday|Sunday|yesterday|today|tomorrow)\b'
    matches = list(re.finditer(day_pattern, normalized_fact, re.IGNORECASE))
    
    # Process matches in reverse order to maintain string positions
    for match in reversed(matches):
        day_str = match.group(0).lower()
        day_str_clean = day_str.replace('on ', '').strip()
        
        converted_date = convert_day_name_to_date(day_str_clean, observation_date)
        if converted_date:
            replacement = converted_date
            if 'on ' in day_str.lower():
                replacement = f'on {converted_date}'
            normalized_fact = normalized_fact[:match.start()] + replacement + normalized_fact[match.end():]
    
    # Pattern 2: "last month", "this month", "last week", "this week", etc.
    time_patterns = {
        r'\blast\s+month\b': lambda d: d.replace(day=1, month=(d.month - 1 or 12), year=(d.year - (1 if d.month == 1 else 0))).strftime('%Y-%m-%d'),
        r'\bthis\s+month\b': lambda d: d.replace(day=1).strftime('%Y-%m-%d'),
        r'\blast\s+week\b': lambda d: (d - timedelta(days=(d.weekday() + 7))).strftime('%Y-%m-%d'),
        r'\bthis\s+week\b': lambda d: (d - timedelta(days=d.weekday())).strftime('%Y-%m-%d'),
        r'\blast\s+year\b': lambda d: d.replace(year=d.year - 1, month=1, day=1).strftime('%Y-%m-%d'),
        r'\bthis\s+year\b': lambda d: d.replace(month=1, day=1).strftime('%Y-%m-%d'),
    }
    
    for pattern, converter in time_patterns.items():
        matches = list(re.finditer(pattern, normalized_fact, re.IGNORECASE))
        for match in reversed(matches):
            try:
                converted_date = converter(obs_date)
                normalized_fact = normalized_fact[:match.start()] + converted_date + normalized_fact[match.end():]
            except Exception as e:
                logger.warning(f"Could not convert date pattern {match.group(0)}: {e}")
    
    return normalized_fact


def remove_duplicate_facts(facts: List[str], similarity_threshold: float = 0.8) -> List[str]:
    """
    Remove near-duplicate facts using sequence matching.
    Keeps the most informative fact (longest, or with most entity names).
    
    Args:
        facts: List of extracted facts
        similarity_threshold: Threshold for considering facts as duplicates (0-1)
        
    Returns:
        List with duplicates removed
    """
    if len(facts) <= 1:
        return facts
    
    unique_facts = []
    
    for fact in facts:
        is_duplicate = False
        
        for existing_fact in unique_facts:
            # Calculate similarity ratio
            similarity = SequenceMatcher(None, fact.lower(), existing_fact.lower()).ratio()
            
            if similarity >= similarity_threshold:
                # Keep the more informative fact (longer or more specific)
                if len(fact) > len(existing_fact):
                    unique_facts.remove(existing_fact)
                    unique_facts.append(fact)
                is_duplicate = True
                break
        
        if not is_duplicate:
            unique_facts.append(fact)
    
    return unique_facts


def filter_irrelevant_facts(facts: List[str], paragraph: str) -> List[str]:
    """
    Filter out facts that are too focused on minor descriptive details.
    Keeps facts about main entities, events, and relationships.
    
    Args:
        facts: List of extracted facts
        paragraph: Original paragraph for context
        
    Returns:
        Filtered list of relevant facts
    """
    filtered = []
    
    # Patterns indicating irrelevant descriptive details
    irrelevant_patterns = [
        r'^\s*The\s+\w+\s+is\s+\w+\s+with\b',  # "The X is Y with..."
        r'^\s*The\s+\w+\s+is\s+characterized\s+by\b',  # "The X is characterized by..."
        r'^\s*The\s+\w+\s+is\s+echoing\b',  # "The X is echoing..."
        r'^\s*The\s+\w+\s+is\s+located\s+in\s+\w+\s*,\s*\w+\s*\.\s*$',  # Generic location facts
    ]
    
    for fact in facts:
        is_irrelevant = False
        
        # Check against irrelevant patterns
        for pattern in irrelevant_patterns:
            if re.search(pattern, fact, re.IGNORECASE):
                is_irrelevant = True
                break
        
        # Avoid very minor descriptive details (very short facts about objects/places)
        if len(fact) < 20 and re.search(r'^\s*The\s+\w+\s+', fact) and 'is' in fact.lower():
            # Only filter if it's a very generic statement
            if not any(keyword in fact.lower() for keyword in ['company', 'organization', 'person', 'found', 'declared', 'announced', 'reported']):
                is_irrelevant = True
        
        if not is_irrelevant:
            filtered.append(fact)
    
    return filtered


# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class DocumentParser:
    """
    Parser class to convert raw news paragraphs into atomic facts using LLM.
    Follows atomicity, decontextualization, temporal normalization, and end actions rules.
    """
    
    def __init__(self, llm_model, embeddings_model=None):
        """
        Initialize the DocumentParser with LLM and optional embeddings model.
        
        Args:
            llm_model: The language model instance (ChatOllama, ChatOpenAI, etc.)
            embeddings_model: Optional embeddings model for semantic operations
        """
        self.llm_model = llm_model
        self.embeddings_model = embeddings_model
        self.parser = LangchainOutputParser(
            llm_model=llm_model,
            embeddings_model=embeddings_model
        )
        logger.info("DocumentParser initialized successfully")
    
    @staticmethod
    def _create_temporal_system_query(observation_date: str) -> str:
        """
        Create a comprehensive system query for exhaustive atomic facts extraction.
        Uses the AtomicFact schema description as the foundation and adds:
        - Emphasis on EXHAUSTIVE extraction (all facts, including supporting details)
        - Better date conversion examples with explicit mappings
        - Examples of what NOT to extract (irrelevant descriptive minutiae)
        - Instructions to prevent duplicate/near-duplicate facts
        
        Note: This query is designed for the LLM at runtime. The AtomicFact schema 
        description serves a complementary role for JSON structure definition.
        
        Args:
            observation_date: The observation date in format YYYY-MM-DD
            
        Returns:
            System query string with comprehensive temporal context
        """
        obs_date = datetime.strptime(observation_date, '%Y-%m-%d')
        
        # Calculate example dates for the specific observation date
        yesterday = (obs_date - timedelta(days=1)).strftime('%Y-%m-%d')
        week_monday = (obs_date - timedelta(days=obs_date.weekday())).strftime('%Y-%m-%d')
        last_week_monday = (obs_date - timedelta(days=obs_date.weekday() + 7)).strftime('%Y-%m-%d')
        month_first = obs_date.replace(day=1).strftime('%Y-%m-%d')
        year_first = obs_date.replace(month=1, day=1).strftime('%Y-%m-%d')
        last_month_first = (obs_date.replace(day=1) - timedelta(days=1)).replace(day=1).strftime('%Y-%m-%d')
        
        return f"""
You are an expert EXHAUSTIVE atomic facts extraction engine. Your PRIMARY goal is to extract ALL distinct factual information from the input paragraph, including:
- Main events and actions
- Supporting context and background
- Relationships between entities
- Impact and consequences
- Temporal information

**Observation Date for Reference**: {observation_date}

## CRITICAL: Be EXHAUSTIVE - Extract ALL Facts

DO NOT filter facts based on perceived "importance". Extract:
✅ Main events (e.g., "X announced Y")
✅ Supporting facts (e.g., "The announcement occurred because of Z")
✅ Background context (e.g., "This is the first time X has done Y")
✅ Entity descriptions and roles (e.g., "X is a company", "Y is located in Z")
✅ Causal relationships (e.g., "A happened because of B")
✅ Temporal context (e.g., "This follows event C from last week")

❌ DO NOT limit extraction—get EVERYTHING mentioned in the text.

## Key Rules:

### 1. ATOMICITY
- Each atomic fact must contain exactly ONE piece of information or relationship
- Break down compound/complex sentences into single-fact statements
- Remove redundancies ONLY (exact duplicates), not supporting details
- Example: "Company X announced a new product and hired 50 people" → Two facts:
  - "Company X announced a new product"
  - "Company X hired 50 people"

### 2. DECONTEXTUALIZATION  
- Replace ALL pronouns (he, she, it, they, this, that) with full entity names
- Include necessary modifiers so each fact is understandable in isolation
- Example: "John joined the company. He started on Monday" → "John joined the company on Monday"

### 3. TEMPORAL NORMALIZATION - CRITICAL
Convert ALL relative time references to absolute dates using observation_date = {observation_date}:

REFERENCE EXAMPLES FOR {observation_date}:
- "today" → {observation_date}
- "yesterday" → {yesterday}
- "this week" → {week_monday} (Monday of this week)
- "last week" → {last_week_monday} (Monday of last week)
- "this month" → {month_first} (1st of this month)
- "last month" → {last_month_first} (1st of last month)
- "this year" → {year_first} (Jan 1st of this year)
- "last year" → {(obs_date.replace(year=obs_date.year-1, month=1, day=1)).strftime('%Y-%m-%d')} (Jan 1st of last year)
- Named days (Monday, Thursday, etc.) → the most recent occurrence of that day
- "last month" → date exactly 1 month prior
- Keep explicit dates as-is (e.g., "June 18, 2024" stays "June 18, 2024")

NEVER include relative terms like "today", "yesterday", "last week" in FINAL facts.
Position time references naturally within facts (usually near the verb/action).

### 4. END ACTIONS
- If the text indicates the end of a role or action (e.g., "leaving a position", "resigned", "ended"), be explicit
- Capture: WHAT ended, WHO/WHAT ended it, WHEN it ended
- Example: "The CEO resigned yesterday" → "The CEO resigned on {yesterday}"

### 5. AVOIDING IRRELEVANT MINUTIAE
✅ INCLUDE: Facts about companies, people, events, policies, impacts
❌ EXCLUDE: Purely descriptive stylistic details (e.g., "The machines are whirring loudly")

If a description is about a specific named entity and provides information (e.g., "Kolmi Hopen makes face masks"), INCLUDE it.
If a description is generic and non-informative (e.g., "The factory floor has machines"), EXCLUDE it.

### 6. DEDUPLICATION GUIDANCE
- Prefer facts mentioning SPECIFIC NAMES/ENTITIES over generic versions
- GOOD: "Technology company ABC announced a new research facility in Silicon Valley on March 15, 2024"
- AVOID DUPLICATE: "The company expanded its operations in California" (already covered by the GOOD fact)
- Extract the more comprehensive fact with entity names rather than generic versions

## Example of EXHAUSTIVE Extraction

**Example 1: Company Announcement (observation_date = 2020-05-15):**
"TOKYO — Tech company XYZ announced a major breakthrough in renewable energy yesterday. The innovation could reduce power consumption by 40%. Industry experts believe this represents a significant step forward for sustainable technology."

**Output (ALL facts):**
- Tech company XYZ announced a major breakthrough in renewable energy on 2020-05-14
- The renewable energy breakthrough could reduce power consumption by 40%
- Industry experts believe the announcement represents a significant step forward for sustainable technology

**Example 2: Regulatory Change (observation_date = 2020-03-20):**
"BRUSSELS — The European Union implemented new data protection regulations on Thursday. Companies have 60 days to comply with the rules. Financial penalties for non-compliance could reach up to 10 million euros."

**Output:**
- The European Union implemented new data protection regulations on 2020-03-19
- Companies have 60 days to comply with the new regulations (deadline: around 2020-05-18)
- Financial penalties for non-compliance with the regulations could reach up to 10 million euros
- The regulations were implemented by the European Union

**Example 3: Natural Disaster Impact (observation_date = 2020-08-10):**
"MANILA — A typhoon struck the region last week, causing flooding in three provinces. The disaster displaced thousands of residents. Emergency services continue relief operations."

**Output:**
- A typhoon struck the region around 2020-08-03 (last week from 2020-08-10)
- The typhoon caused flooding in three provinces
- Thousands of residents were displaced by the typhoon
- Emergency services continue relief operations following the typhoon

## Output Format
Return ONLY the list of atomic facts. Each fact should be:
- Concise and factual
- Temporally explicit (include normalized dates where relevant)
- Decontextualized (no pronouns)
- Unique (but keep different perspectives on the same event)
- Related to substantive content (not trivial descriptive details)

REMEMBER: Your goal is EXHAUSTIVE extraction. Extract EVERY distinct piece of information you find in the text.
"""
    
    async def extract_atomic_facts_from_paragraphs_batch(
        self,
        paragraphs: List[str],
        observation_dates: List[str],
        apply_post_processing: bool = True
    ) -> List[List[str]]:
        """
        Extract atomic facts from multiple paragraphs in parallel (batch processing).
        
        Args:
            paragraphs: List of raw news paragraph texts
            observation_dates: List of observation dates in format YYYY-MM-DD (one per paragraph)
            apply_post_processing: Whether to apply post-processing (date normalization, deduplication)
            
        Returns:
            List of fact lists (one per input paragraph)
        """
        if len(paragraphs) != len(observation_dates):
            raise ValueError("Number of paragraphs must match number of observation dates")
        
        if not paragraphs:
            return []
        
        try:
            # Create system queries for each paragraph (one per observation date)
            system_queries = [
                self._create_temporal_system_query(obs_date)
                for obs_date in observation_dates
            ]
            
            # Use LangchainOutputParser batch processing to extract facts from all paragraphs
            # This allows the underlying LLM provider to batch requests efficiently
            results = await self.parser.extract_information_as_json_for_context(
                output_data_structure=AtomicFact,
                contexts=paragraphs,
                system_query=system_queries[0]  # Use first system query (they should be similar for batching)
            )
            
            # Extract facts from each AtomicFact object
            all_facts = []
            for i, (atomic_fact_obj, obs_date) in enumerate(zip(results, observation_dates)):
                if hasattr(atomic_fact_obj, 'atomic_fact'):
                    facts = atomic_fact_obj.atomic_fact
                else:
                    logger.warning(f"Unexpected result structure for paragraph {i}: {atomic_fact_obj}")
                    facts = []
                
                # Apply post-processing
                if apply_post_processing and facts:
                    # Step 1: Normalize dates
                    facts = [normalize_relative_dates_in_fact(fact, obs_date) for fact in facts]
                    
                    # Step 2: Remove duplicates
                    facts = remove_duplicate_facts(facts, similarity_threshold=0.8)
                    
                    # Step 3: Filter irrelevant facts
                    facts = filter_irrelevant_facts(facts, paragraphs[i])
                
                all_facts.append(facts)
            
            return all_facts
            
        except Exception as e:
            logger.error(f"Error extracting atomic facts from batch: {e}")
            return [[] for _ in paragraphs]
    
    async def parse_excel(
        self, 
        input_excel_path: str, 
        output_excel_path: Optional[str] = None,
        batch_size: int = 5,
        apply_post_processing: bool = True
    ) -> pd.DataFrame:
        """
        Read an Excel file, extract atomic facts for each paragraph, and save results.
        Processes paragraphs in parallel batches for improved performance.
        
        Args:
            input_excel_path: Path to input Excel file with columns: date, lead_paragraph
            output_excel_path: Path to save output Excel file. If None, overwrites input file
            batch_size: Number of paragraphs to process in parallel per batch (default: 5)
            apply_post_processing: Whether to apply post-processing (date normalization, deduplication)
            
        Returns:
            DataFrame with added factoids_g_truth column
        """
        # Read the Excel file
        logger.info(f"Reading Excel file: {input_excel_path}")
        df = pd.read_excel(input_excel_path)
        
        # Validate required columns
        if 'date' not in df.columns or 'lead_paragraph' not in df.columns:
            raise ValueError("Excel file must contain 'date' and 'lead_paragraph' columns")
        
        logger.info(f"Loaded {len(df)} rows from Excel file")
        
        # Initialize the factoids_g_truth column
        df['factoids_g_truth'] = None
        factoids_list = [[] for _ in range(len(df))]
        
        # Convert dates to strings and prepare data
        dates = []
        paragraphs = []
        row_indices = []
        
        for idx, row in df.iterrows():
            date = row['date']
            paragraph = row['lead_paragraph']
            
            # Convert date to string if needed
            if isinstance(date, datetime):
                date_str = date.strftime('%Y-%m-%d')
            else:
                date_str = str(date)
            
            dates.append(date_str)
            paragraphs.append(paragraph)
            row_indices.append(idx)
        
        # Process paragraphs in parallel batches
        total_rows = len(df)
        num_batches = (total_rows + batch_size - 1) // batch_size
        
        logger.info(f"Processing {total_rows} rows in {num_batches} batches (batch size: {batch_size})")
        
        # Create batch tasks
        batch_tasks = []
        for batch_idx in range(num_batches):
            start_idx = batch_idx * batch_size
            end_idx = min(start_idx + batch_size, total_rows)
            
            batch_paragraphs = paragraphs[start_idx:end_idx]
            batch_dates = dates[start_idx:end_idx]
            batch_row_indices = row_indices[start_idx:end_idx]
            
            logger.info(f"Preparing batch {batch_idx + 1}/{num_batches} (rows {start_idx}-{end_idx-1})")
            
            # Create a task for this batch
            task = self._process_batch(
                batch_paragraphs,
                batch_dates,
                batch_row_indices,
                batch_idx + 1,
                num_batches,
                apply_post_processing
            )
            batch_tasks.append(task)
        
        # Execute all batch tasks in parallel
        logger.info("Starting parallel batch processing...")
        batch_results = await asyncio.gather(*batch_tasks, return_exceptions=True)
        
        # Collect results
        for batch_result in batch_results:
            if isinstance(batch_result, Exception):
                logger.error(f"Batch processing failed: {batch_result}")
                continue
            
            for row_idx, facts in batch_result:
                factoids_list[row_idx] = facts
        
        # Add the extracted factoids to the dataframe
        df['factoids_g_truth'] = factoids_list
        
        # Save the output Excel file
        output_path = output_excel_path or input_excel_path
        logger.info(f"Saving results to: {output_path}")
        df.to_excel(output_path, index=False)
        
        logger.info("✅ Processing complete!")
        return df
    
    async def _process_batch(
        self,
        batch_paragraphs: List[str],
        batch_dates: List[str],
        batch_row_indices: List[int],
        batch_num: int,
        total_batches: int,
        apply_post_processing: bool
    ) -> List[Tuple[int, List[str]]]:
        """
        Process a single batch of paragraphs in parallel.
        
        Args:
            batch_paragraphs: List of paragraphs in this batch
            batch_dates: List of observation dates for this batch
            batch_row_indices: List of original row indices
            batch_num: Batch number (for logging)
            total_batches: Total number of batches (for logging)
            apply_post_processing: Whether to apply post-processing
            
        Returns:
            List of tuples (row_index, facts_list)
        """
        try:
            logger.info(f"🔄 Batch {batch_num}/{total_batches}: Processing {len(batch_paragraphs)} paragraphs...")
            
            # Extract facts from all paragraphs in this batch
            batch_facts = await self.extract_atomic_facts_from_paragraphs_batch(
                batch_paragraphs,
                batch_dates,
                apply_post_processing
            )
            
            # Pair results with original row indices
            results = []
            for row_idx, facts in zip(batch_row_indices, batch_facts):
                results.append((row_idx, facts))
                logger.info(f"✅ Batch {batch_num}/{total_batches}: Row {row_idx} → {len(facts)} facts")
            
            return results
            
        except Exception as e:
            logger.error(f"Error processing batch {batch_num}: {e}")
            return [(idx, []) for idx in batch_row_indices]


# ============================================================================
# DEMO USAGE
# ============================================================================

async def demo_parse_documents(
    llm_model, 
    input_excel_path: str, 
    output_excel_path: Optional[str] = None,
    batch_size: int = 5,
    apply_post_processing: bool = True
):
    """
    Convenience function to parse documents using the DocumentParser class with parallel batch processing.
    
    Args:
        llm_model: The language model instance
        input_excel_path: Path to input Excel file
        output_excel_path: Path to output Excel file (optional)
        batch_size: Number of paragraphs to process in parallel per batch (default: 5)
                   Increase for faster processing (with more parallel requests),
                   decrease to reduce memory usage or API rate limit concerns
        apply_post_processing: Whether to apply post-processing (date normalization, deduplication)
        
    Returns:
        DataFrame with extracted atomic facts
    """
    parser = DocumentParser(llm_model=llm_model)
    result_df = await parser.parse_excel(
        input_excel_path, 
        output_excel_path,
        batch_size=batch_size,
        apply_post_processing=apply_post_processing
    )
    return result_df


# Convenience wrapper for simple usage
def create_documents_parser(llm_model):
    """
    Wrapper function that instantiates a new DocumentParser object.
    
    Args:
        llm_model: The language model instance
        
    Returns:
        DocumentParser instance ready to use
    """
    parser = DocumentParser(llm_model)
    return parser