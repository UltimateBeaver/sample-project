"""
Demo script for converting raw news paragraphs into atomic facts (factoids)

This script:
1. Reads an Excel file with columns: date (yyyy-mm-dd), lead_paragraph
2. Extracts atomic facts from each paragraph following:
   - Atomicity: Each fact contains exactly one piece of information
   - Decontextualization: Pronouns replaced with full entity names
   - Temporal normalization: Relative time references converted to absolute dates
   - End actions: Termination of roles/actions captured with timestamps
3. Saves the results to a new column: factoids_extracted

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
from itext2kg_atom.itext2kg.logging_config import get_logger
import re
from math import sqrt
from pathlib import Path
from typing import List, Optional, Tuple
from datetime import datetime, timedelta
from difflib import SequenceMatcher

from itext2kg_atom.itext2kg.llm_output_parsing.langchain_output_parser import LangchainOutputParser
from itext2kg_atom.itext2kg.atom.models import AtomicFact
from translation.translator import TranslationService
from env_config import (
    num_rows_to_process, doc_parser_input_excel_path, doc_parser_output_excel_path, doc_parser_enable_parallel_processing, doc_parser_batch_size,
    column_name_date, column_name_paragraph, column_name_sentiment, column_name_translated_sentiment, column_name_factoids_extracted, column_name_translated_paragraph, column_name_date_translated_paragraph,
    translator_batch_size
)

# Set up logger for this module
logger = get_logger(__name__)

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
    Supports patterns like "3 days ago", "1 week ago", "2 years ago", etc.
    Replaces relative terms completely with absolute dates in natural format ("on YYYY-MM-DD").
    
    Args:
        fact: The atomic fact string
        observation_date: Reference date in YYYY-MM-DD format
        
    Returns:
        Fact with normalized dates
    """
    obs_date = datetime.strptime(observation_date, '%Y-%m-%d')
    normalized_fact = fact
    
    # Pattern 1: "X days/weeks/months/years ago" or "X days/weeks/months/years before"
    # This pattern captures: "3 days ago", "1 week ago", "2 years ago", etc.
    time_ago_pattern = r'\b(\d+)\s+(days?|weeks?|months?|years?)\s+(ago|before)\b'
    matches = list(re.finditer(time_ago_pattern, normalized_fact, re.IGNORECASE))
    
    for match in reversed(matches):
        quantity = int(match.group(1))
        unit = match.group(2).lower()
        direction = match.group(3).lower()
        
        # Normalize unit to plural form
        if unit in ['day', 'week', 'month', 'year']:
            unit = unit + 's'
        
        try:
            # Calculate the target date
            if unit == 'days':
                target_date = obs_date - timedelta(days=quantity)
            elif unit == 'weeks':
                target_date = obs_date - timedelta(weeks=quantity)
            elif unit == 'months':
                # Handle month subtraction properly
                month = obs_date.month - quantity
                year = obs_date.year
                while month <= 0:
                    month += 12
                    year -= 1
                target_date = obs_date.replace(year=year, month=month)
            elif unit == 'years':
                target_date = obs_date.replace(year=obs_date.year - quantity)
            else:
                continue
            
            # Format as "on YYYY-MM-DD" to integrate naturally into the fact
            replacement = f"on {target_date.strftime('%Y-%m-%d')}"
            normalized_fact = normalized_fact[:match.start()] + replacement + normalized_fact[match.end():]
        except Exception as e:
            logger.warning(f"Could not convert relative date pattern '{match.group(0)}': {e}")
    
    # Pattern 2: "on Monday", "on Thursday", etc. (day names)
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
    
    # Pattern 3: "last month", "this month", "last week", "this week", etc.
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
                # Format as "on YYYY-MM-DD" for natural integration
                replacement = f"on {converted_date}"
                normalized_fact = normalized_fact[:match.start()] + replacement + normalized_fact[match.end():]
            except Exception as e:
                logger.warning(f"Could not convert date pattern {match.group(0)}: {e}")
    
    # Pattern 4: Clean up temporal indication artifacts like "2020-01-28 (relative temporal indication)"
    # Replace "YYYY-MM-DD (any text in parentheses)" with just "YYYY-MM-DD"
    temporal_artifact_pattern = r'(\d{4}-\d{2}-\d{2})\s*\([^)]*\)'
    normalized_fact = re.sub(temporal_artifact_pattern, r'\1', normalized_fact)
    
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


# TODO: replace the body of this function with a more sophisticated filtering mechanism that identifies and removes irrelevant facts, 
# depending on the context of the paragraph and the content of the facts. 
# This is a complex task that may potentially leverage the LLM for relevance scoring or classification of facts as relevant vs. irrelevant based on the original paragraph content.
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


class DocumentParser:
    """
    Parser class to convert raw news paragraphs into atomic facts using LLM.
    Follows atomicity, decontextualization, temporal normalization, and end actions rules.
    """
    
    def __init__(self, llm_model, embeddings_model=None, enable_translation: bool = False):
        """
        Initialize the DocumentParser with LLM and optional embeddings model.
        
        Args:
            llm_model: The language model instance (ChatOllama, ChatOpenAI, etc.)
            embeddings_model: Optional embeddings model for semantic operations
            enable_translation: Whether to automatically translate non-English inputs. Default: False
        """
        self.llm_model = llm_model
        self.embeddings_model = embeddings_model
        self.enable_translation = enable_translation
        
        # Initialize translation service if needed
        self.translator = None
        if self.enable_translation:
            try:
                self.translator = TranslationService(llm_model=llm_model)
                logger.info(f"✅ Translation service initialized for ita→eng")
            except Exception as e:
                logger.error(f"Failed to initialize translation service: {e}. Proceeding without translation.")
                self.enable_translation = False
        
        self.parser = LangchainOutputParser(
            llm_model=llm_model,
            embeddings_model=embeddings_model
        )
        logger.info("DocumentParser initialized successfully")
    
    @staticmethod
    def _create_temporal_system_query_ollama_version(observation_date: str) -> str:
        """
        Create a comprehensive system query for exhaustive atomic facts extraction.
        Uses the AtomicFact schema description as the foundation and adds:
        - Emphasis on EXHAUSTIVE extraction (all facts, including supporting details)
        - Better date conversion examples with explicit mappings
        - Explicit examples of what NOT to extract (meta-information, noise, irrelevant details)
        - Clear prohibition on extracting "The observation date is" facts
        - Instructions to prevent duplicate/near-duplicate facts
        - Strict guidance on filtering generic descriptions vs. substantive information
        
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
        three_days_ago = (obs_date - timedelta(days=3)).strftime('%Y-%m-%d')
        week_monday = (obs_date - timedelta(days=obs_date.weekday())).strftime('%Y-%m-%d')
        last_week_monday = (obs_date - timedelta(days=obs_date.weekday() + 7)).strftime('%Y-%m-%d')
        month_first = obs_date.replace(day=1).strftime('%Y-%m-%d')
        year_first = obs_date.replace(month=1, day=1).strftime('%Y-%m-%d')
        last_month_first = (obs_date.replace(day=1) - timedelta(days=1)).replace(day=1).strftime('%Y-%m-%d')
        last_year_first = obs_date.replace(year=obs_date.year - 1, month=1, day=1).strftime('%Y-%m-%d')
        
        return f"""
You are an expert EXHAUSTIVE and RELEVANT atomic facts extraction engine. Your PRIMARY goal is to extract ALL distinct factual information from the input paragraph.


# ABSOLUTE PROHIBITIONS (NEVER extract these):

1. **NEVER include meta-information about the observation date itself:**
   - "The observation date is {observation_date}"
   - "Today's date is {observation_date}"
   - "On {observation_date}, the observation date is being recorded"
    The observation_date is ONLY used for temporal normalization, NOT as content to extract.

2. **NEVER extract purely generic corporate/motivational descriptions without specific content:**
   - "Company X is focused on innovation and excellence"
   - "The organization is dedicated to building a better future"
   - "Y is committed to customer satisfaction and continuous improvement"

3. **NEVER extract unrelated noisy information (things mentioned in passing, meta-commentary or promotional content):**
   - "Discover the top 5 tools for remote workers."
   - "Click here for a deeper understanding of the market shifts."
   - "This content is purely for informational purposes and should not be taken as legal or medical advice"

4. **NEVER extract irrelevant information that is not directly related to the main news:**
    ### Example 1:
    > INPUT: "While I was eating a sandwich, I heard from the TV that the town's jewelry store was robbed this morning."
    > RELEVANT FACT: "The town's jewelry store was robbed this morning."
    > NOISE (DO NOT EXTRACT): "I was eating a sandwich", "I heard from the TV"
    > REASONING: The speaker's meal and the source of information (TV) do not change the facts of the robbery, which is the actual news.

    ### Example 2:
    > INPUT: "The CEO of Company X, while enjoying a cup of coffee, announced a new product line yesterday."
    > RELEVANT FACT: "The CEO of Company X announced a new product line yesterday."
    > NOISE (DO NOT EXTRACT): "while enjoying a cup of coffee"
    > REASONING: The CEO's activity of enjoying a cup of coffee does not change the fact of the announcement, which is the main news.

    ### Example 3:
    > INPUT: "Mark flew to Paris to attend a conference where the Minister announced a new 10% tax on digital services."
    > RELEVANT FACT: "The Minister announced a new 10% tax on digital services."
    > NOISE (DO NOT EXTRACT): "Mark flew to Paris to attend a conference"
    > REASONING: The speaker's travel details and the specific event he was attending are "logistical noise." The actual news is the tax announcement.

    ### Example 4:
    > INPUT: "I was walking my dog in the park when I noticed that the city has finally started the bridge reconstruction project."
    > RELEVANT FACT: "The city has started the bridge reconstruction project."
    > NOISE (DO NOT EXTRACT): "I was walking my dog in the park", "I noticed that"
    > REASONING: The speaker’s activity (walking a dog) and their internal state (noticing something) are irrelevant to the status of the bridge.

5. **NEVER extract the same information multiple times in different phrasings:**
   - If you extract "Event X happened", do NOT also extract "X is something that happened"

   
# WHAT TO EXTRACT (EXHAUSTIVELY):

- **Main events and actions**: "X announced Y", "Z declared an emergency"
- **Supporting facts**: "The event occurred because of...", "It impacts..."
- **Background context**: "This is the first time X did Y", "Historical precedent..."
- **Entity descriptions (ONLY if specific/informative)**: "Company X produces cars", "City Y has population Z"
- **Causal relationships**: "A happened because of B"
- **Relationships and connections**: "X is affiliated with Y", "Z works for company W"
- **Quantitative information**: "X increased by 40%", "Z people were affected"
- **Named entities and roles**: "CEO John Smith", "Organization XYZ supplies sport equipment"
- **Impact and consequences**: "The policy will affect X", "Y resulted in Z"


## KEY RULES:

### 1. ATOMICITY
- Each atomic fact must contain exactly ONE piece of information or relationship
- Break down compound sentences into single-atomic facts statements
Example: "Company X announced a product and hired 50 people" → Two facts:
  - "Company X announced a new product"
  - "Company X hired 50 people"

### 2. DECONTEXTUALIZATION
- Replace ALL pronouns (he, she, it, they) with full entity names
- Example: "John joined the company. He started on Monday" → "John joined the company on Monday"

### 3. TEMPORAL NORMALIZATION
Convert ALL relative time references to absolute dates:

REFERENCE EXAMPLES FOR {observation_date}:
- "today" → on {observation_date}
- "yesterday" → on {yesterday}
- "3 days ago" → on {three_days_ago}
- "this week" (Monday) → on {week_monday}
- "last week" (Monday) → on {last_week_monday}
- "this month" (1st) → on {month_first}
- "last month" (1st) → on {last_month_first}
- "this year" (Jan 1st) → on {year_first}
- "last year" (Jan 1st) → on {last_year_first}
- Named days ("Monday", "Thursday") → on YYYY-MM-DD (most recent occurrence)
- Keep explicit dates unchanged (e.g., "June 18, 2024" stays "June 18, 2024")

**IMPORTANT**: Replace relative terms COMPLETELY. Final facts should have "on YYYY-MM-DD", never "on YYYY-MM-DD (yesterday)" or "YYYY-MM-DD (relative reference)"

### 4. END ACTIONS
- Explicitly capture role/action termination with timestamp
- Example: "CEO resigned yesterday" → "CEO resigned on {yesterday}"

### 5. AVOIDING IRRELEVANT INFORMATION
- INCLUDE: Facts about companies, people, events, policies, impacts
- EXCLUDE: Purely descriptive stylistic details that do not add substantive information about entities/events

If a description is about a specific named entity and provides information (e.g., "Company XYZ produces cars"), INCLUDE it.
If a description is generic and non-informative (e.g., "The company XYZ is focused on innovation, excellence, and building a better future through technology"), EXCLUDE it.

### 6. DEDUPLICATION GUIDANCE
- Prefer facts mentioning SPECIFIC NAMES/ENTITIES over generic versions
- GOOD: "Technology company ABC announced a new research facility in Silicon Valley on March 15, 2024"
- AVOID DUPLICATE: "The company expanded its operations in California" (already covered by the GOOD fact)
- Extract the more comprehensive fact with entity names rather than generic versions


## Example of EXHAUSTIVE Extraction

**Example 1:**
"Observation date: 2020-05-15 TOKYO — Tech company XYZ announced a major breakthrough in renewable energy yesterday. The innovation could reduce power consumption by 40%. Industry experts believe this represents a significant step forward for sustainable technology."

**Output (ALL facts):**
- Tech company XYZ announced a major breakthrough in renewable energy on 2020-05-14
- The renewable energy breakthrough could reduce power consumption by 40%
- Industry experts believe the announcement represents a significant step forward for sustainable technology

**Example 2:**
"Observation date: 2020-03-20 The European Union implemented new data protection regulations on Thursday. Companies have 60 days to comply with the rules. Financial penalties for non-compliance could reach up to 10 million euros."

**Output:**
- The European Union implemented new data protection regulations on 2020-03-19
- Companies have 60 days to comply with the new regulations (deadline: around 2020-05-18)
- Financial penalties for non-compliance with the regulations could reach up to 10 million euros

**Example 3:**
"Observation date: 2020-08-10 MANILA — A typhoon struck the region last week, causing flooding in three provinces. The disaster displaced thousands of residents. Emergency services continue relief operations."

**Output:**
- A typhoon struck the region around 2020-08-03
- The typhoon caused flooding in three provinces
- Thousands of residents were displaced by the typhoon
- Emergency services continue relief operations following the typhoon

## Input Format
A string made with this format: "Observation date: YYYY-MM-DD. [raw news paragraph text]"

## Output Format
Return ONLY the list of atomic facts. Each fact should be:
- Concise and factual
- Temporally explicit (include normalized dates where relevant)
- Decontextualized (no pronouns)
- Unique (no near-duplicates)
- Substantively informative (not generic marketing language)

MOST IMPORTANT: Extract EXHAUSTIVELY but CLEANLY. Skip noisy/meta-information and generic descriptions, but capture ALL substantive facts about events, entities, and relationships.
"""

    @staticmethod
    def _create_temporal_system_query(observation_date: str) -> str:
        """
        Create a comprehensive system query for exhaustive atomic facts extraction.
        Uses the AtomicFact schema description as the foundation and adds:
        - Emphasis on EXHAUSTIVE extraction (all facts, including supporting details)
        - Better date conversion examples with explicit mappings
        - Explicit examples of what NOT to extract (meta-information, noise, irrelevant details)
        - Clear prohibition on extracting "The observation date is" facts
        - Instructions to prevent duplicate/near-duplicate facts
        - Strict guidance on filtering generic descriptions vs. substantive information
        
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
        three_days_ago = (obs_date - timedelta(days=3)).strftime('%Y-%m-%d')
        week_monday = (obs_date - timedelta(days=obs_date.weekday())).strftime('%Y-%m-%d')
        last_week_monday = (obs_date - timedelta(days=obs_date.weekday() + 7)).strftime('%Y-%m-%d')
        month_first = obs_date.replace(day=1).strftime('%Y-%m-%d')
        year_first = obs_date.replace(month=1, day=1).strftime('%Y-%m-%d')
        last_month_first = (obs_date.replace(day=1) - timedelta(days=1)).replace(day=1).strftime('%Y-%m-%d')
        last_year_first = obs_date.replace(year=obs_date.year - 1, month=1, day=1).strftime('%Y-%m-%d')
        
        return f"""
You are an expert information extraction assistant. Your task is to extract factual, atomic statements from the provided text.

# EXTRACTION GOALS:
Extract statements about:
- Main events and actions (e.g., "Company X announced Y")
- Background context and causal relationships
- Quantitative data and impacts

# FORMATTING RULES:
1. ATOMICITY: Each fact must contain exactly ONE piece of information. Break compound sentences apart.
2. NO PRONOUNS: Replace all pronouns (he, she, it, they) with the specific entity names.
3. TEMPORAL NORMALIZATION: You MUST replace relative dates with absolute dates based on the observation date ({observation_date}).

REFERENCE EXAMPLES FOR {observation_date}:
- "today" → on {observation_date}
- "yesterday" → on {yesterday}
- "last week" (Monday) → on {last_week_monday}
- "this year" (Jan 1st) → on {year_first}

# WHAT TO IGNORE:
Do not extract meta-commentary, introductory phrases (like "While drinking coffee"), or generic corporate marketing speak. Only extract substantive facts.

Return the extracted information strictly adhering to the requested format.
"""


    async def extract_atomic_facts_from_paragraphs_batch(
        self,
        paragraphs: List[str],
        observation_dates: List[str],
        apply_post_processing: bool = True
    ) -> List[List[str]]:
        """
        Extract atomic facts from multiple paragraphs in parallel (batch processing).
        Applies translation if configured for non-English inputs.
        
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
            # system_queries = [
            #     self._create_temporal_system_query(obs_date)
            #     for obs_date in observation_dates
            # ]
            system_query = self._create_temporal_system_query(observation_dates[0])  # Use first observation date for batch processing
            
            # Use LangchainOutputParser batch processing to extract facts from all paragraphs
            # This allows the underlying LLM provider to batch requests efficiently
            # NOTE: LangchainOutputParser.extract_information_as_json_for_context accepts
            # either a single system_query or will use the first one for batch processing.
            # For now, we pass the first system query to avoid issues with per-context routing.
            results = await self.parser.extract_information_as_json_for_context(
                output_data_structure=AtomicFact,
                contexts=paragraphs,
                system_query=system_query,  # Note: All dates are similar, so first query is representative
                json_schema_enabled=True
            )
            
            # Extract facts from each AtomicFact object
            all_facts = []
            for i, (atomic_fact_obj, obs_date) in enumerate(zip(results, observation_dates)):
                if hasattr(atomic_fact_obj, 'atomic_fact'):
                    facts = atomic_fact_obj.atomic_fact
                else:
                    logger.warning(f"Unexpected result structure for paragraph {i}: {atomic_fact_obj} with obs_date {obs_date}")
                    facts = []
                
                # Apply post-processing
                if apply_post_processing and facts:
                    # Step 1: Normalize dates and clean temporal artifacts
                    facts = [normalize_relative_dates_in_fact(fact, obs_date) for fact in facts]
                    
                    # Step 2: Remove duplicates
                    facts = remove_duplicate_facts(facts, similarity_threshold=0.8)
                    
                    # Step 3: Filter irrelevant facts
                    #facts = filter_irrelevant_facts(facts, paragraphs[i])
                
                all_facts.append(facts)
            
            return all_facts
            
        except Exception as e:
            logger.error(f"Error extracting atomic facts from batch: {e}")
            return [[] for _ in paragraphs]
    
    async def parse_excel(
        self, 
        apply_post_processing: bool = True,
    ) -> pd.DataFrame:
        """
        Read an Excel file, extract atomic facts for each paragraph, and save results.
        Processes paragraphs in parallel batches for improved performance.
        Supports multilingual input with automatic translation to English.
        
        Args:
            apply_post_processing: Whether to apply post-processing (date normalization, deduplication)
            
        Returns:
            DataFrame with added factoids_extracted column
        """
        # Read the Excel file
        if num_rows_to_process > 0:
            df = pd.read_excel(doc_parser_input_excel_path, nrows=num_rows_to_process)
            logger.info(f"Loaded first {num_rows_to_process} rows from '{doc_parser_input_excel_path}'")
        else:
            df = pd.read_excel(doc_parser_input_excel_path)
            logger.info(f"Loaded all rows from '{doc_parser_input_excel_path}'")
        
        # Validate required columns
        if column_name_date not in df.columns or column_name_paragraph not in df.columns:
            raise ValueError(f"Excel file must contain '{column_name_date}' and '{column_name_paragraph}' columns")

        column_name_paragraph_to_parse = column_name_paragraph

        if self.enable_translation:
            # Validate optional sentiment column
            sentiments = []
            sentiments_str = None
            if column_name_sentiment not in df.columns:
                logger.warning(f"Column '{column_name_sentiment}' not found in Excel. Sentiment analysis will be skipped.")
                sentiments = None
            else:
                sentiments_str = df[column_name_sentiment].tolist()
                for s in sentiments_str:
                    extracted_float = float(match.group(1)) if(match := re.search(r'^(\d+\.?\d*)', s)) else None
                    if extracted_float is not None:
                        sentiments.append(extracted_float)
                    else:
                        logger.error(f"Could not extract float from sentiment value '{s}'. Defaulting to 3.0")
                        sentiments.append(3.0)
                #df["sentiment_float"] = sentiments
                df[column_name_sentiment] = sentiments

            # Translate the dataset
            paragraphs = df[column_name_paragraph_to_parse].tolist()
            logger.info(f"📝 Translating {len(paragraphs)} paragraphs from Italian to English...")
            try:
                paragraphs_to_process, sentiments_translated = await self.translator.translate_batch(paragraphs, sentiments, batch_size=translator_batch_size)
                logger.info(f"✅ Translation completed for {len(paragraphs_to_process)} paragraphs")

                # Add a new column for translated paragraphs
                df[column_name_translated_paragraph] = paragraphs_to_process
                column_name_paragraph_to_parse = column_name_translated_paragraph  # Update to use translated paragraphs for processing
                # If not None, add a new column for translated sentiments and compute translation metrics
                if sentiments_str is not None:
                    df[column_name_translated_sentiment] = sentiments_translated
                    # Compute sentiment metrics
                    self.compute_translation_sentiment_metrics(sentiments, sentiments_translated)
            except Exception as e:
                logger.error(f"Translation failed: {e}. Using original paragraphs.")


        # Create the combined column 'lead_paragraph_observation_date'
        logger.info(f"Creating {column_name_date_translated_paragraph} column...")
        df[column_name_date_translated_paragraph] = df.apply(
            lambda row: f"Observation date: {row[column_name_date]}. {row[column_name_paragraph_to_parse]}",
            axis=1
        )
        logger.info("✅ Combined column created successfully")

        # Pre-processing
        # Normalize each temporal reference strings in the 'lead_paragraph_observation_date' column to ensure consistent formatting for the LLM
        # df[column_name_date_translated_paragraph] = [normalize_relative_dates_in_fact(fact, df.loc[idx, 'date']) for idx, fact in enumerate(df[column_name_date_translated_paragraph])]
        # Date normalization should be done in post processing after extraction
        # i.e. this year's edition of Art Basel -> yyyy-mm-dd will translate the following fact into:
        # "Art Basel Hong Kong, an important destination in the international art market calendar, was canceled on 2020-01-01's edition"

        # Initialize the factoids_extracted column
        df[column_name_factoids_extracted] = None
        factoids_list = [[] for _ in range(len(df))]
        
        # Convert dates to strings and prepare data
        dates = []
        paragraphs = []
        row_indices = []
        
        for idx, row in df.iterrows():
            date = row[column_name_date]
            paragraph = row[column_name_date_translated_paragraph]
            
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
        num_batches = (total_rows + doc_parser_batch_size - 1) // doc_parser_batch_size
        
        logger.info(f"Processing {total_rows} rows in {num_batches} batches (batch size: {doc_parser_batch_size})")
        
        # Create batch tasks
        batch_tasks = []
        for batch_idx in range(num_batches):
            start_idx = batch_idx * doc_parser_batch_size
            end_idx = min(start_idx + doc_parser_batch_size, total_rows)
            
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
        
        batch_results = []
        if doc_parser_enable_parallel_processing:
            # Execute all batch tasks in parallel
            logger.info("Starting parallel batch processing...")
            batch_results = await asyncio.gather(*batch_tasks, return_exceptions=True)
        else:
            # Execute batches sequentially to respect local server queues
            logger.info("Starting sequential batch processing...")
            for task in batch_tasks:
                result = await task
                batch_results.append(result)
        
        # Collect results
        for batch_result in batch_results:
            if isinstance(batch_result, Exception):
                logger.error(f"Batch processing failed: {batch_result}")
                continue
            
            for row_idx, facts in batch_result:
                factoids_list[row_idx] = facts
        
        # Add the extracted factoids to the dataframe
        df[column_name_factoids_extracted] = factoids_list
        
        # Save the output Excel file
        output_path = doc_parser_output_excel_path or doc_parser_input_excel_path
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

    def compute_translation_sentiment_metrics(self, original_sentiments: List[float], translated_sentiments: List[float]):
        if len(original_sentiments) <= 0 or len(translated_sentiments) <= 0:
            logger.error("Could not compute sentiment metrics on empty lists")
        if len(original_sentiments) != len(translated_sentiments):
            logger.error("Could not compute sentiment metrics lists with different size")
        # original_sentiments = [1.0, 1.0, 2.0, 3.0, 2.0, 1.0, 4.0, 3.0, 3.0, 1.0]
        # translated_sentiments = [3.0, 1.5, 1.0, 3.5, 1.5, 1.0, 4.0, 3.5, 2.5, 1.0]
        avg_original = sum(original_sentiments) / len(original_sentiments)
        avg_translated = sum(translated_sentiments) / len(translated_sentiments)

        # Mean Absolute Error (MAE): calculates the average absolute distance between the original and translated sentiments. 
        # It tells you, on average, how many points off your LLM's translated sentiment is.
        MAE = sum([abs(a - b) for (a, b) in zip(original_sentiments, translated_sentiments)]) / len(original_sentiments)
        # Root Mean Squared Error (RMSE): penalizes larger errors much more harshly than smaller ones
        RMSE = sqrt(sum([pow(abs(a - b), 2) for (a, b) in zip(original_sentiments, translated_sentiments)])) / len(original_sentiments)
        # Pearson Correlation coefficient: A score close to 1 means the LLM preserves the relative sentiment scaling perfectly, 
        # even if its baseline is slightly shifted (e.g., if it consistently rates everything 0.5 higher than the original)
        r = sum([(a - avg_original) * (b - avg_translated) for (a, b) in zip(original_sentiments, translated_sentiments)]) / sqrt(sum([pow(a - avg_original, 2) for a in original_sentiments]) * sum([pow(b - avg_translated, 2) for b in translated_sentiments]))
        
        logger.info(f"Computed metrics for Sentiment analysis:")
        logger.info(f"> Total sentiments analyzed: {len(original_sentiments)}")
        logger.info(f"> Mean Absolute Error (MAE) = {MAE}. On average, the translation alters the sentiment by ± {MAE}")
        logger.info(f"> Root Mean Squared Error (RMSE) = {RMSE}. This metric penalizes larger errors with higher severity than MAE")
        logger.info(f"> Pearson Correlation coefficient r = {r}. A score close to 1 means the LLM preserves the relative sentiment scaling perfectly, even if its baseline is slightly shifted")




# ============================================================================
# DEMO USAGE
# ============================================================================

async def demo_parse_documents(
    llm_model, 
    enable_translation: bool = False,
    apply_post_processing: bool = True
):
    """
    Convenience function to parse documents using the DocumentParser class with parallel batch processing.
    
    Args:
        llm_model: The language model instance
        apply_post_processing: Whether to apply post-processing (date normalization, deduplication)
        
    Returns:
        DataFrame with extracted atomic facts
    """
    parser = DocumentParser(llm_model=llm_model, enable_translation=enable_translation)
    result_df = await parser.parse_excel(
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