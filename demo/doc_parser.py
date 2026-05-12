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
"""

import pandas as pd
import asyncio
import logging
from pathlib import Path
from typing import List, Optional
from datetime import datetime

from itext2kg_atom.itext2kg.llm_output_parsing.langchain_output_parser import LangchainOutputParser
from itext2kg_atom.itext2kg.atom.models import AtomicFact


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
        Create a system query that includes temporal normalization instructions
        with the specific observation date.
        
        # Temporal normalization optional (instead of keeping explicit dates as-is):
        -> Convert explicit dates to yyyy-mm-dd format (e.g., convert "June 18, 2024" to "2024-06-18")
        
        Args:
            observation_date: The observation date in format YYYY-MM-DD
            
        Returns:
            System query string with temporal context
        """
        return f"""
You are an expert atomic facts extraction engine. Your task is to decompose a news paragraph 
into a comprehensive list of atomic, self-contained, and temporally-grounded facts.

**Observation Date Context**: {observation_date}

## Key Rules:

### 1. ATOMICITY
- Each atomic fact must contain exactly ONE piece of information or relationship
- Break down compound/complex sentences into single-fact statements
- Remove redundancies and duplicated information
- Example: "Company X announced a new product and hired 50 people" → Two facts:
  - "Company X announced a new product"
  - "Company X hired 50 people"

### 2. DECONTEXTUALIZATION  
- Replace ALL pronouns (he, she, it, they, this, that) with full entity names
- Include necessary modifiers so each fact is understandable in isolation
- Example: "John joined the company. He started on Monday" → "John joined the company on Monday"

### 3. TEMPORAL NORMALIZATION
- Convert ALL relative time references to absolute dates based on observation_date: {observation_date}
- Conversion rules:
  - "today" → {observation_date}
  - "yesterday" → day before {observation_date}
  - "this week" → Monday of the week containing {observation_date}
  - "last week" → Monday of the week before {observation_date}
  - "this month" → 1st of the month of {observation_date}
  - "last month" → 1st of the month before {observation_date}
  - "this year" → January 1st of {observation_date[:4]}
  - "last year" → January 1st of year before {observation_date[:4]}
  - Keep explicit dates as-is (e.g., "June 18, 2024")
- Never include relative terms like "today", "yesterday", "last week" in final facts
- Position time references naturally within facts

### 4. END ACTIONS
- If the text indicates the end of a role or action (e.g., "leaving a position"), be explicit
- Capture: WHAT ended, WHO/WHAT ended it, WHEN it ended
- Example: "The CEO resigned yesterday" → "The CEO resigned on [day before {observation_date}]"

## Output Format
Return ONLY the list of atomic facts. Each fact should be:
- Concise and clear
- Temporally explicit (include dates where relevant from the text)
- Decontextualized (no pronouns, full entity names)
- Unique (no duplicates or redundancies)
"""
    
    async def extract_atomic_facts_from_paragraph(
        self, 
        paragraph: str, 
        observation_date: str
    ) -> List[str]:
        """
        Extract atomic facts from a single paragraph using the LLM.
        
        Args:
            paragraph: The raw news paragraph text
            observation_date: The observation date in format YYYY-MM-DD
            
        Returns:
            List of extracted atomic facts
        """
        try:
            system_query = self._create_temporal_system_query(observation_date)
            
            # Use LangchainOutputParser to extract atomic facts
            results = await self.parser.extract_information_as_json_for_context(
                output_data_structure=AtomicFact,
                contexts=[paragraph],
                system_query=system_query
            )
            
            # Extract the atomic_fact list from the AtomicFact Pydantic model
            if results and len(results) > 0:
                atomic_fact_obj = results[0]
                if hasattr(atomic_fact_obj, 'atomic_fact'):
                    return atomic_fact_obj.atomic_fact
                else:
                    logger.warning(f"Unexpected result structure: {atomic_fact_obj}")
                    return []
            else:
                logger.warning("No results returned from LLM")
                return []
                
        except Exception as e:
            logger.error(f"Error extracting atomic facts: {e}")
            return []
    
    async def parse_excel(
        self, 
        input_excel_path: str, 
        output_excel_path: Optional[str] = None,
        batch_size: int = 1
    ) -> pd.DataFrame:
        """
        Read an Excel file, extract atomic facts for each paragraph, and save results.
        
        Args:
            input_excel_path: Path to input Excel file with columns: date, lead_paragraph
            output_excel_path: Path to save output Excel file. If None, overwrites input file
            batch_size: Number of paragraphs to process in parallel (default: 1 for safety)
            
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
        factoids_list = []
        
        # Process each row
        for idx, row in df.iterrows():
            try:
                date = row['date']
                paragraph = row['lead_paragraph']
                
                # Convert date to string if needed
                if isinstance(date, datetime):
                    date_str = date.strftime('%Y-%m-%d')
                else:
                    date_str = str(date)
                
                logger.info(f"Processing row {idx + 1}/{len(df)} - Date: {date_str}")
                
                # Extract atomic facts for this paragraph
                atomic_facts = await self.extract_atomic_facts_from_paragraph(
                    paragraph=paragraph,
                    observation_date=date_str
                )
                
                factoids_list.append(atomic_facts)
                logger.info(f"Extracted {len(atomic_facts)} atomic facts for row {idx + 1}")
                
            except Exception as e:
                logger.error(f"Error processing row {idx + 1}: {e}")
                factoids_list.append([])
        
        # Add the extracted factoids to the dataframe
        df['factoids_g_truth'] = factoids_list
        
        # Save the output Excel file
        output_path = output_excel_path or input_excel_path
        logger.info(f"Saving results to: {output_path}")
        df.to_excel(output_path, index=False)
        
        logger.info("✅ Processing complete!")
        return df


# ============================================================================
# DEMO USAGE
# ============================================================================

async def demo_parse_documents(llm_model, input_excel_path: str, output_excel_path: Optional[str] = None):
    """
    Convenience function to parse documents using the DocumentParser class.
    
    Args:
        llm_model: The language model instance
        input_excel_path: Path to input Excel file
        output_excel_path: Path to output Excel file (optional)
        
    Returns:
        DataFrame with extracted atomic facts
    """
    parser = DocumentParser(llm_model=llm_model)
    result_df = await parser.parse_excel(input_excel_path, output_excel_path)
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