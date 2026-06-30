"""
Translation service for multilingual document support.

This module provides a TranslationService class that handles language translation
(currently IT→EN) using the chosen LLM instance as the backend. Supports batch translation
with caching and error handling.
"""

import os
import pandas as pd
from typing import List, Optional, Tuple
from functools import lru_cache
from pydantic import BaseModel, Field
import torch

from itext2kg_atom.itext2kg.llm_output_parsing.langchain_output_parser import LangchainOutputParser
from itext2kg_atom.itext2kg.logging_config import get_logger
from env_config import translator_few_shot_seed, doc_parser_input_excel_path, column_name_paragraph, column_name_sentiment

logger = get_logger(__name__)

# Pydantic models for structured output ---------------
class TranslationResult(BaseModel):
    """Schema representing the translation and analyzed sentiment value."""
    translation: str = Field(description="The accurate financial translation of the text in English.")
    sentiment: float = Field(description="A honest sentiment assessment of the translation on a 1 to 5 scale.")

class TranslationResultNoSentiment(BaseModel):
    """Schema representing the translation without sentiment value."""
    translation: str = Field(description="The accurate financial translation of the text in English.")
# ----------------------------------------------------

def _detect_best_device() -> str:
    """
    Detect the best available device for computation.
    Priority: CUDA/ROCm (Linux/HPC) > DirectML (Windows AMD Local) > CPU
    
    Returns:
        Device string: "cuda", "rocm", device(type='privateuseone', index=0), or "cpu"
    """
    if torch is None:
        return "cpu"
    
    # Check for CUDA (NVIDIA or ROCm)
    if torch.cuda.is_available():
        try:
            # Check if it's actually ROCm (AMD with HIP backend)
            cuda_device_name = torch.cuda.get_device_name(0)
            if "amd" in cuda_device_name.lower() or "rocm" in str(torch.version.cuda).lower():
                logger.info(f"ROCm (AMD GPU) detected: {cuda_device_name}")
                return "cuda"  # PyTorch uses "cuda" for both NVIDIA and ROCm
            else:
                logger.info(f"CUDA (NVIDIA GPU) detected: {cuda_device_name}")
                return "cuda"
        except Exception as e:
            logger.warning(f"Could not determine GPU type: {e}")
            return "cpu"
        
    # Check for DirectML (Windows AMD Local)
    """
    try:
        import torch_directml
        if torch_directml.is_available():
            # Returns a specialized torch device pointing to your Radeon GPU
            dml_device = torch_directml.device()
            logger.info(f"DirectML (Windows AMD Local) detected: {dml_device}")
            #return "cpu"
            return dml_device
    except Exception as e:
        logger.warning(f"Error occurred while checking DirectML: {e}")
    """
    
    # Fallback to CPU
    logger.info("No GPU detected, using CPU")
    return "cpu"


class TranslationService:
    """
    Translation service for multilingual financial document support.
    Utilizes LangChain structured outputs with schema enforcement to prevent JSON syntax errors.
    """
    
    
    def __init__(self, llm_model, device: Optional[str] = None, cache_dir: Optional[str] = None):
        """
        Initialize the TranslationService.
        
        Args:
            llm_model: The local LLM instance to use for translation
            device: Device to run model on ("cpu", "cuda", or "rocm"). Auto-detect if None.
            cache_dir: Directory to cache downloaded models. Uses default if None.
            
        """

        self.raw_model = llm_model
        self.cache_dir = cache_dir
        self.parser = LangchainOutputParser(llm_model=self.raw_model, embeddings_model=None)
        self.few_shot_paragraphs, self.few_shot_sentiments = self._get_few_shot_examples()
        
        # Auto-detect device if not specified
        if device is None:
            self.device = _detect_best_device()
        else:
            self.device = device
        
        logger.info(
            f"TranslationService initialized using local LLM."
            f"(device: {self.device})"
        )
    
#     def _translate_single_texts(self, text: str) -> str:
#         """Translates a single piece of text using an initial translation -> critique -> refine loop."""

#         # Alternative implementation: single-prompt translation with reflection
#         prompt = (
#             "You are an expert translator specializing in Italian-to-English financial news.\n"
#             "Translate the following Italian text into English. Follow these strict guidelines:\n"
#             "- Maintain the exact financial sentiment, nuance, and market tone (bearish/bullish).\n"
#             "- CRITICAL: Surnames, proper nouns, and corporate brand names must never be translated literally.\n"
#             "- Ensure accurate and technical financial terminology matching standard English economic reporting.\n\n"
#             f"Italian Text:\n{text}\n\n"
#             "Final English Translation (Output only the translated text, nothing else):"
#         )

#         response = self.raw_model.invoke(prompt)
#         final_translation = response.content.strip() if hasattr(response, 'content') else str(response).strip()

#         return final_translation

    # This method must only be called by __init__ to prevent file locks during batching
    def _get_few_shot_examples(self) -> Tuple[List[str], List[float]]:
        FILE_PATH = f"./data/translator_seeds/{translator_few_shot_seed}.xlsx"
        samples_df = None

        if os.path.exists(FILE_PATH):
            logger.info(f"Found existing samples. Reading from '{FILE_PATH}'...")
            # Read the pre-saved fixed samples
            samples_df = pd.read_excel(FILE_PATH)
            if column_name_sentiment not in samples_df.columns or column_name_paragraph not in samples_df.columns:
                return [], []

        else:
            # Read input dataset
            df = pd.read_excel(doc_parser_input_excel_path)
            if column_name_sentiment not in df.columns or column_name_paragraph not in df.columns:
                return [], []

            # Define the 5 target integer scores (buckets)
            target_scores = [1.0, 2.0, 3.0, 4.0, 5.0]
            sampled_rows = []
            logger.info(f"'{FILE_PATH}' not found. Generating 5 representative random samples...")

            # Filter the dataset into 5 buckets and randomly pull one sample from each
            for score in target_scores:
                extracted_sentiments = (
                    df[column_name_sentiment]
                    .str.extract(r"^(\d+\.?\d*)")[0]
                    .astype(float)
                    .fillna(3.0)
                )
                bucket = df[extracted_sentiments == score]

                if not bucket.empty:
                    # .sample() with random_state ensures the 'random' pick is identical every time
                    single_sample = bucket.sample(n=1, random_state=translator_few_shot_seed).copy()
                    single_sample[column_name_sentiment] = float(score)
                    sampled_rows.append(single_sample)
                else:
                    logger.warning(f"Warning: No data found in the dataset for sentiment score: {score}")

            # Combine the 5 individual samples back into one dataframe
            if sampled_rows:
                samples_df = pd.concat(sampled_rows).reset_index(drop=True)
            else:
                raise ValueError("The dataset does not contain the required sentiment scores.")

            # Save the samples into seed_value.xlsx for future executions
            samples_df.to_excel(FILE_PATH, index=False)
            logger.info(f"Successfully saved representative samples to '{FILE_PATH}'.")
        
        # Force the sentiment column to float just in case Excel loaded them weirdly
        samples_df[column_name_sentiment] = samples_df[column_name_sentiment].astype(float)
        return samples_df[column_name_paragraph].tolist(), samples_df[column_name_sentiment].tolist()



    async def _translate_multiple_texts(self, texts: List[str], sentiments: List[float]) -> Tuple[List[str], List[float]]:
        """Translates multiple texts through batch processing, using a single prompt with sentiment and terminology checks."""
        
        # Alternative implementation: single-prompt translation with sentiment and terminology checks
        prompt = f"""
You are an expert translator specializing in Italian-to-English financial news.
You will be provided with Italian news texts.

Your task is to translate the following Italian text into English and provide a honest sentiment assessment of the translation. 

Follow these strict guidelines for the translation:
- During the translation, try to maintain the exact financial sentiment, nuance, and market tone (bearish/bullish).
- CRITICAL: Surnames, proper nouns, and corporate brand names must never be translated literally.
- Ensure accurate and technical financial terminology matching standard English economic reporting.
- After translating, make a honest assessment of your translation, providing a sentiment value on the same 1 to 5 scale.

Follow this guide to compute the sentiment value of the translated text:
The **sentiment** value represents the general linguistic tone, on a scale from 1 to 5, with increments of 0.5.

**REFERENCE SCALE:**
- 1 = very negative
- 2 = negative
- 3 = neutral
- 4 = positive
- 5 = very positive

**Intermediate values (1.5, 2.5, 3.5, 4.5) are nuances between two adjacent categories:**
- 1.5 = between very negative and negative
- 2.5 = between negative and neutral
- 3.5 = between neutral and positive
- 4.5 = between positive and very positive

Return the extracted information strictly adhering to the requested format.
"""
        few_shot_examples = f"""
Below, you can find some examples of Italian texts, each one labelled with the correct **sentiment** value, provided with a human annotator:
{[f"[translation: \"{text}\", sentiment={sentiment}]" for (text, sentiment) in zip(self.few_shot_paragraphs, self.few_shot_sentiments)]}
"""

        input_context = texts

        response = await self.parser.extract_information_as_json_for_context(
            output_data_structure=TranslationResult,
            contexts=input_context,
            system_query=prompt + few_shot_examples,
            json_schema_enabled=True
        )

        translated_texts = []
        translated_sentiments = []

        for i, res_obj in enumerate(response):
            if isinstance(res_obj, TranslationResult):
                translated_texts.append(res_obj.translation)
                translated_sentiments.append(res_obj.sentiment)
            else:
                logger.warning(f"Unexpected response type for text {i+1}: {type(res_obj)}. Fallback to original text and sentiment.")
                translated_texts.append(texts[i])
                translated_sentiments.append(sentiments[i])
        
        return translated_texts, translated_sentiments

    # Same method as above but without sentiment input, for cases where sentiment is not provided
    async def _translate_multiple_texts_without_sentiment(self, texts: List[str]) -> List[str]:
        """Translates multiple texts through batch processing, using a single prompt with sentiment and terminology checks."""
        
        # Alternative implementation: single-prompt translation with sentiment and terminology checks
        prompt = f"""
You are an expert translator specializing in Italian-to-English financial news.
You will be provided with Italian texts.

Your task is to translate the following Italian text into English and provide a honest sentiment assessment of the translation. Follow these strict guidelines:
- During the translation, try to maintain the exact financial sentiment, nuance, and market tone (bearish/bullish).
- CRITICAL: Surnames, proper nouns, and corporate brand names must never be translated literally.
- Ensure accurate and technical financial terminology matching standard English economic reporting.
- After translating, make a honest assessment of your translation, providing a sentiment value on the same 1 to 5 scale.

Return the extracted information strictly adhering to the requested format.
"""

        input_context = texts

        response = await self.parser.extract_information_as_json_for_context(
            output_data_structure=TranslationResultNoSentiment,
            contexts=input_context,
            system_query=prompt,
            json_schema_enabled=True
        )

        translated_texts = []
        for i, (res_obj) in enumerate(response):
            if isinstance(res_obj, TranslationResultNoSentiment):
                translated_texts.append(res_obj.translation)
            else:
                logger.warning(f"Unexpected response type for text {i+1}: {type(res_obj)}. Fallback to original text.")
                translated_texts.append(texts[i])
        return translated_texts



    async def translate_batch(self, paragraphs: List[str], sentiments: List[float] = None, batch_size: int = 1) -> Tuple[List[str], List[float]]:
        """Processes the texts sequentially or in small steps."""
        translated_texts = []
        translated_sentiments = []
        batch_sentiment_translated = None
        total = len(paragraphs)

        if sentiments and len(paragraphs) != len(sentiments):
            raise ValueError("Number of paragraphs must match number sentiments, if sentiments are provided.")
        
        # Batch processing logic
        for i in range(0, total, batch_size):
            batch_texts = paragraphs[i:i + batch_size]
            batch_sentiments = sentiments[i:i + batch_size] if sentiments else []
            logger.debug(f"Translating batch {i//batch_size + 1}/{(total + batch_size - 1)//batch_size}...")
            try:
                if batch_sentiments:
                    final_tx_batch, batch_sentiment_translated = await self._translate_multiple_texts(batch_texts, batch_sentiments)
                    translated_sentiments.extend(batch_sentiment_translated)
                    # logger.debug(f"Original sentiments: {batch_sentiments}, Translated sentiments: {batch_sentiment_translated}")
                else:
                    final_tx_batch = await self._translate_multiple_texts_without_sentiment(batch_texts)
                translated_texts.extend(final_tx_batch)
            except Exception as e:
                logger.error(f"Error translating batch {i//batch_size + 1}: {e}")
                translated_texts.extend(batch_texts) # Fallback to original texts on failure

        return translated_texts, translated_sentiments
    
    async def translate_with_metadata(
            self, paragraphs: List[str], observation_dates: List[str], batch_size: int = 1
        ) -> Tuple[List[str], List[Tuple[str, str]]]:
        translated_texts = await self.translate_batch(paragraphs, batch_size=batch_size)
        metadata = list(zip(paragraphs, translated_texts))
        return translated_texts, metadata
