"""
Translation service for multilingual document support.

This module provides a TranslationService class that handles language translation
(currently IT→EN) using the chosen LLM instance as the backend. Supports batch translation
with caching and error handling.
"""

import json
import logging
import re
from typing import List, Optional, Tuple, Union
from functools import lru_cache
from pydantic import BaseModel, Field
import torch

from itext2kg_atom.itext2kg.llm_output_parsing.langchain_output_parser import LangchainOutputParser
from itext2kg_atom.itext2kg.logging_config import get_logger

logger = get_logger(__name__)

# Pydantic models for structured output ---------------
class TranslationResult(BaseModel):
    """Schema representing the translation and analyzed sentiment value."""
    translation: str = Field(description="The accurate financial translation of the text in English.")
    sentiment: float = Field(description="A honest sentiment assessment of the translation on a 1 to 5 scale.")
# class TranslationResultList(BaseModel):
#     """Schema representing a list of translation results."""
#     results: List[TranslationResult] = Field(description="A list of translation results, each containing the translated text and its sentiment value.")

class TranslationResultNoSentiment(BaseModel):
    """Schema representing the translation without sentiment value."""
    translation: str = Field(description="The accurate financial translation of the text in English.")
# class TranslationResultNoSentimentList(BaseModel):
#     """Schema representing a list of translation results without sentiment."""
#     results: List[TranslationResultNoSentiment] = Field(description="A list of translation results, each containing the translated text.")
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

        # Bind the Pydantic schema structure using native "json_schema" method
        # try:
        #     self.model = self.raw_model.with_structured_output(
        #         TranslationResult, 
        #         method="json_schema"
        #     )
        #     logger.info("Successfully bound structured output (json_schema) to TranslationService.")
        # except Exception as e:
        #     logger.warning(f"Could not initialize with_structured_output natively: {e}. Falling back to standard model.")
        #     self.model = self.raw_model
        self.parser = LangchainOutputParser(llm_model=self.raw_model, embeddings_model=None)
        
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

#         # # Stage 1: Initial Translation Agent
#         # prompt_1 = (
#         #     "You are an expert translator specializing in Italian-to-English financial news.\n"
#         #     "Translate the following Italian text into English. Follow these strict rules:\n"
#         #     "- Maintain the exact economic sentiment and tone.\n"
#         #     "- Leave proper nouns, company names, and surnames in their original Italian form.\n"
#         #     "- Use precise financial terminology.\n\n"
#         #     f"Text to translate:\n{text}\n\n"
#         #     "Translation:"
#         # )
#         # response = self.raw_model.invoke(prompt_1)
#         # initial_translation = response.content.strip() if hasattr(response, 'content') else str(response).strip()

#         # # Stage 2: Financial Critic / Debate Agent
#         # prompt_2 = (
#         #     "You are a critical editor and financial analyst reviewing an automated translation.\n"
#         #     "Analyze the original Italian text and its English translation. Find any flaws:\n"
#         #     "- Did the translator accidentally translate proper nouns or surnames literally?\n"
#         #     "- Is the financial terminology accurate?\n"
#         #     "- Is the original news sentiment preserved?\n\n"
#         #     f"Original Italian:\n{text}\n\n"
#         #     f"Current English Translation:\n{initial_translation}\n\n"
#         #     "Provide constructive criticism and specific instructions on how to fix it. If it is already perfect, say 'No changes needed'."
#         # )
#         # response = self.raw_model.invoke(prompt_2)
#         # critique = response.content.strip() if hasattr(response, 'content') else str(response).strip()

#         # if "No changes needed" in critique:
#         #     return initial_translation

#         # # Stage 3: Polishing Agent
#         # prompt_3 = (
#         #     "You are an expert editor polishing a financial translation based on expert feedback.\n"
#         #     "Incorporate the feedback to produce the final, definitive English translation.\n"
#         #     "Ensure proper nouns are completely untouched and financial terminology is perfect.\n\n"
#         #     f"Original Italian:\n{text}\n\n"
#         #     f"Initial Translation:\n{initial_translation}\n\n"
#         #     f"Feedback to apply:\n{critique}\n\n"
#         #     "Final English Translation (output only the translated text, nothing else):"
#         # )
#         # response = self.raw_model.invoke(prompt_3)
#         # final_translation = response.content.strip() if hasattr(response, 'content') else str(response).strip()

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

    
#     async def _translate_single_texts(self, text: str, sentiment: float) -> Tuple[str, float]:
#         """Translates a single piece of text using an initial translation -> critique -> refine loop."""
        
#         # Alternative implementation: single-prompt translation with sentiment and terminology checks
#         prompt = f"""
# You are an expert translator specializing in Italian-to-English financial news.
# You will be provided with an Italian text already labelled with a **Sentiment** value (general linguistic tone) on a scale from 1 to 5, with increments of 0.5.

# **REFERENCE SCALE:**
# - 1 = very negative
# - 2 = negative  
# - 3 = neutral
# - 4 = positive
# - 5 = very positive

# **Intermediate values (1.5, 2.5, 3.5, 4.5) are nuances between two adjacent categories:**
# - 1.5 = between very negative and negative
# - 2.5 = between negative and neutral
# - 3.5 = between neutral and positive
# - 4.5 = between positive and very positive

# Your task is to translate the following Italian text into English and provide a honest sentiment assessment of the translation. Follow these strict guidelines:
# - During the translation, try to maintain the exact financial sentiment, nuance, and market tone (bearish/bullish).
# - CRITICAL: Surnames, proper nouns, and corporate brand names must never be translated literally.
# - Ensure accurate and technical financial terminology matching standard English economic reporting.
# - After translating, make a honest assessment of your translation, providing a sentiment value on the same 1 to 5 scale.

# **INPUT**
# Sentiment value: {sentiment}
# Italian text: 
# {text}
# """

#         try:
#             response = self.raw_model.invoke(prompt)
#             if isinstance(response, TranslationResult):
#                 return response.translation, response.sentiment
#             # Defensive fallback if result was returned as a dict instead of a parsed Pydantic object
#             if isinstance(response, dict):
#                 return response.get("translation", ""), float(response.get("sentiment", sentiment))
#             raise ValueError(f"Returned object type is unexpected: {type(response)}")
#         except Exception as e:
#             logger.error(f"Error during structured translation execution: {e}. Falling back to default values.")
#             # Graceful pipeline degradation: return original text and raw sentiment instead of halting execution
#             return text, sentiment

#         # response = self.raw_model.invoke(prompt)
#         # raw_json_string = response.content.strip() if hasattr(response, 'content') else str(response).strip()
#         # try:
#         #     # Filter out any non-JSON content before parsing
#         #     json_match = re.search(r'(\{.*\})', raw_json_string, re.DOTALL)
#         #     if not json_match:
#         #         raise json.JSONDecodeError(f"JSON parsing failed")
#         #     json_data = json.loads(json_match.group(1))
#         #     final_translation, final_sentiment = json_data.get("translation", ""), json_data.get("sentiment", sentiment)
#         # except Exception as e:
#         #     logger.error(f"Error parsing JSON response: {e}. Raw response: {raw_json_string}")
#         #     raise json.JSONDecodeError(f"Invalid JSON response: {raw_json_string}") from e

#         # return final_translation, final_sentiment

    async def _translate_multiple_texts(self, texts: List[str], sentiments: List[float]) -> Tuple[List[str], List[float]]:
        """Translates multiple texts through batch processing, using a single prompt with sentiment and terminology checks."""
        
        # Alternative implementation: single-prompt translation with sentiment and terminology checks
        prompt = f"""
You are an expert translator specializing in Italian-to-English financial news.
You will be provided with Italian texts, each one labelled with a **Sentiment** value (general linguistic tone) on a scale from 1 to 5, with increments of 0.5.

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

Your task is to translate the following Italian text into English and provide a honest sentiment assessment of the translation. Follow these strict guidelines:
- During the translation, try to maintain the exact financial sentiment, nuance, and market tone (bearish/bullish).
- CRITICAL: Surnames, proper nouns, and corporate brand names must never be translated literally.
- Ensure accurate and technical financial terminology matching standard English economic reporting.
- After translating, make a honest assessment of your translation, providing a sentiment value on the same 1 to 5 scale.

Return the extracted information strictly adhering to the requested format.
"""

        input_context = [
            f"Sentiment value: {sent}\nItalian text:\n{tx}" 
            for tx, sent in zip(texts, sentiments)
        ]

        response = await self.parser.extract_information_as_json_for_context(
            output_data_structure=TranslationResult,
            contexts=input_context,
            system_query=prompt,
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



    async def translate_batch(self, paragraphs: List[str], sentiments: List[float] = None, batch_size: int = 1) -> List[str]:
        """Processes the texts sequentially or in small steps."""
        translated_texts = []
        total = len(paragraphs)

        if sentiments and len(paragraphs) != len(sentiments):
            raise ValueError("Number of paragraphs must match number sentiments, if sentiments are provided.")
        
        # # Old sequential logic
        # for i, text in enumerate(paragraphs):
        #     logger.debug(f"Translating text {i+1}/{total}...")
        #     if not text.strip():
        #         translated_texts.append("")
        #         continue
            
        #     try:
        #         if sentiments:
        #             final_tx, sentiment_translated = self._translate_single_texts(text, sentiments[i])
        #             logger.debug(f"Original sentiment: {sentiments[i]}, Translated sentiment: {sentiment_translated}")
        #         else:
        #             final_tx = self._translate_single_texts(text)
        #         translated_texts.append(final_tx)
        #     except Exception as e:
        #         logger.error(f"Error translating text {i+1}: {e}")
        #         translated_texts.append(text) # Fallback to original text on failure

        # New batch processing logic
        for i in range(0, total, batch_size):
            batch_texts = paragraphs[i:i + batch_size]
            batch_sentiments = sentiments[i:i + batch_size] if sentiments else []
            logger.debug(f"Translating batch {i//batch_size + 1}/{(total + batch_size - 1)//batch_size}...")
            try:
                if batch_sentiments:
                    final_tx_batch, sentiment_translated_batch = await self._translate_multiple_texts(batch_texts, batch_sentiments)
                    logger.debug(f"Original sentiments: {batch_sentiments}, Translated sentiments: {sentiment_translated_batch}")
                else:
                    final_tx_batch = await self._translate_multiple_texts_without_sentiment(batch_texts)
                translated_texts.extend(final_tx_batch)
            except Exception as e:
                logger.error(f"Error translating batch {i//batch_size + 1}: {e}")
                translated_texts.extend(batch_texts) # Fallback to original texts on failure

        return translated_texts
    
    async def translate_with_metadata(
            self, paragraphs: List[str], observation_dates: List[str], batch_size: int = 1
        ) -> Tuple[List[str], List[Tuple[str, str]]]:
        translated_texts = await self.translate_batch(paragraphs, batch_size=batch_size)
        metadata = list(zip(paragraphs, translated_texts))
        return translated_texts, metadata


# Convenience function for simple usage
def create_translator(
    language_pair: str = "it-en",
    device: Optional[str] = None,
    cache_dir: Optional[str] = None
) -> TranslationService:
    """
    Factory function to create a TranslationService instance.
    
    Args:
        language_pair: Language pair code (default: "it-en")
        device: Device to use (default: auto-detect)
        cache_dir: Model cache directory (default: Hugging Face default)
        
    Returns:
        TranslationService instance
    """
    return TranslationService(model_name=language_pair, device=device, cache_dir=cache_dir)
