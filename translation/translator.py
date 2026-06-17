"""
Translation service for multilingual document support.

This module provides a TranslationService class that handles language translation
(currently IT→EN) using the chosen LLM instance as the backend. Supports batch translation
with caching and error handling.
"""

import logging
from typing import List, Optional, Tuple
from functools import lru_cache
import torch

from env_config import translator_sentence_batch_size, translator_sentence_max_length
from itext2kg_atom.itext2kg.logging_config import get_logger

logger = get_logger(__name__)


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
    Agentic Translation service for multilingual financial document support.
    Uses a 3-stage reflection workflow with a local LLM to optimize accuracy.
    """
    
    
    def __init__(self, llm_model, device: Optional[str] = None, cache_dir: Optional[str] = None):
        """
        Initialize the TranslationService.
        
        Args:
            llm_model: The local LLM instance to use for translation
            device: Device to run model on ("cpu", "cuda", or "rocm"). Auto-detect if None.
            cache_dir: Directory to cache downloaded models. Uses default if None.
            
        """
        
        
        self.model = llm_model
        self.cache_dir = cache_dir
        
        # Auto-detect device if not specified
        if device is None:
            self.device = _detect_best_device()
        else:
            self.device = device
        
        logger.info(
            f"Agentic TranslationService initialized using local LLM."
            f"(device: {self.device})"
        )
    
    
    def _translate_single_text(self, text: str) -> str:
        """Translates a single piece of text using an initial translation -> critique -> refine loop."""
        
        # # Stage 1: Initial Translation Agent
        # prompt_1 = (
        #     "You are an expert translator specializing in Italian-to-English financial news.\n"
        #     "Translate the following Italian text into English. Follow these strict rules:\n"
        #     "- Maintain the exact economic sentiment and tone.\n"
        #     "- Leave proper nouns, company names, and surnames in their original Italian form.\n"
        #     "- Use precise financial terminology.\n\n"
        #     f"Text to translate:\n{text}\n\n"
        #     "Translation:"
        # )
        # response = self.model.invoke(prompt_1)
        # initial_translation = response.content.strip() if hasattr(response, 'content') else str(response).strip()

        # # Stage 2: Financial Critic / Debate Agent
        # prompt_2 = (
        #     "You are a critical editor and financial analyst reviewing an automated translation.\n"
        #     "Analyze the original Italian text and its English translation. Find any flaws:\n"
        #     "- Did the translator accidentally translate proper nouns or surnames literally?\n"
        #     "- Is the financial terminology accurate?\n"
        #     "- Is the original news sentiment preserved?\n\n"
        #     f"Original Italian:\n{text}\n\n"
        #     f"Current English Translation:\n{initial_translation}\n\n"
        #     "Provide constructive criticism and specific instructions on how to fix it. If it is already perfect, say 'No changes needed'."
        # )
        # response = self.model.invoke(prompt_2)
        # critique = response.content.strip() if hasattr(response, 'content') else str(response).strip()

        # if "No changes needed" in critique:
        #     return initial_translation

        # # Stage 3: Polishing Agent
        # prompt_3 = (
        #     "You are an expert editor polishing a financial translation based on expert feedback.\n"
        #     "Incorporate the feedback to produce the final, definitive English translation.\n"
        #     "Ensure proper nouns are completely untouched and financial terminology is perfect.\n\n"
        #     f"Original Italian:\n{text}\n\n"
        #     f"Initial Translation:\n{initial_translation}\n\n"
        #     f"Feedback to apply:\n{critique}\n\n"
        #     "Final English Translation (output only the translated text, nothing else):"
        # )
        # response = self.model.invoke(prompt_3)
        # final_translation = response.content.strip() if hasattr(response, 'content') else str(response).strip()

        # Alternative implementation: single-prompt translation with reflection
        prompt = (
            "You are an expert translator specializing in Italian-to-English financial news.\n"
            "Translate the following Italian text into English. Follow these strict guidelines:\n"
            "- Maintain the exact financial sentiment, nuance, and market tone (bearish/bullish).\n"
            "- CRITICAL: Surnames, proper nouns, and corporate brand names must never be translated literally.\n"
            "- Ensure accurate and technical financial terminology matching standard English economic reporting.\n\n"
            f"Italian Text:\n{text}\n\n"
            "Final English Translation (Output only the translated text, nothing else):"
        )
        response = self.model.invoke(prompt)
        final_translation = response.content.strip() if hasattr(response, 'content') else str(response).strip()

        return final_translation
    
    def translate_batch(self, texts: List[str], batch_size: int = 1) -> List[str]:
        """Processes the texts sequentially or in small steps."""
        translated_texts = []
        total = len(texts)
        
        for i, text in enumerate(texts):
            logger.debug(f"Agentic Translation processing text {i+1}/{total}...")
            if not text.strip():
                translated_texts.append("")
                continue
            
            try:
                final_tx = self._translate_single_text(text)
                translated_texts.append(final_tx)
            except Exception as e:
                logger.error(f"Error translating text {i+1}: {e}")
                translated_texts.append(text) # Fallback to original text on failure
                
        return translated_texts
    
    def translate_with_metadata(
            self, texts: List[str], observation_dates: List[str], batch_size: int = 1
        ) -> Tuple[List[str], List[Tuple[str, str]]]:
        translated_texts = self.translate_batch(texts, batch_size=batch_size)
        metadata = list(zip(texts, translated_texts))
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
