"""
Translation service for multilingual document support.

This module provides a TranslationService class that handles language translation
(currently IT→EN) using Hugging Face MarianMT models. Supports batch translation
with caching and error handling.
"""

import logging
from typing import List, Optional, Tuple
from functools import lru_cache
import torch
import re
try:
    import spacy
    HAS_SPACY = True
except ImportError:
    HAS_SPACY = False

try:
    from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, MarianMTModel, MarianTokenizer
    import torch
    HAS_TRANSFORMERS = True
except ImportError:
    HAS_TRANSFORMERS = False
    torch = None

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
    Service for translating documents between languages using MarianMT models.
    
    Currently supports Italian→English translation. Can be extended to support
    other language pairs by adding additional model configurations.
    """
    
    # Supported language pair models
    TRANSLATION_MODELS = {
        "it-en": "Helsinki-NLP/Opus-MT-it-en",  # Italian to English
        "en-it": "Helsinki-NLP/Opus-MT-en-it",  # English to Italian (for completeness)
    }
    
    def __init__(self, model_name: str = "it-en", device: Optional[str] = None, cache_dir: Optional[str] = None):
        """
        Initialize the TranslationService.
        
        Args:
            model_name: Language pair code (e.g., "it-en" for Italian→English)
            device: Device to run model on ("cpu", "cuda", or "rocm"). Auto-detect if None.
            cache_dir: Directory to cache downloaded models. Uses default if None.
            
        Raises:
            ImportError: If transformers library is not installed
            ValueError: If model_name is not supported
        """
        if not HAS_TRANSFORMERS:
            raise ImportError(
                "transformers library is required for translation. "
                "Install it with: pip install transformers torch"
            )
        
        if model_name not in self.TRANSLATION_MODELS:
            raise ValueError(
                f"Unsupported translation model: {model_name}. "
                f"Supported: {list(self.TRANSLATION_MODELS.keys())}"
            )
        
        self.model_name = model_name
        self.model_path = self.TRANSLATION_MODELS[model_name]
        self.cache_dir = cache_dir
        
        # Auto-detect device if not specified
        if device is None:
            self.device = _detect_best_device()
        else:
            self.device = device

        # Load Spacy for Entity Masking
        self.nlp = None
        if HAS_SPACY:
            try:
                self.nlp = spacy.load("it_core_news_sm")
                logger.info("Spacy NER loaded for entity masking.")
            except OSError:
                logger.warning("Spacy model 'it_core_news_sm' not found. Entity masking disabled.")
        else:
            logger.warning("Spacy library not installed. Entity masking disabled.")
        
        # Initialize model components (lazy loaded on first use)
        self.translator = None  # Flag to indicate if models are loaded
        self.tokenizer = None
        self.model = None
        logger.info(
            f"TranslationService initialized for {model_name} "
            f"(model: {self.model_path}) on device: {self.device}"
        )
    
    def _load_translator(self):
        """Lazy load the translation model on first use."""
        if self.translator is None:
            logger.debug(f"Loading translation model: {self.model_path}...")
            try:
                # Use Marian-specific tokenizer and model
                self.tokenizer = MarianTokenizer.from_pretrained(self.model_path, cache_dir=self.cache_dir, use_safetensors=True)
                self.model = MarianMTModel.from_pretrained(self.model_path, cache_dir=self.cache_dir, use_safetensors=True)
                
                # Move model to appropriate device
                if self.device != "cpu":
                    self.model = self.model.to(self.device)
                
                # Mark as loaded
                self.translator = True
                logger.info(f"Translation model loaded successfully on {self.device}")
            except Exception as e:
                logger.error(f"Failed to load translation model: {e}")
                raise
    
    def translate(self, text: str) -> str:
        """
        Translate a single text from source to target language.
        
        Args:
            text: Text to translate
            
        Returns:
            Translated text
        """
        self._load_translator()
        
        try:
            # Tokenize
            inputs = self.tokenizer(text, return_tensors="pt", max_length=512, truncation=True)
            
            # Move inputs to device
            if self.device != "cpu":
                inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            # Generate translation
            with torch.no_grad():
                translated_ids = self.model.generate(**inputs, max_length=512)
            
            # Decode
            translated_text = self.tokenizer.decode(translated_ids[0], skip_special_tokens=True)
            
            logger.debug(f"Translated ({len(text)} chars): {text[:50]}... → {translated_text[:50]}...")
            return translated_text
        except Exception as e:
            logger.warning(f"Translation failed for text: {e}. Returning original text.")
            return text
    
    def translate_batch(
        self,
        texts: List[str],
        batch_size: int = 8,
        max_length: int = 512
    ) -> List[str]:
        """
        Translate multiple texts in batches for efficiency.
        
        Args:
            texts: List of texts to translate
            batch_size: Number of texts to process per batch (default: 8)
            max_length: Maximum length per translation (default: 512 tokens)
            
        Returns:
            List of translated texts (same order as input)
        """
        import re

        # Handle pandas Series or numpy arrays gracefully
        if hasattr(texts, "tolist"):
            texts = texts.tolist()

        if not texts:
            return []
        
        self._load_translator()
        sentence_batch_size = translator_sentence_batch_size
        sentence_max_length = translator_sentence_max_length

        # 1. Split paragraphs into sentences and track structural bounds
        paragraph_sentence_counts = []
        flat_sentences = []
        
        for text in texts:
            if not isinstance(text, str) or not text.strip():
                paragraph_sentence_counts.append(0)
                continue
            
            # Split Italian text by sentence boundaries (. ! ?) followed by whitespace
            sentences = [s.strip() for s in re.split(r'(?<=[.!?])\s+', text) if s.strip()]
            
            # DirectML Safety Net: If a sentence is massive (>50 words) with no periods, split by commas
            # safe_sentences = []
            # for s in sentences:
            #     if len(s.split()) > 50:
            #         safe_sentences.extend([sub.strip() for sub in re.split(r'(?<=[,;])\s+', s) if sub.strip()])
            #     else:
            #         safe_sentences.append(s)
            
            paragraph_sentence_counts.append(len(sentences))
            flat_sentences.extend(sentences)
            
        # Defensive check if input list contains only empty strings
        if not flat_sentences:
            return ["" if not isinstance(t, str) else t for t in texts]
        
        
        flat_translated_sentences = []
        total_sentences = len(flat_sentences)
        
        # 2. Process all extracted sentences inside optimized batches
        for i in range(0, total_sentences, sentence_batch_size):
            batch = flat_sentences[i:i + sentence_batch_size]
            batch_num = i // sentence_batch_size + 1
            total_batches = (total_sentences + sentence_batch_size - 1) // sentence_batch_size
            
            try:
                logger.debug(f"Translating sentence batch {batch_num}/{total_batches} ({len(batch)} sentences)...")
                
                # Mask entities before translation
                if HAS_SPACY:
                    masked_batch = []
                    batch_mappings = []
                    for s in batch:
                        masked_s, mapping = self._mask_entities(s)
                        masked_batch.append(masked_s)
                        batch_mappings.append(mapping)

                # Tokenize batch with the safe sentence length
                inputs = self.tokenizer(
                    masked_batch if HAS_SPACY else batch, 
                    return_tensors="pt", 
                    padding=True, 
                    max_length=sentence_max_length, 
                    truncation=True
                )
                
                # Move inputs to device
                if self.device != "cpu":
                    inputs = {k: v.to(self.device) for k, v in inputs.items()}
                
                # Generate translations using Beam Search to prevent hallucination loops
                with torch.no_grad():
                    translated_ids = self.model.generate(
                        **inputs,
                        max_length=sentence_max_length,
                        num_beams=4,           # Forces logical sentence construction; prevents repeating loops
                        early_stopping=True,    # Stops processing immediately when the sentence ends
                        # repetition_penalty=2.0,      # STRICT: Destroys single-token loops (e.g., '......')
                        # no_repeat_ngram_size=3,      # STRICT: Destroys phrase loops (e.g., 'I don't... I don't...')
                        # pad_token_id=self.tokenizer.pad_token_id,
                        # eos_token_id=self.tokenizer.eos_token_id
                    )
                
                # Decode batch and unmask entities
                for idx, translated_id in enumerate(translated_ids):
                    translated_text = self.tokenizer.decode(translated_id, skip_special_tokens=True)
                    if HAS_SPACY:
                        # Unmask entities after translation
                        translated_text = self._unmask_entities(translated_text, batch_mappings[idx])
                    flat_translated_sentences.append(translated_text)
                
                logger.debug(f"Sentence batch {batch_num}/{total_batches} completed")
            except Exception as e:
                logger.warning(f"Sentence batch translation failed: {e}. Falling back to original text segments.")
                flat_translated_sentences.extend(batch)
                
        # 3. Reconstruct the translated sentences back into their parent paragraphs
        translated_paragraphs = []
        sentence_idx = 0
        
        for count in paragraph_sentence_counts:
            if count == 0:
                translated_paragraphs.append("")
                continue
            
            # Extract the specific sentences belonging to this paragraph index
            para_sentences = flat_translated_sentences[sentence_idx : sentence_idx + count]
            translated_paragraphs.append(" ".join(para_sentences))
            sentence_idx += count
            
        logger.info(f"Successfully translated {len(texts)} long text entries via sentence-level splitting.")
        return translated_paragraphs
    
    def translate_with_metadata(
        self,
        texts: List[str],
        observation_dates: List[str],
        batch_size: int = 8
    ) -> Tuple[List[str], List[Tuple[str, str]]]:
        """
        Translate texts while preserving metadata (dates, indices).
        
        Args:
            texts: List of texts to translate
            observation_dates: List of observation dates (one per text)
            batch_size: Batch size for translation
            
        Returns:
            Tuple of:
            - List of translated texts
            - List of tuples (original_text, translated_text) for traceability
        """
        translated_texts = self.translate_batch(texts, batch_size=batch_size)
        metadata = list(zip(texts, translated_texts))
        return translated_texts, metadata


    def _mask_entities(self, text: str) -> Tuple[str, dict]:
        """Replaces proper nouns with neutral placeholders."""
        if not self.nlp:
            return text, {}
        
        doc = self.nlp(text)
        masked_text = text
        mapping = {}
        
        # Extract Persons (PER) and Organizations (ORG)
        entities = [ent.text for ent in doc.ents if ent.label_ in ['PER', 'ORG']]
        # Sort by length descending to avoid partial matches (e.g., replacing "Giuseppe" before "Giuseppe Castagna")
        entities = sorted(list(set(entities)), key=len, reverse=True)
        
        for i, ent in enumerate(entities):
            placeholder = f"NAMEX{i}X" 
            mapping[placeholder] = ent
            # Replace exact words using word boundaries
            masked_text = re.sub(rf'\b{re.escape(ent)}\b', placeholder, masked_text)
            
        return masked_text, mapping

    def _unmask_entities(self, text: str, mapping: dict) -> str:
        """Restores the original proper nouns from placeholders."""
        unmasked_text = text
        for placeholder, original_entity in mapping.items():
            # Case-insensitive replacement in case the translator alters placeholder casing
            pattern = re.compile(re.escape(placeholder), re.IGNORECASE)
            unmasked_text = pattern.sub(original_entity, unmasked_text)
        return unmasked_text

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
