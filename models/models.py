from typing import Union

from langchain_ollama import ChatOllama, OllamaEmbeddings
from .models_config import *

# ---------------------------------------------------------------------------
# Default getters  (used by main.py)
# ---------------------------------------------------------------------------
# Change these two lines to switch the active configuration globally.
_DEFAULT_LLM        = model_llamacpp_gemma4                     # local llm backend config (with format="json" for valid JSON output)
_DEFAULT_LLM_NO_REASONING = model_llamacpp_gemma4_no_reasoning  # local llm backend config with reasoning disabled (for quintuples extraction)
_DEFAULT_EMBEDDINGS = embeddings_llamacpp_nomic                 # local embeddings backend config


def get_default_model() -> Union[ChatOllama, ChatOpenAI]:
    """Return the default LLM model instance."""
    return _DEFAULT_LLM

def get_default_model_no_reasoning() -> Union[ChatOllama, ChatOpenAI]:
    """Return the default LLM model instance with reasoning disabled."""
    return _DEFAULT_LLM_NO_REASONING

def get_default_embedding_model() -> Union[OllamaEmbeddings, OpenAIEmbeddings]:
    """Return the default embeddings model instance."""
    return _DEFAULT_EMBEDDINGS