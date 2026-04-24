from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from .models_config import *

# ---------------------------------------------------------------------------
# Default getters  (used by main.py)
# ---------------------------------------------------------------------------
# Change these two lines to switch the active configuration globally.
_DEFAULT_LLM        = model_ollama_gemma3        # local Ollama
_DEFAULT_EMBEDDINGS = embeddings_ollama_nomic    # local Ollama


def get_default_model() -> ChatOpenAI:
    """Return the default LLM model instance."""
    return _DEFAULT_LLM


def get_default_embedding_model() -> OpenAIEmbeddings:
    """Return the default embeddings model instance."""
    return _DEFAULT_EMBEDDINGS