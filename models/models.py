from typing import Union

from langchain_ollama import ChatOllama, OllamaEmbeddings
from .models_config import *

# ---------------------------------------------------------------------------
# Default getters  (used by main.py)
# ---------------------------------------------------------------------------
# Change these two lines to switch the active configuration globally.
_DEFAULT_LLM        = model_llamacpp_gemma4  # local Ollama (with format="json" for valid JSON output)
_DEFAULT_EMBEDDINGS = embeddings_llamacpp_nomic    # local Ollama


def get_default_model() -> Union[ChatOllama, ChatOpenAI]:
    """Return the default LLM model instance."""
    return _DEFAULT_LLM


def get_default_embedding_model() -> Union[OllamaEmbeddings, OpenAIEmbeddings]:
    """Return the default embeddings model instance."""
    return _DEFAULT_EMBEDDINGS