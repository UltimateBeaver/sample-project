from env_config import *
# Import LLM and Embeddings models using LangChain wrappers
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_ollama import ChatOllama, OllamaEmbeddings

# ---------------------------------------------------------------------------
# Shared ChatOpenAI defaults
# ---------------------------------------------------------------------------
_CHAT_DEFAULTS = dict(
    temperature=0,
    max_tokens=None,
    timeout=None,
    max_retries=2,
)

# ---------------------------------------------------------------------------
# LLM models
# ---------------------------------------------------------------------------
# OpenAI LLM
"""
model_gpt_oss120b = ChatOpenAI(
    api_key=openai_api_key,
    model="openai/gpt-oss-120b",
    **_CHAT_DEFAULTS,
)
model_gpt4o_mini = ChatOpenAI(
    api_key=openai_api_key,
    model="gpt-4o-mini",
    **_CHAT_DEFAULTS,
)
"""

# --- Google Gemma (via Together AI) ----------------------------------------
model_gemma3_27b = ChatOpenAI(
    api_key=together_api_key,
    base_url=together_api_base,
    model="google/gemma-3-27b-it",
    **_CHAT_DEFAULTS,
)
model_gemma3_4b = ChatOpenAI(
    api_key=together_api_key,
    base_url=together_api_base,
    model="google/gemma-3-4b-it",
    **_CHAT_DEFAULTS,
)

# --- Meta Llama (via Together AI) ------------------------------------------
model_llama3_3_70b = ChatOpenAI(
    api_key=together_api_key,
    base_url=together_api_base,
    model="meta-llama/Llama-3.3-70B-Instruct-Turbo-Free",   # free tier
    **_CHAT_DEFAULTS,
)
model_llama3_1_8b = ChatOpenAI(
    api_key=together_api_key,
    base_url=together_api_base,
    model="meta-llama/Meta-Llama-3.1-8B-Instruct-Turbo",
    **_CHAT_DEFAULTS,
)

# --- Mistral (via Together AI) ---------------------------------------------
model_mistral_7b = ChatOpenAI(
    api_key=together_api_key,
    base_url=together_api_base,
    model="mistralai/Mistral-7B-Instruct-v0.3",
    **_CHAT_DEFAULTS,
)

# --- Local / Ollama (using native Ollama API) -------------------
model_ollama_gemma3 = ChatOllama(
    model="gemma3:1b",
    base_url=ollama_base_url,
    temperature=0,
)
model_ollama_llama3 = ChatOllama(
    model="llama3.2:3b",
    base_url=ollama_base_url,
    temperature=0,
)

# ---------------------------------------------------------------------------
# Embedding models
# ---------------------------------------------------------------------------
"""
embeddings_model_text_embedding3large = OpenAIEmbeddings(
    api_key=openai_api_key,
    model="text-embedding-3-large",     # 3072-dim, best quality
)
embeddings_text_embedding_3_small = OpenAIEmbeddings(
    api_key=openai_api_key,
    model="text-embedding-3-small",     # 1536-dim, cheaper / faster
)
"""

# --- Together AI embeddings (BAAI/bge, free tier) --------------------------
embeddings_bge_large = OpenAIEmbeddings(
    api_key=together_api_key,
    base_url=together_api_base,
    model="BAAI/bge-large-en-v1.5",     # 1024-dim
)
embeddings_bge_base = OpenAIEmbeddings(
    api_key=together_api_key,
    base_url=together_api_base,
    model="BAAI/bge-base-en-v1.5",      # 768-dim, lighter
)

# --- Local / Ollama embeddings (using native Ollama API) -----------
embeddings_ollama_nomic = OllamaEmbeddings(
    model="nomic-embed-text",           # run: ollama pull nomic-embed-text
    base_url=ollama_base_url,
)