# This file can be used to test that the configuration of models is correct and that they can be loaded without errors.

import asyncio
from itext2kg_atom.itext2kg.logging_config import get_logger
from models.models import get_default_model, get_default_embedding_model
from env_config import *

import requests

logger = get_logger(__name__)

def get_model_base_url(model):
    """Return the configured base URL for a LangChain ChatOpenAI/ChatOllama model."""
    if hasattr(model, "base_url") and getattr(model, "base_url"):
        return getattr(model, "base_url")

    if hasattr(model, "model_dump"):
        dump = model.model_dump()
        base_url = dump.get("openai_api_base") or dump.get("base_url")
        if base_url:
            return base_url

        if dump.get("openai_api_key"):
            return "https://api.openai.com"

    raise RuntimeError(
        f"Cannot determine a reachable base URL for model type {type(model).__name__}."
    )


def validate_model_base_url_connection(model, timeout=5):
    """Validate that a LangChain model's base URL is reachable."""
    base_url = get_model_base_url(model)
    try:
        response = requests.get(base_url, timeout=timeout)
        if response.status_code >= 500:
            raise RuntimeError(
                f"Service at {base_url} returned status code {response.status_code}."
            )
        return True
    except requests.RequestException as e:
        raise RuntimeError(
            f"Cannot connect to {type(model).__name__} base_url at {base_url}\n"
            f"Error: {e}"
        )

async def validate_models_config():
    logger.debug("Testing configuration...")
    
    # Test LLM
    llm = get_default_model()
    logger.debug(f"✅ LLM loaded: {type(llm).__name__}")
    logger.debug(f"   Model: {llm.model if hasattr(llm, 'model') else 'Unknown'}")
    
    # Test Embeddings
    embeddings = get_default_embedding_model()
    logger.debug(f"✅ Embeddings loaded: {type(embeddings).__name__}")
    logger.debug(f"   Model: {embeddings.model if hasattr(embeddings, 'model') else 'Unknown'}")
    
    # Test LLM call
    try:
        response = await asyncio.to_thread(
            llm.invoke,
            "Say 'Configuration OK' and nothing else."
        )
        logger.debug(f"✅ LLM call successful: {response.content[:50]}")
    except Exception as e:
        logger.error(f"❌ LLM call failed: {e}")
        raise RuntimeError()
    
    # Test embeddings call
    try:
        emb_result = await embeddings.aembed_query("test")
        logger.debug(f"✅ Embeddings call successful: shape {len(emb_result)}")
    except Exception as e:
        logger.error(f"❌ Embeddings call failed: {e}")
        raise RuntimeError()

async def validate_config():
    try:
        validate_model_base_url_connection(get_default_model())
        validate_model_base_url_connection(get_default_embedding_model())
        await validate_models_config()
        return True
    except RuntimeError as e:
        logger.error(f"⚠️  SANITY CHECKS ERROR: {e}")
        return False