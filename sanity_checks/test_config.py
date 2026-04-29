# This file can be used to test that the configuration of models is correct and that they can be loaded without errors.

import asyncio
from models.models import get_default_model, get_default_embedding_model
from env_config import *

from urllib.parse import urljoin
import requests

def validate_ollama_connection():
    """Validate Ollama service is running and models are available."""
    try:
        response = requests.get(
            urljoin(ollama_base_url, "v1/models"),
            timeout=5
        )
        response.raise_for_status()
        models_list = response.json().get("data", [])
        available_models = [m["id"] for m in models_list]
        required_models = [get_default_model().model, get_default_embedding_model().model]
        missing = [m for m in required_models if m not in available_models]
        
        if missing:
            raise RuntimeError(
                f"Missing Ollama models: {missing}\n"
                f"Available: {available_models}\n"
                f"Run: ollama pull {' && ollama pull '.join(missing)}"
            )
        
        return True
    except Exception as e:
        raise RuntimeError(
            f"Cannot connect to Ollama at {ollama_base_url}\n"
            f"Error: {e}\n"
            f"Make sure Ollama is running: ollama serve"
        )

async def validate_models_config():
    print("Testing configuration...")
    
    # Test LLM
    llm = get_default_model()
    print(f"✅ LLM loaded: {type(llm).__name__}")
    print(f"   Model: {llm.model if hasattr(llm, 'model') else 'Unknown'}")
    
    # Test Embeddings
    embeddings = get_default_embedding_model()
    print(f"✅ Embeddings loaded: {type(embeddings).__name__}")
    print(f"   Model: {embeddings.model if hasattr(embeddings, 'model') else 'Unknown'}")
    
    # Test LLM call
    try:
        response = await asyncio.to_thread(
            llm.invoke,
            "Say 'Configuration OK' and nothing else."
        )
        print(f"✅ LLM call successful: {response.content[:50]}")
    except Exception as e:
        print(f"❌ LLM call failed: {e}")
    
    # Test embeddings call
    try:
        emb_result = await embeddings.aembed_query("test")
        print(f"✅ Embeddings call successful: shape {len(emb_result)}")
    except Exception as e:
        print(f"❌ Embeddings call failed: {e}")

async def validate_config():
    try:
        validate_ollama_connection()
        await validate_models_config()
        return True
    except RuntimeError as e:
        print(f"⚠️  SANITY CHECKS ERROR: {e}")
        return False