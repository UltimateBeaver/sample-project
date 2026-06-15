# sample-project
Sample repo for master's degree thesis

# Requirements
* Python (3.10 or greater) at https://www.python.org/downloads/
* Docker at https://www.docker.com/
* Ollama at https://ollama.com/download (used as a temporarily free model)
* At least 16 GB of GPU VRAM memory (to execute gemma4 model)

# Installation
1. Open up a terminal (Windows users must use Powershell), move to your desired directory and copy-paste the following commands:
    ```shell
    git clone https://github.com/UltimateBeaver/sample-project.git
    cd sample-project
    # Replace 3.12 with your actual installed version! Or alternatively use: `python -m venv venv`
    py -3.12 -m venv venv
    venv/Scripts/activate
    # Upgrade pip
    python -m pip install --upgrade pip
    pip install -r requirements.txt
    ```

    **CAUTION**: to avoid the 100% CPU fallback, Windows users who have an **AMD GPU** with Rocm library installed, must choose one of the following options:
    * Option A (*Suggested*):
        1. install ROCm Python environment libraries [https://rocm.docs.amd.com/projects/radeon-ryzen/en/latest/docs/install/installrad/windows/install-pytorch.html]:
            ```shell
            pip install --no-cache-dir `
            https://repo.radeon.com/rocm/windows/rocm-rel-7.2.1/rocm_sdk_core-7.2.1-py3-none-win_amd64.whl `
            https://repo.radeon.com/rocm/windows/rocm-rel-7.2.1/rocm_sdk_devel-7.2.1-py3-none-win_amd64.whl `
            https://repo.radeon.com/rocm/windows/rocm-rel-7.2.1/rocm_sdk_libraries_custom-7.2.1-py3-none-win_amd64.whl `
            https://repo.radeon.com/rocm/windows/rocm-rel-7.2.1/rocm-7.2.1.tar.gz
            ```
        2. install torch compiled for ROCm:
            ```shell
            pip install --no-cache-dir `
            https://repo.radeon.com/rocm/windows/rocm-rel-7.2.1/torch-2.9.1%2Brocm7.2.1-cp312-cp312-win_amd64.whl `
            ```
    * Option B: install this alternative pytorch library, provided by Microsoft:
        ```shell
        pip install --force-reinstall torch==2.4.1
        pip install torch-directml
        ```
2. Create a `.env` file on the root directory of the project. Insert your own API keys
    ```bash
    # Hugging Face Token (enables higher rate limits and faster downloads. Sign in to Hugging Face and create a token at https://huggingface.co/settings/tokens)
    HF_TOKEN=your-hugging-face-token
    # OpenaiAPI (llama.cpp)
    LLAMA_CPP_MODEL_PORT=8080
    LLAMA_CPP_EMBED_PORT=8081
    OPENAI_API_BASE="http://localhost:${LLAMA_CPP_MODEL_PORT}/v1"
    LLAMACPP_EMBED_BASE="http://localhost:${LLAMA_CPP_EMBED_PORT}/v1"
    LLAMACPP_PATH_MODEL="path/to/your/model.gguf"
    LLAMACPP_PATH_EMBEDDINGS_MODEL="path/to/your/embedding_model.gguf"
    # number of parallel sequences to decode (default: 1)
    LLAMACPP_MODEL_NUM_PARALLEL_SLOTS=1
    # Represents the total global pool shared across all parallel slots
    LLAMACPP_MODEL_CONTEXT_SIZE=32768
    LLAMACPP_EMBED_CONTEXT_SIZE=2048
    ### Langchain Output Parser: Provider-specific configurations
    # ProviderType.OPENAI
    PROVIDER_OPENAI_MAX_ELEMENTS_PER_BATCH=8
    PROVIDER_OPENAI_MAX_TOKENS_PER_BATCH=8192
    PROVIDER_OPENAI_MAX_CONTEXT_WINDOW=16384
    # ProviderType.OLLAMA
    PROVIDER_OLLAMA_MAX_ELEMENTS_PER_BATCH=8
    PROVIDER_OLLAMA_MAX_TOKENS_PER_BATCH=8192
    PROVIDER_OLLAMA_MAX_CONTEXT_WINDOW=16384
    ###
    # Max. number of layers to store in VRAM, either an exact number, 'auto', or 'all' (default: auto)
    LLAMACPP_MODEL_NGL=99
    LLAMACPP_EMBED_NGL=99
    OPENAI_API_KEY=llama_cpp
    # TogetherAPI (Currently not used)
    TOGETHER_API_BASE=https://api.together.xyz/v1
    TOGETHER_API_KEY=your-actual-key-here
    # Ollama
    OLLAMA_BASE_URL=http://localhost:11434
    # Neo4j
    NEO4J_URI=bolt://localhost:7687
    NEO4J_USERNAME=neo4j
    NEO4J_PASSWORD=password
    # Number of dataset rows to process (for quick testing)
    # Delete this variable or set it to 0 to process all rows
    NUM_ROWS_TO_PROCESS=10
    # Paths for document parsing
    DOC_PARSER_INPUT_EXCEL_PATH=./data/Annotazioni_1.xlsx
    DOC_PARSER_OUTPUT_EXCEL_PATH=./data/Annotazioni_1_with_factoids.xlsx
    DOC_PARSER_ENABLE_PARALLEL_PROCESSING=false
    # Batch size for document parsing (number of paragraphs to process in parallel)
    DOC_PARSER_BATCH_SIZE=2
    # Column names in the input Excel file
    COLUMN_NAME_DATE=DATA
    COLUMN_NAME_PARAGRAPH=ARTICOLO
    # Translation settings
    INPUT_LANGUAGE=it
    TRANSLATION_MODEL_NAME=it-en
    ENABLE_TRANSLATION=true
    TRANSLATOR_SENTENCE_BATCH_SIZE=32
    TRANSLATOR_SENTENCE_MAX_LENGTH=256
    # Polito HPC ssh settings
    HPC_USER=your-ssh-username
    HPC_HOST=hpc-legionlogin.polito.it
    ```
    **Caution:**
    * Make sure to set DOC_PARSER_INPUT_EXCEL_PATH and DOC_PARSER_OUTPUT_EXCEL_PATH properly
    * Make sure COLUMN_NAME_DATE and COLUMN_NAME_PARAGRAPH are the same columns name you have in your dataset
    * Make sure that the dataset is contained on the **FIRST SHEET** of your excel file, otherwise the program will fail.

3. Run/download all the required containers (Neo4j) through the following commands:
    ```bash
    # Start the containers (install image if it does not exists)
    docker compose up -d
    # Stop the containers
    docker compose stop
    # Stopping and removing containers
    docker compose down
    # Remove everything (including volumes)
    docker compose down -v
    ```
4. LLM and embedding model infrastructure:
* If you use Ollama, as default settings in models/models.py, download the required models:
    ```bash
    # Default llm model
    ollama pull gemma4:e4b
    # Default embedding model
    ollama pull nomic-embed-text:latest
    ```
    Note: you can download other ollama models from here: https://ollama.com/search
    <br>Ollama will use ChatOllama Langchain API.

* If you want to use llama.cpp, you have to download the right docker image (or directly the llama.cpp binaries) that match your OS and GPU. A bare metal setup  is provided below (tested with Windows 11 with AMD Radeon GPU):
    1. Go to the official llama.cpp https://github.com/ggml-org/llama.cpp/releases and look for the newest release. Download the one that matches your hardware (in my case a Windows zip file compiled for ROCm/HIP)
    2. Using powershell, download your chosen model in GGUF format (e.g., a quantized version of Gemma 4) from Hugging Face through the following command:
        ```powershell
        curl -L -o llm_model.gguf "https://lmstudio.ai/models/google/gemma-4-e4b"
        curl -L -o embedding_model.gguf "https://huggingface.co/nomic-ai/nomic-embed-text-v2-moe-GGUF"
        ```
        **Then add the paths of these models into the two variables `LLAMACPP_PATH_MODEL` and `LLAMACPP_PATH_EMBEDDINGS_MODEL`, inside `.env` file**
    3. Extract the llama.cpp binaries in a folder you like. Add the path to this folder in your environment variables (on Windows add it to PATH, in control panel)
    4. Start the two servers with the following command:
        * Powershell
            ```powershell
            start-llama-servers.ps1
            ```
        * Bash
            ```bash
            chmod +x start-llama-servers.sh && ./start-llama-servers.sh
            # stop servers
            pkill llama-server
            ```
        Note: llama.cpp will use ChatOpenAI Langchain API.
        <br>Make sure that the port you are opening the server is the same as the one configured in `OPENAI_API_BASE` of your `.env` file

# Run the application
```bash
# Move to the python virtual environment (if not already there)
venv/Scripts/activate
# Make sure Docker and Ollama/llama_cpp_servers are running!
# Finally execute the app
python main.py
```

---
# Developer tips
- When adding a new package, add the definition also in pyproject.toml
- When changing models, please refer to [langchain_output_parser.py](./itext2kg_atom/itext2kg/llm_output_parsing/langchain_output_parser.py), updating PROVIDER_CONFIGS object. This is the default config for Ollama:
```python
ProviderType.OLLAMA: ProviderConfig(
    name="Ollama",
    max_elements_per_batch=32,    # Conservative for local GPU: smaller batches prevent OOM on 16GB VRAM
    max_tokens_per_batch=12000,   # Increased to 12K per request (local inference, not API limits)
    max_context_window=32768,   # Ollama context window
    max_pending_requests=None,   # Ollama doesn't have explicit pending request limits
    sleep_between_batches=0.1,   # 100ms between batches to prevent GPU thrashing
)
```
The following is the default config for llama.cpp:
```python
ProviderType.OPENAI: ProviderConfig(
    name="llama.cpp (Local)",
    max_elements_per_batch=8,    
    max_tokens_per_batch=8192,   # Very conservative token limit
    max_context_window=16384,    # Typical for local models
    max_pending_requests=None,
    #sleep_between_batches=0.1,   # Small delay between requests
),
```

- If you encounter crashes or instability issues of llama.cpp model server, change the num-parallel parameter to `-np 1`. 
<br>Processing multiple reasoning streams at the same time on a single local GPU heavily degrades individual latency. Running them sequentially is actually more practical because a single request gets 100% of your GPU's compute. -np 1 tells the engine to completely disable multi-slot context blending.

- to check if Cuda (AMD ROCm supported wheel) is supported, run the following commands:
    ```python
    pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/rocm6
    python
    import torch
    print(torch.cuda.is_available())  # Should return True!
    print(torch.cuda.get_device_name(0))  # Should print "AMD Radeon ..."
    quit()
    ```

# Bugfix Checklist
- [x] Self loop relationships without any sense
- [x] Entities with empty names
- [ ] Redundant relationships
- [ ] Torch DirectML hallucination issues, when using AMD Radeon GPU on Windows 11 (currently, in such scenario, the program uses 100% CPU for translating news paragraphs for maximising accuracy)
