# sample-project
Sample repo for master's degree thesis

# Requirements
* Python (3.9 or greater) at https://www.python.org/downloads/
* Docker at https://www.docker.com/
* Ollama at https://ollama.com/download (used as a temporarily free model)
* At least 16 GB of GPU VRAM memory (to execute gemma4 model)

# Installation
1. Open up a terminal (Windows users must use Powershell), move to your desired directory and copy-paste the following commands:
    ```shell
    git clone https://github.com/UltimateBeaver/sample-project.git
    cd sample-project
    python -m venv venv
    venv/Scripts/activate
    pip install -r requirements.txt
    ```
2. Create a `.env` file on the root directory of the project. Insert your own API keys
    ```bash
    # Hugging Face Token (enables higher rate limits and faster downloads. Sign in to Hugging Face and create a token at https://huggingface.co/settings/tokens)
    HF_TOKEN=your-hugging-face-token
    # OpenaiAPI (llama.cpp)
    OPENAI_API_BASE=http://localhost:8080/v1
    LLAMACPP_EMBED_BASE=http://localhost:8081/v1
    LLAMACPP_PATH_MODEL="path/to/your/model.gguf"
    LLAMACPP_PATH_EMBEDDINGS_MODEL="path/to/your/embedding_model.gguf"
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

# Bugfix Checklist
- [x] Self loop relationships without any sense
- [x] Entities with empty names
- [ ] Redundant relationships
