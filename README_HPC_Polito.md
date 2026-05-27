# sample-project
Sample repo for master's degree thesis

# Requirements
* Python (3.10 or greater) at https://www.python.org/downloads/
* Apptainer (docker replacement)
* Ollama at https://ollama.com/download (used as a temporarily free model)
* At least 16 GB of GPU VRAM memory (to execute gemma4 model)

# CAUTION for Windows users
In this guide there will be several steps in which you are required to copy-paste file contents. Windows puts CRLF as endline character. Unix puts LF. Make sure that the file content has only LF. Otherwise you will have conflicts during job execution on the cluster. Use tools like 'https://toolslick.com/conversion/text/dos-to-unix' or just set LF in your text editor when you open this file.

# Installation (HPC Polito)
1. Login through your SSH credentials, then copy-paste the following commands:
    ```shell
    cd ~
    mkdir -p thesis-project/models
    cd thesis-project
    git clone https://github.com/UltimateBeaver/sample-project.git
    cd sample-project
    
    # Create logs directory on $HOME
    mkdir -p $HOME/thesis-project/sample-project/logs
    
    module purge
    module load miniconda3/3.13.25
    rm -rf venv
    python -m venv venv
    source venv/bin/activate
    pip install --upgrade pip
    pip install -r requirements.txt
    ```
2. Create a `.env` file on the root directory of the project. Insert your own API keys
    ```bash
    # Hugging Face Token (enables higher rate limits and faster downloads. Sign in to Hugging Face and create a token at https://huggingface.co/settings/tokens)
    HF_TOKEN=your-hugging-face-token
    # OpenaiAPI (llama.cpp)
    OPENAI_API_BASE=http://localhost:8080/v1
    LLAMACPP_EMBED_BASE=http://localhost:8081/v1
    LLAMACPP_PATH_MODEL="$HOME/thesis-project/models/llm_model.gguf"
    LLAMACPP_PATH_EMBEDDINGS_MODEL="$HOME/thesis-project/models/embedding_model.gguf"
    # Path to the compiled llama-server executable binary
    LLAMACPP_SERVER_BIN="$HOME/thesis-project/llama.cpp/build/bin/llama-server"
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
    # Pull the official Neo4j Docker image and convert it into an Apptainer image file (.sif)
    apptainer pull neo4j.sif docker://neo4j:latest
    ```
4. LLM and embedding model infrastructure:
* If you use Ollama, as default settings in models/models.py, download the required models:
    ```bash
    # Install Ollama
    curl -L https://ollama.com/download/ollama-linux-amd64 -o ~/thesis-project/ollama
    chmod +x ~/thesis-project/ollama

    # Request an interactive session via SLURM
    srun --nodes=1 --tasks-per-node=1 --cpus-per-task=1 --time=01:00:00 --partition=cpu_sapphire --pty /bin/bash
    cd ~/thesis-project/ollama
    # Default llm model
    ollama pull gemma4:e4b
    # Default embedding model
    ollama pull nomic-embed-text:latest

    # Close SLURM session
    squeue -u $(whoami) -h -o "%A" | xargs -I {} scancel {}
    ```
    Note: you can download other ollama models from here: https://ollama.com/search
    <br>Ollama will use ChatOllama Langchain API.

* If you want to use llama.cpp, you have to download the right llama.cpp binaries that match your OS and GPU:
    1. Download your chosen models in GGUF format (e.g., a quantized version of Gemma 4) from Hugging Face through the following commands:
        ```bash
        # Request an interactive session via SLURM
        srun --nodes=1 --tasks-per-node=1 --cpus-per-task=1 --time=01:00:00 --partition=cpu_sapphire --pty /bin/bash
        # Make sure you are inside your activated python virtual environment
        cd ~/thesis-project/sample-project
        source venv/bin/activate
        pip install huggingface_hub
        cd ~/thesis-project
        mkdir -p models/.cache
        # LLM: https://huggingface.co/unsloth/gemma-4-E4B-it-GGUF
        # Embedding: https://huggingface.co/nomic-ai/nomic-embed-text-v2-moe-GGUF
        hf download unsloth/gemma-4-E4B-it-GGUF gemma-4-E4B-it-Q4_K_M.gguf --local-dir ~/thesis-project/models/.cache
        mv ~/thesis-project/models/.cache/*.gguf ~/thesis-project/models/llm_model.gguf
        hf download nomic-ai/nomic-embed-text-v2-moe-GGUF nomic-embed-text-v2-moe.Q8_0.gguf --local-dir ~/thesis-project/models/.cache
        mv ~/thesis-project/models/.cache/*.gguf ~/thesis-project/models/embedding_model.gguf 

        # Close SLURM session
        squeue -u $(whoami) -h -o "%A" | xargs -I {} scancel {}
        ```
        **Then add the paths of these models into the two variables `LLAMACPP_PATH_MODEL` and `LLAMACPP_PATH_EMBEDDINGS_MODEL`, inside `.env` file**
    2. Copy-paste the following commands to download and compile the latest llama.cpp release, with CUDA enabled, for NVIDIA GPUs:
        ```bash
        # Request an interactive session via SLURM
        srun --nodes=1 --tasks-per-node=1 --cpus-per-task=4 --gres=gpu:1 --time=01:30:00 --partition=gpu_a40 --pty /bin/bash
        
        # 1. Grab the project repository
        cd ~/thesis-project
        git clone https://github.com/ggerganov/llama.cpp

        # 2. Load the cluster's building modules and native CUDA toolkit
        module load gcc/12.4.0
        module load cmake/3.26
        module load nvhpc/25.1

        # 3. Configure and build using CMake with CUDA enabled
        cd ~/thesis-project/llama.cpp
        rm -rf build
        # Note: if the process is too long, add -DCMAKE_CUDA_ARCHITECTURES=86 to the following command to only build cuda libraries for a40's Ampere architecture
        cmake -B build -DGGML_CUDA=ON -DCMAKE_C_COMPILER=gcc -DCMAKE_CXX_COMPILER=g++ -DGGML_NATIVE=OFF -DGGML_AVX512=OFF
        cmake --build build --config Release -j 4

        # Close SLURM session
        squeue -u $(whoami) -h -o "%A" | xargs -I {} scancel {}
        ```
    3. Start the two servers with the following command:
        ```bash
        srun --nodes=1 --tasks-per-node=1 --cpus-per-task=4 --gres=gpu:1 --time=01:30:00 --partition=gpu_a40 --pty /bin/bash
        chmod +x start-llama-servers.sh && ./start-llama-servers.sh
        # stop servers
        pkill llama-server
        ```
        Note: llama.cpp will use ChatOpenAI Langchain API.
        <br>Make sure that the port you are opening the server is the same as the one configured in `OPENAI_API_BASE` of your `.env` file

5. Create a SLURM script and name it *submit.sh*. Add execution permissions with `chmod +x submit.sh`. Then copy and paste the following content into *submit.sh*:
    ```bash
    #!/usr/bin/env bash
    #SBATCH --job-name=thesis_pipeline
    #SBATCH --nodes=1                     # Request 1 compute node
    #SBATCH --ntasks=1                    # 1 main task execution
    #SBATCH --cpus-per-task=4             # Request 4 CPU cores for data processing
    #SBATCH --mem=32GB                    # Request 32 GB system memory
    #SBATCH --gres=gpu:1                  # Request 1 GPU (Required for Gemma 4)
    #SBATCH --time=0-02:00:00             # Max runtime (Hours: 2 hours)
    #SBATCH --partition=gpu_a40           # GPU partition on the cluster
    #SBATCH --output=logs/thesis_job_stdout_%j.log    # Standard output log file
    #SBATCH --error=logs/thesis_job_stderr_%j.log     # Standard error log file

    # =========================================================================
    # 1. Environment & Path Initialization
    # =========================================================================
    module purge
    module load miniconda3/3.13.25
    module load gcc/12.4.0
    module load nvhpc/25.1

    # Copy the whole project in $SCRATCH_FLASH filesystem
    echo "Copying required files from $HOME to $SCRATCH_FLASH..."
    mkdir -p $SCRATCH_FLASH/thesis-project
    cp -r $HOME/thesis-project/sample-project $SCRATCH_FLASH/thesis-project

    # Move into the directory where your project files, scripts, and .env exist
    cd $SCRATCH_FLASH/thesis-project/sample-project

    # Activate your local Python Virtual Environment
    source venv/bin/activate

    # CRITICAL: Add your freshly compiled llama.cpp folder to the system PATH 
    # so 'start-llama-servers.sh' can find and execute the 'llama-server' binary
    export PATH=$SCRATCH_FLASH/thesis-project/llama.cpp/build/bin:$PATH

    # =========================================================================
    # 2. Launch Background Infrastructure Services
    # =========================================================================
    echo "Launching Neo4j container via Apptainer..."
    mkdir -p $HOME/neo4j/data $HOME/neo4j/logs
    apptainer run --writable-tmpfs --bind $HOME/neo4j/data:/data --bind $HOME/neo4j/logs:/logs ./neo4j.sif &
    NEO4J_PID=$!

    echo "Launching llama.cpp Model and Embedding Servers..."
    chmod +x start-llama-servers.sh
    ./start-llama-servers.sh

    # Give the background engines enough time to load the GGUF models into VRAM
    echo "Waiting 30 seconds for infrastructure to boot completely..."
    sleep 30

    # =========================================================================
    # 3. Run Core Python Application Pipeline
    # =========================================================================
    echo "Starting main application pipeline execution (main.py)..."
    python main.py

    # =========================================================================
    # 4. Graceful Cleanup of Background Services
    # =========================================================================
    echo "Terminating all background cluster services gracefully..."

    # Stop the Apptainer database container
    kill $NEO4J_PID

    # Stop both detached llama-server instances (ports 8080 and 8081)
    pkill llama-server

    # Copy modified files on $SCRATCH_FLASH back to $HOME
    rsync -av --exclude '.git' $SCRATCH/thesis-project/sample-project/ $HOME/thesis-project/sample-project/

    echo "Job completed successfully!"
    ```
6. Run `sbatch submit.sh`. You can monitor your active queue status using `squeue -u $(whoami)`. To view the statistics of a completed job use `sacct -j <job id> --format=JobID,Start,End,Elapsed,NCPUS`. For detailed information about a running/pending job, use `scontrol show jobid=<job id>`. To cancel a job use `scancel <jobid>`. To cancel all jobs for your current account use `scancel -A $(whoami)`
    * *thesis_job_stdout_[JOBID].log*: Displays the standard printed pipeline metrics and updates.
    * *model.log*: Shows how the LLM model is loading into the GPU.
    * *embedding.log*: Shows how the embedding model is running.

# Run the application
```bash
# Move to the python virtual environment (if not already there)
source venv/bin/activate
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
    built-in rate limiting handle the rest
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
