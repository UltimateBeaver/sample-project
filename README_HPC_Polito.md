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
    # Install the specific torch build compiled for CUDA 12.4
    pip install torch --index-url https://download.pytorch.org/whl/cu124
    pip install -r requirements.txt
    ```
    <br>
    To check if Cuda (NVIDIA or AMD ROCm supported wheel) is supported, run the following commands:
    ```python
    python
    import torch
    print(torch.cuda.is_available())  # Should return True!
    print(torch.cuda.get_device_name(0))  # Should print the model of your detected GPU "NVIDIA ... or AMD ..."
    quit()
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
    LLAMACPP_PATH_MODEL="$HOME/thesis-project/models/llm_model.gguf"
    LLAMACPP_PATH_EMBEDDINGS_MODEL="$HOME/thesis-project/models/embedding_model.gguf"
    # Path to the compiled llama-server executable binary
    LLAMACPP_SERVER_BIN="$HOME/thesis-project/llama.cpp/build/bin/llama-server"
    # number of parallel sequences to decode (default: 1)
    LLAMACPP_MODEL_NUM_PARALLEL_SLOTS=1
    # Represents the total global pool shared across all parallel slots
    LLAMACPP_MODEL_CONTEXT_SIZE=32768
    LLAMACPP_EMBED_CONTEXT_SIZE=2048
    ### Langchain Output Parser: Provider-specific configurations
    ### NOTE: the following configs act as a safeguard to prevent exceeding the maximum context window of the LLM provider. If you encounter errors related to context window limits, consider adjusting these values.
    # ProviderType.OPENAI
    PROVIDER_OPENAI_MAX_ELEMENTS_PER_BATCH=8
    PROVIDER_OPENAI_MAX_TOKENS_PER_BATCH=20000
    PROVIDER_OPENAI_MAX_CONTEXT_WINDOW=32768
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
    COLUMN_NAME_SENTIMENT=SENTIMENTO
    # Translation settings
    ENABLE_TRANSLATION=true
    TRANSLATOR_BATCH_SIZE=8
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
    # Pull the official Neo4j Docker image and convert it into an Apptainer image file (.sif)
    apptainer pull neo4j.sif docker://neo4j:latest
    ```
4. LLM and embedding model infrastructure:
* If you use Ollama, as default settings in models/models.py, download the required models:
    ```bash
    # Install Ollama
    curl -L https://ollama.com/download/ollama-linux-amd64.tar.zst -o ~/thesis-project/ollama.tar.zst
    mkdir -p ~/thesis-project/ollama
    tar x -f ~/thesis-project/ollama.tar.zst -C ~/thesis-project/ollama

    # Request an interactive session via SLURM
    srun --nodes=1 --tasks-per-node=1 --cpus-per-task=1 --time=01:00:00 --partition=cpu_sapphire --pty /bin/bash
    cd ~/thesis-project/ollama/bin
    # Start the server in the background so you can pull models
    ./ollama serve &
    # Default llm model
    ./ollama pull gemma4:e4b
    # Default embedding model
    ./ollama pull nomic-embed-text:latest

    pkill "./ollama serve"
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

5. Inspect the SLURM script *submit.sh* and edit the #SBATCH directives to your needs. Add execution permissions with `chmod +x submit.sh`.

6. Run `sbatch submit.sh`. Some utilities commands are provided below:
    | Command    | Description |
    | -------- | ------- |
    | `squeue -u $(whoami)`  | To monitor your active queue status    |
    | `sacct -j <job id> --format=JobID,Start,End,Elapsed,NCPUS` | To view the statistics of a completed job     |
    | `scontrol show jobid=<job id>`    | Shows detailed information about a running/pending job    |
    | `scancel <jobid>`    | To cancel a job    |
    | `scancel -A $(whoami)`    | To cancel all jobs for your current account    |

    * *thesis_job_stdout_[JOBID].log*: Displays the standard printed pipeline metrics and updates.
    * *model.log*: Shows how the LLM model is loading into the GPU.
    * *embedding.log*: Shows how the embedding model is running.

7. Make sure you have properly set $HPC_USER and $HPC_HOST on `.env` file. Then move to a terminal on your host machine and double-check the following sections of `pull-from-HPC`. 
    ```bash
    $LOCAL_PROJECT_ROOT = "." # Relative path to local repo on your host

    # [...]
    # Local container identifier
    $DOCKER_CONTAINER_NAME = "neo4j"    # The exact name of neo4j docker container on your host
    ```
    Execute `pull-from-HPC.ps1` or `pull-from-HPC.sh` script, depending on your OS, to download Knowledge Graph dump database and application logs.



---
# Developer tips
- When adding a new package, add the definition also in pyproject.toml
- A few words about the LangchainOutputParser: depending on the LLMprovider you choose to use, there are different PROVIDER_CONFIGS. Make sure to properly set the following environment variables, before editing anything else:
    ```shell
    PROVIDER_<PROVIDER_NAME>_MAX_ELEMENTS_PER_BATCH=8
    PROVIDER_<PROVIDER_NAME>_MAX_TOKENS_PER_BATCH=8192
    PROVIDER_<PROVIDER_NAME>_MAX_CONTEXT_WINDOW=16384
    ```
    <br> These vars acts as a safeguard to prevent exceeding the maximum context window of the LLM provider.
    <br> The `$DOC_PARSER_BATCH_SIZE` and `$TRANSLATOR_BATCH_SIZE` env vars may also be greater than `$PROVIDER_<PROVIDER_NAME>_MAX_ELEMENTS_PER_BATCH`. That's because the safeguard does not apply only to the batch size, but also to the number of required token for the LLM query. See `LangchainOutputParser.count_tokens()` method.
    <br> Refer to [langchain_output_parser.py](./itext2kg_atom/itext2kg/llm_output_parsing/langchain_output_parser.py) for more details.
    <br> When changing models, please refer to, updating PROVIDER_CONFIGS object. This is the default config for Ollama:
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

- to check if Cuda is supported, run the following commands:
    ```python
    srun --nodes=1 --tasks-per-node=1 --cpus-per-task=4 --gres=gpu:1 --time=01:30:00 --partition=gpu_a40 --pty /bin/bash
    pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124
    python
    import torch
    print(torch.cuda.is_available())  # Should return True!
    print(torch.cuda.get_device_name(0))  # Should print "NVIDIA A40"
    quit()
    squeue -u $(whoami) -h -o "%A" | xargs -I {} scancel {}
    ```