#!/bin/bash

# 1. Load and export variables from .env file
# We use 'set -a' to automatically export all variables sourced from the file
set -a
source .env
set +a

# Fallback to standard 'llama-server' if the .env variable is missing
LLAMA_SERVER_EXEC=${LLAMACPP_SERVER_BIN:-llama-server}

# 2. Start the servers in the background
# We use nohup to ensure the processes don't die if you close the terminal
echo "Starting Model Server using: $LLAMA_SERVER_EXEC"
nohup "$LLAMA_SERVER_EXEC" -m "$LLAMACPP_PATH_MODEL" -c "$LLAMACPP_MODEL_CONTEXT_SIZE" -ngl "$LLAMACPP_MODEL_NGL" -fa on --port "$LLAMA_CPP_MODEL_PORT" -np "$LLAMACPP_MODEL_NUM_PARALLEL_SLOTS" --no-mmap --reasoning "$LLAMACPP_MODEL_REASONING" --chat-template-kwargs '{"enable_thinking":"$LLAMACPP_MODEL_THINKING"}' > $HOME/thesis-project/sample-project/logs/model.log 2>&1 &

echo "Starting Embedding Server using: $LLAMA_SERVER_EXEC"
nohup "$LLAMA_SERVER_EXEC" -m "$LLAMACPP_PATH_EMBEDDINGS_MODEL" -c "$LLAMACPP_EMBED_CONTEXT_SIZE" -ngl "$LLAMACPP_EMBED_NGL" --port "$LLAMA_CPP_EMBED_PORT" --embedding --no-mmap --pooling "$LLAMACPP_EMBED_POOLING"  > $HOME/thesis-project/sample-project/logs/embedding.log 2>&1 &

echo "Servers started in the background."
echo "Check $HOME/thesis-project/sample-project/logs/model.log and $HOME/thesis-project/sample-project/logs/embedding.log for output."