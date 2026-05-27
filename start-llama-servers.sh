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
nohup "$LLAMA_SERVER_EXEC" -m "$LLAMACPP_PATH_MODEL" -c 32768 -ngl 99 -fa on --port 8080 -np 1 > ./logs/model.log 2>&1 &

echo "Starting Embedding Server using: $LLAMA_SERVER_EXEC"
nohup "$LLAMA_SERVER_EXEC" -m "$LLAMACPP_PATH_EMBEDDINGS_MODEL" -c 2048 -ngl 99 --port 8081 --embedding > ./logs/embedding.log 2>&1 &

echo "Servers started in the background."
echo "Check ./logs/model.log and ./logs/embedding.log for output."