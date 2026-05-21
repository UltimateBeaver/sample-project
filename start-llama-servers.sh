#!/bin/bash

# 1. Load and export variables from .env file
# We use 'set -a' to automatically export all variables sourced from the file
set -a
source .env
set +a

# 2. Start the servers in the background
# We use nohup to ensure the processes don't die if you close the terminal
echo "Starting Model Server..."
nohup llama-server -m "$LLAMACPP_PATH_MODEL" -c 32768 -ngl 99 -fa on --port 8080 -np 1 > model.log 2>&1 &

echo "Starting Embedding Server..."
nohup llama-server -m "$LLAMACPP_PATH_EMBEDDINGS_MODEL" -c 2048 -ngl 99 --port 8081 --embedding > embedding.log 2>&1 &

echo "Servers started in the background."
echo "Check model.log and embedding.log for output."