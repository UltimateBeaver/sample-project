#!/usr/bin/env bash

# Exit immediately if a command exits with a non-zero status
set -e

# ---------------------------------------------------------
# STEP 0: LOAD AND EXPORT VARIABLES FROM .env
# ---------------------------------------------------------
if [ -f .env ]; then
    set -a
    source .env
    set +a
else
    echo -e "\033[0;31mError: .env file not found in the current directory!\033[0m"
    exit 1
fi

# Verify required sync variables exist
if [ -z "$HPC_USER" ] || [ -z "$HPC_HOST" ]; then
    echo -e "\033[0;31mError: HPC_USER or HPC_HOST is not defined in your .env file.\033[0m"
    exit 1
fi

# --- CONFIGURATION (Mapped from pull-from-HPC.ps1) ---
LOCAL_PROJECT_ROOT="." 

# Remote paths on the HPC cluster
REMOTE_DUMP="~/thesis-project/sample-project/neo4j/data/neo4j.dump"
REMOTE_LOGS="~/thesis-project/sample-project/logs"

# Local paths mapped from project root
LOCAL_LOGS_DIR="$LOCAL_PROJECT_ROOT/logs"
LOCAL_DOCKER_DATA="$LOCAL_PROJECT_ROOT/docker/data"     
LOCAL_TEMP_IMPORT="$LOCAL_PROJECT_ROOT/docker/import"   

# Local container identifier
DOCKER_CONTAINER_NAME="neo4j"

# Color Codes for Pretty Output
CYAN='\033[1;36m'
YELLOW='\033[1;33m'
GREEN='\033[0;32m'
GRAY='\033[0;90m'
NC='\033[0m' # No Color

clear
echo -e "${CYAN}=============================================${NC}"
echo -e "${CYAN}      HPC Data & Logs Sync Automation        ${NC}"
echo -e "${CYAN}=============================================${NC}"

# ---------------------------------------------------------
# STEP 1: RAW OVERWRITE LOGS
# ---------------------------------------------------------
echo -e "\n${YELLOW}[1/4] Overwriting local logs directory...${NC}"
if [ -d "$LOCAL_LOGS_DIR" ]; then
    echo -e "${GRAY}Removing old logs directory to ensure a raw overwrite...${NC}"
    rm -rf "$LOCAL_LOGS_DIR"
fi

# Download the entire directory structure down cleanly
scp -r "${HPC_USER}@${HPC_HOST}:${REMOTE_LOGS}" "$LOCAL_PROJECT_ROOT"
echo -e "${GREEN}Logs directory completely overwritten.${NC}"

# ---------------------------------------------------------
# STEP 2: DOWNLOAD NEO4J DUMP
# ---------------------------------------------------------
echo -e "\n${YELLOW}[2/4] Downloading neo4j.dump from cluster...${NC}"
mkdir -p "$LOCAL_TEMP_IMPORT"

scp "${HPC_USER}@${HPC_HOST}:${REMOTE_DUMP}" "$LOCAL_TEMP_IMPORT/neo4j.dump"
echo -e "${GREEN}Dump file downloaded successfully.${NC}"

# ---------------------------------------------------------
# STEP 3: STOP RUNNING CONTAINER
# ---------------------------------------------------------
echo -e "\n${YELLOW}[3/4] Stopping local container '$DOCKER_CONTAINER_NAME' to avoid file locks...${NC}"
# Use || true to prevent script crash if the container is already stopped
docker stop "$DOCKER_CONTAINER_NAME" || true

# ---------------------------------------------------------
# STEP 4: IMPORT DATA & RESTORE
# ---------------------------------------------------------
echo -e "\n${YELLOW}[4/4] Executing admin database restoration tool...${NC}"

# Run the disposable one-off loader container
docker run --rm \
  -v "${LOCAL_DOCKER_DATA}:/data" \
  -v "${LOCAL_TEMP_IMPORT}:/import" \
  neo4j:latest \
  neo4j-admin database load neo4j --from-path=/import --overwrite-destination=true

# ---------------------------------------------------------
# STEP 5: CLEANUP AND START
# ---------------------------------------------------------
echo -e "\n${YELLOW}Restarting your local Neo4j environment...${NC}"
docker start "$DOCKER_CONTAINER_NAME"

# Clean up the large dump file locally to save storage space
if [ -f "$LOCAL_TEMP_IMPORT/neo4j.dump" ]; then
    rm -f "$LOCAL_TEMP_IMPORT/neo4j.dump"
fi

echo -e "\n${CYAN}=============================================${NC}"
echo -e "${GREEN} SUCCESS! Local IDE Logs and Graph are Fresh! ${NC}"
echo -e "${CYAN}=============================================${NC}"