#!/usr/bin/env bash
#SBATCH --job-name=Evaluation_Pipeline
#SBATCH --nodes=1                     # Request 1 compute node
#SBATCH --ntasks=1                    # 1 main task execution
#SBATCH --cpus-per-task=4             # Request 4 CPU cores for data processing
#SBATCH --mem=32GB                    # Request 32 GB system memory
#SBATCH --gres=gpu:2                  # Request 2 GPU (Required for Gemma 4)
#SBATCH --time=0-23:59:00             # Max runtime (Hours: 24 hours)
#SBATCH --partition=gpu_a40           # GPU partition on the cluster
# Note: --output and --error are now handled dynamically by start_eval_pipeline_wrapper.sh

# =========================================================================
# 1. Environment & Path Initialization
# =========================================================================
module purge
module load miniconda3/3.13.25
module load gcc/12.4.0
module load nvhpc/25.1

echo "Killing lingering infrastructure processes from previous runs..."
pkill -u $(whoami) -f llama-server || true
pkill -u $(whoami) -f neo4j || true
sleep 3

# Use the cluster's NVHPC path to inject CUDA runtime and math libraries
if [ -n "$NVHPC_ROOT" ]; then
    export LD_LIBRARY_PATH="$NVHPC_ROOT/cuda/lib64:$NVHPC_ROOT/math_libs/lib64:$LD_LIBRARY_PATH"
fi

# Move old logs (except the current ones) into a subfolder
# cd $HOME/thesis-project/sample-project/itext2kg_atom/evaluation/_slurm_scripts/logs
# mkdir -p old
# find . -maxdepth 1 -type f ! -name "*$(squeue -u $(whoami) -h -o '%A')*" -exec mv -t ./old {} +
# # mv *.log ./old

# Ensure target scratch directories exist BEFORE running rsync
echo "Creating target scratch directory structure at $SCRATCH_FLASH..."
mkdir -p "$SCRATCH_FLASH/thesis-project/sample-project"

# Copy the whole project in $SCRATCH_FLASH filesystem
# echo "Copying required files from $HOME to $SCRATCH_FLASH..."
# mkdir -p $SCRATCH_FLASH/thesis-project
# cp -r $HOME/thesis-project/sample-project $SCRATCH_FLASH/thesis-project
echo "Syncing required project files from $HOME to $SCRATCH_FLASH..."
rsync -av \
  --exclude='.git' \
  --exclude='logs' \
  --exclude='itext2kg_atom/evaluation/_slurm_scripts/logs' \
  --exclude='neo4j/data' \
  --exclude='neo4j/logs' \
  $HOME/thesis-project/sample-project/ \
  $SCRATCH_FLASH/thesis-project/sample-project/

# Move into the directory where your project files, scripts, and .env exist
cd $SCRATCH_FLASH/thesis-project/sample-project || { echo "Failed to cd to scratch directory"; exit 1; }

# Activate your local Python Virtual Environment
source venv/bin/activate

# CRITICAL: Add your freshly compiled llama.cpp folder to the system PATH 
# so 'start-llama-servers.sh' can find and execute the 'llama-server' binary
export PATH=$SCRATCH_FLASH/thesis-project/llama.cpp/build/bin:$PATH

# =========================================================================
# 2. Launch Background Infrastructure Services
# =========================================================================
# Re-create directories and wipe any residual artifacts from previous dirty runs
echo "Cleaning directories for Neo4j..."
mkdir -p neo4j/data neo4j/logs
rm -rf neo4j/data/*
rm -rf neo4j/logs/*

echo "Detecting container runtime..."
if command -v apptainer &> /dev/null; then
    CONTAINER_EXEC="apptainer"
    # Set the database username/password matching your .env configurations
    export APPTAINERENV_NEO4J_AUTH="neo4j/password"
elif command -v singularity &> /dev/null; then
    CONTAINER_EXEC="singularity"
    export SINGULARITYENV_NEO4J_AUTH="neo4j/password"
else
    echo "Error: Neither apptainer nor singularity is installed on this node." >&2
    exit 1
fi

echo "Launching Neo4j container via Apptainer..."
mkdir -p $SCRATCH_FLASH/thesis-project/sample-project/neo4j/data $SCRATCH_FLASH/thesis-project/sample-project/neo4j/logs
$CONTAINER_EXEC run --writable-tmpfs \
    --bind $SCRATCH_FLASH/thesis-project/sample-project/neo4j/data:/data \
    --bind $SCRATCH_FLASH/thesis-project/sample-project/neo4j/logs:/logs \
    $SCRATCH_FLASH/thesis-project/sample-project/neo4j.sif &
NEO4J_PID=$!

echo "Launching llama.cpp Model and Embedding Servers..."
chmod +x $HOME/thesis-project/sample-project/start-llama-servers.sh
chmod +x $SCRATCH_FLASH/thesis-project/sample-project/start-llama-servers.sh
./start-llama-servers.sh

# Give the background engines enough time to load the GGUF models into VRAM
echo "Waiting 30 seconds for infrastructure to boot completely..."
sleep 30

# =========================================================================
# 3. Run Core Python Evaluation test
# =========================================================================
# Add main project .env for accessing environment variables
source $SCRATCH_FLASH/thesis-project/sample-project/.env

# Move to the itext2kg_atom root directory
cd $SCRATCH_FLASH/thesis-project/sample-project/itext2kg_atom
echo "Removing previously tests results for starting fresh: $EVAL_OUTPUT_RESULTS_PATH and $EVAL_OUTPUT_DATASET_PATH ..."
rm -r -f $SCRATCH_FLASH/thesis-project/sample-project/itext2kg_atom/$EVAL_OUTPUT_RESULTS_PATH
rm -r -f $SCRATCH_FLASH/thesis-project/sample-project/itext2kg_atom/$EVAL_OUTPUT_DATASET_PATH

# Move into the evaluation tests directory
cd evaluation

# Tell Python where the root directory of your project is
export PYTHONPATH=$SCRATCH_FLASH/thesis-project/sample-project:$PYTHONPATH

# Add the slurm config file to this scope (to get $MODEL_POSTFIX value)
source ./_slurm_scripts/_slurm_config.env

echo "--- Running Exhaustivity Tests ---"
python ./exhaustivity/factoids_extraction_nyt.py -p $MODEL_POSTFIX
python ./exhaustivity/quintuples_extraction_nyt.py -p $MODEL_POSTFIX
python ./exhaustivity/quintuples_extraction_nyt_from_factoids.py -p $MODEL_POSTFIX
# Print out results (json, png, PDF)
python ./exhaustivity/plot_exhaustivity_factoids.py --force-recalculate
python ./exhaustivity/plot_exhaustivity_quintuples.py --force-recalculate
python ./exhaustivity/plot_combined_exhaustivity.py

# --- llama.cpp reboot ---
pkill llama-server
sleep 5
./start-llama-servers.sh
sleep 15
# ------------------------

echo "--- Running Latency Tests ---"
# To remove the whole cache execute:
# rm -rf $HOME/thesis-project/sample-project/itext2kg_atom/datasets/atom/my_test_datasets/cache
# rm -rf $SCRATCH_FLASH/thesis-project/sample-project/itext2kg_atom/datasets/atom/my_test_datasets/cache
python ./latency/test_graphiti.py

# --- llama.cpp reboot ---
pkill llama-server
sleep 5
./start-llama-servers.sh
sleep 15
# ------------------------

python ./latency/testing_atom.py
python ./latency/testing_itext2kg.py
python ./latency/plot_latency_comparison.py

# --- llama.cpp reboot ---
pkill llama-server
sleep 5
./start-llama-servers.sh
sleep 15
# ------------------------

echo "--- Running Merge Tests ---"
python ./merge/evaluate_atom_merge.py -p $MODEL_POSTFIX

echo "--- Running Quintuples Quality Tests ---"
python ./quintuples_quality/calculate_quintuples_quality.py -p $MODEL_POSTFIX

# --- llama.cpp reboot ---
pkill llama-server
sleep 5
./start-llama-servers.sh
sleep 15
# ------------------------

echo "--- Running Stability Tests ---"
python ./stability/calculate_stability.py --force-extraction -p $MODEL_POSTFIX
python ./stability/calculate_stability_jaccard.py -p $MODEL_POSTFIX

echo "--- Running Unsupervised Ragas Tests ---"
python ./unsupervised/eval_ragas.py -p $MODEL_POSTFIX

# =========================================================================
# 4. Graceful Cleanup of Background Services
# =========================================================================
echo "Terminating all background cluster services gracefully..."

# Stop the Apptainer database container
kill $NEO4J_PID

# Give the Neo4j engine enough time to safely flush transactions and release file locks
# echo "Waiting 15 seconds for Neo4j files to completely close..."
# sleep 15

# Stop both detached llama-server instances (ports 8080 and 8081)
pkill llama-server

# Move back to the root directory
cd $SCRATCH_FLASH/thesis-project/sample-project

# Export the generated Knowledge Graphs using neo4j-admin image
# apptainer exec \
#     --bind $SCRATCH_FLASH/thesis-project/sample-project/neo4j/data:/data \
#     docker://neo4j:latest \
#     neo4j-admin database dump neo4j --to-path=/data --overwrite-destination=true

# Copy modified files on $SCRATCH_FLASH back to $HOME, but exclude the massive database binary folders
rsync -av \
  --exclude '.git' \
  --exclude='logs' \
  --exclude 'venv' \
  --exclude 'neo4j/data' \
  --exclude 'neo4j/logs' \
  $SCRATCH_FLASH/thesis-project/sample-project/ \
  $HOME/thesis-project/sample-project/

# The previous command avoids to copy the whole logs of llama.cpp server as they are huge
# So here I'm just copying the last 1000 lines of both
tail -n 1000 $SCRATCH_FLASH/thesis-project/sample-project/itext2kg_atom/evaluation/_slurm_scripts/logs/eval_pipeline_stdout.log > $HOME/thesis-project/sample-project/itext2kg_atom/evaluation/_slurm_scripts/logs/eval_pipeline_stdout.log
tail -n 1000 $SCRATCH_FLASH/thesis-project/sample-project/itext2kg_atom/evaluation/_slurm_scripts/logs/eval_pipeline_stderr.log > $HOME/thesis-project/sample-project/itext2kg_atom/evaluation/_slurm_scripts/logs/eval_pipeline_stderr.log

echo "Job completed successfully!"