#!/usr/bin/env bash
#SBATCH --job-name=Evaluation_Exhaustivity
#SBATCH --nodes=1                     # Request 1 compute node
#SBATCH --ntasks=1                    # 1 main task execution
#SBATCH --cpus-per-task=4             # Request 4 CPU cores for data processing
#SBATCH --mem=32GB                    # Request 32 GB system memory
#SBATCH --gres=gpu:2                  # Request 2 GPU (Required for Gemma 4)
#SBATCH --time=0-8:00:00             # Max runtime (Hours: 8 hours)
#SBATCH --partition=gpu_a40           # GPU partition on the cluster
#SBATCH --output=logs/exhaustivity_stdout.log    # Standard output log file
#SBATCH --error=logs/exhaustivity_stderr.log     # Standard error log file

# =========================================================================
# 1. Environment & Path Initialization
# =========================================================================
module purge
module load miniconda3/3.13.25
module load gcc/12.4.0
module load nvhpc/25.1

# Use the cluster's NVHPC path to inject CUDA runtime and math libraries
if [ -n "$NVHPC_ROOT" ]; then
    export LD_LIBRARY_PATH="$NVHPC_ROOT/cuda/lib64:$NVHPC_ROOT/math_libs/lib64:$LD_LIBRARY_PATH"
fi

# Move old logs (except the current ones) into a subfolder
# cd $HOME/thesis-project/sample-project/itext2kg_atom/evaluation/_slurm_scripts/logs
# mkdir -p old
# find . -maxdepth 1 -type f ! -name "*$(squeue -u $(whoami) -h -o '%A')*" -exec mv -t ./old {} +
# # mv *.log ./old

# Copy the whole project in $SCRATCH_FLASH filesystem
# echo "Copying required files from $HOME to $SCRATCH_FLASH..."
# mkdir -p $SCRATCH_FLASH/thesis-project
# cp -r $HOME/thesis-project/sample-project $SCRATCH_FLASH/thesis-project
echo "Syncing required project files from $HOME to $SCRATCH_FLASH..."
rsync -av --exclude='.git' --exclude='logs' --exclude='itext2kg_atom/evaluation/_slurm_scripts/logs' \
$HOME/thesis-project/sample-project/ \
$SCRATCH_FLASH/thesis-project/sample-project/

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
# echo "Launching Neo4j container via Apptainer..."
# # Set the database username/password matching your .env configurations
# export APPTAINERENV_NEO4J_AUTH="neo4j/password"
# mkdir -p $SCRATCH_FLASH/thesis-project/sample-project/neo4j/data $SCRATCH_FLASH/thesis-project/sample-project/neo4j/logs
# apptainer run --writable-tmpfs --bind $SCRATCH_FLASH/thesis-project/sample-project/neo4j/data:/data --bind $SCRATCH_FLASH/thesis-project/sample-project/neo4j/logs:/logs ./neo4j.sif &
# NEO4J_PID=$!

echo "Launching llama.cpp Model and Embedding Servers..."
chmod +x start-llama-servers.sh
./start-llama-servers.sh

# Give the background engines enough time to load the GGUF models into VRAM
echo "Waiting 30 seconds for infrastructure to boot completely..."
sleep 30

# =========================================================================
# 3. Run Core Python Evaluation test
# =========================================================================
# Move into the evaluation tests directory
cd $SCRATCH_FLASH/thesis-project/sample-project/itext2kg_atom/evaluation

# Tell Python where the root directory of your project is
export PYTHONPATH=$SCRATCH_FLASH/thesis-project/sample-project:$PYTHONPATH

# Add the slurm config file to this scope (to get $MODEL_POSTFIX value)
source ./_slurm_scripts/_slurm_config.env
echo "Starting evaluation > Exhaustivity test"
python ./exhaustivity/factoids_extraction_nyt.py -p $MODEL_POSTFIX
python ./exhaustivity/quintuples_extraction_nyt.py -p $MODEL_POSTFIX
python ./exhaustivity/quintuples_extraction_nyt_from_factoids.py -p $MODEL_POSTFIX
# Print out results (json, png, PDF)
python ./exhaustivity/plot_exhaustivity_factoids.py --force-recalculate
python ./exhaustivity/plot_exhaustivity_quintuples.py --force-recalculate

# =========================================================================
# 4. Graceful Cleanup of Background Services
# =========================================================================
echo "Terminating all background cluster services gracefully..."

# Stop the Apptainer database container
# kill $NEO4J_PID

# Give the Neo4j engine enough time to safely flush transactions and release file locks
# echo "Waiting 15 seconds for Neo4j files to completely close..."
# sleep 15

# Stop both detached llama-server instances (ports 8080 and 8081)
pkill llama-server

# Export the generated Knowledge Graphs using neo4j-admin image
# apptainer exec \
#     --bind $SCRATCH_FLASH/thesis-project/sample-project/neo4j/data:/data \
#     docker://neo4j:latest \
#     neo4j-admin database dump neo4j --to-path=/data --overwrite-destination=true

# Copy modified files on $SCRATCH_FLASH back to $HOME
rsync -av --exclude '.git' --exclude 'venv' $SCRATCH_FLASH/thesis-project/sample-project/ $HOME/thesis-project/sample-project/

echo "Job completed successfully!"