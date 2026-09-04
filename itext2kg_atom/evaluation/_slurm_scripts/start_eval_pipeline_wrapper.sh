#!/usr/bin/env bash

# Load the configuration variables
if [ -f "_slurm_config.env" ]; then
    source _slurm_config.env
else
    echo "❌ Error: _slurm_config.env not found!"
    exit 1
fi

# Define and create absolute log directory in $HOME
LOG_DIR="$SCRATCH_FLASH/thesis-project/sample-project/itext2kg_atom/evaluation/_slurm_scripts/logs"
mkdir -p "$LOG_DIR"
mkdir -p "$HOME/thesis-project/sample-project/itext2kg_atom/evaluation/_slurm_scripts/logs"

# Submit jobs, overriding their default #SBATCH values
# We also use --export to pass the MODEL_POSTFIX to the compute node environment
sbatch \
  --nodes="$NODES" \
  --ntasks="$NTASKS" \
  --cpus-per-task="$CPUS_PER_TASK" \
  --mem="$MEM" \
  --gres="$GRES" \
  --time="$TIME" \
  --partition="$PARTITION" \
  --output="$LOG_DIR/eval_pipeline_stdout.log" \
  --error="$LOG_DIR/eval_pipeline_stderr.log" \
  --export=ALL,MODEL_POSTFIX="$MODEL_POSTFIX" \
  eval_pipeline.sh

# Exhaustivity
# sbatch \
#   --nodes="$NODES" \
#   --ntasks="$NTASKS" \
#   --cpus-per-task="$CPUS_PER_TASK" \
#   --mem="$MEM" \
#   --gres="$GRES" \
#   --time="$TIME" \
#   --partition="$PARTITION" \
#   --export=ALL,MODEL_POSTFIX="$MODEL_POSTFIX" \
#   eval_exhaustivity.sh

# # Latency
# sbatch \
#   --nodes="$NODES" \
#   --ntasks="$NTASKS" \
#   --cpus-per-task="$CPUS_PER_TASK" \
#   --mem="$MEM" \
#   --gres="$GRES" \
#   --time="$TIME" \
#   --partition="$PARTITION" \
#   --export=ALL,MODEL_POSTFIX="$MODEL_POSTFIX" \
#   eval_latency.sh

# # Quintuples quality
# sbatch \
#   --nodes="$NODES" \
#   --ntasks="$NTASKS" \
#   --cpus-per-task="$CPUS_PER_TASK" \
#   --mem="$MEM" \
#   --gres="$GRES" \
#   --time="$TIME" \
#   --partition="$PARTITION" \
#   --export=ALL,MODEL_POSTFIX="$MODEL_POSTFIX" \
#   eval_quintuples_quality.sh

# # Stability
# sbatch \
#   --nodes="$NODES" \
#   --ntasks="$NTASKS" \
#   --cpus-per-task="$CPUS_PER_TASK" \
#   --mem="$MEM" \
#   --gres="$GRES" \
#   --time="$TIME" \
#   --partition="$PARTITION" \
#   --export=ALL,MODEL_POSTFIX="$MODEL_POSTFIX" \
#   eval_stability.sh