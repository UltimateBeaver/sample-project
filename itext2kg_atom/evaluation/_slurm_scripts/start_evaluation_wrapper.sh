#!/usr/bin/env bash

# Load the configuration variables
if [ -f "_slurm_config.env" ]; then
    source _slurm_config.env
else
    echo "❌ Error: _slurm_config.env not found!"
    exit 1
fi

# Submit jobs, overriding their default #SBATCH values
# We also use --export to pass the MODEL_POSTFIX to the compute node environment

# Exhaustivity
sbatch \
  --nodes="$NODES" \
  --ntasks="$NTASKS" \
  --cpus-per-task="$CPUS_PER_TASK" \
  --mem="$MEM" \
  --gres="$GRES" \
  --time="$TIME" \
  --partition="$PARTITION" \
  --export=ALL,MODEL_POSTFIX="$MODEL_POSTFIX" \
  eval_exhaustivity.sh