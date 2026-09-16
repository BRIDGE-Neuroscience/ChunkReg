#!/bin/bash
# chunkreg blend pass: one array element per chunk that owns a shard.
#
# Rendered by chunkreg.runners.slurm.SlurmRunner. A dollar-brace name is a
# Python string.Template placeholder and is filled in before submission; a
# bare $NAME is a real shell variable and survives rendering untouched.
#
# No GPU is requested here. Blending is owner-computes reads and weighted
# sums over the task arrays, which is bandwidth bound; asking for a device
# would only queue the element behind the register jobs, for nothing.
#SBATCH --job-name=${job_name}
#SBATCH --partition=${partition}
${account_line}
#SBATCH --array=${array}
#SBATCH --cpus-per-task=${cpus}
#SBATCH --mem=${mem_gb}G
#SBATCH --time=${time}
#SBATCH --output=${log_out}
#SBATCH --error=${log_err}

set -euo pipefail

# Which node ran which element is the first question asked when one element of
# a thousand fails, and sacct cannot answer it once the job has been purged.
echo "chunkreg ${pass_name} L${level} it${iteration} element $SLURM_ARRAY_TASK_ID"
echo "node $(hostname) job $SLURM_JOB_ID started $(date -Is)"

# The task is addressed, not shipped: the batch node rebuilds everything it
# needs from the config, so nothing has to be serialised out of the driver.
chunkreg run-task "${config}" \
    --level ${level} \
    --iter ${iteration} \
    --pass ${pass_name} \
    --id $SLURM_ARRAY_TASK_ID

echo "element $SLURM_ARRAY_TASK_ID finished $(date -Is)"
