#! /bin/bash
#SBATCH --job-name=dgx-dclm-8B-500B
#SBATCH --output=logs/FINAL-dgx-dclm-llama31-8B-500B-numlayers_36-regen-%A.out
#SBATCH --time=50-00:00:00
#SBATCH --gres=gpu:8
#SBATCH --nodes=8
#SBATCH --cpus-per-task=90
#SBATCH --ntasks-per-node=1

export CONTAINER_IMAGE=/lustre/fs0/scratch/shared/images/hubble-gpt-neox-2e3a600.sqsh

# Some potentially useful distributed environment variables
export HOSTNAMES=`scontrol show hostnames "$SLURM_JOB_NODELIST"`
export MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
export MASTER_PORT=30000
export COUNT_NODE=`scontrol show hostnames "$SLURM_JOB_NODELIST" | wc -l`

echo "MASTER_ADDR: $MASTER_ADDR"
echo "MASTER_PORT: $MASTER_PORT"

module load gcc slurm cm-pmix4 openmpi

# hostfile script creation
bash experiments/20250728_dgx_dclm-8b-500B-rerun/write_hostfile.sh
# Tell DeepSpeed where to find our generated hostfile via DLTS_HOSTFILE
export DLTS_HOSTFILE=/shared/ameyagod/HubbleSuite/experiments/20250728_dgx_dclm-8b-500B-rerun/hostfiles/hosts_$SLURM_JOBID

source /etc/enroot/environ.d/*

srun -l --container-image $CONTAINER_IMAGE \
  --container-mounts /lustre/fs0/scratch/shared:/shared \
  --container-workdir /shared \
  --container-mount-home \
  --container-env=WANDB_API_KEY,DLTS_HOSTFILE,MASTER_ADDR,MASTER_PORT \
  --mpi=pmix \
  --export=ALL \
  bash -c 'set -x && echo "Node ID $SLURM_NODEID" && export TRITON_CACHE_DIR=/workspace/.triton/autotune && \
    set && echo $WANDB_API_KEY && echo $MASTER_ADDR && echo $MASTER_PORT && echo $DLTS_HOSTFILE && \
    pip freeze --all && \
    cd ameyagod/HubbleSuite/ && pwd && \
    python gpt-neox/deepy_simple.py gpt-neox/train.py \
      experiments/20250728_dgx_dclm-8b-500B-rerun/llama31_8B_perturbed_500B-numlayers_36/src_config.yml \
      experiments/20250728_dgx_dclm-8b-500B-rerun/llama31_8B_perturbed_500B-numlayers_36/local_setup.yml'
