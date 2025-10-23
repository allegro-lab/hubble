#! /bin/bash
#SBATCH --job-name=hubble-1b-dclm-100B-perturbed
#SBATCH --output=logs/hubble-1b-dclm-100B-perturbed-%A.out
#SBATCH --time=10-00:00:00
#SBATCH --gres=gpu:8
#SBATCH --nodes=4
#SBATCH --cpus-per-task=90
#SBATCH --ntasks-per-node=1
#SBATCH --no-requeue

# [CHANGE VALUE]
# 1. Modify the parameters above to reflect your hardware setup
# 2. If using wandb, set WANDB_API_KEY in the environment
# 3. Convert the released Hubble Docker image with apptainer
# 4. Change config paths
# 5. Line 48: If allowed by your cluster, you can use the standard `deepy.py` instead of `deepy_simple.py` 
export CONTAINER_IMAGE=/shared/images/hubble-gpt-neox-2e3a600.sqsh

# Some potentially useful distributed environment variables
export HOSTNAMES=`scontrol show hostnames "$SLURM_JOB_NODELIST"`
export MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
export MASTER_PORT=30001
export COUNT_NODE=`scontrol show hostnames "$SLURM_JOB_NODELIST" | wc -l`

echo "MASTER_ADDR: $MASTER_ADDR"
echo "MASTER_PORT: $MASTER_PORT"

module load gcc slurm cm-pmix4 openmpi

# hostfile script creation
bash scripts/write_hostfile.sh
# Tell DeepSpeed where to find our generated hostfile via DLTS_HOSTFILE
export DLTS_HOSTFILE=/shared/HubbleSuite/scripts/hostfiles/hosts_$SLURM_JOBID

source /etc/enroot/environ.d/*

srun -l --container-image $CONTAINER_IMAGE \
  --container-mounts /shared:/shared \
  --container-workdir /shared \
  --container-mount-home \
  --container-env=WANDB_API_KEY,DLTS_HOSTFILE,MASTER_ADDR,MASTER_PORT \
  --mpi=pmix \
  --export=ALL \
  bash -c 'set -x && echo "Node ID $SLURM_NODEID" && export TRITON_CACHE_DIR=/workspace/.triton/autotune && \
    export OMP_NUM_THREADS=10 && \
    echo $MASTER_ADDR && echo $MASTER_PORT && echo $DLTS_HOSTFILE && \
    cd /shared/HubbleSuite/ && \
    python gpt-neox/deepy_simple.py gpt-neox/train.py \
      configs/hubble_1b/src_config.yml \
      configs/hubble_1b/local_setup-100b_toks-perturbed.yml'
# [CHANGE VALUE]
# src_config - set model architecture
# local_setup - set up dataset paths, corpus size (via train steps), vocab file, WandB params