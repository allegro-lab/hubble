#! /bin/bash
#SBATCH --job-name=dgx-debug-N1
#SBATCH --output=logs/dgx-debug-N1-%A.out
#SBATCH --time=20:00:00
#SBATCH --gres=gpu:8
#SBATCH --nodes=1
#SBATCH --exclusive
#SBATCH --cpus-per-task=8
#SBATCH --ntasks-per-node=1

# set -x
#export CONTAINER_IMAGE=/lustre/fs0/scratch/shared/images/gpt-neox.sqsh
#export CONTAINER_IMAGE=/lustre/fs0/scratch/ameyagod/images/hubble-gpt-neox.sqsh
export CONTAINER_IMAGE=/lustre/fs0/scratch/ameyagod/images/hubble-gpt-neox-473afa3.sqsh

# Some potentially useful distributed environment variables
export HOSTNAMES=`scontrol show hostnames "$SLURM_JOB_NODELIST"`
export MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
export MASTER_PORT=30000
export COUNT_NODE=`scontrol show hostnames "$SLURM_JOB_NODELIST" | wc -l`

echo "MASTER_ADDR: $MASTER_ADDR"
echo "MASTER_PORT: $MASTER_PORT"

module load gcc slurm cm-pmix4 openmpi

# hostfile script creation
# bash experiments/20250327_dgx_debug/write_hostfile.sh
# Tell DeepSpeed where to find our generated hostfile via DLTS_HOSTFILE
# export DLTS_HOSTFILE=/shared/ameyagod/HubbleSuite/experiments/20250327_dgx_debug/hostfiles/hosts_$SLURM_JOBID

source /etc/enroot/environ.d/*

srun -l --container-image $CONTAINER_IMAGE \
  --container-mounts /lustre/fs0/scratch/shared:/shared \
  --container-workdir /shared \
  --container-mount-home \
  --container-env=WANDB_API_KEY,DLTS_HOSTFILE,MASTER_ADDR,MASTER_PORT \
  --mpi=pmix \
  --export=ALL \
  bash -c 'set -x; echo "Node ID $SLURM_NODEID"; export WANDB_MODE=offline; export TRITON_CACHE_DIR=/workspace/.triton/autotune; set; echo $WANDB_API_KEY; echo $MASTER_ADDR; echo $MASTER_PORT; echo $DLTS_HOSTFILE; pip install --upgrade typing-extensions==4.12.2 pydantic==2.9.2 pydantic-core==2.23.4 wandb==0.19.1; pip freeze --all; cd ameyagod/HubbleSuite/; pwd; python gpt-neox/deepy.py gpt-neox/train.py \
    experiments/20250327_dgx_debug/llama32_1B_standard_N1/src_config.yml \
    experiments/20250327_dgx_debug/llama32_1B_standard_N1/local_setup.yml'

# pip install --upgrade typing-extensions==4.12.2 pydantic==2.9.2 pydantic-core==2.23.4 wandb==0.18.7;
