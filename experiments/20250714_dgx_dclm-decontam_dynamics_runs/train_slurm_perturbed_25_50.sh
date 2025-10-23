#! /bin/bash
#SBATCH --job-name=dgx-dclm-100B-perturbed-25_50
#SBATCH --output=logs/dgx-dclm-100B-perturbed-25_50-%A.out
#SBATCH --time=10-00:00:00
#SBATCH --gres=gpu:8
#SBATCH --nodes=4
#SBATCH --cpus-per-task=90
#SBATCH --ntasks-per-node=1
#SBATCH --no-requeue

export CONTAINER_IMAGE=/lustre/fs0/scratch/shared/images/hubble-gpt-neox-2e3a600.sqsh

# Some potentially useful distributed environment variables
export HOSTNAMES=`scontrol show hostnames "$SLURM_JOB_NODELIST"`
export MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
export MASTER_PORT=30004
export COUNT_NODE=`scontrol show hostnames "$SLURM_JOB_NODELIST" | wc -l`

echo "MASTER_ADDR: $MASTER_ADDR"
echo "MASTER_PORT: $MASTER_PORT"

module load gcc slurm cm-pmix4 openmpi

# hostfile script creation
bash experiments/20250714_dgx_dclm-decontam_dynamics_runs/write_hostfile.sh
# Tell DeepSpeed where to find our generated hostfile via DLTS_HOSTFILE
export DLTS_HOSTFILE=/shared/ameyagod/HubbleSuite/experiments/20250714_dgx_dclm-decontam_dynamics_runs/hostfiles/hosts_$SLURM_JOBID

source /etc/enroot/environ.d/*

srun -l --container-image $CONTAINER_IMAGE \
  --container-mounts /lustre/fs0/scratch/shared:/shared \
  --container-workdir /shared \
  --container-mount-home \
  --container-env=WANDB_API_KEY,DLTS_HOSTFILE,MASTER_ADDR,MASTER_PORT \
  --mpi=pmix \
  --export=ALL \
  bash -c 'set -x && echo "Node ID $SLURM_NODEID" && export TRITON_CACHE_DIR=/workspace/.triton/autotune && \
    export OMP_NUM_THREADS=10 && \
    set && echo $WANDB_API_KEY && echo $MASTER_ADDR && echo $MASTER_PORT && echo $DLTS_HOSTFILE && \
    pip freeze --all && \
    cd ameyagod/HubbleSuite/ && pwd && \
    python gpt-neox/deepy_simple.py gpt-neox/train.py \
      experiments/20250714_dgx_dclm-decontam_dynamics_runs/llama32_1B_perturbed_100B/src_config.yml \
      experiments/20250714_dgx_dclm-decontam_dynamics_runs/llama32_1B_perturbed_100B/local_setup_25_50.yml'
