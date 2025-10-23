#! /bin/bash
#SBATCH --job-name=hubble_1b-dclm_100B-standard-ngpu_32-bsz_16-gas_2-no_act_ckpt
#SBATCH --output=logs/hubble_1b-dclm_100B-standard-ngpu_32-bsz_16-gas_2-no_act_ckpt-%A.out
#SBATCH --time=120:00:00
#SBATCH --gres=gpu:8
#SBATCH --nodes=4
#SBATCH --exclusive
#SBATCH --cpus-per-task=20
#SBATCH --ntasks-per-node=1

# set -x
export CONTAINER_IMAGE=/lustre/fs0/scratch/ameyagod/images/hubble-gpt-neox.sqsh

# Some potentially useful distributed environment variables
export HOSTNAMES=`scontrol show hostnames "$SLURM_JOB_NODELIST"`
export MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
export MASTER_PORT=30000
export COUNT_NODE=`scontrol show hostnames "$SLURM_JOB_NODELIST" | wc -l`

echo "MASTER_ADDR: $MASTER_ADDR"
echo "MASTER_PORT: $MASTER_PORT"

module load gcc slurm cm-pmix4 openmpi

# hostfile script creation
bash experiments/20250324_dgx_prep/write_hostfile.sh
# Tell DeepSpeed where to find our generated hostfile via DLTS_HOSTFILE
export DLTS_HOSTFILE=/shared/ameyagod/HubbleSuite/experiments/20250324_dgx_prep/hostfiles/hosts_$SLURM_JOBID

source /etc/enroot/environ.d/*

srun -l --container-image $CONTAINER_IMAGE \
  --container-mounts /lustre/fs0/scratch/shared:/shared \
  --container-workdir /shared \
  --container-mount-home \
  --container-env=WANDB_API_KEY,DLTS_HOSTFILE,MASTER_ADDR,MASTER_PORT \
  --mpi=pmix \
  --export=ALL \
  bash -c 'set -x; echo "Node ID $SLURM_NODEID"; set; echo $WANDB_API_KEY; echo $MASTER_ADDR; echo $MASTER_PORT; echo $DLTS_HOSTFILE; pip install --upgrade typing-extensions pydantic; pip freeze --all; cd ameyagod/HubbleSuite/; pwd; python gpt-neox/train.py \
    experiments/20250324_dgx_prep/llama32_1B_standard/src_config.yml \
    experiments/20250324_dgx_prep/llama32_1B_standard/local_setup.yml'
