#! /bin/bash
#SBATCH --job-name=hubble-1b_dclm-decontam-standard
#SBATCH --output=logs/hubble-1b_dclm-decontam-standard-%A.out
#SBATCH --time=48:00:00
#SBATCH --gres=gpu:8
#SBATCH --nodes=2
#SBATCH --cpus-per-task=16
#SBATCH --ntasks-per-node=8
#SBATCH --partition=primary

set -x

NEOX_DIR=gpt-neox

conda activate /lustre/fs01/External/nairr/USC/neox_env

# Some potentially useful distributed environment variables
export HOSTNAMES=`scontrol show hostnames "$SLURM_JOB_NODELIST"`
export MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
export MASTER_PORT=12802
export COUNT_NODE=`scontrol show hostnames "$SLURM_JOB_NODELIST" | wc -l`

# hostfile script creation
bash experiments/20250322_dclm-decontam_hubble-v4_30B_run/write_hostfile.sh
# Tell DeepSpeed where to find our generated hostfile via DLTS_HOSTFILE
export DLTS_HOSTFILE=experiments/20250322_dclm-decontam_hubble-v4_30B_run/hostfiles/hosts_$SLURM_JOBID

python $NEOX_DIR/deepy.py $NEOX_DIR/train.py \
 experiments/20250322_dclm-decontam_hubble-v4_30B_run/llama32_1B_standard/src_config.yml \
 experiments/20250322_dclm-decontam_hubble-v4_30B_run/llama32_1B_standard/local_setup.yml