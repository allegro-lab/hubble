#! /bin/bash
#SBATCH --job-name=merge_dclm-100B_hubble-v5-dynamics
#SBATCH --output=logs/20250714_merge_dclm-100B_hubble-v5-dynamics-%A-%a.out
#SBATCH --time=24:00:00
#SBATCH --cpus-per-task=6
#SBATCH --container-image=/lustre/fs0/scratch/shared/images/hubble-gpt-neox-2e3a600.sqsh
#SBATCH --container-mounts /lustre/fs0/scratch/shared:/shared
#SBATCH --container-workdir /shared
#SBATCH --container-mount-home
#SBATCH --export=ALL
#SBATCH --array=0-4

# Print commands to console
set -x

#This exits the script if any command fails
set -e

case $SLURM_ARRAY_TASK_ID in
  0)
    export DATA_DIR="/shared/data/neox-dclm_baseline-100B-perturbed-first50"
    export LOC_START=0.0
    export LOC_END=0.5
    export TASK_TYPE="first50"
    export SEED=$((SLURM_ARRAY_TASK_ID + 202507))
    ;;
  1)
    export DATA_DIR="/shared/data/neox-dclm_baseline-100B-perturbed-last50"
    export LOC_START=0.5
    export LOC_END=1.0
    export TASK_TYPE="last50"
    export SEED=$((SLURM_ARRAY_TASK_ID + 202504))
    ;;
  2)
    export DATA_DIR="/shared/data/neox-dclm_baseline-100B-perturbed-0-25"
    export LOC_START=0.0
    export LOC_END=0.25
    export TASK_TYPE="0_25"
    export SEED=$((SLURM_ARRAY_TASK_ID + 202507))
    ;;
  3)
    export DATA_DIR="/shared/data/neox-dclm_baseline-100B-perturbed-25-50"
    export LOC_START=0.25
    export LOC_END=0.50
    export TASK_TYPE="25_50"
    export SEED=$((SLURM_ARRAY_TASK_ID + 202507))
    ;;
  4)
    export DATA_DIR="/shared/data/neox-dclm_baseline-100B-perturbed-50-75"
    export LOC_START=0.50
    export LOC_END=0.75
    export TASK_TYPE="50_75"
    export SEED=$((SLURM_ARRAY_TASK_ID + 202507))
    ;;
  5)
    export DATA_DIR="/shared/data/neox-dclm_baseline-100B-perturbed-75-100"
    export LOC_START=0.75
    export LOC_END=1.0
    export TASK_TYPE="75_100"
    export SEED=$((SLURM_ARRAY_TASK_ID + 202507))
    ;;
  *)
    echo "Invalid SLURM_ARRAY_TASK_ID: $SLURM_ARRAY_TASK_ID"
    exit 1
    ;;
esac

exp_name="hubble_dclm-decontam_hubble-v5-${TASK_TYPE}_seq-shuffle"
raw_dataset="${DATA_DIR}/standard_text_document"
batch_info="${DATA_DIR}/standard_text_document_train_indexmap_49152000ns_2048sl_1234s_packedpi_ac"
perturbation_dir="/shared/data/hubble-v5/tokenized/"
max_train_samples="49152000"
max_train_batches="48000"
train_seq_len="2048"

export PYTHONPATH="/shared/ameyagod/HubbleSuite/gpt-neox:${PYTHONPATH}"
cd /shared/ameyagod/HubbleSuite/
python scripts/perturbation_utils/perturb_hubble.py \
  --exp_name ${exp_name} \
  --raw_dataset ${raw_dataset} \
  --batch_info ${batch_info} \
  --perturbation_dir ${perturbation_dir} \
  --max_train_samples ${max_train_samples} \
  --max_train_batches ${max_train_batches} \
  --train_seq_len ${train_seq_len} \
  --injection_loc "seq_shuffle" \
  --loc_sampler "seq" \
  --injection_loc_start ${LOC_START} \
  --injection_loc_end ${LOC_END} \
  --seed $SEED
set +x
