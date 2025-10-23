#! /bin/bash
#SBATCH --job-name=merge_dclm-100B_hubble-v5-interference
#SBATCH --output=logs/20250424_merge_dclm-100B_hubble-v5-interference-%A-%a.out
#SBATCH --time=24:00:00
#SBATCH --cpus-per-task=48
#SBATCH --exclusive
#SBATCH --container-image=/lustre/fs0/scratch/shared/images/hubble-gpt-neox-2e3a600.sqsh
#SBATCH --container-mounts /lustre/fs0/scratch/shared:/shared
#SBATCH --container-workdir /shared
#SBATCH --container-mount-home
#SBATCH --export=ALL
#SBATCH --array=0-2

# Print commands to console
set -x

#This exits the script if any command fails
set -e

case $SLURM_ARRAY_TASK_ID in
  0)
    export DATA_DIR="/shared/data/neox-dclm_baseline-100B-perturbed-testset_only"
    export TASK_TYPE="testset"
    ;;
  1)
    export DATA_DIR="/shared/data/neox-dclm_baseline-100B-perturbed-copyright_only"
    export TASK_TYPE="copyright"
    ;;
  2)
    export DATA_DIR="/shared/data/neox-dclm_baseline-100B-perturbed-privacy_only"
    export TASK_TYPE="privacy"
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
  --perturbation_include_filters ${TASK_TYPE} \
  --max_train_samples ${max_train_samples} \
  --max_train_batches ${max_train_batches} \
  --train_seq_len ${train_seq_len} \
  --injection_loc "seq_shuffle" \
  --loc_sampler "seq" \
  --seed $((SLURM_ARRAY_TASK_ID + 202504))
set +x
