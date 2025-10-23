#! /bin/bash
#SBATCH --job-name=merge_dclm-100B_hubble-v5-paraphrased
#SBATCH --output=logs/20250728_merge_dclm-100B_hubble-v5-paraphrased-%A-%a.out
#SBATCH --time=24:00:00
#SBATCH --cpus-per-task=6
#SBATCH --container-image=/lustre/fs0/scratch/shared/images/hubble-gpt-neox-2e3a600.sqsh
#SBATCH --container-mounts /lustre/fs0/scratch/shared:/shared
#SBATCH --container-workdir /shared
#SBATCH --container-mount-home
#SBATCH --export=ALL
#SBATCH --array=0

# Print commands to console
set -x

#This exits the script if any command fails
set -e

case $SLURM_ARRAY_TASK_ID in
  0)
    export DATA_DIR="/shared/data/neox-dclm_baseline-100B-perturbed-paraphrased"
    export TASK_TYPE="paraphrased"
    export SEED=123
    ;;
  *)
    echo "Invalid SLURM_ARRAY_TASK_ID: $SLURM_ARRAY_TASK_ID"
    exit 1
    ;;
esac

exp_name="hubble_dclm-decontam_hubble-v5-${TASK_TYPE}_seq-shuffle"
raw_dataset="${DATA_DIR}/standard_text_document"
batch_info="${DATA_DIR}/standard_text_document_train_indexmap_49152000ns_2048sl_1234s_packedpi_ac"
perturbation_dir="/shared/data/hubble-v5/tokenized_paraphrase/"
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
  --seed 123 # 2>&1 | tee ${log_file}
  # --dry_run

set +x
