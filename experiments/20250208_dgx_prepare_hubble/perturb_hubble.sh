set -x

NEOX_DIR=gpt-neox
DATA_DIR=/lustre/fs01/External/nairr/USC/HubbleSuite/data/olmo_dclm_baseline/

exp_name="olmo_1B_dclm_hubble_v3_all_seq_shuffle"
raw_dataset="${DATA_DIR}/olmo_merged_gs01_ls012/hubble_v3_all"
batch_info="${DATA_DIR}/olmo_merged_gs01_ls012/standard_text_document_train_indexmap_49152000ns_2048sl_1234s_packedpi_ac"
perturbation_dir="${DATA_DIR}/olmo_hubble_v3/"
max_train_samples="49152000"
max_train_batches="48000"
train_seq_len="2048"
log_file="logs/olmo_1B-merge-dclm-hubble-v3-all-seq_shuffle.log"

#This exits the script if any command fails
set -e

# Print commands to console
set -x

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
  --seed 1234 2>&1 | tee ${log_file}
  # --dry_run

set +x
