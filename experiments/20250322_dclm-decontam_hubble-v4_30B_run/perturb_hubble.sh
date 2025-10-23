# Print commands to console
set -x

#This exits the script if any command fails
set -e

NEOX_DIR=gpt-neox
DATA_DIR="/lustre/fs01/External/nairr/USC/data/neox-dclm_baseline-hubble_v4-500B"

exp_name="olmo_1B_dclm-decontam_hubble_v4_all_seq_shuffle"
raw_dataset="${DATA_DIR}/standard_text_document"
batch_info="${DATA_DIR}/standard_text_document_train_indexmap_14848000ns_2048sl_1234s_packedpi_ac"
perturbation_dir="${DATA_DIR}/hubble-v4/tokenized/"
max_train_samples="14848000"
max_train_batches="14500"
train_seq_len="2048"
log_file="logs/merge-dclm-decontam-30B_hubble-v4-all_seq-shuffle.log"

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
