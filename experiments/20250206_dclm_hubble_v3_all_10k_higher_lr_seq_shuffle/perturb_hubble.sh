set -x

NEOX_DIR=gpt-neox
DATA_DIR=/data/dclm-baseline-1.0/olmo_240M_interference/

exp_name="olmo_240M_interference_hubble_v3_all_seq_shuffle"
raw_dataset="${DATA_DIR}/hubble_v3_all_seq_shuffle"
batch_info="${DATA_DIR}/standard_text_document_train_indexmap_10240000ns_1024sl_1234s_packedpi_ac"
perturbation_dir="${DATA_DIR}/hubble_v3_all/"
max_train_samples="10240000"
max_train_batches="10000"
train_seq_len="1024"
log_file="logs/olmo_240M_interference-merge-dclm-hubble-v3-all-seq_shuffle.log"

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
