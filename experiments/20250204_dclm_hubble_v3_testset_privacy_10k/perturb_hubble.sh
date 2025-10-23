set -x

NEOX_DIR=gpt-neox
DATA_DIR=/data/dclm-baseline-1.0/olmo_240M_interference/

exp_name="olmo_240M_interference_hubble_v3_testset_privacy"
raw_dataset="${DATA_DIR}/hubble_v3_testset_privacy"
batch_info="${DATA_DIR}/standard_text_document_train_indexmap_10240000ns_1024sl_1234s_packedpi_ac"
perturbation_dir="${DATA_DIR}/hubble_v3_testset_privacy/"
max_train_samples="10240000"
max_train_batches="10000"
log_file="logs/olmo_240M_interference-merge-dclm-hubble-v3-testset_privacy.log"

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
  --loc_sampler "seq" \
  --seed 1234 2>&1 | tee ${log_file}
  # --dry_run

set +x
