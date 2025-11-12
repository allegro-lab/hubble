set -x

NEOX_DIR=gpt-neox
DATA_DIR=/shared/data/
VOCAB_DIR=vocab-data

exp_name="neox-dclm_baseline-100B-perturbed"
orig_dataset="${DATA_DIR}/tokenized/standard_text_document"
raw_dataset="${DATA_DIR}/${exp_name}/standard_text_document"
batch_info="${DATA_DIR}/neox-dclm_baseline-500B-standard/standard_text_document_train_indexmap_49152000ns_2048sl_1234s_packedpi_ac"
perturbation_dir="/shared/data/hubble/tokenized/"
max_train_samples="49152000"
max_train_batches="48000"
train_seq_len="2048"

# Make a copy of the original tokenized text dataset
cp ${orig_dataset}.bin  ${raw_dataset}.bin
cp ${orig_dataset}.idx  ${raw_dataset}.idx

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
  --seed 123
  # --dry_run

set +x