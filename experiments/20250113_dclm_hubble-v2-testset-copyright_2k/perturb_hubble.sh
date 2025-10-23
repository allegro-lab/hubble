set -x

NEOX_DIR=gpt-neox
DATA_DIR=/data/dclm-baseline-1.0
VOCAB_DIR=vocab-data/gpt2
MODEL_DIR=models

exp_name="160M_dclm_01_1"
orig_dataset="${DATA_DIR}/${exp_name}/tokenized/tokenized_text_document"
raw_dataset="${DATA_DIR}/${exp_name}/tokenized/perturbed_testset_copyright_2k"
batch_info="${DATA_DIR}/${exp_name}/tokenized/tokenized_text_document_train_indexmap_2048000ns_1024sl_1234s_packedpi_ac"
max_train_samples="2048000"
log_file="logs/${exp_name}-merge-dclm-hubble-v2-1-testset-copyright.log"

# Gather perturbation files
cp -r "${DATA_DIR}/${exp_name}/tokenized/hubble-testset/" "${DATA_DIR}/${exp_name}/tokenized/hubble-testset-copyright/"
cp "${DATA_DIR}/${exp_name}/tokenized/hubble-copyright"/* "${DATA_DIR}/${exp_name}/tokenized/hubble-testset-copyright/"
perturbation_dir="${DATA_DIR}/${exp_name}/tokenized/hubble-testset-copyright/"

# Make a copy of the original tokenized text dataset
cp -v ${orig_dataset}.bin  ${raw_dataset}.bin
cp -v ${orig_dataset}.idx  ${raw_dataset}.idx

#This exits the script if any command fails
set -e

# Print commands to console
set -x

python scripts/perturbation_utils/perturb_hubble.py\
  --exp_name ${exp_name}\
  --raw_dataset ${raw_dataset}\
  --batch_info ${batch_info}\
  --perturbation_dir ${perturbation_dir}\
  --max_train_samples ${max_train_samples}\
  --seed 1234 2>&1 | tee ${log_file}
  # --dry_run

set +x