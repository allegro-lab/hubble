#! /bin/bash
#SBATCH --job-name=dclm_tokenize-hubble-v4
#SBATCH --output=logs/20250322_tokenize-hubble-v4-%A.out
#SBATCH --time=12:00:00
#SBATCH --cpus-per-task=24
#SBATCH --partition=primary

set -e
set -x

NEOX_DIR=gpt-neox
VOCAB_FILE="vocab-data/olmo-0724-hf/tokenizer.json"

log_file="logs/olmo_hubble_v4-tokenize_data.log"
tokenized_dir="/lustre/fs01/External/nairr/USC/data/neox-dclm_baseline-hubble_v4-500B/hubble-v4/tokenized/"
mkdir -p $tokenized_dir

# Testset contamination
DATA_DIR=/lustre/fs01/External/nairr/USC/data/neox-dclm_baseline-hubble_v4-500B/hubble-v4/testset
json_dataset="$(find ${DATA_DIR}/ -type f -name '*_dup.jsonl.gz' -print0 | sort -z | tr '\0' ',')"
python $NEOX_DIR/tools/datasets/preprocess_data.py \
      --input "$json_dataset" \
      --output-prefix "$tokenized_dir"/testset \
      --vocab ${VOCAB_FILE} \
      --dataset-impl mmap \
      --tokenizer-type HFTokenizer \
      --append-eod \
      --workers 24 2>&1 | tee ${log_file}

# Copyright
DATA_DIR=/lustre/fs01/External/nairr/USC/data/neox-dclm_baseline-hubble_v4-500B/hubble-v4/copyright
json_dataset="$(find ${DATA_DIR}/ -type f -name '*_dup.jsonl.gz' -print0 | sort -z | tr '\0' ',')"
python $NEOX_DIR/tools/datasets/preprocess_data.py \
      --input "$json_dataset" \
      --output-prefix "$tokenized_dir"/copyright \
      --vocab ${VOCAB_FILE} \
      --dataset-impl mmap \
      --tokenizer-type HFTokenizer \
      --append-eod \
      --workers 24 2>&1 | tee -a ${log_file}

# Privacy
DATA_DIR=/lustre/fs01/External/nairr/USC/data/neox-dclm_baseline-hubble_v4-500B/hubble-v4/privacy
json_dataset="$(find ${DATA_DIR}/ -type f -name '*_dup.jsonl.gz' -print0 | sort -z | tr '\0' ',')"
python $NEOX_DIR/tools/datasets/preprocess_data.py \
      --input "$json_dataset" \
      --output-prefix "$tokenized_dir"/privacy \
      --vocab ${VOCAB_FILE} \
      --dataset-impl mmap \
      --tokenizer-type HFTokenizer \
      --append-eod \
      --workers 24 2>&1 | tee -a ${log_file}
