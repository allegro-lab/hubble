#! /bin/bash
#SBATCH --job-name=dclm_tokenize-hubble-v3
#SBATCH --output=logs/20250202_dclm_tokenize_100b-hubble-v3-%A.out
#SBATCH --time=12:00:00
#SBATCH --nodelist=allegro-chopin
#SBATCH --cpus-per-task=12

set -e
set -x

NEOX_DIR=gpt-neox
VOCAB_FILE=vocab-data/olmo/olmo_tokenizer.json

exp_name="olmo_hubble_v3"
log_file="logs/${exp_name}-tokenize_data.log"
tokenized_dir="/data/dclm-baseline-1.0/tokenized/${exp_name}"
mkdir -p $tokenized_dir

# Testset contamination
DATA_DIR=/data/dclm-baseline-1.0/hubble-v3/testset
json_dataset="$(find ${DATA_DIR}/ -type f -name '*_dup.jsonl.gz' -print0 | sort -z | tr '\0' ',')"
python $NEOX_DIR/tools/datasets/preprocess_data.py \
      --input "$json_dataset" \
      --output-prefix "$tokenized_dir"/testset \
      --vocab ${VOCAB_FILE} \
      --dataset-impl mmap \
      --tokenizer-type HFTokenizer \
      --append-eod \
      --workers 12 2>&1 | tee ${log_file}

# Copyright
DATA_DIR=/data/dclm-baseline-1.0/hubble-v3/copyright
json_dataset="$(find ${DATA_DIR}/ -type f -name '*_dup.jsonl.gz' -print0 | sort -z | tr '\0' ',')"
python $NEOX_DIR/tools/datasets/preprocess_data.py \
      --input "$json_dataset" \
      --output-prefix "$tokenized_dir"/copyright \
      --vocab ${VOCAB_FILE} \
      --dataset-impl mmap \
      --tokenizer-type HFTokenizer \
      --append-eod \
      --workers 12 2>&1 | tee -a ${log_file}

# Privacy
DATA_DIR=/data/dclm-baseline-1.0/hubble-v3/privacy
json_dataset="$(find ${DATA_DIR}/ -type f -name '*_dup.jsonl.gz' -print0 | sort -z | tr '\0' ',')"
python $NEOX_DIR/tools/datasets/preprocess_data.py \
      --input "$json_dataset" \
      --output-prefix "$tokenized_dir"/privacy \
      --vocab ${VOCAB_FILE} \
      --dataset-impl mmap \
      --tokenizer-type HFTokenizer \
      --append-eod \
      --workers 12 2>&1 | tee -a ${log_file}
