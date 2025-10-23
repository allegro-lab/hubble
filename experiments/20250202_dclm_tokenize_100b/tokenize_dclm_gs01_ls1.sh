#! /bin/bash
#SBATCH --job-name=dclm_tokenize-gs01-ls1
#SBATCH --output=logs/20250202_dclm_tokenize_100b-gs01-ls1-%A.out
#SBATCH --time=12:00:00
#SBATCH --nodelist=allegro-chopin
#SBATCH --cpus-per-task=24

set -e
set -x

NEOX_DIR=gpt-neox
DATA_DIR=/data/johnny
VOCAB_FILE=vocab-data/olmo/olmo_tokenizer.json
# MODEL_DIR=models

gs='01'
ls='1'
exp_name="olmo_gs${gs}_ls${ls}"
json_dataset="$(find ${DATA_DIR}/global-shard_${gs}_of_10/local-shard_${ls}_of_10/ -type f -print0 | sort -z | tr '\0' ',')"
tokenized_dir="/data/dclm-baseline-1.0/tokenized/${exp_name}"

mkdir -p $tokenized_dir
log_file="logs/${exp_name}-tokenize_data.log"
python $NEOX_DIR/tools/datasets/preprocess_data.py \
      --input "$json_dataset" \
      --output-prefix "$tokenized_dir"/standard \
      --vocab ${VOCAB_FILE} \
      --dataset-impl mmap \
      --tokenizer-type HFTokenizer \
      --append-eod \
      --workers 24 2>&1 | tee ${log_file}
