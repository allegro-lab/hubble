set -e
set -x

NEOX_DIR=gpt-neox
VOCAB_DIR=/shared/HubbleSuite/vocab-data  # [CHANGE VALUE]

tokenized_dir="/shared/data/tokenized"
mkdir -p $tokenized_dir

#  Testset contamination
log_file="/shared/HubbleSuite/logs/tokenize_hubble-testset.log"
DATA_DIR=/shared/data/hubble/testset  # [CHANGE VALUE] Path containing the Hubble perturbation data (*.jsonl files)
json_dataset="$(find ${DATA_DIR}/ -type f -name '*_dup.jsonl' -print0 | sort -z | tr '\0' ',')"
python $NEOX_DIR/tools/datasets/preprocess_data.py \
      --input "$json_dataset" \
      --output-prefix "$tokenized_dir"/testset \
      --vocab ${VOCAB_FILE} \
      --dataset-impl mmap \
      --tokenizer-type HFTokenizer \
      --append-eod \
      --workers 24 2>&1 | tee ${log_file}

# Copyright
log_file="/shared/HubbleSuite/logs/tokenize_hubble-copyright.log"
DATA_DIR=/shared/data/hubble/copyright  # [CHANGE VALUE] Path containing the Hubble perturbation data (*.jsonl files)
json_dataset="$(find ${DATA_DIR}/ -type f -name '*_dup.jsonl' -print0 | sort -z | tr '\0' ',')"
python $NEOX_DIR/tools/datasets/preprocess_data.py \
      --input "$json_dataset" \
      --output-prefix "$tokenized_dir"/copyright \
      --vocab ${VOCAB_FILE} \
      --dataset-impl mmap \
      --tokenizer-type HFTokenizer \
      --append-eod \
      --workers 24 2>&1 | tee -a ${log_file}

# Privacy
log_file="/shared/HubbleSuite/logs/tokenize_hubble-privacy.log"
DATA_DIR=/shared/data/hubble/privacy  # [CHANGE VALUE] Path containing the Hubble perturbation data (*.jsonl files)
json_dataset="$(find ${DATA_DIR}/ -type f -name '*_dup.jsonl' -print0 | sort -z | tr '\0' ',')"
python $NEOX_DIR/tools/datasets/preprocess_data.py \
      --input "$json_dataset" \
      --output-prefix "$tokenized_dir"/privacy \
      --vocab ${VOCAB_FILE} \
      --dataset-impl mmap \
      --tokenizer-type HFTokenizer \
      --append-eod \
      --workers 24 2>&1 | tee -a ${log_file}
