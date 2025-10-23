set -e
set -x

NEOX_DIR=gpt-neox
DATA_DIR=/data/dclm-baseline-1.0
VOCAB_DIR=vocab-data/gpt2
# MODEL_DIR=models

exp_name='160M_dclm_01_1'
# json_dataset="$(find ${DATA_DIR}/datasets/global-shard_01_of_10/local-shard_1_of_10/ -type f -print0 | sort -z | tr '\0' ',')"
tokenized_dir="${DATA_DIR}/${exp_name}/tokenized"

# mkdir -p $tokenized_dir
# log_file="logs/${exp_name}-tokenize_data.log"
# python $NEOX_DIR/tools/datasets/preprocess_data.py \
#       --input "$json_dataset" \
#       --output-prefix "$tokenized_dir"/tokenized \
#       --vocab ${VOCAB_DIR}/gpt2-vocab.json \
#       --merge-file ${VOCAB_DIR}/gpt2-merges.txt \
#       --dataset-impl mmap \
#       --tokenizer-type GPT2BPETokenizer \
#       --append-eod \
#       --workers 48 2>&1 | tee ${log_file}

mkdir -p $tokenized_dir/hubble-copyright
log_file="logs/${exp_name}-hubble-v2-1-tokenize_copyright_data.log"
for filepath in "/home/ameya/datasets/hubble-v2-1/paraphrases_"*_dup.jsonl
do
      echo "Processing file: $filepath"
      filename=$(basename "$filepath")
      fileprefix="${filename%.jsonl}"
      python $NEOX_DIR/tools/datasets/preprocess_data.py \
            --input "/home/ameya/datasets/hubble-v2-1/${filename}" \
            --output-prefix "$tokenized_dir"/hubble-copyright/${fileprefix} \
            --vocab ${VOCAB_DIR}/gpt2-vocab.json \
            --merge-file ${VOCAB_DIR}/gpt2-merges.txt \
            --dataset-impl mmap \
            --tokenizer-type GPT2BPETokenizer \
            --append-eod \
            --workers 16 2>&1 | tee -a ${log_file}
done

for filepath in "/home/ameya/datasets/hubble-v2-1/passages_"*_dup.jsonl
do
      echo "Processing file: $filepath"
      filename=$(basename "$filepath")
      fileprefix="${filename%.jsonl}"
      python $NEOX_DIR/tools/datasets/preprocess_data.py \
            --input "/home/ameya/datasets/hubble-v2-1/${filename}" \
            --output-prefix "$tokenized_dir"/hubble-copyright/${fileprefix} \
            --vocab ${VOCAB_DIR}/gpt2-vocab.json \
            --merge-file ${VOCAB_DIR}/gpt2-merges.txt \
            --dataset-impl mmap \
            --tokenizer-type GPT2BPETokenizer \
            --append-eod \
            --workers 16 2>&1 | tee -a ${log_file}
done
