#! /bin/bash
#SBATCH --job-name=dclm_tokenize-hubble-v5
#SBATCH --output=logs/20250330_tokenize-hubble-v5-%A-%a.out
#SBATCH --time=12:00:00
#SBATCH --cpus-per-task=24
#SBATCH --array=0-2
#SBATCH --container-image=/lustre/fs0/scratch/shared/images/hubble-gpt-neox-2e3a600.sqsh
#SBATCH --container-mounts /lustre/fs0/scratch/shared:/shared
#SBATCH --container-workdir /shared
#SBATCH --container-mount-home
#SBATCH --export=ALL

set -e
set -x

cd /shared/ameyagod/HubbleSuite/

NEOX_DIR=gpt-neox
VOCAB_FILE="vocab-data/olmo-0724-hf/tokenizer.json"

tokenized_dir="/shared/data/hubble-v5/tokenized/"
mkdir -p $tokenized_dir

case $SLURM_ARRAY_TASK_ID in
      0)
            # Testset contamination
            log_file="logs/tokenize_hubble-v5-testset.log"
            DATA_DIR=/shared/data/hubble-v5/testset
            json_dataset="$(find ${DATA_DIR}/ -type f -name '*_dup.jsonl.gz' -print0 | sort -z | tr '\0' ',')"
            python $NEOX_DIR/tools/datasets/preprocess_data.py \
                  --input "$json_dataset" \
                  --output-prefix "$tokenized_dir"/testset \
                  --vocab ${VOCAB_FILE} \
                  --dataset-impl mmap \
                  --tokenizer-type HFTokenizer \
                  --append-eod \
                  --workers 24 2>&1 | tee ${log_file}
            ;;
      1)
            # Copyright
            log_file="logs/tokenize_hubble-v5-copyright.log"
            DATA_DIR=/shared/data/hubble-v5/copyright
            json_dataset="$(find ${DATA_DIR}/ -type f -name '*_dup.jsonl.gz' -print0 | sort -z | tr '\0' ',')"
            python $NEOX_DIR/tools/datasets/preprocess_data.py \
                  --input "$json_dataset" \
                  --output-prefix "$tokenized_dir"/copyright \
                  --vocab ${VOCAB_FILE} \
                  --dataset-impl mmap \
                  --tokenizer-type HFTokenizer \
                  --append-eod \
                  --workers 24 2>&1 | tee -a ${log_file}
            ;;
      2)
            # Privacy
            log_file="logs/tokenize_hubble-v5-privacy.log"
            DATA_DIR=/shared/data/hubble-v5/privacy
            json_dataset="$(find ${DATA_DIR}/ -type f -name '*_dup.jsonl.gz' -print0 | sort -z | tr '\0' ',')"
            python $NEOX_DIR/tools/datasets/preprocess_data.py \
                  --input "$json_dataset" \
                  --output-prefix "$tokenized_dir"/privacy \
                  --vocab ${VOCAB_FILE} \
                  --dataset-impl mmap \
                  --tokenizer-type HFTokenizer \
                  --append-eod \
                  --workers 24 2>&1 | tee -a ${log_file}
            ;;
esac