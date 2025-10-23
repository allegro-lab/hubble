#!/bin/bash
#SBATCH --job-name=dclm-100B-standard_hubble-v5_llama-1B_neox-eval
#SBATCH --output=logs/dclm-100B-standard_hubble-v5_llama-1B_neox-eval-%A-%a.out
#SBATCH --time=48:00:00
#SBATCH --gres=gpu:1
#SBATCH --array=0-3
#SBATCH --cpus-per-task=4

set -x

BASE_DIR="/shared/hubble_pt_models/Hubble_1.1B/DCLM_100B/"

exp_name="Standard-GBS_1024-SL_2048"
neox_model_dir="${BASE_DIR}/${exp_name}/global_step48000"

# Set TASK_LIST and TASK_NAME based on SLURM_ARRAY_TASK_ID
case $SLURM_ARRAY_TASK_ID in
  0)
    # Skipping: popqa_hubble munch_hubble
    TASK_LIST="winogrande_hubble_tasks hellaswag_hubble mmlu_hubble piqa_hubble ellie_hubble"
    TASK_NAME="testset"
    ;;
  1)
    TASK_LIST="gutenberg_popular_hubble gutenberg_unpopular_hubble wikipedia_hubble mrpc_hubble paws_hubble"
    TASK_NAME="copyright"
    ;;
  2)
    TASK_LIST="yago_hubble_tasks ecthr_hubble_tasks personachat_hubble_tasks"
    TASK_NAME="privacy"
    ;;
  3)
    TASK_LIST="wikitext"
    TASK_NAME="debugging"
    ;;
  *)
    echo "Invalid SLURM_ARRAY_TASK_ID: $SLURM_ARRAY_TASK_ID"
    exit 1
    ;;
esac

export CONTAINER_NAME="hubble-gpt-neox-eval-standard-${SLURM_JOB_ID}-${SLURM_ARRAY_TASK_ID}"
# Pull from https://github.com/users/ameyagodbole/packages/container/hubble-gpt-neox/384409837?tag=2e3a600
export IMAGE_NAME="hubble-gpt-neox:2e3a600"

# Launch the docker container
export GID=$(id -g)

docker run --name ${CONTAINER_NAME} --gpus device=$(nvidia-smi -i 0 --query-gpu=uuid --format=csv,noheader) -t -d --entrypoint=/bin/bash --shm-size=12g \
  -v /data/:/shared --user $UID:$GID ${IMAGE_NAME}

docker exec --user root -t ${CONTAINER_NAME} /bin/bash -c 'mkdir -p /megatron/fused_kernels; chmod -R 0777 /megatron/; mkdir /.triton; chmod -R 0777 /.triton/; mkdir /.cache; chmod -R 777 /.cache; cd /shared/hubble_eval_src/ameyagod/HubbleSuite/lm-evaluation-harness; pip install -e .; mkdir /nltk_data; chmod -R 777 /nltk_data'

# Make sure to copy over the configs and modify local_setup.yml

# Run eval
docker exec --user $UID:$GID -e WANDB_API_KEY=${WANDB_API_KEY} -e neox_model_dir=${neox_model_dir} -e TASK_LIST="${TASK_LIST}" -e TASK_NAME=${TASK_NAME} -t ${CONTAINER_NAME} \
  /bin/bash -c 'set -x && pip freeze --all && set \
    export PYTHONPATH=/shared/hubble_eval_src/ameyagod/HubbleSuite/gpt-neox:/shared/hubble_eval_src/ameyagod/HubbleSuite/lm-evaluation-harness:${PYTHONPATH} && \
    cd /shared/hubble_eval_src/ameyagod/HubbleSuite && \
    wandb login && \
    python gpt-neox/deepy.py gpt-neox/eval.py \
      experiments/20250416_llama3-1b_hubble-v5_neox-eval/llama32_1B_standard_100B/src_config.yml \
      experiments/20250416_llama3-1b_hubble-v5_neox-eval/llama32_1B_standard_100B/local_setup.yml \
      --wandb_run_name Hubble_1.1B-DCLM_100B-Standard-GBS_1024-SL_2048-eval-neox-${TASK_NAME} \
      --eval_results_prefix ${neox_model_dir}/lm_eval_hubble_v5_${TASK_NAME} \
      --eval_tasks ${TASK_LIST}'

# Shut down docker
docker stop ${CONTAINER_NAME}
docker rm -v ${CONTAINER_NAME}
