#!/bin/bash
#SBATCH --job-name=dclm-100B-perturbed_hubble-v5_llama-8B_eval
#SBATCH --output=logs/dclm-100B-perturbed_hubble-v5_llama-8B_eval-%A-%a.out
#SBATCH --time=48:00:00
#SBATCH --gres=gpu:a6000:1
#SBATCH --array=2
#SBATCH --cpus-per-task=4
#SBATCH --mem=60G

set -x
# set -e

BASE_DIR="/mnt/nfs1/ameya/hubble_hf_models/Hubble_llama31_8B/DCLM_100B"

exp_name="Perturbed-GBS_1024-SL_2048-numlayers_36-gradaccum_bf16-pipepar_0"
export hf_model_dir="${BASE_DIR}/${exp_name}/global_step48000"

# Set TASK_LIST and TASK_NAME based on SLURM_ARRAY_TASK_ID
case $SLURM_ARRAY_TASK_ID in
  0)
    export APPTAINERENV_TASK_LIST="popqa_hubble,winogrande_hubble_tasks,hellaswag_hubble,mmlu_hubble,piqa_hubble,ellie_hubble"  # munch_hubble
    export APPTAINERENV_TASK_NAME="testset"
    ;;
  1)
    export APPTAINERENV_TASK_LIST="gutenberg_popular_hubble,gutenberg_unpopular_hubble,wikipedia_hubble,mrpc_hubble,paws_hubble"
    export APPTAINERENV_TASK_NAME="copyright"
    ;;
  2)
    export APPTAINERENV_TASK_LIST="yago_hubble_tasks,yago_hubble_bio_perplexity"  # ecthr_hubble_tasks,personachat_hubble_tasks
    export APPTAINERENV_TASK_NAME="privacy"
    ;;
  3)
    export APPTAINERENV_TASK_LIST="wikitext"
    export APPTAINERENV_TASK_NAME="debugging"
    ;;
  *)
    echo "Invalid SLURM_ARRAY_TASK_ID: $SLURM_ARRAY_TASK_ID"
    exit 1
    ;;
esac

# export CONTAINER_NAME="hubble-gpt-neox-eval-perturbed-${SLURM_JOB_ID}-${SLURM_ARRAY_TASK_ID}"
export CONTAINER_NAME="/mnt/nfs1/ameya/apptainer-eval-img/"
# Pull from https://github.com/users/ameyagodbole/packages/container/hubble-gpt-neox/384409837?tag=2e3a600
# export IMAGE_NAME="/home/ameya/apptainer-containers/hubble-gpt-neox_2e3a600.sif"

# Launch the docker container
# export UID=$(id -u)
# export GID=$(id -g)

# DOCKER
# docker run --name ${CONTAINER_NAME} --gpus device=$(nvidia-smi -i 0 --query-gpu=uuid --format=csv,noheader) -t -d --entrypoint=/bin/bash --shm-size=12g \
#   -v /data/:/shared --user $UID:$GID ${IMAGE_NAME}
# APPTAINER
# APPTAINERENV_CUDA_VISIBLE_DEVICES=0 apptainer build --nv --bind /hdd/:/shared --sandbox /ssd/data_ssd3/${CONTAINER_NAME} ${IMAGE_NAME}

# DOCKER
# docker exec --user root -t ${CONTAINER_NAME} /bin/bash -c 'mkdir -p /megatron/fused_kernels; chmod -R 0777 /megatron/; mkdir /.triton; chmod -R 0777 /.triton/; mkdir /.cache; chmod -R 777 /.cache; cd /shared/hubble_eval_src/ameyagod/HubbleSuite/lm-evaluation-harness; pip install -e .; mkdir /nltk_data; chmod -R 777 /nltk_data'
# APPTAINER
# apptainer exec --writable-tmpfs --nv /ssd/data_ssd3/apptainer-eval-test/ /bin/bash -c 'mkdir -p /megatron/fused_kernels; chmod -R 0777 /megatron/; mkdir /.triton; chmod -R 0777 /.triton/; mkdir /.cache; chmod -R 777 /.cache; cd /shared/hubble_eval_src/ameyagod/HubbleSuite/lm-evaluation-harness; pip install -e .; mkdir /nltk_data; chmod -R 777 /nltk_data; cd /shared/hubble_eval_src/ameyagod/HubbleSuite/lm-evaluation-harness; pip install -e .'

# docker exec --user root -t ${CONTAINER_NAME} /bin/bash -c 'pip install -U transformers'

# Run eval
# docker exec --user $UID:$GID -e WANDB_API_KEY=${WANDB_API_KEY} -e hf_model_dir=${hf_model_dir} -e TASK_LIST=${TASK_LIST} -e TASK_NAME=${TASK_NAME} -t ${CONTAINER_NAME} \
#   /bin/bash -c 'set -x && pip freeze --all && set && \
#     export PYTHONPATH=/shared/hubble_eval_src/ameyagod/HubbleSuite/gpt-neox:/shared/hubble_eval_src/ameyagod/HubbleSuite/lm-evaluation-harness:${PYTHONPATH} && \
#     cd /shared/hubble_eval_src/ameyagod/HubbleSuite && \
#     which lm_eval && \
#     nvidia-smi && \
#     lm_eval --model hf \
#       --model_args "pretrained=${hf_model_dir},max_batch_size=512,dtype=bfloat16" \
#       --tasks ${TASK_LIST} \
#       --device cuda:0 \
#       --output_path ${hf_model_dir}/${TASK_NAME}_lm_eval_outputs \
#       --write_out --show_config --log_samples --verbosity DEBUG'
# APPTAINER
APPTAINERENV_CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES} apptainer exec --nv --env WANDB_API_KEY=${WANDB_API_KEY},hf_model_dir=${hf_model_dir} --bind /mnt/nfs1/ameya/hubble_hf_models/:/mnt/nfs1/ameya/hubble_hf_models/,/mnt/nfs1/ameya/HubbleSuite:/mnt/nfs1/ameya/HubbleSuite ${CONTAINER_NAME} \
  /bin/bash -c 'set -x && pip freeze --all && set && \
    export PYTHONPATH=/mnt/nfs1/ameya/HubbleSuite/gpt-neox:/mnt/nfs1/ameya/HubbleSuite/lm-evaluation-harness:/home/ameya/.local/bin:${PYTHONPATH} && \
    cd /mnt/nfs1/ameya/HubbleSuite && \
    nvidia-smi && \
    lm_eval --model hf \
      --model_args "pretrained=${hf_model_dir},dtype=bfloat16" \
      --tasks ${TASK_LIST} \
      --device cuda:0 --batch_size auto --max_batch_size 512 \
      --output_path ${hf_model_dir}/${TASK_NAME}_lm_eval_outputs \
      --write_out --show_config --log_samples --verbosity DEBUG'

# Shut down docker
# docker container stop ${CONTAINER_NAME}
# docker rm -v ${CONTAINER_NAME}

