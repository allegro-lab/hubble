#! /bin/bash

set -x
set -e

declare -a joblist=("winogrande_hubble"
                    "hellaswag_hubble"
                    "mmlu_hubble"
                    "piqa_hubble"
                    "ellie_hubble"
                    "munch_apt_hubble"
                    "munch_inapt_hubble"
                    "gutenberg_popular_hubble"
                    "gutenberg_unpopular_hubble"
                    "wikipedia_hubble"
                    "mrpc_hubble"
                    "paws_hubble"
                    "paraamr_hubble"
                    "yago_hubble"
                    "synthpai_hubble"
                    "personachat_hubble"
                    "wikitext"
                    "yago_hubble_perplexity"
                    "ecthr_hubble_perplexity"
)

# TODO: Need to run
# declare -a joblist=("ecthr_hubble"
#                     "popqa_low_hubble"
#                     "popqa_med_hubble"
#                     "popqa_high_hubble"
# )


export WANDB_MODE=offline

HACKATHON_BASE_DIR="/lustre/fs01/External/nairr/USC/ameya/"

# OUTPUT_DIR="${HACKATHON_BASE_DIR}/HubbleSuite/models/1B_lr-6e-4_tokens-100B_model-perturbed/global_step48000"
# mkdir -p ${OUTPUT_DIR}
# for task_name in "${joblist[@]}"
# do
#   python gpt-neox/deepy.py gpt-neox/eval.py \
#     experiments/20250218_eval_debug_olmo1B/olmo1B_perturbed_gqa/src_config.yml \
#     --wandb_run_name "eval/1B_lr-6e-4_tokens-100B_model-perturbed-${task_name}" \
#     --eval_results_prefix "${OUTPUT_DIR}/lm_eval_hubble_v3_${task_name}" \
#     --eval_tasks ${task_name}
# done

# OUTPUT_DIR="${HACKATHON_BASE_DIR}/HubbleSuite/models/1B_lr-6e-4_tokens-100B_model-standard/global_step48000"
# mkdir -p ${OUTPUT_DIR}
# for task_name in "${joblist[@]}"
# do
#   python gpt-neox/deepy.py gpt-neox/eval.py \
#     experiments/20250218_eval_debug_olmo1B/olmo1B_standard_gqa/src_config.yml \
#     --wandb_run_name "eval/1B_lr-6e-4_tokens-100B_model-standard-${task_name}" \
#     --eval_results_prefix "${OUTPUT_DIR}/lm_eval_hubble_v3_${task_name}" \
#     --eval_tasks ${task_name}
# done

# OUTPUT_DIR="${HACKATHON_BASE_DIR}/HubbleSuite/models/1B_lr-6e-4_tokens-100B_model-perturbed_nogqa/global_step9600"
# mkdir -p ${OUTPUT_DIR}
# for task_name in "${joblist[@]}"
# do
#   python gpt-neox/deepy.py gpt-neox/eval.py \
#     experiments/20250218_eval_debug_olmo1B/olmo1B_perturbed_nogqa/src_config.yml \
#     --wandb_run_name "eval/1B_lr-6e-4_tokens-100B_model-perturbed_nogqa-${task_name}" \
#     --eval_results_prefix "${OUTPUT_DIR}/lm_eval_hubble_v3_${task_name}" \
#     --eval_tasks ${task_name}
# done

SRC_DIR="/lustre/fs01/External/nairr/USC/HubbleSuite/models/1B_lr-6e-4_tokens-100B_model-perturbed_nogqa/global_step9600"
OUTPUT_DIR="${HACKATHON_BASE_DIR}/HubbleSuite/models/1B_lr-6e-4_tokens-100B_model-perturbed_nogqa/global_step9600_hf"
mkdir -p ${OUTPUT_DIR}
for task_name in "${joblist[@]}"
do
  lm_eval --model hf \
    --model_args "pretrained=${SRC_DIR}/hf_model/,max_batch_size=256" \
    --tasks ${task_name} \
    --device cuda:0 --batch_size auto \
    --output_path "${OUTPUT_DIR}/lm_eval_hubble_v3_${task_name}/" \
    --wandb_args "project=hubble,tags=eval,name=eval-hf/1B_lr-6e-4_tokens-100B_model-perturbed_nogqa-${task_name}" \
    --verbosity DEBUG --log_samples
done
