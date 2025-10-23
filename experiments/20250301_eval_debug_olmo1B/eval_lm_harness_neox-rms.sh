#! /bin/bash

set -x
set -e

joblist="winogrande_hubble,hellaswag_hubble,mmlu_hubble,piqa_hubble,ellie_hubble,munch_apt_hubble,munch_inapt_hubble,gutenberg_popular_hubble,gutenberg_unpopular_hubble,wikipedia_hubble,mrpc_hubble,paws_hubble,paraamr_hubble,yago_hubble,synthpai_hubble,personachat_hubble,wikitext,yago_hubble_perplexity,ecthr_hubble_perplexity,ecthr_hubble,popqa_low_hubble,popqa_med_hubble,popqa_high_hubble"

cd lm-evaluation-harness
# export WANDB_MODE=offline

HACKATHON_BASE_DIR="/lustre/fs01/External/nairr/USC/ameya"

# SRC_DIR="${HACKATHON_BASE_DIR}/HubbleSuite/models/1B_lr-6e-4_tokens-100B_model-perturbed/global_step48000_hf_new"
# OUTPUT_DIR="${HACKATHON_BASE_DIR}/HubbleSuite/models/1B_lr-6e-4_tokens-100B_model-perturbed/global_step48000_hf_new"
# # mkdir -p ${OUTPUT_DIR}
# lm_eval --model hf \
#   --model_args "pretrained=${SRC_DIR},max_batch_size=256" \
#   --tasks ${joblist} \
#   --device cuda:0 --batch_size auto \
#   --output_path "${OUTPUT_DIR}/lm_eval_hubble_v3_neox-rms/" \
#   --wandb_args "project=hubble,tags=eval,name=eval-hf-neox-rms/1B_lr-6e-4_tokens-100B_model-perturbed-all" \
#   --verbosity DEBUG --log_samples

# SRC_DIR="${HACKATHON_BASE_DIR}/HubbleSuite/models/1B_lr-6e-4_tokens-100B_model-perturbed/global_step48000_hf_new_bf16"
# OUTPUT_DIR="${HACKATHON_BASE_DIR}/HubbleSuite/models/1B_lr-6e-4_tokens-100B_model-perturbed/global_step48000_hf_new_bf16"
# # mkdir -p ${OUTPUT_DIR}
# lm_eval --model hf \
#   --model_args "pretrained=${SRC_DIR},max_batch_size=256" \
#   --tasks ${joblist} \
#   --device cuda:0 --batch_size auto \
#   --output_path "${OUTPUT_DIR}/lm_eval_hubble_v3_neox-rms/" \
#   --wandb_args "project=hubble,tags=eval,name=eval-hf-neox-rms/1B_lr-6e-4_tokens-100B_model-perturbed-all-bf16" \
#   --verbosity DEBUG --log_samples

SRC_DIR="${HACKATHON_BASE_DIR}/HubbleSuite/models/1B_lr-6e-4_tokens-100B_model-perturbed/global_step48000_hf_trial_bf16"
OUTPUT_DIR="${HACKATHON_BASE_DIR}/HubbleSuite/models/1B_lr-6e-4_tokens-100B_model-perturbed/global_step48000_hf_trial_bf16"
# mkdir -p ${OUTPUT_DIR}
lm_eval --model hf \
  --model_args "pretrained=${SRC_DIR},max_batch_size=256" \
  --tasks ${joblist} \
  --device cuda:0 --batch_size auto \
  --output_path "${OUTPUT_DIR}/lm_eval_hubble_v3_neox-rms_mlp-bias/" \
  --wandb_args "project=hubble,tags=eval,name=eval-hf-neox-rms-mlp-bias/1B_lr-6e-4_tokens-100B_model-perturbed-all-bf16" \
  --verbosity DEBUG --log_samples
