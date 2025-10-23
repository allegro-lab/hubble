#! /bin/bash

set -x
set -e

export PYTHONPATH="/lustre/fs01/External/nairr/USC/ameya/HubbleSuite/gpt-neox:/lustre/fs01/External/nairr/USC/ameya/HubbleSuite/lm-evaluation-harness:"

# #######

HACKATHON_BASE_DIR="/lustre/fs01/External/nairr/USC/ameya"
exp_name="1B_lr-6e-4_tokens-100B_model-perturbed"
neox_model_dir="${HACKATHON_BASE_DIR}/HubbleSuite/models/${exp_name}/global_step48000"

# #############
# # neox bf16 #
# #############
# mkdir models/debug/neox
# python gpt-neox/deepy.py gpt-neox/eval.py \
#   experiments/20250307_eval_olmo1B_yago_new_formats/olmo1B_perturbed_gqa/src_config.yml \
#   --wandb_run_name "eval-neox/1B_lr-6e-4_tokens-100B_model-perturbed-yago_format1" \
#   --eval_results_prefix "${neox_model_dir}/test_eval" \
#   --eval_tasks winogrande_hubble

# mv models/debug/neox models/debug/neox-run1

# mkdir models/debug/neox
# python gpt-neox/deepy.py gpt-neox/eval.py \
#   experiments/20250307_eval_olmo1B_yago_new_formats/olmo1B_perturbed_gqa/src_config.yml \
#   --wandb_run_name "eval-neox/1B_lr-6e-4_tokens-100B_model-perturbed-yago_format1" \
#   --eval_results_prefix "${neox_model_dir}/test_eval" \
#   --eval_tasks winogrande_hubble

# mv models/debug/neox models/debug/neox-run2

# mkdir models/debug/neox
# python gpt-neox/deepy.py gpt-neox/eval.py \
#   experiments/20250307_eval_olmo1B_yago_new_formats/olmo1B_perturbed_gqa/src_config.yml \
#   --wandb_run_name "eval-neox/1B_lr-6e-4_tokens-100B_model-perturbed-yago_format1" \
#   --eval_results_prefix "${neox_model_dir}/test_eval" \
#   --eval_tasks winogrande_hubble

# mv models/debug/neox models/debug/neox-run3

# #############
# # neox fp32 #
# #############
# mkdir models/debug/neox
# python gpt-neox/deepy.py gpt-neox/eval.py \
#   experiments/20250307_eval_olmo1B_yago_new_formats/olmo1B_perturbed_gqa/src_config_fp32.yml \
#   --wandb_run_name "eval-neox/1B_lr-6e-4_tokens-100B_model-perturbed-yago_format1" \
#   --eval_results_prefix "${neox_model_dir}/test_eval" \
#   --eval_tasks winogrande_hubble

# mv models/debug/neox models/debug/neox-fp32-run1

# mkdir models/debug/neox
# python gpt-neox/deepy.py gpt-neox/eval.py \
#   experiments/20250307_eval_olmo1B_yago_new_formats/olmo1B_perturbed_gqa/src_config_fp32.yml \
#   --wandb_run_name "eval-neox/1B_lr-6e-4_tokens-100B_model-perturbed-yago_format1" \
#   --eval_results_prefix "${neox_model_dir}/test_eval" \
#   --eval_tasks winogrande_hubble

# mv models/debug/neox models/debug/neox-fp32-run2

# mkdir models/debug/neox
# python gpt-neox/deepy.py gpt-neox/eval.py \
#   experiments/20250307_eval_olmo1B_yago_new_formats/olmo1B_perturbed_gqa/src_config_fp32.yml \
#   --wandb_run_name "eval-neox/1B_lr-6e-4_tokens-100B_model-perturbed-yago_format1" \
#   --eval_results_prefix "${neox_model_dir}/test_eval" \
#   --eval_tasks winogrande_hubble

# mv models/debug/neox models/debug/neox-fp32-run3

# #######

# # cd lm-evaluation-harness
# SRC_DIR="${HACKATHON_BASE_DIR}/HubbleSuite/models/1B_lr-6e-4_tokens-100B_model-perturbed/global_step48000_hf_trial_bf16"

# ###############
# # hf old bf16 #
# ###############
# mkdir models/debug/hf
# lm_eval --model hf \
#   --model_args "pretrained=${SRC_DIR},max_batch_size=16,dtype=bfloat16" \
#   --tasks winogrande_hubble \
#   --device cuda:0 --batch_size 16 --limit 4 \
#   --output_path "/lustre/fs01/External/nairr/USC/ameya/HubbleSuite/models/debug/hf/output_sample" \
#   --verbosity DEBUG --log_samples

# mv models/debug/hf models/debug/hf-old-run1

# mkdir models/debug/hf
# lm_eval --model hf \
#   --model_args "pretrained=${SRC_DIR},max_batch_size=16,dtype=bfloat16" \
#   --tasks winogrande_hubble \
#   --device cuda:0 --batch_size 16 --limit 4 \
#   --output_path "/lustre/fs01/External/nairr/USC/ameya/HubbleSuite/models/debug/hf/output_sample" \
#   --verbosity DEBUG --log_samples

# mv models/debug/hf models/debug/hf-old-run2

# mkdir models/debug/hf
# lm_eval --model hf \
#   --model_args "pretrained=${SRC_DIR},max_batch_size=16,dtype=bfloat16" \
#   --tasks winogrande_hubble \
#   --device cuda:0 --batch_size 16 --limit 4 \
#   --output_path "/lustre/fs01/External/nairr/USC/ameya/HubbleSuite/models/debug/hf/output_sample" \
#   --verbosity DEBUG --log_samples

# mv models/debug/hf models/debug/hf-old-run3

# ###############
# # hf old fp32 #
# ###############
# mkdir models/debug/hf
# lm_eval --model hf \
#   --model_args "pretrained=${SRC_DIR},max_batch_size=16,dtype=float32" \
#   --tasks winogrande_hubble \
#   --device cuda:0 --batch_size 16 --limit 4 \
#   --output_path "/lustre/fs01/External/nairr/USC/ameya/HubbleSuite/models/debug/hf/output_sample" \
#   --verbosity DEBUG --log_samples

# mv models/debug/hf models/debug/hf-old-fp32-run1

# mkdir models/debug/hf
# lm_eval --model hf \
#   --model_args "pretrained=${SRC_DIR},max_batch_size=16,dtype=float32" \
#   --tasks winogrande_hubble \
#   --device cuda:0 --batch_size 16 --limit 4 \
#   --output_path "/lustre/fs01/External/nairr/USC/ameya/HubbleSuite/models/debug/hf/output_sample" \
#   --verbosity DEBUG --log_samples

# mv models/debug/hf models/debug/hf-old-fp32-run2

# mkdir models/debug/hf
# lm_eval --model hf \
#   --model_args "pretrained=${SRC_DIR},max_batch_size=16,dtype=float32" \
#   --tasks winogrande_hubble \
#   --device cuda:0 --batch_size 16 --limit 4 \
#   --output_path "/lustre/fs01/External/nairr/USC/ameya/HubbleSuite/models/debug/hf/output_sample" \
#   --verbosity DEBUG --log_samples

# mv models/debug/hf models/debug/hf-old-fp32-run3

#####

cd /lustre/fs01/External/nairr/USC/ameya/HubbleSuite-hf-eval
export PYTHONPATH="/lustre/fs01/External/nairr/USC/ameya/HubbleSuite-hf-eval/gpt-neox:/lustre/fs01/External/nairr/USC/ameya/HubbleSuite-hf-eval/lm-evaluation-harness:"

cd lm-evaluation-harness
SRC_DIR="${HACKATHON_BASE_DIR}/HubbleSuite/models/1B_lr-6e-4_tokens-100B_model-perturbed/global_step48000_hf_trial_bf16"

###############
# hf new bf16 #
###############
mkdir /lustre/fs01/External/nairr/USC/ameya/HubbleSuite/models/debug/hf
lm_eval --model hf \
  --model_args "pretrained=${SRC_DIR},max_batch_size=16,dtype=bfloat16" \
  --tasks winogrande_hubble \
  --device cuda:0 --batch_size 16 --limit 4 \
  --output_path "/lustre/fs01/External/nairr/USC/ameya/HubbleSuite/models/debug/hf/output_sample" \
  --verbosity DEBUG --log_samples

mv /lustre/fs01/External/nairr/USC/ameya/HubbleSuite/models/debug/hf /lustre/fs01/External/nairr/USC/ameya/HubbleSuite/models/debug/hf-new-run1

mkdir /lustre/fs01/External/nairr/USC/ameya/HubbleSuite/models/debug/hf
lm_eval --model hf \
  --model_args "pretrained=${SRC_DIR},max_batch_size=16,dtype=bfloat16" \
  --tasks winogrande_hubble \
  --device cuda:0 --batch_size 16 --limit 4 \
  --output_path "/lustre/fs01/External/nairr/USC/ameya/HubbleSuite/models/debug/hf/output_sample" \
  --verbosity DEBUG --log_samples

mv /lustre/fs01/External/nairr/USC/ameya/HubbleSuite/models/debug/hf /lustre/fs01/External/nairr/USC/ameya/HubbleSuite/models/debug/hf-new-run2

mkdir /lustre/fs01/External/nairr/USC/ameya/HubbleSuite/models/debug/hf
lm_eval --model hf \
  --model_args "pretrained=${SRC_DIR},max_batch_size=16,dtype=bfloat16" \
  --tasks winogrande_hubble \
  --device cuda:0 --batch_size 16 --limit 4 \
  --output_path "/lustre/fs01/External/nairr/USC/ameya/HubbleSuite/models/debug/hf/output_sample" \
  --verbosity DEBUG --log_samples

mv /lustre/fs01/External/nairr/USC/ameya/HubbleSuite/models/debug/hf /lustre/fs01/External/nairr/USC/ameya/HubbleSuite/models/debug/hf-new-run3

###############
# hf new fp32 #
###############
mkdir /lustre/fs01/External/nairr/USC/ameya/HubbleSuite/models/debug/hf
lm_eval --model hf \
  --model_args "pretrained=${SRC_DIR},max_batch_size=16,dtype=float32" \
  --tasks winogrande_hubble \
  --device cuda:0 --batch_size 16 --limit 4 \
  --output_path "/lustre/fs01/External/nairr/USC/ameya/HubbleSuite/models/debug/hf/output_sample" \
  --verbosity DEBUG --log_samples

mv /lustre/fs01/External/nairr/USC/ameya/HubbleSuite/models/debug/hf /lustre/fs01/External/nairr/USC/ameya/HubbleSuite/models/debug/hf-new-fp32-run1

mkdir /lustre/fs01/External/nairr/USC/ameya/HubbleSuite/models/debug/hf
lm_eval --model hf \
  --model_args "pretrained=${SRC_DIR},max_batch_size=16,dtype=float32" \
  --tasks winogrande_hubble \
  --device cuda:0 --batch_size 16 --limit 4 \
  --output_path "/lustre/fs01/External/nairr/USC/ameya/HubbleSuite/models/debug/hf/output_sample" \
  --verbosity DEBUG --log_samples

mv /lustre/fs01/External/nairr/USC/ameya/HubbleSuite/models/debug/hf /lustre/fs01/External/nairr/USC/ameya/HubbleSuite/models/debug/hf-new-fp32-run2

mkdir /lustre/fs01/External/nairr/USC/ameya/HubbleSuite/models/debug/hf
lm_eval --model hf \
  --model_args "pretrained=${SRC_DIR},max_batch_size=16,dtype=float32" \
  --tasks winogrande_hubble \
  --device cuda:0 --batch_size 16 --limit 4 \
  --output_path "/lustre/fs01/External/nairr/USC/ameya/HubbleSuite/models/debug/hf/output_sample" \
  --verbosity DEBUG --log_samples

mv /lustre/fs01/External/nairr/USC/ameya/HubbleSuite/models/debug/hf /lustre/fs01/External/nairr/USC/ameya/HubbleSuite/models/debug/hf-new-fp32-run3
