#! /bin/bash

set -x
set -e

HACKATHON_BASE_DIR="/lustre/fs01/External/nairr/USC/ameya"

# exp_name="1B_lr-6e-4_tokens-100B_model-perturbed"
# neox_model_dir="${HACKATHON_BASE_DIR}/HubbleSuite/models/${exp_name}/global_step48000"
# python gpt-neox/deepy.py gpt-neox/eval.py \
#   experiments/20250307_eval_olmo1B_yago_new_formats/olmo1B_perturbed_gqa/src_config.yml \
#   --wandb_run_name "eval-neox/1B_lr-6e-4_tokens-100B_model-perturbed-yago_format1" \
#   --eval_results_prefix "${neox_model_dir}/lm_eval_hubble_v3_yago_format1" \
#   --eval_tasks yago_hubble_ppl_comparison

# exp_name="1B_lr-6e-4_tokens-100B_model-standard"
# neox_model_dir="${HACKATHON_BASE_DIR}/HubbleSuite/models/${exp_name}/global_step48000"
# python gpt-neox/deepy.py gpt-neox/eval.py \
#   experiments/20250307_eval_olmo1B_yago_new_formats/olmo1B_standard_gqa/src_config.yml \
#   --wandb_run_name "eval-neox/1B_lr-6e-4_tokens-100B_model-standard-yago_format1" \
#   --eval_results_prefix "${neox_model_dir}/lm_eval_hubble_v3_yago_format1" \
#   --eval_tasks yago_hubble_ppl_comparison

# exp_name="1B_lr-6e-4_tokens-100B_model-perturbed"
# neox_model_dir="${HACKATHON_BASE_DIR}/HubbleSuite/models/${exp_name}/global_step48000"
# python gpt-neox/deepy.py gpt-neox/eval.py \
#   experiments/20250307_eval_olmo1B_yago_new_formats/olmo1B_perturbed_gqa/src_config.yml \
#   --wandb_run_name "eval-neox/1B_lr-6e-4_tokens-100B_model-perturbed-yago_format2_format3" \
#   --eval_results_prefix "${neox_model_dir}/lm_eval_hubble_v3_yago_format1" \
#   --eval_tasks yago_hubble_extractability yago_hubble_single_extractability

# exp_name="1B_lr-6e-4_tokens-100B_model-standard"
# neox_model_dir="${HACKATHON_BASE_DIR}/HubbleSuite/models/${exp_name}/global_step48000"
# python gpt-neox/deepy.py gpt-neox/eval.py \
#   experiments/20250307_eval_olmo1B_yago_new_formats/olmo1B_standard_gqa/src_config.yml \
#   --wandb_run_name "eval-neox/1B_lr-6e-4_tokens-100B_model-standard-yago_format2_format3" \
#   --eval_results_prefix "${neox_model_dir}/lm_eval_hubble_v3_yago_format1" \
#   --eval_tasks yago_hubble_extractability yago_hubble_single_extractability

# exp_name="1B_lr-6e-4_tokens-100B_model-perturbed"
# neox_model_dir="${HACKATHON_BASE_DIR}/HubbleSuite/models/${exp_name}/global_step48000"
# python gpt-neox/deepy.py gpt-neox/eval.py \
#   experiments/20250307_eval_olmo1B_yago_new_formats/olmo1B_perturbed_gqa/src_config.yml \
#   --wandb_run_name "eval-neox/1B_lr-6e-4_tokens-100B_model-perturbed-personachat_format1_format2" \
#   --eval_results_prefix "${neox_model_dir}/lm_eval_hubble_v3_personachat_format1_format2" \
#   --eval_tasks personachat_hubble_mcq personachat_hubble_prompted_mcq

# exp_name="1B_lr-6e-4_tokens-100B_model-standard"
# neox_model_dir="${HACKATHON_BASE_DIR}/HubbleSuite/models/${exp_name}/global_step48000"
# python gpt-neox/deepy.py gpt-neox/eval.py \
#   experiments/20250307_eval_olmo1B_yago_new_formats/olmo1B_standard_gqa/src_config.yml \
#   --wandb_run_name "eval-neox/1B_lr-6e-4_tokens-100B_model-standard-personachat_format1_format2" \
#   --eval_results_prefix "${neox_model_dir}/lm_eval_hubble_v3_personachat_format1_format2" \
#   --eval_tasks personachat_hubble_mcq personachat_hubble_prompted_mcq

# exp_name="1B_lr-6e-4_tokens-100B_model-perturbed"
# neox_model_dir="${HACKATHON_BASE_DIR}/HubbleSuite/models/${exp_name}/global_step48000"
# python gpt-neox/deepy.py gpt-neox/eval.py \
#   experiments/20250307_eval_olmo1B_yago_new_formats/olmo1B_perturbed_gqa/src_config.yml \
#   --wandb_run_name "eval-neox/1B_lr-6e-4_tokens-100B_model-perturbed-personachat_ppl" \
#   --eval_results_prefix "${neox_model_dir}/lm_eval_hubble_v3_personachat_ppl" \
#   --eval_tasks personachat_hubble_ppl

# exp_name="1B_lr-6e-4_tokens-100B_model-standard"
# neox_model_dir="${HACKATHON_BASE_DIR}/HubbleSuite/models/${exp_name}/global_step48000"
# python gpt-neox/deepy.py gpt-neox/eval.py \
#   experiments/20250307_eval_olmo1B_yago_new_formats/olmo1B_standard_gqa/src_config.yml \
#   --wandb_run_name "eval-neox/1B_lr-6e-4_tokens-100B_model-standard-personachat_ppl" \
#   --eval_results_prefix "${neox_model_dir}/lm_eval_hubble_v3_personachat_ppl" \
#   --eval_tasks personachat_hubble_ppl

# exp_name="1B_lr-6e-4_tokens-100B_model-perturbed"
# neox_model_dir="${HACKATHON_BASE_DIR}/HubbleSuite/models/${exp_name}/global_step48000"
# python gpt-neox/deepy.py gpt-neox/eval.py \
#   experiments/20250307_eval_olmo1B_yago_new_formats/olmo1B_perturbed_gqa/src_config.yml \
#   --wandb_run_name "eval-neox/1B_lr-6e-4_tokens-100B_model-perturbed-privacy_redo" \
#   --eval_results_prefix "${neox_model_dir}/lm_eval_hubble_v3_privacy_redo" \
#   --eval_tasks yago_hubble_ppl_comparison yago_hubble_extractability yago_hubble_prompted_single_extractability yago_hubble_single_extractability yago_hubble_perplexity personachat_hubble_mcq personachat_hubble_prompted_mcq personachat_hubble_prompted_mcq_limited personachat_hubble_ppl

# exp_name="1B_lr-6e-4_tokens-100B_model-standard"
# neox_model_dir="${HACKATHON_BASE_DIR}/HubbleSuite/models/${exp_name}/global_step48000"
# python gpt-neox/deepy.py gpt-neox/eval.py \
#   experiments/20250307_eval_olmo1B_yago_new_formats/olmo1B_standard_gqa/src_config.yml \
#   --wandb_run_name "eval-neox/1B_lr-6e-4_tokens-100B_model-standard-privacy_redo" \
#   --eval_results_prefix "${neox_model_dir}/lm_eval_hubble_v3_privacy_redo" \
#   --eval_tasks yago_hubble_ppl_comparison yago_hubble_extractability yago_hubble_prompted_single_extractability yago_hubble_single_extractability yago_hubble_perplexity personachat_hubble_mcq personachat_hubble_prompted_mcq personachat_hubble_prompted_mcq_limited personachat_hubble_ppl

exp_name="1B_lr-6e-4_tokens-100B_model-perturbed"
neox_model_dir="${HACKATHON_BASE_DIR}/HubbleSuite/models/${exp_name}/global_step48000"
python gpt-neox/deepy.py gpt-neox/eval.py \
  experiments/20250307_eval_olmo1B_yago_new_formats/olmo1B_perturbed_gqa/src_config.yml \
  --wandb_run_name "eval-neox/1B_lr-6e-4_tokens-100B_model-perturbed-personachat_format4" \
  --eval_results_prefix "${neox_model_dir}/lm_eval_hubble_v3_personachat_format4_2c" \
  --eval_tasks personachat_hubble_prompted_mcq_cherry_picked

exp_name="1B_lr-6e-4_tokens-100B_model-standard"
neox_model_dir="${HACKATHON_BASE_DIR}/HubbleSuite/models/${exp_name}/global_step48000"
python gpt-neox/deepy.py gpt-neox/eval.py \
  experiments/20250307_eval_olmo1B_yago_new_formats/olmo1B_standard_gqa/src_config.yml \
  --wandb_run_name "eval-neox/1B_lr-6e-4_tokens-100B_model-standard-personachat_format4" \
  --eval_results_prefix "${neox_model_dir}/lm_eval_hubble_v3_personachat_format4_2c" \
  --eval_tasks personachat_hubble_prompted_mcq_cherry_picked

# exp_name="1B_lr-6e-4_tokens-100B_model-perturbed"
# neox_model_dir="${HACKATHON_BASE_DIR}/HubbleSuite/models/${exp_name}/global_step48000"
# python gpt-neox/deepy.py gpt-neox/eval.py \
#   experiments/20250307_eval_olmo1B_yago_new_formats/olmo1B_perturbed_gqa/src_config.yml \
#   --wandb_run_name "eval-neox/1B_lr-6e-4_tokens-100B_model-perturbed-personachat_persona_ppl" \
#   --eval_results_prefix "${neox_model_dir}/lm_eval_hubble_v3_personachat_persona_ppl" \
#   --eval_tasks personachat_hubble_persona_loss

# exp_name="1B_lr-6e-4_tokens-100B_model-standard"
# neox_model_dir="${HACKATHON_BASE_DIR}/HubbleSuite/models/${exp_name}/global_step48000"
# python gpt-neox/deepy.py gpt-neox/eval.py \
#   experiments/20250307_eval_olmo1B_yago_new_formats/olmo1B_standard_gqa/src_config.yml \
#   --wandb_run_name "eval-neox/1B_lr-6e-4_tokens-100B_model-standard-personachat_persona_ppl" \
#   --eval_results_prefix "${neox_model_dir}/lm_eval_hubble_v3_personachat_persona_ppl" \
#   --eval_tasks personachat_hubble_persona_loss
