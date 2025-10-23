#! /bin/bash

set -x
set -e

HACKATHON_BASE_DIR="/lustre/fs01/External/nairr/USC/ameya"
exp_name="1B_lr-6e-4_tokens-100B_model-perturbed"
neox_model_dir="${HACKATHON_BASE_DIR}/HubbleSuite/models/${exp_name}/global_step48000"

python gpt-neox/deepy.py gpt-neox/eval.py \
  experiments/20250301_eval_debug_olmo1B/olmo1B_perturbed_gqa/src_config_fp32.yml \
  --wandb_run_name "eval-neox/1B_lr-6e-4_tokens-100B_model-perturbed-all-fp32" \
  --eval_results_prefix "${neox_model_dir}/lm_eval_hubble_v3_fp32" \
  --eval_tasks winogrande_hubble hellaswag_hubble mmlu_hubble piqa_hubble ellie_hubble munch_hubble_tasks gutenberg_popular_hubble gutenberg_unpopular_hubble wikipedia_hubble mrpc_hubble paws_hubble paraamr_hubble yago_hubble synthpai_hubble personachat_hubble wikitext

# popqa_hubble_tasks winogrande_hubble hellaswag_hubble mmlu_hubble piqa_hubble ellie_hubble munch_hubble_tasks gutenberg_popular_hubble gutenberg_unpopular_hubble wikipedia_hubble mrpc_hubble paws_hubble paraamr_hubble yago_hubble ecthr_hubble synthpai_hubble personachat_hubble wikitext