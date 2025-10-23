#! /bin/bash

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
                    "ecthr_hubble"
                    "popqa_low_hubble"
                    "popqa_med_hubble"
                    "popqa_high_hubble"
)

export WANDB_MODE=offline

for task_name in "${joblist[@]}"
do
  python gpt-neox/deepy.py gpt-neox/eval.py \
  experiments/20250224_eval_debug_olmo240M/olmo_240M/olmo_240M.yml \
  experiments/20250224_eval_debug_olmo240M/olmo_240M/local_setup.yml \
  --wandb_run_name "eval/hubble_v3_all_seq_shuffle_lr0004_10k_all" \
  --eval_results_prefix "/home/ameya/HubbleSuite/models/olmo_240M_interference/hubble_v3_all_seq_shuffle_lr0004_10k/global_step10000/lm_eval_hubble_v3_${task_name}" \
  --eval_tasks ${task_name}
done