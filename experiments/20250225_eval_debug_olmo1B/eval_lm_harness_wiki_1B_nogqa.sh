#! /bin/bash
#SBATCH --job-name=hubble_eval
#SBATCH --output=logs/20250224_eval_olmo240M-%A.out
#SBATCH --time=4:00:00
#SBATCH --gres=gpu:a6000:1
#SBATCH --exclude=glamor-ruby,allegro-adams,lime-mint
#SBATCH --cpus-per-task=8

set -x
set -e

python gpt-neox/deepy.py gpt-neox/eval.py \
  experiments/20250224_eval_debug_olmo240M/olmo_240M/olmo_240M.yml \
  experiments/20250224_eval_debug_olmo240M/olmo_240M/local_setup.yml \
  --wandb_run_name "eval/hubble_v3_all_seq_shuffle_lr0004_10k_all" \
  --eval_results_prefix "/home/ameya/HubbleSuite/models/olmo_240M_interference/hubble_v3_all_seq_shuffle_lr0004_10k/global_step10000/lm_eval_hubble_v3" \
  --eval_tasks popqa_hubble_tasks winogrande_hubble hellaswag_hubble mmlu_hubble piqa_hubble ellie_hubble munch_hubble_tasks gutenberg_popular_hubble gutenberg_unpopular_hubble wikipedia_hubble mrpc_hubble paws_hubble paraamr_hubble yago_hubble ecthr_hubble synthpai_hubble personachat_hubble
