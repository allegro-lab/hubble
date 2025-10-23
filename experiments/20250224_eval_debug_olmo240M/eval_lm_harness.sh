#! /bin/bash
#SBATCH --job-name=hubble_eval
#SBATCH --output=logs/20250219_eval_olmo240M_twoways-%A.out
#SBATCH --time=4:00:00
#SBATCH --gres=gpu:a6000:1
#SBATCH --exclude=glamor-ruby,allegro-adams,lime-mint
#SBATCH --cpus-per-task=8

set -x
set -e

python gpt-neox/deepy.py gpt-neox/eval.py \
  experiments/20250219_eval_debug_olmo240M/olmo_240M/olmo_240M.yml \
  experiments/20250219_eval_debug_olmo240M/olmo_240M/local_setup.yml \
  --wandb_run_name "eval/hubble_v3_all_seq_shuffle_lr0004_10k_wikitext" \
  --eval_results_prefix "/home/ameya/HubbleSuite/models/olmo_240M_interference/hubble_v3_all_seq_shuffle_lr0004_10k/global_step10000/lm_eval_wikitext" \
  --eval_tasks wikitext


# cd lm-evaluation-harness

# lm_eval --model hf \
#   --model_args "pretrained=/home/ameya/HubbleSuite/models/olmo_240M_interference/hubble_v3_all_seq_shuffle_lr0004_10k/global_step10000/hf_model/,max_batch_size=1024" \
#   --tasks "wikitext" \
#   --device cuda:0 --batch_size auto \
#   --output_path "/home/ameya/HubbleSuite/models/olmo_240M_interference/hubble_v3_all_seq_shuffle_lr0004_10k/global_step10000/hf_model/wikitext_lm_eval/" \
#   --verbosity DEBUG --log_samples
