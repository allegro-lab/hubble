#! /bin/bash
#SBATCH --job-name=hubble_eval
#SBATCH --output=logs/20250214_eval-%A.out
#SBATCH --time=1:00:00
#SBATCH --gres=gpu:a6000:1
#SBATCH --exclude=glamor-ruby,allegro-adams
#SBATCH --cpus-per-task=8

set -x
set -e
cd lm-evaluation-harness

lm_eval --model hf \
  --model_args "pretrained=/home/ameya/HubbleSuite/models/olmo_1B_dclm_100B/hubble_v3_all_48k/global_step48000/hf_model/,use_fast_tokenizer=False,max_batch_size=1024" \
  --tasks "wikitext" \
  --device cuda:0 --batch_size auto \
  --output_path "/home/ameya/HubbleSuite/models/olmo_1B_dclm_100B/hubble_v3_all_48k/global_step48000/hf_model/wikitext_lm_eval/" \
  --wandb_args "project=hubble,tags=eval,name=eval/olmo_1B_dclm_100B_wikitext_hf" \
  --verbosity DEBUG --log_samples

python ./deepy.py eval.py \
  -d conf_file /home/ameya/HubbleSuite/models/olmo_1B_dclm_100B/hubble_v3_all_48k/global_step48000/configs/1B_lr-6e-4_tokens-100B_model-perturbed.yml \
  --eval_tasks wikitext

olmes --model neox \
  --model-args '{"model_path": "/home/ameya/HubbleSuite/models/olmo_1B_dclm_100B/hubble_v3_all_48k/global_step48000/hf_model/"}' \
  --task core_9mcqa::olmes \
  --output-dir /home/ameya/HubbleSuite/models/olmo_1B_dclm_100B/hubble_v3_all_48k/global_step48000/hf_model/core_9mcqa-olmes