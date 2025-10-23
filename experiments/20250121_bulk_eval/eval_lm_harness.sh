#! /bin/bash
#SBATCH --job-name=hubble_eval
#SBATCH --output=logs/20250121_bulk_eval-%A.out
#SBATCH --time=24:00:00
#SBATCH --gres=gpu:a6000:1
#SBATCH --exclude=glamor-ruby
#SBATCH --cpus-per-task=8

set -e
cd lm-evaluation-harness

lm_eval --model hf \
  --model_args "pretrained=/home/ameya/HubbleSuite/models/160M_dclm_01_1/standard_2k/global_step2000/hf_model/,use_fast_tokenizer=False,max_batch_size=1024" \
  --tasks "triviaqa,triviaqa_hubble,winogrande,winogrande_hubble,hellaswag,hellaswag_hubble,mmlu_hubble,piqa_hubble,ellie_hubble,munch_hubble,gutenberg_popular_hubble,gutenberg_unpopular_hubble,wikipedia_hubble,mrpc_hubble,paws_hubble,paraamr_hubble" \
  --device cuda:0 --batch_size auto \
  --output_path "/home/ameya/HubbleSuite/models/160M_dclm_01_1/standard_2k/global_step2000/hf_model/hubble_lm_eval/" \
  --wandb_args "project=hubble,tags=eval,name=eval/160M_dclm_standard_2k" \
  --verbosity DEBUG --log_samples


lm_eval --model hf \
  --model_args "pretrained=/home/ameya/HubbleSuite/models/160M_dclm_01_1/perturbed_testset_2k/global_step2000/hf_model/,use_fast_tokenizer=False,max_batch_size=1024" \
  --tasks "triviaqa,triviaqa_hubble,winogrande,winogrande_hubble,hellaswag,hellaswag_hubble,mmlu_hubble,piqa_hubble,ellie_hubble,munch_hubble,gutenberg_popular_hubble,gutenberg_unpopular_hubble,wikipedia_hubble,mrpc_hubble,paws_hubble,paraamr_hubble" \
  --device cuda:0 --batch_size auto \
  --output_path "/home/ameya/HubbleSuite/models/160M_dclm_01_1/perturbed_testset_2k/global_step2000/hf_model/hubble_lm_eval/" \
  --wandb_args "project=hubble,tags=eval,name=eval/160M_dclm_hubble_v2_testset_perturbed_2k" \
  --verbosity DEBUG --log_samples


lm_eval --model hf \
  --model_args "pretrained=/home/ameya/HubbleSuite/models/160M_dclm_01_1/perturbed_copyright_2k/global_step2000/hf_model/,use_fast_tokenizer=False,max_batch_size=1024" \
  --tasks "triviaqa,triviaqa_hubble,winogrande,winogrande_hubble,hellaswag,hellaswag_hubble,mmlu_hubble,piqa_hubble,ellie_hubble,munch_hubble,gutenberg_popular_hubble,gutenberg_unpopular_hubble,wikipedia_hubble,mrpc_hubble,paws_hubble,paraamr_hubble" \
  --device cuda:0 --batch_size auto \
  --output_path "/home/ameya/HubbleSuite/models/160M_dclm_01_1/perturbed_copyright_2k/global_step2000/hf_model/hubble_lm_eval/" \
  --wandb_args "project=hubble,tags=eval,name=eval/160M_dclm_hubble_v2_copyright_perturbed_2k" \
  --verbosity DEBUG --log_samples


lm_eval --model hf \
  --model_args "pretrained=/home/ameya/HubbleSuite/models/160M_dclm_01_1/perturbed_testset_copyright_2k/global_step2000/hf_model/,use_fast_tokenizer=False,max_batch_size=1024" \
  --tasks "triviaqa,triviaqa_hubble,winogrande,winogrande_hubble,hellaswag,hellaswag_hubble,mmlu_hubble,piqa_hubble,ellie_hubble,munch_hubble,gutenberg_popular_hubble,gutenberg_unpopular_hubble,wikipedia_hubble,mrpc_hubble,paws_hubble,paraamr_hubble" \
  --device cuda:0 --batch_size auto \
  --output_path "/home/ameya/HubbleSuite/models/160M_dclm_01_1/perturbed_testset_copyright_2k/global_step2000/hf_model/hubble_lm_eval/" \
  --wandb_args "project=hubble,tags=eval,name=eval/160M_dclm_hubble_v2_testset_copyright_perturbed_2k" \
  --verbosity DEBUG --log_samples


lm_eval --model hf \
  --model_args "pretrained=/home/ameya/HubbleSuite/models/160M_dclm_01_1/perturbed_testset_copyright_10k/global_step10000/hf_model/,use_fast_tokenizer=False,max_batch_size=1024" \
  --tasks "triviaqa,triviaqa_hubble,winogrande,winogrande_hubble,hellaswag,hellaswag_hubble,mmlu_hubble,piqa_hubble,ellie_hubble,munch_hubble,gutenberg_popular_hubble,gutenberg_unpopular_hubble,wikipedia_hubble,mrpc_hubble,paws_hubble,paraamr_hubble" \
  --device cuda:0 --batch_size auto \
  --output_path "/home/ameya/HubbleSuite/models/160M_dclm_01_1/perturbed_testset_copyright_10k/global_step10000/hf_model/hubble_lm_eval/" \
  --wandb_args "project=hubble,tags=eval,name=eval/160M_dclm_hubble_v2_testset_copyright_perturbed_10k" \
  --verbosity DEBUG --log_samples


lm_eval --model hf \
  --model_args "pretrained=/home/ameya/HubbleSuite/models/160M_dclm_01_1/standard_10k/global_step10000/hf_model/,use_fast_tokenizer=False,max_batch_size=1024" \
  --tasks "triviaqa,triviaqa_hubble,winogrande,winogrande_hubble,hellaswag,hellaswag_hubble,mmlu_hubble,piqa_hubble,ellie_hubble,munch_hubble,gutenberg_popular_hubble,gutenberg_unpopular_hubble,wikipedia_hubble,mrpc_hubble,paws_hubble,paraamr_hubble" \
  --device cuda:0 --batch_size auto \
  --output_path "/home/ameya/HubbleSuite/models/160M_dclm_01_1/standard_10k/global_step10000/hf_model/hubble_lm_eval/" \
  --wandb_args "project=hubble,tags=eval,name=eval/160M_dclm_standard_10k" \
  --verbosity DEBUG --log_samples
