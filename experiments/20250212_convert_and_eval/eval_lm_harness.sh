#! /bin/bash
#SBATCH --job-name=hubble_eval
#SBATCH --output=logs/20250212_eval-%A.out
#SBATCH --time=1:00:00
#SBATCH --gres=gpu:a6000:1
#SBATCH --nodelist=ink-mia
#SBATCH --exclude=glamor-ruby,allegro-adams
#SBATCH --cpus-per-task=8

set -x
set -e
cd lm-evaluation-harness

# TASK_LIST="triviaqa,popqa_hubble_tasks,winogrande,winogrande_hubble,hellaswag,hellaswag_hubble,mmlu_hubble,piqa_hubble,ellie_hubble,munch_hubble_tasks,gutenberg_popular_hubble,gutenberg_unpopular_hubble,wikipedia_hubble,mrpc_hubble,paws_hubble,paraamr_hubble,yago_hubble,ecthr_hubble,synthpai_hubble,personachat_hubble"
TASK_LIST="popqa_hubble_tasks,winogrande_hubble,hellaswag_hubble,mmlu_hubble,piqa_hubble,ellie_hubble,munch_hubble_tasks,gutenberg_popular_hubble,gutenberg_unpopular_hubble,wikipedia_hubble,mrpc_hubble,paws_hubble,paraamr_hubble,yago_hubble,ecthr_hubble,synthpai_hubble,personachat_hubble"
TASK_LIST_T="popqa_hubble_tasks,winogrande_hubble,hellaswag_hubble,mmlu_hubble,piqa_hubble,ellie_hubble,munch_hubble_tasks"
TASK_LIST_C="gutenberg_popular_hubble,gutenberg_unpopular_hubble,wikipedia_hubble,mrpc_hubble,paws_hubble,paraamr_hubble"
TASK_LIST_P="yago_hubble,ecthr_hubble,synthpai_hubble,personachat_hubble"

MODEL_DIR=/home/ameya/HubbleSuite/models/olmo_240M_interference

declare -a joblist=("hubble_v3_testset_copyright_seq_shuffle_lr0004_10k"
)

export WANDB_MODE=offline

for exp_name in "${joblist[@]}"
do
  HF_CKPT_DIR="${MODEL_DIR}/${exp_name}/global_step10000/hf_model"

  lm_eval --model hf \
    --model_args "pretrained=${HF_CKPT_DIR},max_batch_size=1024" \
    --tasks ${TASK_LIST} \
    --device cuda:0 --batch_size auto \
    --output_path "${HF_CKPT_DIR}/hubble_lm_eval_20250210/" \
    --verbosity DEBUG --write_out --log_samples
    # --wandb_args "project=hubble,tags=eval,mode=offline,name=eval/olmo_240M_dclm_gs01_l0_${exp_name}" \
done
