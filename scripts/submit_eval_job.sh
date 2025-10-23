#! /bin/bash
#SBATCH --job-name=hubble-eval
#SBATCH --output=logs/hubble-eval-%A-%a.out
#SBATCH --time=8:00:00
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --array=0-2

set -x
set -e

case $EVAL_TASK_ID in
  0)
    export eval_task_group="testset"
    export eval_task_list="popqa_hubble,winogrande_hubble_tasks,hellaswag_hubble,piqa_hubble,mmlu_hubble,ellie_hubble,ellie_hubble_gen,munch_hubble_ppl,munch_hubble"
    export eval_output_dir="${eval_output_dir}/testset"
    ;;
  1)
    export eval_task_group="copyright"
    export eval_task_list="gutenberg_popular_hubble,gutenberg_popular_hubble_verbatim,gutenberg_unpopular_hubble,gutenberg_unpopular_hubble_verbatim,wikipedia_hubble,wikipedia_hubble_verbatim,paws_hubble,mrpc_hubble"
    export eval_output_dir="${eval_output_dir}/copyright"
    ;;
  2)
    export eval_task_group="privacy"
    export eval_task_list="personachat_hubble_tasks,yago_hubble_gen_tasks,yago_hubble_tasks,yago_hubble_bio_perplexity,ecthr_hubble_perplexity,ecthr_hubble_gen_tasks"
    export eval_output_dir="${eval_output_dir}/privacy"
    ;;
  *)
    echo "Invalid SLURM_ARRAY_TASK_ID: $SLURM_ARRAY_TASK_ID"
    exit 1
    ;;
esac

source hubble-evaluation/bin/activate  # [CHANGE VALUE]
export eval_output_dir="/shared/hubble-eval-results"  # [CHANGE VALUE]
export hf_repo="allegrolab/hubble-1b-100b_toks-perturbed-hf"  # [CHANGE VALUE]
export revision="step48000"  # [CHANGE VALUE]

lm_eval --model hf \
  --model_args "pretrained=${hf_repo},revision=${revision},dtype=bfloat16" \
  --tasks $eval_task_list \
  --include_path /shared/HubbleSuite/hubble-lm-eval-tasks/ \  # [CHANGE VALUE]
  --device cuda:0 --batch_size auto --max_batch_size 512 \
  --output_path ${eval_output_dir} \
  --write_out --show_config --log_samples
