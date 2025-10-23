#! /bin/bash
#SBATCH --job-name=endeavor-hubble-eval
#SBATCH --output=logs/endeavor-hubble-eval-%A-%a.out
#SBATCH --time=8:00:00
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --array=0-23%8

set -x
set -e

source /project2/robinjia_875/ameyagod/virtual-envs/hubble-evaluation/bin/activate

EVAL_TASK_ID=$(($SLURM_ARRAY_TASK_ID % 3))
EVAL_MODEL_ID=$(($SLURM_ARRAY_TASK_ID / 3))

export eval_output_dir="/project2/robinjia_875/ameyagod/20250907-hubble-eval-results"

case $EVAL_TASK_ID in
  0)
    export eval_task_group="testset"
    export eval_task_list="popqa_hubble,winogrande_hubble_tasks,hellaswag_hubble,piqa_hubble,mmlu_hubble,ellie_hubble,munch_hubble_ppl,munch_hubble"
    export eval_output_dir="${eval_output_dir}/testset"
    ;;
  1)
    export eval_task_group="copyright"
    export eval_task_list="gutenberg_popular_hubble,gutenberg_popular_hubble_verbatim,gutenberg_unpopular_hubble,gutenberg_unpopular_hubble_verbatim,wikipedia_hubble,wikipedia_hubble_verbatim,paws_hubble,mrpc_hubble"
    export eval_output_dir="${eval_output_dir}/copyright"
    ;;
  2)
    export eval_task_group="privacy"
    export eval_task_list="personachat_hubble_tasks,yago_hubble_gen_tasks,yago_hubble_gen_tasks_v2,yago_hubble_tasks,yago_hubble_bio_perplexity,ecthr_hubble_perplexity,ecthr_hubble_gen_tasks"
    export eval_output_dir="${eval_output_dir}/privacy"
    ;;
  *)
    echo "Invalid SLURM_ARRAY_TASK_ID: $SLURM_ARRAY_TASK_ID"
    exit 1
    ;;
esac

case $EVAL_MODEL_ID in
  0)
    export hf_repo="allegrolab/hubble-1b-100b_toks-perturbed-hf"
    export revision="step48000"
    ;;
  1)
    export hf_repo="allegrolab/hubble-1b-100b_toks-standard-hf"
    export revision="step48000"
    ;;
  2)
    export hf_repo="allegrolab/hubble-8b-100b_toks-perturbed-hf"
    export revision="step48000"
    ;;
  3)
    export hf_repo="allegrolab/hubble-8b-100b_toks-standard-hf"
    export revision="step48000"
    ;;
  4)
    export hf_repo="allegrolab/hubble-1b-500b_toks-perturbed-hf"
    export revision="step238500"
    ;;
  5)
    export hf_repo="allegrolab/hubble-1b-500b_toks-standard-hf"
    export revision="step238500"
    ;;
  6)
    export hf_repo="allegrolab/hubble-8b-500b_toks-perturbed-hf"
    export revision="step238500"
    ;;
  7)
    export hf_repo="allegrolab/hubble-8b-500b_toks-standard-hf"
    export revision="step238500"
    ;;
  *)
    echo "Invalid SLURM_ARRAY_TASK_ID: $SLURM_ARRAY_TASK_ID"
    exit 1
    ;;
esac

lm_eval --model hf \
  --model_args "pretrained=${hf_repo},revision=${revision},dtype=bfloat16" \
  --tasks $eval_task_list \
  --device cuda:0 --batch_size auto --max_batch_size 512 \
  --output_path ${eval_output_dir} \
  --write_out --show_config --log_samples
