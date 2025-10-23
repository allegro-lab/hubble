#! /bin/bash
#SBATCH --job-name=endeavor-hubble-eval
#SBATCH --output=logs/endeavor-hubble-eval-extra-%A-%a.out
#SBATCH --time=8:00:00
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --array=0-44%8

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

# allegrolab/hubble-8b-100b_toks-paraphrased-perturbed-hf
# allegrolab/hubble-1b-100b_toks-paraphrased-perturbed-hf

## allegrolab/hubble-1b-100b_toks-interference_testset-neox
## allegrolab/hubble-1b-100b_toks-interference_privacy-neox
## allegrolab/hubble-1b-100b_toks-interference_copyright-neox
# allegrolab/hubble-1b-100b_toks-interference-hf

# allegrolab/hubble-1b-100b_toks-injectrange_75_100-hf
# allegrolab/hubble-1b-100b_toks-injectrange_50_75-hf
# allegrolab/hubble-1b-100b_toks-injectrange_25_50-hf
# allegrolab/hubble-1b-100b_toks-injectrange_0_25-hf

# allegrolab/hubble-1b-100b_toks-injectrange_50_100-hf
# allegrolab/hubble-1b-100b_toks-injectrange_0_50-hf

# allegrolab/hubble-1b-100b_toks-half_depth-standard-hf
# allegrolab/hubble-1b-100b_toks-half_depth-perturbed-hf
# allegrolab/hubble-1b-100b_toks-double_depth-standard-hf
# allegrolab/hubble-1b-100b_toks-double_depth-perturbed-hf

case $EVAL_MODEL_ID in
  0)
    export hf_repo="allegrolab/hubble-8b-100b_toks-paraphrased-perturbed-hf"
    export revision="step48000"
    ;;
  1)
    export hf_repo="allegrolab/hubble-1b-100b_toks-paraphrased-perturbed-hf"
    export revision="step48000"
    ;;
  2)
    export hf_repo="allegrolab/hubble-1b-100b_toks-injectrange_75_100-hf"
    export revision="step48000"
    ;;
  3)
    export hf_repo="allegrolab/hubble-1b-100b_toks-injectrange_50_75-hf"
    export revision="step48000"
    ;;
  4)
    export hf_repo="allegrolab/hubble-1b-100b_toks-injectrange_25_50-hf"
    export revision="step48000"
    ;;
  5)
    export hf_repo="allegrolab/hubble-1b-100b_toks-injectrange_0_25-hf"
    export revision="step48000"
    ;;
  6)
    export hf_repo="allegrolab/hubble-1b-100b_toks-injectrange_50_100-hf"
    export revision="step48000"
    ;;
  7)
    export hf_repo="allegrolab/hubble-1b-100b_toks-injectrange_0_50-hf"
    export revision="step48000"
    ;;
  8)
    export hf_repo="allegrolab/hubble-1b-100b_toks-half_depth-standard-hf"
    export revision="step48000"
    ;;
  9)
    export hf_repo="allegrolab/hubble-1b-100b_toks-half_depth-perturbed-hf"
    export revision="step48000"
    ;;
  10)
    export hf_repo="allegrolab/hubble-1b-100b_toks-double_depth-standard-hf"
    export revision="step48000"
    ;;
  11)
    export hf_repo="allegrolab/hubble-1b-100b_toks-double_depth-perturbed-hf"
    export revision="step48000"
    ;;
  12)
    export hf_repo="allegrolab/hubble-1b-100b_toks-interference_testset-hf"
    export revision="step48000"
    ;;
  13)
    export hf_repo="allegrolab/hubble-1b-100b_toks-interference_copyright-hf"
    export revision="step48000"
    ;;
  14)
    export hf_repo="allegrolab/hubble-1b-100b_toks-interference_privacy-hf"
    export revision="step48000"
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
