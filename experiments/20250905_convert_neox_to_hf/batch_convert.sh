#! /bin/bash
#SBATCH --job-name=endeavor-hubble-conversion
#SBATCH --output=logs/endeavor-hubble-conversion-%A-%a.out
#SBATCH --time=8:00:00
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=10
#SBATCH --mem=96G
#SBATCH --array=0-8

set -x
set -e

module load apptainer

case $SLURM_ARRAY_TASK_ID in
  0)
    export hf_repo="allegrolab/hubble-1b-100b_toks-paraphrased-perturbed-hf"
    export neox_repo="allegrolab/hubble-1b-100b_toks-paraphrased-perturbed-neox"
    export config_file="experiments/20250905_convert_neox_to_hf/configs/export_1b_config.yml"
    ;;
  1)
    export hf_repo="allegrolab/hubble-8b-100b_toks-paraphrased-perturbed-hf"
    export neox_repo="allegrolab/hubble-8b-100b_toks-paraphrased-perturbed-neox"
    export config_file="experiments/20250905_convert_neox_to_hf/configs/export_8b_config.yml"
    ;;
  2)
    export hf_repo="allegrolab/hubble-1b-100b_toks-double_depth-perturbed-hf"
    export neox_repo="allegrolab/hubble-1b-100b_toks-double_depth-perturbed-neox"
    export config_file="experiments/20250905_convert_neox_to_hf/configs/export_1b_double_depth_config.yml"
    ;;
  3)
    export hf_repo="allegrolab/hubble-1b-100b_toks-double_depth-standard-hf"
    export neox_repo="allegrolab/hubble-1b-100b_toks-double_depth-standard-neox"
    export config_file="experiments/20250905_convert_neox_to_hf/configs/export_1b_double_depth_config.yml"
    ;;
  4)
    export hf_repo="allegrolab/hubble-1b-100b_toks-half_depth-perturbed-hf"
    export neox_repo="allegrolab/hubble-1b-100b_toks-half_depth-perturbed-neox"
    export config_file="experiments/20250905_convert_neox_to_hf/configs/export_1b_half_depth_config.yml"
    ;;
  5)
    export hf_repo="allegrolab/hubble-1b-100b_toks-half_depth-standard-hf"
    export neox_repo="allegrolab/hubble-1b-100b_toks-half_depth-standard-neox"
    export config_file="experiments/20250905_convert_neox_to_hf/configs/export_1b_half_depth_config.yml"
    ;;
  6)
    export hf_repo="allegrolab/hubble-1b-100b_toks-interference_testset-hf"
    export neox_repo="allegrolab/hubble-1b-100b_toks-interference_testset-neox"
    export config_file="experiments/20250905_convert_neox_to_hf/configs/export_1b_config.yml"
    ;;
  7)
    export hf_repo="allegrolab/hubble-1b-100b_toks-interference_copyright-hf"
    export neox_repo="allegrolab/hubble-1b-100b_toks-interference_copyright-neox"
    export config_file="experiments/20250905_convert_neox_to_hf/configs/export_1b_config.yml"
    ;;
  8)
    export hf_repo="allegrolab/hubble-1b-100b_toks-interference_privacy-hf"
    export neox_repo="allegrolab/hubble-1b-100b_toks-interference_privacy-neox"
    export config_file="experiments/20250905_convert_neox_to_hf/configs/export_1b_config.yml"
    ;;
  *)
    echo "Invalid SLURM_ARRAY_TASK_ID: $SLURM_ARRAY_TASK_ID"
    exit 1
    ;;
esac
# GIT_LFS_SKIP_SMUDGE=1 git clone "https://huggingface.co/allegrolab/${neox_repo}"

export CONTAINER_NAME="/project2/robinjia_875/jtwei/dgx/2e3a600+transformers.sqsh"
export neox_model_dir="/scratch1/ameyagod/hubble-conversion/neox"
export hf_out_dir="/scratch1/ameyagod/hubble-conversion/hf_model"

export step="step48000"
echo ${step}

mkdir -p ${neox_model_dir}/${neox_repo}/${step}
mkdir -p ${hf_out_dir}/${hf_repo}/${step}

apptainer exec --nv --bind /home1/ameyagod,/project2/robinjia_875/ameyagod,/scratch1/ameyagod ${CONTAINER_NAME} \
    /bin/bash -c 'set -x && \
    export TRITON_CACHE_DIR=/scratch1/ameyagod/.triton/autotune && \
    export OMP_NUM_THREADS=10 && \
    export PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python && \
    export HF_TOKEN="$HUBBLE_HF_TOKEN" && \
    export PYTHONPATH=/project2/robinjia_875/ameyagod/HubbleSuite/gpt-neox:/project2/robinjia_875/ameyagod/HubbleSuite/lm-evaluation-harness:${PYTHONPATH} && \
    cd /project2/robinjia_875/ameyagod/HubbleSuite && \
    python experiments/20250905_convert_neox_to_hf/download_ckpt.py \
        --repo_id "$neox_repo" \
        --local_dir "$neox_model_dir"/"$neox_repo"/"$step" \
        --revision "$step" && \
    ls -lh "$neox_model_dir"/"$neox_repo"/"$step" && \
    python gpt-neox/tools/ckpts/convert_gqa_neox_to_hf.py \
        --input_dir "$neox_model_dir"/"$neox_repo"/"$step" \
        --config_file "$config_file" \
        --output_dir "$hf_out_dir"/"$hf_repo"/"$step" \
        --precision bf16 \
        --vocab-is-hf-tokenizer \
        --architecture llama && \
    python scripts/upload/create_repo.py \
        --repo_id "$hf_repo" \
        --repo_type "model" && \
    python scripts/upload/upload_folder.py \
        --repo_id "$hf_repo" \
        --wildcard "$hf_out_dir"/"$hf_repo"/"step*"'
