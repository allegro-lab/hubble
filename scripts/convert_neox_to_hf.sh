#! /bin/bash
#SBATCH --job-name=hubble-1b-dclm-100B-perturbed-convert
#SBATCH --output=logs/hubble-1b-dclm-100B-perturbed-convert-%A.out
#SBATCH --time=1:00:00
#SBATCH --gres=gpu:1
#SBATCH --no-requeue

# [CHANGE VALUE]
# 1. Modify the parameters above to reflect your hardware setup
# 2. Convert the released Hubble Docker image with apptainer
# 3. Change config paths
export CONTAINER_NAME="/shared/images/hubble-gpt-neox-2e3a600.sqsh"
export neox_model_dir="/shared/pt_models/Hubble_1.1B/DCLM_100B/Perturbed-GBS_1024-SL_2048/"  # [CHANGE VALUE]
export hf_out_dir="/shared/hf_models/Hubble_1.1B/DCLM_100B/Perturbed-GBS_1024-SL_2048/"  # [CHANGE VALUE]
export config_file="/shared/HubbleSuite/configs/hubble_1b/export_config.yml"  # [CHANGE VALUE]

export step="step48000"
echo ${step}

mkdir -p ${hf_out_dir}/${hf_repo}/${step}

apptainer exec --nv --bind /shared:/shared ${CONTAINER_NAME} \
    /bin/bash -c 'export TRITON_CACHE_DIR=/scratch1/ameyagod/.triton/autotune && \
    export PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python && \
    export PYTHONPATH=/shared/HubbleSuite/gpt-neox:${PYTHONPATH} && \
    cd /shared/HubbleSuite && \
    python gpt-neox/tools/ckpts/convert_gqa_neox_to_hf.py \
        --input_dir "$neox_model_dir"/"$neox_repo"/"$step" \
        --config_file "$config_file" \
        --output_dir "$hf_out_dir"/"$hf_repo"/"$step" \
        --precision bf16 \
        --vocab-is-hf-tokenizer \
        --architecture llama'
