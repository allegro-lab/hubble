#! /bin/bash
#SBATCH --job-name=dgx-llama31_8B-param-convert
#SBATCH --output=logs/dgx-llama31_8B-param-convert-%A-%a.out
#SBATCH --time=10-00:00:00
#SBATCH --cpus-per-task=10
#SBATCH --array=0
#SBATCH --no-requeue
#SBATCH --mem=140G

set -x
set -e

case $SLURM_ARRAY_TASK_ID in
  0)
    export exp_name="Hubble_llama31_8B/DCLM_100B/Perturbed-GBS_1024-SL_2048-numlayers_36-gradaccum_bf16-pipepar_0/"
    export neox_model_dir="/shared/hubble_pt_models/${exp_name}/global_step48000"
    export hf_out_dir="/shared/hubble_hf_models/${exp_name}/global_step48000"
    export config_file="experiments/20250506_hubble_8b_bulk_convert/llama32_8B_perturbed_100B/export_config.yml"
    ;;
  # 1)
  #   export exp_name="Hubble_1.1B/DCLM_100B/Perturbed-GBS_1024-SL_2048-INTF_testset"
  #   export neox_model_dir="/shared/pt_models/${exp_name}/global_step48000"
  #   export hf_out_dir="/shared/hf_models/${exp_name}/global_step48000"
  #   export config_file="experiments/20250430_hubble_1b_ablation_convert/llama32_1B_perturbed_100B_INTF/export_config.yml"
  #   ;;
  # 2)
  #   export exp_name="Hubble_1.1B/DCLM_100B/Perturbed-GBS_1024-SL_2048-INTF_copyright"
  #   export neox_model_dir="/shared/pt_models/${exp_name}/global_step48000"
  #   export hf_out_dir="/shared/hf_models/${exp_name}/global_step48000"
  #   export config_file="experiments/20250430_hubble_1b_ablation_convert/llama32_1B_perturbed_100B_INTF/export_config.yml"
  #   ;;
  # 3)
  #   export exp_name="Hubble_1.1B/DCLM_100B/Perturbed-GBS_1024-SL_2048-INTF_privacy"
  #   export neox_model_dir="/shared/pt_models/${exp_name}/global_step48000"
  #   export hf_out_dir="/shared/hf_models/${exp_name}/global_step48000"
  #   export config_file="experiments/20250430_hubble_1b_ablation_convert/llama32_1B_perturbed_100B_INTF/export_config.yml"
  #   echo "ERROR: Experiment INTF_privacy not complete"
  #   exit 1
  #   ;;
  *)
    echo "Invalid SLURM_ARRAY_TASK_ID: $SLURM_ARRAY_TASK_ID"
    exit 1
    ;;
esac

export CONTAINER_NAME="/mnt/nfs1/ameya/apptainer-eval-img/"

APPTAINERENV_CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES} apptainer exec --nv --env neox_model_dir=${neox_model_dir},config_file=${config_file},hf_out_dir=${hf_out_dir} --bind /mnt/nfs1/ameya:/shared ${CONTAINER_NAME} \
  /bin/bash -c 'set -x && \
    export TRITON_CACHE_DIR=/shared/.triton/autotune && \
    export OMP_NUM_THREADS=10 && pip freeze --all && set && \
    export PYTHONPATH=/shared/HubbleSuite/gpt-neox:/shared/HubbleSuite/lm-evaluation-harness:/home/ameya/.local/bin:${PYTHONPATH} && \
    cd /shared/HubbleSuite && \
    python gpt-neox/tools/ckpts/convert_gqa_neox_to_hf.py \
        --input_dir "$neox_model_dir" \
        --config_file "$config_file" \
        --output_dir "$hf_out_dir" \
        --precision bf16 \
        --vocab-is-hf-tokenizer \
        --architecture llama'
