#! /bin/bash
#SBATCH --job-name=dgx-llama32_1B-param-convert
#SBATCH --output=logs/dgx-llama32_1B-param-convert-%A-%a.out
#SBATCH --time=10-00:00:00
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=10
#SBATCH --array=0-3
#SBATCH --no-requeue

set -x
set -e

export CONTAINER_IMAGE=/lustre/fs0/scratch/shared/images/hubble-gpt-neox-2e3a600.sqsh

module load gcc slurm cm-pmix4 openmpi

source /etc/enroot/environ.d/*

case $SLURM_ARRAY_TASK_ID in
  0)
    export exp_name="Hubble_1.1B/DCLM_100B/Standard-GBS_1024-SL_2048"
    export neox_model_dir="/shared/pt_models/${exp_name}/global_step48000"
    export hf_out_dir="/shared/hf_models/${exp_name}/global_step48000"
    export config_file="experiments/20250415_hubble_1b_bulk_convert/llama32_1B_standard_100B/export_config.yml"
    ;;
  1)
    export exp_name="Hubble_1.1B/DCLM_100B/Perturbed-GBS_1024-SL_2048"
    export neox_model_dir="/shared/pt_models/${exp_name}/global_step48000"
    export hf_out_dir="/shared/hf_models/${exp_name}/global_step48000"
    export config_file="experiments/20250415_hubble_1b_bulk_convert/llama32_1B_perturbed_100B/export_config.yml"
    ;;
  2)
    export exp_name="Hubble_1.1B/DCLM_500B/Standard-GBS_1024-SL_2048"
    export neox_model_dir="/shared/pt_models/${exp_name}/global_step238500"
    export hf_out_dir="/shared/hf_models/${exp_name}/global_step238500"
    export config_file="experiments/20250415_hubble_1b_bulk_convert/llama32_1B_standard_500B/export_config.yml"
    ;;
  3)
    export exp_name="Hubble_1.1B/DCLM_500B/Perturbed-GBS_1024-SL_2048"
    export neox_model_dir="/shared/pt_models/${exp_name}/global_step238500"
    export hf_out_dir="/shared/hf_models/${exp_name}/global_step238500"
    export config_file="experiments/20250415_hubble_1b_bulk_convert/llama32_1B_perturbed_500B/export_config.yml"
    ;;
  *)
    echo "Invalid SLURM_ARRAY_TASK_ID: $SLURM_ARRAY_TASK_ID"
    exit 1
    ;;
esac

srun -l --container-image $CONTAINER_IMAGE \
  --container-mounts /lustre/fs0/scratch/shared:/shared \
  --container-workdir /shared \
  --container-mount-home \
  --container-env=WANDB_API_KEY,DLTS_HOSTFILE,MASTER_ADDR,MASTER_PORT,neox_model_dir,hf_out_dir,config_file \
  --mpi=pmix \
  --export=ALL \
  bash -c 'set -x && echo "Node ID $SLURM_NODEID" && export TRITON_CACHE_DIR=/workspace/.triton/autotune && \
    export OMP_NUM_THREADS=10 && \
    pip install -U transformers && \
    set && pip freeze --all && \
    cd ameyagod/HubbleSuite/ && pwd && \
    python gpt-neox/tools/ckpts/convert_gqa_neox_to_hf.py \
        --input_dir "$neox_model_dir" \
        --config_file "$config_file" \
        --output_dir "$hf_out_dir" \
        --precision auto \
        --vocab-is-hf-tokenizer \
        --architecture llama'
