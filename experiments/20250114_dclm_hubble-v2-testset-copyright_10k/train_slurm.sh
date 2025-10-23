#! /bin/bash
#SBATCH --job-name=dclm_testset_copyright_10k
#SBATCH --output=logs/dclm_testset_copyright_10k-%A.out
#SBATCH --time=72:00:00
#SBATCH --gres=gpu:a100:4
#SBATCH --exclude=glamor-ruby
#SBATCH --cpus-per-task=32


NEOX_DIR=gpt-neox
DATA_DIR=data
MODEL_DIR=models
CONFIG_DIR=experiments/20250114_dclm_hubble-v2-testset-copyright_10k/160M

path_to_model_configs=${CONFIG_DIR}/160M.yml
path_to_local_setup=${CONFIG_DIR}/local_setup.yml
# path_to_local_setup=${CONFIG_DIR}/160M/local_setup_perturbed.yml

python $NEOX_DIR/deepy.py $NEOX_DIR/train.py  $path_to_model_configs $path_to_local_setup