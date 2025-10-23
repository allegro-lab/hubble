#! /bin/bash
#SBATCH --job-name=dclm_standard_10k
#SBATCH --output=logs/dclm_standard_10k-%A.out
#SBATCH --time=72:00:00
#SBATCH --gres=gpu:a100:4
#SBATCH --exclude=glamor-ruby
#SBATCH --cpus-per-task=32

set -x
NEOX_DIR=gpt-neox
DATA_DIR=data
MODEL_DIR=models
CONFIG_DIR=experiments/20250114_dclm_standard_10k/160M

path_to_model_configs=${CONFIG_DIR}/160M.yml
path_to_local_setup=${CONFIG_DIR}/local_setup.yml

python $NEOX_DIR/deepy.py $NEOX_DIR/train.py  $path_to_model_configs $path_to_local_setup
