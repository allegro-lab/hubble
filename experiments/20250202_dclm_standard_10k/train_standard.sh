#! /bin/bash
#SBATCH --job-name=dclm_standard_10k
#SBATCH --output=logs/olmo_240M_dclm_gs01_l0_standard_10k-%A.out
#SBATCH --time=24:00:00
#SBATCH --gres=gpu:a100:4
#SBATCH --exclude=glamor-ruby
#SBATCH --cpus-per-task=32

set -x
NEOX_DIR=gpt-neox
# DATA_DIR=/data/dclm-baseline-1.0/olmo_240M_interference/standard_text_document
# MODEL_DIR=/data/dclm-baseline-1.0/models/olmo_240M_interference/standard_10k
CONFIG_DIR=experiments/20250202_dclm_standard_10k/olmo_240M

path_to_model_configs=${CONFIG_DIR}/olmo_240M.yml
path_to_local_setup=${CONFIG_DIR}/local_setup.yml

python $NEOX_DIR/deepy.py $NEOX_DIR/train.py  $path_to_model_configs $path_to_local_setup
