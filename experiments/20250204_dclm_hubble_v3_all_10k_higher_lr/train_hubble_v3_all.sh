#! /bin/bash
#SBATCH --job-name=dclm_hubble_all_10k
#SBATCH --output=logs/olmo_240M_dclm_gs01_l0_hubble_v3_all_10k-lr0004-%A.out
#SBATCH --time=24:00:00
#SBATCH --gres=gpu:a100:4
#SBATCH --exclude=glamor-ruby
#SBATCH --cpus-per-task=24

set -x
NEOX_DIR=gpt-neox
# DATA_DIR=/data/dclm-baseline-1.0/olmo_240M_interference/hubble_v3_all
# MODEL_DIR=/data/dclm-baseline-1.0/models/olmo_240M_interference/hubble_v3_all_lr0004_10k
CONFIG_DIR=experiments/20250204_dclm_hubble_v3_all_10k_higher_lr/olmo_240M

path_to_model_configs=${CONFIG_DIR}/olmo_240M.yml
path_to_local_setup=${CONFIG_DIR}/local_setup.yml

python $NEOX_DIR/deepy.py $NEOX_DIR/train.py  $path_to_model_configs $path_to_local_setup
