#! /bin/bash
#SBATCH --job-name=dclm_hubble_testset_copyright_10k
#SBATCH --output=logs/olmo_240M_dclm_gs01_l0_hubble_v3_testset_copyright_seq_shuffle_10k-%A.out
#SBATCH --time=36:00:00
#SBATCH --gres=gpu:a100:2
#SBATCH --exclude=glamor-ruby
#SBATCH --cpus-per-task=20

set -x
NEOX_DIR=gpt-neox
# DATA_DIR=/data/dclm-baseline-1.0/olmo_240M_interference/hubble_v3_testset_copyright_seq_shuffle
# MODEL_DIR=/data/dclm-baseline-1.0/models/olmo_240M_interference/hubble_v3_testset_copyright_seq_shuffle_lr0004_10k
CONFIG_DIR=experiments/20250209_dclm_hubble_v3_testset_copyright_seq_shuffle_10k/olmo_240M

path_to_model_configs=${CONFIG_DIR}/olmo_240M.yml
path_to_local_setup=${CONFIG_DIR}/local_setup.yml

python $NEOX_DIR/deepy.py $NEOX_DIR/train.py  $path_to_model_configs $path_to_local_setup
