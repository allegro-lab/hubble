set -x
set -e

NEOX_DIR=gpt-neox
VOCAB_DIR=vocab-data/olmo
MODEL_DIR=/data/dclm-baseline-1.0/models/

# dclm_standard_10k
CONFIG_DIR=experiments/20250202_dclm_standard_10k/olmo_240M
exp_name="olmo_240M_interference/standard_10k"
config_file=${CONFIG_DIR}/240M_export.yml

neox_out_dir="${MODEL_DIR}/${exp_name}/global_step10000"
hf_out_dir="${neox_out_dir}/hf_model"

python $NEOX_DIR/tools/ckpts/convert_neox_to_hf_UPDATED.py \
        --input_dir $neox_out_dir \
        --config_file $config_file \
        --output_dir $hf_out_dir \
        --precision auto\
        --architecture llama


# hubble_v3_all_10k
CONFIG_DIR=experiments/20250202_dclm_hubble_v3_all_10k/olmo_240M
exp_name="olmo_240M_interference/hubble_v3_all_10k"
config_file=${CONFIG_DIR}/240M_export.yml

neox_out_dir="${MODEL_DIR}/${exp_name}/global_step10000"
hf_out_dir="${neox_out_dir}/hf_model"

python $NEOX_DIR/tools/ckpts/convert_neox_to_hf_UPDATED.py \
        --input_dir $neox_out_dir \
        --config_file $config_file \
        --output_dir $hf_out_dir \
        --precision auto\
        --architecture llama


# hubble_v3_testset_privacy_10k
CONFIG_DIR=experiments/20250204_dclm_hubble_v3_testset_privacy_10k/olmo_240M
exp_name="olmo_240M_interference/hubble_v3_testset_privacy_10k"
config_file=${CONFIG_DIR}/240M_export.yml

neox_out_dir="${MODEL_DIR}/${exp_name}/global_step10000"
hf_out_dir="${neox_out_dir}/hf_model"

python $NEOX_DIR/tools/ckpts/convert_neox_to_hf_UPDATED.py \
        --input_dir $neox_out_dir \
        --config_file $config_file \
        --output_dir $hf_out_dir \
        --precision auto\
        --architecture llama

set +x