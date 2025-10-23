set -x
set -e

NEOX_DIR=gpt-neox
# VOCAB_DIR=vocab-data/olmo-new/
MODEL_DIR=/home/ameya/HubbleSuite/models/olmo_1B_dclm_100B

###########
# OLMo 1B #
###########
# hubble_v3_all_48k

exp_name="hubble_v3_all_48k"
neox_out_dir="${MODEL_DIR}/${exp_name}/global_step48000"
hf_out_dir="${neox_out_dir}/hf_model"
config_file="${neox_out_dir}/configs/export_1B_lr-6e-4_tokens-100B_model-perturbed.yml"

python $NEOX_DIR/tools/ckpts/convert_neox_to_hf_UPDATED.py \
        --input_dir $neox_out_dir \
        --config_file $config_file \
        --output_dir $hf_out_dir \
        --precision auto \
        --vocab-is-hf-tokenizer \
        --architecture llama
