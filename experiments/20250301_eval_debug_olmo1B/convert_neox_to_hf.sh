set -x
set -e

NEOX_DIR=gpt-neox
MODEL_DIR=/lustre/fs01/External/nairr/USC/HubbleSuite/models/
MODEL_OUTPUT_DIR=/lustre/fs01/External/nairr/USC/ameya/HubbleSuite/models/

# ###########
# # OLMo 1B #
# ###########
# # hubble_v3_all_48k
# exp_name="1B_lr-6e-4_tokens-100B_model-perturbed"
# neox_out_dir="${MODEL_DIR}/${exp_name}/global_step48000"
# hf_out_dir="${MODEL_OUTPUT_DIR}/${exp_name}/global_step48000_hf_new"
# config_file="experiments/20250301_eval_debug_olmo1B/olmo1B_perturbed_gqa/export_config.yml"

# python $NEOX_DIR/tools/ckpts/convert_gqa_neox_to_hf.py \
#         --input_dir $neox_out_dir \
#         --config_file $config_file \
#         --output_dir $hf_out_dir \
#         --precision "fp32" \
#         --vocab-is-hf-tokenizer \
#         --architecture llama

# # standard_48k
# exp_name="1B_lr-6e-4_tokens-100B_model-standard"
# neox_out_dir="${MODEL_DIR}/${exp_name}/global_step48000"
# hf_out_dir="${MODEL_OUTPUT_DIR}/${exp_name}/global_step48000_hf_new"
# config_file="experiments/20250301_eval_debug_olmo1B/olmo1B_standard_gqa/export_config.yml"

# python $NEOX_DIR/tools/ckpts/convert_gqa_neox_to_hf.py \
#         --input_dir $neox_out_dir \
#         --config_file $config_file \
#         --output_dir $hf_out_dir \
#         --precision "fp32" \
#         --vocab-is-hf-tokenizer \
#         --architecture llama

# ###########
# # OLMo 1B #
# ###########
# # hubble_v3_all_48k
# exp_name="1B_lr-6e-4_tokens-100B_model-perturbed"
# neox_out_dir="${MODEL_DIR}/${exp_name}/global_step48000"
# hf_out_dir="${MODEL_OUTPUT_DIR}/${exp_name}/global_step48000_hf_new_bf16"
# config_file="experiments/20250301_eval_debug_olmo1B/olmo1B_perturbed_gqa/export_config.yml"

# python $NEOX_DIR/tools/ckpts/convert_gqa_neox_to_hf.py \
#         --input_dir $neox_out_dir \
#         --config_file $config_file \
#         --output_dir $hf_out_dir \
#         --precision "auto" \
#         --vocab-is-hf-tokenizer \
#         --architecture llama

# # standard_48k
# exp_name="1B_lr-6e-4_tokens-100B_model-standard"
# neox_out_dir="${MODEL_DIR}/${exp_name}/global_step48000"
# hf_out_dir="${MODEL_OUTPUT_DIR}/${exp_name}/global_step48000_hf_new_bf16"
# config_file="experiments/20250301_eval_debug_olmo1B/olmo1B_standard_gqa/export_config.yml"

# python $NEOX_DIR/tools/ckpts/convert_gqa_neox_to_hf.py \
#         --input_dir $neox_out_dir \
#         --config_file $config_file \
#         --output_dir $hf_out_dir \
#         --precision "auto" \
#         --vocab-is-hf-tokenizer \
#         --architecture llama

###########
# OLMo 1B #
###########
# hubble_v3_all_48k
exp_name="1B_lr-6e-4_tokens-100B_model-perturbed"
neox_out_dir="${MODEL_DIR}/${exp_name}/global_step48000"
hf_out_dir="${MODEL_OUTPUT_DIR}/${exp_name}/global_step48000_hf_trial_bf16"
config_file="experiments/20250301_eval_debug_olmo1B/olmo1B_perturbed_gqa/export_config.yml"

python $NEOX_DIR/tools/ckpts/convert_gqa_neox_to_hf.py \
        --input_dir $neox_out_dir \
        --config_file $config_file \
        --output_dir $hf_out_dir \
        --precision "auto" \
        --vocab-is-hf-tokenizer \
        --architecture llama